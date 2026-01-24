#!/usr/bin/env python3
"""
Live pipeline for CryptoScreener-X.

Connects to Binance WebSocket streams and processes market data through
the full pipeline: BinanceStreamManager → StreamRouter → FeatureEngine →
BaselineRunner → Ranker → Alerter.

Usage:
    python -m scripts.run_live --symbols BTCUSDT,ETHUSDT
    python -m scripts.run_live --top 50  # Top 50 by volume
    python -m scripts.run_live --dry-run  # No LLM calls

Per DEC-005:
- LLM is disabled by default in live mode (--llm to enable)
- LLM calls are rate-limited per symbol (60s cooldown)
- LLM failures do not break the pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from cryptoscreener.contracts.events import (
        FeatureSnapshot,
        PredictionSnapshot,
        RankEvent,
    )

from cryptoscreener.alerting.alerter import (
    Alerter,
    AlerterConfig,
    ExplainLLMProtocol,
)
from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager
from cryptoscreener.connectors.binance.types import ConnectorConfig
from cryptoscreener.features.engine import FeatureEngine, FeatureEngineConfig
from cryptoscreener.model_runner import BaselineRunner, ModelRunnerConfig
from cryptoscreener.ranker.ranker import Ranker, RankerConfig
from cryptoscreener.stream_router.router import StreamRouter, StreamRouterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class LivePipelineConfig:
    """Configuration for live pipeline."""

    # Symbols to track (empty = use --top)
    symbols: list[str] = field(default_factory=list)

    # Top N symbols by volume (if symbols is empty)
    top_n: int = 50

    # Snapshot cadence in milliseconds
    snapshot_cadence_ms: int = 1000

    # Whether to enable LLM explanations
    llm_enabled: bool = False

    # Output file for RankEvents (None = stdout)
    output_file: Path | None = None

    # Dry-run mode (no actual connections)
    dry_run: bool = False

    # Verbose logging
    verbose: bool = False


@dataclass
class PipelineMetrics:
    """Aggregated metrics for live pipeline."""

    # Timing
    start_ts: int = 0
    last_snapshot_ts: int = 0

    # Counts
    market_events_received: int = 0
    snapshots_emitted: int = 0
    predictions_made: int = 0
    rank_events_emitted: int = 0
    alerts_emitted: int = 0

    # Latencies (ms)
    max_event_latency_ms: int = 0
    total_event_latency_ms: int = 0

    # LLM
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_failures: int = 0

    def avg_event_latency_ms(self) -> float:
        """Get average event latency."""
        if self.market_events_received == 0:
            return 0.0
        return self.total_event_latency_ms / self.market_events_received

    def events_per_second(self) -> float:
        """Get events per second."""
        if self.start_ts == 0 or self.last_snapshot_ts == 0:
            return 0.0
        elapsed_s = (self.last_snapshot_ts - self.start_ts) / 1000
        if elapsed_s <= 0:
            return 0.0
        return self.market_events_received / elapsed_s


class LivePipeline:
    """
    Live pipeline connecting all components.

    Data flow:
        BinanceStreamManager
            ↓ (MarketEvent)
        StreamRouter
            ↓ (validates, filters)
        FeatureEngine
            ↓ (FeatureSnapshot on cadence)
        BaselineRunner.predict()
            ↓ (PredictionSnapshot)
        Ranker.update()
            ↓ (RankEvent: SYMBOL_ENTER/EXIT)
        Alerter.process_prediction()
            ↓ (RankEvent: alerts)
        Output
    """

    def __init__(
        self,
        config: LivePipelineConfig,
        explainer: ExplainLLMProtocol | None = None,
    ) -> None:
        """
        Initialize live pipeline.

        Args:
            config: Pipeline configuration.
            explainer: Optional LLM explainer for alert text.
        """
        self._config = config
        self._metrics = PipelineMetrics()
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Initialize components
        self._stream_manager = BinanceStreamManager(
            config=ConnectorConfig(),
            on_event=None,  # We'll use async iteration
        )

        self._feature_engine = FeatureEngine(
            config=FeatureEngineConfig(
                snapshot_cadence_ms=config.snapshot_cadence_ms,
                max_symbols=500,
            )
        )

        self._router = StreamRouter(
            feature_engine=self._feature_engine,
            config=StreamRouterConfig(
                stale_threshold_ms=5000,
                drop_stale=False,
            ),
        )

        self._model_runner = BaselineRunner(
            config=ModelRunnerConfig()
        )

        self._ranker = Ranker(
            config=RankerConfig(
                top_k=20,
                enter_ms=1500,
                exit_ms=3000,
            )
        )

        self._alerter = Alerter(
            config=AlerterConfig(
                cooldown_ms=120_000,
                stable_ms=2000,
                llm_enabled=config.llm_enabled,
            ),
            explainer=explainer if config.llm_enabled else None,
        )

        # Predictions storage for ranker
        self._predictions: dict[str, PredictionSnapshot] = {}

        # Output file handle
        self._output_file: Path | None = config.output_file
        self._output_handle: TextIO | None = None

    @property
    def metrics(self) -> PipelineMetrics:
        """Get pipeline metrics."""
        return self._metrics

    async def _on_snapshot(self, snapshot: FeatureSnapshot) -> None:
        """
        Handle emitted FeatureSnapshot.

        Args:
            snapshot: FeatureSnapshot from engine.
        """
        self._metrics.snapshots_emitted += 1
        self._metrics.last_snapshot_ts = snapshot.ts

        # Run prediction
        prediction = self._model_runner.predict(snapshot)
        self._metrics.predictions_made += 1

        # Store for ranker batch
        self._predictions[snapshot.symbol] = prediction

    async def _process_predictions(self, ts: int) -> list[RankEvent]:
        """
        Process predictions through ranker and alerter.

        Args:
            ts: Current timestamp.

        Returns:
            List of emitted RankEvents.
        """
        all_events: list[RankEvent] = []

        if not self._predictions:
            return all_events

        # Ranker: update with all predictions
        rank_events = self._ranker.update(self._predictions, ts)
        self._metrics.rank_events_emitted += len(rank_events)
        all_events.extend(rank_events)

        # Alerter: process each prediction
        for symbol, prediction in self._predictions.items():
            state = self._ranker.get_state(symbol)
            # RankEvent.rank must be >= 0; use 0 as default for unranked symbols
            rank = max(0, state.rank) if state else 0
            score = state.score if state else 0.0

            alert_events = self._alerter.process_prediction(
                prediction=prediction,
                ts=ts,
                rank=rank,
                score=score,
            )
            self._metrics.alerts_emitted += len(alert_events)
            all_events.extend(alert_events)

        # Clear predictions for next cycle
        self._predictions.clear()

        return all_events

    def _write_event(self, event: RankEvent) -> None:
        """Write RankEvent to output."""
        line = event.to_json().decode() + "\n"
        if self._output_handle:
            self._output_handle.write(line)
            self._output_handle.flush()
        else:
            sys.stdout.write(line)
            sys.stdout.flush()

    async def _main_loop(self) -> None:
        """Main processing loop."""
        logger.info("Starting main loop")

        last_process_ts = 0

        async for event in self._stream_manager.events():
            if not self._running:
                break

            # Track metrics
            self._metrics.market_events_received += 1
            latency = event.recv_ts - event.ts
            self._metrics.total_event_latency_ms += latency
            self._metrics.max_event_latency_ms = max(
                self._metrics.max_event_latency_ms, latency
            )

            # Route to feature engine
            await self._router.route(event)

            # Process on cadence
            current_ts = int(time.time() * 1000)
            if current_ts - last_process_ts >= self._config.snapshot_cadence_ms:
                # Emit snapshots
                snapshots = await self._feature_engine.emit_snapshots(current_ts)

                # Process each snapshot
                for snapshot in snapshots:
                    await self._on_snapshot(snapshot)

                # Process predictions through ranker/alerter
                rank_events = await self._process_predictions(current_ts)

                # Write events
                for rank_event in rank_events:
                    self._write_event(rank_event)

                last_process_ts = current_ts

    async def start(self) -> None:
        """Start the live pipeline."""
        if self._running:
            return

        logger.info("Starting live pipeline")
        self._running = True
        self._metrics.start_ts = int(time.time() * 1000)

        # Open output file if specified
        if self._output_file:
            self._output_handle = self._output_file.open("w")

        # Start stream manager
        await self._stream_manager.start()

        # Bootstrap symbols
        if self._config.symbols:
            symbols = self._config.symbols
        else:
            # Get top N by volume
            symbol_infos = await self._stream_manager.bootstrap()
            # Sort by quote volume (approximation: just take first N)
            symbols = [s.symbol for s in symbol_infos[: self._config.top_n]]

        logger.info("Subscribing to %d symbols", len(symbols))

        # Set symbol filter on router
        self._router.set_symbol_filter(set(symbols))

        # Subscribe to streams
        await self._stream_manager.subscribe(symbols)

        # Register snapshot callback
        self._feature_engine.on_snapshot(self._on_snapshot)

        # Run main loop
        try:
            await self._main_loop()
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")

    async def stop(self) -> None:
        """Stop the live pipeline gracefully."""
        if not self._running:
            return

        logger.info("Stopping live pipeline")
        self._running = False

        # Stop components
        await self._feature_engine.stop()
        await self._stream_manager.stop()

        # Close output file
        if self._output_handle:
            self._output_handle.close()
            self._output_handle = None

        # Log final metrics
        self._log_metrics()

        logger.info("Live pipeline stopped")

    def _log_metrics(self) -> None:
        """Log pipeline metrics summary."""
        m = self._metrics
        logger.info("=" * 60)
        logger.info("PIPELINE METRICS SUMMARY")
        logger.info("=" * 60)
        logger.info("Runtime: %.1f seconds", (m.last_snapshot_ts - m.start_ts) / 1000)
        logger.info("Market events: %d (%.1f/s)", m.market_events_received, m.events_per_second())
        logger.info("Snapshots: %d", m.snapshots_emitted)
        logger.info("Predictions: %d", m.predictions_made)
        logger.info("Rank events: %d", m.rank_events_emitted)
        logger.info("Alerts: %d", m.alerts_emitted)
        logger.info("Latency: avg=%.1fms max=%dms", m.avg_event_latency_ms(), m.max_event_latency_ms)

        # LLM metrics
        alerter_m = self._alerter.metrics
        logger.info("LLM calls: %d (cache hits: %d, failures: %d)",
                    alerter_m.llm_calls, alerter_m.llm_cache_hits, alerter_m.llm_failures)

        # Router metrics
        router_m = self._router.metrics
        logger.info("Router: %d events, %d stale, %d late",
                    router_m.events_routed, router_m.stale_events, router_m.late_events)

        # Ranker metrics
        ranker_m = self._ranker.metrics
        logger.info("Ranker: %d updates, %d enters, %d exits",
                    ranker_m.updates_processed, ranker_m.enter_events, ranker_m.exit_events)
        logger.info("=" * 60)

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()


def setup_signal_handlers(pipeline: LivePipeline, loop: asyncio.AbstractEventLoop) -> None:
    """
    Setup signal handlers for graceful shutdown.

    Args:
        pipeline: LivePipeline instance.
        loop: Event loop.
    """

    def signal_handler(sig: int, frame: object) -> None:
        logger.info("Received signal %s", signal.Signals(sig).name)
        pipeline.request_shutdown()
        # Schedule stop
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(pipeline.stop())
        )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def run_pipeline(config: LivePipelineConfig) -> int:
    """
    Run the live pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        Exit code (0 = success).
    """
    # Setup LLM explainer if enabled
    explainer: ExplainLLMProtocol | None = None
    if config.llm_enabled:
        try:
            from cryptoscreener.explain_llm.explainer import AnthropicExplainer
            explainer = AnthropicExplainer()
            logger.info("LLM explainer enabled (Anthropic)")
        except ImportError:
            logger.warning("anthropic package not installed, LLM disabled")
        except Exception as e:
            logger.warning("Failed to initialize LLM: %s", e)

    pipeline = LivePipeline(config=config, explainer=explainer)

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    setup_signal_handlers(pipeline, loop)

    try:
        await pipeline.start()
        return 0
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 1
    finally:
        await pipeline.stop()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run live CryptoScreener pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of top symbols by volume (default: 50)",
    )
    parser.add_argument(
        "--cadence",
        type=int,
        default=1000,
        help="Snapshot cadence in ms (default: 1000)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM explanations (disabled by default)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for RankEvents (default: stdout)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode (no actual connections)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse symbols
    symbols: list[str] = []
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    config = LivePipelineConfig(
        symbols=symbols,
        top_n=args.top,
        snapshot_cadence_ms=args.cadence,
        llm_enabled=args.llm,
        output_file=args.output,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    logger.info("Starting CryptoScreener Live Pipeline")
    logger.info("  Symbols: %s", symbols if symbols else f"top {config.top_n}")
    logger.info("  Cadence: %dms", config.snapshot_cadence_ms)
    logger.info("  LLM: %s", "enabled" if config.llm_enabled else "disabled")

    return asyncio.run(run_pipeline(config))


if __name__ == "__main__":
    sys.exit(main())
