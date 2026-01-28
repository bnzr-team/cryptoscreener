#!/usr/bin/env python3
"""
Live pipeline for CryptoScreener-X.

Connects to Binance WebSocket streams and processes market data through
the full pipeline: BinanceStreamManager → StreamRouter → FeatureEngine →
BaselineRunner → Ranker → Alerter.

Usage:
    python -m scripts.run_live --symbols BTCUSDT,ETHUSDT
    python -m scripts.run_live --top 50  # Top 50 by volume (uses REST bootstrap)
    python -m scripts.run_live --duration-s 60  # Run for 60 seconds

Per DEC-005:
- LLM is disabled by default in live mode (--llm to enable)
- LLM calls are rate-limited per symbol (60s cooldown)
- LLM failures do not break the pipeline

Per DEC-006:
- --top N mode uses one-time REST call to fetch symbol list at startup
- All data streaming is via WebSocket (no REST polling)
- Graceful shutdown via SIGINT/SIGTERM or --duration-s timeout
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import resource
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

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
from cryptoscreener.connectors.binance.types import ConnectorConfig, ConnectorMetrics
from cryptoscreener.connectors.exporter import MetricsExporter
from cryptoscreener.connectors.metrics_server import start_metrics_server, stop_metrics_server
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


# DEC-030: Env var names that must never be logged
REDACTED_ENV_VARS = frozenset({"ANTHROPIC_API_KEY", "BINANCE_API_KEY", "BINANCE_SECRET_KEY"})


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

    # Duration in seconds (None = run until SIGINT/SIGTERM)
    duration_s: int | None = None

    # Verbose logging
    verbose: bool = False

    # Metrics server port (0 = disabled)
    metrics_port: int = 9090

    # DEC-027: Fault injection
    fault_drop_ws_every_s: int | None = None
    fault_slow_consumer_ms: int | None = None

    # DEC-027: Soak summary output
    summary_json: Path | None = None

    # DEC-030: Rollout knobs
    dry_run: bool = False
    graceful_timeout_s: int = 10

    # DEC-030: Readiness staleness window (seconds)
    readiness_staleness_s: int = 30

    # DEC-032: Override WebSocket base URL (for offline soak with FakeWSServer)
    ws_url: str | None = None

    def __post_init__(self) -> None:
        """DEC-030: Validate config values at construction time."""
        import os
        import re

        if not 0 <= self.metrics_port <= 65535:
            msg = f"metrics_port must be 0..65535, got {self.metrics_port}"
            raise ValueError(msg)
        if not 1 <= self.top_n <= 2000:
            msg = f"top_n must be 1..2000, got {self.top_n}"
            raise ValueError(msg)
        if not 100 <= self.snapshot_cadence_ms <= 60000:
            msg = f"snapshot_cadence_ms must be 100..60000, got {self.snapshot_cadence_ms}"
            raise ValueError(msg)
        if self.duration_s is not None and self.duration_s <= 0:
            msg = f"duration_s must be > 0, got {self.duration_s}"
            raise ValueError(msg)
        if self.graceful_timeout_s < 0:
            msg = f"graceful_timeout_s must be >= 0, got {self.graceful_timeout_s}"
            raise ValueError(msg)
        # Fault flags require ALLOW_FAULTS=1 or ENV=dev
        has_faults = self.fault_drop_ws_every_s is not None or self.fault_slow_consumer_ms is not None
        if has_faults:
            allow = os.environ.get("ALLOW_FAULTS") == "1" or os.environ.get("ENV") == "dev"
            if not allow:
                msg = (
                    "Fault injection flags require ALLOW_FAULTS=1 or ENV=dev. "
                    "Set environment variable to enable."
                )
                raise ValueError(msg)
        # Symbol format validation
        symbol_re = re.compile(r"^[A-Z0-9]+$")
        for s in self.symbols:
            if not symbol_re.match(s):
                msg = f"Invalid symbol format: {s!r} (must be uppercase alphanumeric)"
                raise ValueError(msg)


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

    # DEC-028: Backpressure instrumentation
    max_event_queue_depth: int = 0
    max_snapshot_queue_depth: int = 0
    max_tick_drift_ms: int = 0
    max_rss_mb: float = 0.0
    events_dropped: int = 0
    snapshots_dropped: int = 0

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


@dataclass
class SoakSummary:
    """DEC-027: Summary written to --summary-json at pipeline exit."""

    duration_s: float = 0.0
    market_events: int = 0
    events_per_second: float = 0.0
    snapshots: int = 0
    predictions: int = 0
    alerts: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: int = 0
    ws_disconnects: int = 0
    ws_reconnect_attempts: int = 0
    ws_reconnects_denied: int = 0
    ws_ping_timeouts: int = 0
    cb_transitions_to_open: int = 0
    max_reconnect_rate_per_min: float = 0.0
    # DEC-028: Backpressure fields
    max_event_queue_depth: int = 0
    max_snapshot_queue_depth: int = 0
    max_tick_drift_ms: int = 0
    max_rss_mb: float = 0.0
    events_dropped: int = 0
    snapshots_dropped: int = 0
    faults_injected: dict[str, Any] = field(default_factory=dict)
    passed: bool = False


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
        metrics_exporter: MetricsExporter | None = None,
    ) -> None:
        """
        Initialize live pipeline.

        Args:
            config: Pipeline configuration.
            explainer: Optional LLM explainer for alert text.
            metrics_exporter: Optional Prometheus metrics exporter (DEC-026).
        """
        self._config = config
        self._metrics = PipelineMetrics()
        self._running = False
        self._metrics_exporter = metrics_exporter

        # DEC-027: Reconnect rate tracking (deque of (ts_s, cumulative_attempts))
        # Pruned to last 10 minutes to bound memory and O(n) computation.
        self._reconnect_samples: deque[tuple[float, int]] = deque()
        self._last_fault_drop_ts: float = 0.0
        self._start_monotonic: float = 0.0
        self._stop_monotonic: float = 0.0
        self._final_connector_metrics: ConnectorMetrics | None = None
        self._final_shard_count: int = 1
        self._last_tick_monotonic: float = 0.0  # DEC-028: for tick drift

        # Initialize components
        connector_config = ConnectorConfig()
        if config.ws_url is not None:
            connector_config = ConnectorConfig(base_ws_url=config.ws_url)
        self._stream_manager = BinanceStreamManager(
            config=connector_config,
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

        self._model_runner = BaselineRunner(config=ModelRunnerConfig())

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

    def get_health_info(self) -> dict[str, Any]:
        """DEC-029: Return pipeline health info for /healthz endpoint."""
        uptime_s = round(time.monotonic() - self._start_monotonic, 1) if self._start_monotonic else 0.0
        # Check if any shard is connected
        ws_connected = False
        if self._stream_manager:
            cm = self._stream_manager.get_metrics()
            ws_connected = cm.active_shards > 0
        return {
            "status": "ok" if self._running else "stopped",
            "uptime_s": uptime_s,
            "ws_connected": ws_connected,
            "last_event_ts": self._metrics.last_snapshot_ts,
        }

    def get_ready_info(self) -> tuple[bool, dict[str, Any]]:
        """DEC-030: Return readiness status for /readyz endpoint.

        Ready when: running AND ws_connected AND last_event_ts within staleness window.
        Returns (is_ready, info_dict).
        """
        if not self._running:
            return False, {"ready": False, "reason": "pipeline not running"}

        ws_connected = False
        if self._stream_manager:
            cm = self._stream_manager.get_metrics()
            ws_connected = cm.active_shards > 0

        if not ws_connected:
            return False, {"ready": False, "ws_connected": False, "reason": "no WS shards connected"}

        last_ts = self._metrics.last_snapshot_ts
        now_ms = int(time.time() * 1000)
        staleness_ms = self._config.readiness_staleness_s * 1000
        last_event_age_s = round((now_ms - last_ts) / 1000, 1) if last_ts > 0 else -1.0

        if last_ts == 0:
            return False, {
                "ready": False,
                "ws_connected": True,
                "last_event_age_s": -1.0,
                "reason": "no events received yet",
            }

        if (now_ms - last_ts) > staleness_ms:
            return False, {
                "ready": False,
                "ws_connected": True,
                "last_event_age_s": last_event_age_s,
                "reason": f"stale: last event {last_event_age_s}s ago",
            }

        return True, {
            "ready": True,
            "ws_connected": True,
            "last_event_age_s": last_event_age_s,
        }

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

        # Calculate deadline if duration specified
        deadline_ts: int | None = None
        if self._config.duration_s is not None:
            deadline_ts = self._metrics.start_ts + (self._config.duration_s * 1000)
            logger.info("Pipeline will stop after %d seconds", self._config.duration_s)

        async for event in self._stream_manager.events():
            if not self._running:
                break

            # Check duration deadline
            current_ts = int(time.time() * 1000)
            if deadline_ts is not None and current_ts >= deadline_ts:
                logger.info("Duration limit reached, stopping pipeline")
                break

            # Track metrics
            self._metrics.market_events_received += 1
            latency = event.recv_ts - event.ts
            self._metrics.total_event_latency_ms += latency
            self._metrics.max_event_latency_ms = max(self._metrics.max_event_latency_ms, latency)

            # Route to feature engine
            await self._router.route(event)

            # Process on cadence
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

                # DEC-028: Tick drift, queue depth, RSS sampling
                now_s = time.monotonic()
                if self._last_tick_monotonic > 0:
                    expected_s = self._config.snapshot_cadence_ms / 1000
                    drift_ms = max(
                        0, int(((now_s - self._last_tick_monotonic) - expected_s) * 1000)
                    )
                    self._metrics.max_tick_drift_ms = max(self._metrics.max_tick_drift_ms, drift_ms)
                self._last_tick_monotonic = now_s

                eq_depth = self._stream_manager.event_queue_depth
                self._metrics.max_event_queue_depth = max(
                    self._metrics.max_event_queue_depth, eq_depth
                )
                sq_depth = self._feature_engine.snapshot_queue_depth
                self._metrics.max_snapshot_queue_depth = max(
                    self._metrics.max_snapshot_queue_depth, sq_depth
                )

                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                self._metrics.max_rss_mb = max(self._metrics.max_rss_mb, rss_mb)
                self._metrics.events_dropped = self._stream_manager.events_dropped
                self._metrics.snapshots_dropped = self._feature_engine.snapshots_dropped

                # DEC-026: Push infrastructure metrics to Prometheus
                cm = self._stream_manager.get_metrics()
                if self._metrics_exporter is not None:
                    self._metrics_exporter.update(
                        governor=self._stream_manager.governor,
                        circuit_breaker=self._stream_manager.circuit_breaker,
                        connector_metrics=cm,
                        pipeline_metrics={
                            "event_queue_depth": eq_depth,
                            "snapshot_queue_depth": sq_depth,
                            "tick_drift_ms": self._metrics.max_tick_drift_ms,
                            "rss_mb": rss_mb,
                            "events_dropped": self._metrics.events_dropped,
                            "snapshots_dropped": self._metrics.snapshots_dropped,
                        },
                    )

                # DEC-027: Sample reconnect rate for soak summary
                self._reconnect_samples.append((now_s, cm.total_reconnect_attempts))
                # Prune samples older than 10 minutes
                cutoff = now_s - 600.0
                while self._reconnect_samples and self._reconnect_samples[0][0] < cutoff:
                    self._reconnect_samples.popleft()

                # DEC-027: Fault injection — periodic WS disconnect
                if (
                    self._config.fault_drop_ws_every_s is not None
                    and now_s - self._last_fault_drop_ts >= self._config.fault_drop_ws_every_s
                ):
                    self._last_fault_drop_ts = now_s
                    logger.warning("DEC-027 FAULT: forcing WS disconnect")
                    await self._stream_manager.force_disconnect()

            # DEC-027: Fault injection — slow consumer
            if self._config.fault_slow_consumer_ms is not None:
                await asyncio.sleep(self._config.fault_slow_consumer_ms / 1000)

    async def start(self) -> None:
        """Start the live pipeline."""
        if self._running:
            return

        logger.info("Starting live pipeline")
        self._running = True
        self._metrics.start_ts = int(time.time() * 1000)
        self._start_monotonic = time.monotonic()

        # Open output file if specified
        if self._output_file:
            self._output_handle = self._output_file.open("w")

        # Start stream manager
        await self._stream_manager.start()

        # Bootstrap symbols
        if self._config.symbols:
            symbols = self._config.symbols
        else:
            # Get top N by 24h quote volume (sorted descending)
            symbols = await self._stream_manager.get_top_symbols_by_volume(self._config.top_n)

        logger.info("Subscribing to %d symbols", len(symbols))

        # Set symbol filter on router
        self._router.set_symbol_filter(set(symbols))

        # Subscribe to streams
        await self._stream_manager.subscribe(symbols)

        # NOTE: We do NOT register on_snapshot callback here because
        # we manually call emit_snapshots() in the main loop and process
        # the returned snapshots directly. Registering a callback would
        # cause double processing.

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

        # DEC-027: Snapshot connector metrics BEFORE stopping (stop clears shards)
        self._final_connector_metrics = self._stream_manager.get_metrics()
        self._final_shard_count = max(1, len(self._stream_manager._shards))

        # Stop components
        await self._feature_engine.stop()
        await self._stream_manager.stop()

        # Close output file
        if self._output_handle:
            self._output_handle.close()
            self._output_handle = None

        self._stop_monotonic = time.monotonic()

        # Log final metrics
        self._log_metrics()

        # DEC-027: Write soak summary JSON
        if self._config.summary_json is not None:
            summary = self._build_soak_summary()
            self._config.summary_json.write_text(json.dumps(asdict(summary), indent=2))
            logger.info("Soak summary written to %s", self._config.summary_json)

        logger.info("Live pipeline stopped")

    def _build_soak_summary(self) -> SoakSummary:
        """DEC-027: Build soak summary from pipeline and connector metrics."""
        m = self._metrics
        cm = self._final_connector_metrics or ConnectorMetrics()
        cb = self._stream_manager.circuit_breaker

        # Use monotonic clock for accurate wall-time duration
        duration_s = self._stop_monotonic - self._start_monotonic

        # Compute max reconnect rate per minute per shard from samples.
        # Samples are (monotonic_ts, cumulative_total_attempts).
        # We find the max delta over any 60s+ window, then divide by shard count.
        shard_count = self._final_shard_count
        max_rate_total = 0.0
        samples = list(self._reconnect_samples)
        for i, (ts_i, count_i) in enumerate(samples):
            for ts_j, count_j in samples[i + 1 :]:
                window = ts_j - ts_i
                if window >= 60.0:
                    rate = (count_j - count_i) / (window / 60.0)
                    max_rate_total = max(max_rate_total, rate)
                    break
        max_rate_per_shard = max_rate_total / shard_count

        faults: dict[str, Any] = {}
        if self._config.fault_drop_ws_every_s is not None:
            faults["drop_ws_every_s"] = self._config.fault_drop_ws_every_s
        if self._config.fault_slow_consumer_ms is not None:
            faults["slow_consumer_ms"] = self._config.fault_slow_consumer_ms

        return SoakSummary(
            duration_s=round(duration_s, 1),
            market_events=m.market_events_received,
            events_per_second=round(m.events_per_second(), 1),
            snapshots=m.snapshots_emitted,
            predictions=m.predictions_made,
            alerts=m.alerts_emitted,
            avg_latency_ms=round(m.avg_event_latency_ms(), 1),
            max_latency_ms=m.max_event_latency_ms,
            ws_disconnects=cm.total_disconnects,
            ws_reconnect_attempts=cm.total_reconnect_attempts,
            ws_reconnects_denied=cm.total_reconnects_denied,
            ws_ping_timeouts=cm.total_ping_timeouts,
            cb_transitions_to_open=cb.metrics.transitions_closed_to_open,
            max_reconnect_rate_per_min=round(max_rate_per_shard, 1),
            # DEC-028: Backpressure fields
            max_event_queue_depth=m.max_event_queue_depth,
            max_snapshot_queue_depth=m.max_snapshot_queue_depth,
            max_tick_drift_ms=m.max_tick_drift_ms,
            max_rss_mb=round(m.max_rss_mb, 1),
            events_dropped=m.events_dropped,
            snapshots_dropped=m.snapshots_dropped,
            faults_injected=faults,
            passed=max_rate_per_shard <= 6.0,
        )

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
        logger.info(
            "Latency: avg=%.1fms max=%dms", m.avg_event_latency_ms(), m.max_event_latency_ms
        )

        # LLM metrics
        alerter_m = self._alerter.metrics
        logger.info(
            "LLM calls: %d (cache hits: %d, failures: %d)",
            alerter_m.llm_calls,
            alerter_m.llm_cache_hits,
            alerter_m.llm_failures,
        )

        # Router metrics
        router_m = self._router.metrics
        logger.info(
            "Router: %d events, %d stale, %d late",
            router_m.events_routed,
            router_m.stale_events,
            router_m.late_events,
        )

        # Ranker metrics
        ranker_m = self._ranker.metrics
        logger.info(
            "Ranker: %d updates, %d enters, %d exits",
            ranker_m.updates_processed,
            ranker_m.enter_events,
            ranker_m.exit_events,
        )
        logger.info("=" * 60)

    def request_shutdown(self) -> None:
        """Request graceful shutdown by setting running flag to False."""
        logger.info("Shutdown requested")
        self._running = False


def setup_signal_handlers(pipeline: LivePipeline) -> None:
    """
    Setup signal handlers for graceful shutdown.

    Signal handler only sets the running flag to False.
    The main loop will exit naturally and finally block will call stop().
    This avoids race conditions from concurrent stop() calls.

    Args:
        pipeline: LivePipeline instance.
    """

    def signal_handler(sig: int, frame: object) -> None:
        logger.info("Received signal %s, initiating shutdown", signal.Signals(sig).name)
        # Only set flag - do NOT call stop() here to avoid double-stop race
        pipeline.request_shutdown()

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

    # DEC-026: Create MetricsExporter + registry for /metrics endpoint
    exporter: MetricsExporter | None = None
    metrics_runner = None
    registry = None
    if config.metrics_port > 0:
        from prometheus_client.registry import CollectorRegistry

        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)

    pipeline = LivePipeline(
        config=config,
        explainer=explainer,
        metrics_exporter=exporter,
    )

    # DEC-029/030: Start metrics server after pipeline so we can pass health_fn + ready_fn
    if registry is not None:
        metrics_runner = await start_metrics_server(
            registry,
            port=config.metrics_port,
            health_fn=pipeline.get_health_info,
            ready_fn=pipeline.get_ready_info,
        )

    # DEC-030: --dry-run exits after validation + metrics server start
    if config.dry_run:
        logger.info("Dry-run mode: config valid, metrics server up, exiting")
        if metrics_runner is not None:
            await stop_metrics_server(metrics_runner)
        return 0

    # Setup signal handlers (only sets flags, does not call stop())
    setup_signal_handlers(pipeline)

    try:
        await pipeline.start()
        return 0
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 1
    finally:
        await pipeline.stop()
        if metrics_runner is not None:
            await stop_metrics_server(metrics_runner)


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
        "--duration-s",
        type=int,
        default=None,
        help="Run for N seconds then stop gracefully (default: run until SIGINT/SIGTERM)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9090,
        help="Prometheus /metrics port (0 to disable, default: 9090)",
    )
    parser.add_argument(
        "--fault-drop-ws-every-s",
        type=int,
        default=None,
        help="DEC-027: Force-close all WS connections every N seconds",
    )
    parser.add_argument(
        "--fault-slow-consumer-ms",
        type=int,
        default=None,
        help="DEC-027: Add N ms delay after each event (slow consumer simulation)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="DEC-027: Write soak summary JSON to this path at exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DEC-030: Validate config, start metrics server, exit without processing",
    )
    parser.add_argument(
        "--graceful-timeout-s",
        type=int,
        default=10,
        help="DEC-030: Graceful shutdown timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default=None,
        help="DEC-032: Override WebSocket base URL (for offline soak with FakeWSServer)",
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
        duration_s=args.duration_s,
        verbose=args.verbose,
        metrics_port=args.metrics_port,
        fault_drop_ws_every_s=args.fault_drop_ws_every_s,
        fault_slow_consumer_ms=args.fault_slow_consumer_ms,
        summary_json=args.summary_json,
        dry_run=args.dry_run,
        graceful_timeout_s=args.graceful_timeout_s,
        ws_url=args.ws_url,
    )

    logger.info("Starting CryptoScreener Live Pipeline")
    logger.info("  Symbols: %s", symbols if symbols else f"top {config.top_n}")
    logger.info("  Cadence: %dms", config.snapshot_cadence_ms)
    logger.info("  Duration: %s", f"{config.duration_s}s" if config.duration_s else "until signal")
    logger.info("  LLM: %s", "enabled" if config.llm_enabled else "disabled")

    return asyncio.run(run_pipeline(config))


if __name__ == "__main__":
    sys.exit(main())
