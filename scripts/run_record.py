#!/usr/bin/env python3
"""
Record harness for CryptoScreener-X.

Records market events and expected rank events into fixture files
for determinism verification via replay.

Usage:
    # Synthetic data (default, for testing)
    python -m scripts.run_record --symbols BTCUSDT,ETHUSDT --duration-s 10 --out-dir tests/fixtures/my_run/

    # Live data (requires configured data source)
    python -m scripts.run_record --source live --symbols BTCUSDT --duration-s 60 --out-dir tests/fixtures/live_run/

Output files:
    market_events.jsonl       - Recorded market events
    expected_rank_events.jsonl - RankEvents from pipeline processing
    manifest.json             - Metadata with SHA256 checksums and digest
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from collections.abc import Iterator

from cryptoscreener.contracts import (
    MarketEvent,
    MarketEventType,
    RankEvent,
    RankEventPayload,
    RankEventType,
    compute_rank_events_digest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Schema version for manifest format
MANIFEST_SCHEMA_VERSION = "1.0.0"


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class SyntheticMarketEventGenerator:
    """
    Generates synthetic market events for testing.

    Produces deterministic sequences of trade and book events
    for specified symbols.
    """

    def __init__(
        self,
        symbols: list[str],
        cadence_ms: int = 100,
        seed: int = 42,
    ) -> None:
        """
        Initialize synthetic generator.

        Args:
            symbols: List of symbols to generate events for.
            cadence_ms: Milliseconds between events.
            seed: Random seed for deterministic generation.
        """
        self.symbols = symbols
        self.cadence_ms = cadence_ms
        self.seed = seed
        self._base_prices = {
            "BTCUSDT": 42000.0,
            "ETHUSDT": 2500.0,
            "SOLUSDT": 95.0,
            "BNBUSDT": 300.0,
            "XRPUSDT": 0.5,
        }
        self._event_counter = 0

    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol (default to 100.0 for unknown)."""
        return self._base_prices.get(symbol, 100.0)

    def generate(
        self,
        start_ts: int,
        duration_ms: int,
    ) -> Iterator[MarketEvent]:
        """
        Generate synthetic market events.

        Args:
            start_ts: Start timestamp in milliseconds.
            duration_ms: Duration to generate events for in milliseconds.

        Yields:
            MarketEvent objects.
        """
        current_ts = start_ts
        end_ts = start_ts + duration_ms

        while current_ts < end_ts:
            for symbol in self.symbols:
                base_price = self._get_base_price(symbol)
                # Deterministic price variation based on counter
                price_offset = (self._event_counter % 100) / 100.0 * base_price * 0.001
                price = base_price + price_offset

                # Alternate between trade and book events
                if self._event_counter % 2 == 0:
                    yield MarketEvent(
                        ts=current_ts,
                        source="binance_usdm",
                        symbol=symbol,
                        type=MarketEventType.TRADE,
                        payload={
                            "price": f"{price:.2f}",
                            "qty": f"{0.1 + (self._event_counter % 10) * 0.1:.1f}",
                            "side": "buy" if self._event_counter % 3 == 0 else "sell",
                        },
                        recv_ts=current_ts + 5,
                    )
                else:
                    spread = base_price * 0.0001  # 1 bps spread
                    yield MarketEvent(
                        ts=current_ts,
                        source="binance_usdm",
                        symbol=symbol,
                        type=MarketEventType.BOOK,
                        payload={
                            "bid": f"{price - spread:.2f}",
                            "ask": f"{price + spread:.2f}",
                            "bid_qty": f"{5.0 + (self._event_counter % 5):.1f}",
                            "ask_qty": f"{3.0 + (self._event_counter % 5):.1f}",
                        },
                        recv_ts=current_ts + 5,
                    )

                self._event_counter += 1

            current_ts += self.cadence_ms


class MinimalRecordPipeline:
    """
    Minimal pipeline for recording expected RankEvents.

    This mirrors MinimalReplayPipeline logic to ensure deterministic output.
    """

    def __init__(self, seed: int = 42, llm_enabled: bool = False) -> None:
        """Initialize pipeline with deterministic seed."""
        self.seed = seed
        self.llm_enabled = llm_enabled
        self._symbol_state: dict[str, dict[str, float | int | bool]] = {}
        self._rank_counter = 0

    def process_event(self, event: MarketEvent) -> Iterator[RankEvent]:
        """
        Process a single market event and yield any resulting RankEvents.

        Uses the same deterministic logic as MinimalReplayPipeline.

        Args:
            event: MarketEvent to process.

        Yields:
            RankEvent objects (if any state transitions occur).
        """
        symbol = event.symbol

        # Initialize state for new symbols
        if symbol not in self._symbol_state:
            self._symbol_state[symbol] = {
                "trade_count": 0,
                "last_ts": 0,
                "score": 0.0,
                "in_top_k": False,
            }

        state = self._symbol_state[symbol]

        # Simple deterministic logic (mirrors MinimalReplayPipeline)
        if event.type.value == "trade":
            state["trade_count"] = int(state["trade_count"]) + 1
            state["last_ts"] = event.ts

            # Deterministic score based on trade count
            base_score = min(0.5 + int(state["trade_count"]) * 0.1, 0.95)
            # Add symbol-specific offset for determinism
            symbol_offset = sum(ord(c) for c in symbol) % 100 / 1000
            state["score"] = round(base_score + symbol_offset, 2)

            # Emit SYMBOL_ENTER after 2 trades if not already in top-k
            if state["trade_count"] == 2 and not state["in_top_k"]:
                state["in_top_k"] = True
                yield RankEvent(
                    ts=event.ts,
                    event=RankEventType.SYMBOL_ENTER,
                    symbol=symbol,
                    rank=self._rank_counter,
                    score=float(state["score"]),
                    payload=RankEventPayload(prediction={"status": "WATCH"}, llm_text=""),
                )
                self._rank_counter += 1

            # Emit ALERT_TRADABLE after 4 trades
            elif state["trade_count"] == 4 and state["in_top_k"]:
                yield RankEvent(
                    ts=event.ts,
                    event=RankEventType.ALERT_TRADABLE,
                    symbol=symbol,
                    rank=0,  # Promoted to top
                    score=min(float(state["score"]) + 0.1, 0.95),
                    payload=RankEventPayload(prediction={"status": "TRADEABLE"}, llm_text=""),
                )

    def record(self, events: list[MarketEvent]) -> list[RankEvent]:
        """
        Process all market events and collect RankEvents.

        Args:
            events: List of MarketEvent objects (should be sorted by ts).

        Returns:
            List of RankEvent objects emitted during processing.
        """
        rank_events: list[RankEvent] = []

        for event in events:
            for rank_event in self.process_event(event):
                rank_events.append(rank_event)
                logger.debug(
                    f"RankEvent: {rank_event.event.value} {rank_event.symbol} "
                    f"rank={rank_event.rank} score={rank_event.score}"
                )

        logger.info(f"Recording complete: {len(rank_events)} rank events generated")
        return rank_events


def write_jsonl(filepath: Path, events: list[MarketEvent] | list[RankEvent]) -> int:
    """
    Write events to JSONL file.

    Args:
        filepath: Output file path.
        events: List of events to write.

    Returns:
        Number of events written.
    """
    with filepath.open("wb") as f:
        for event in events:
            f.write(event.to_json())
            f.write(b"\n")
    return len(events)


def run_record(
    symbols: list[str],
    duration_s: int,
    out_dir: Path,
    cadence_ms: int = 100,
    source: str = "synthetic",
    llm_enabled: bool = False,
) -> tuple[Path, str]:
    """
    Run recording session.

    Args:
        symbols: List of symbols to record.
        duration_s: Recording duration in seconds.
        out_dir: Output directory for fixture files.
        cadence_ms: Milliseconds between events (synthetic mode).
        source: Data source ("synthetic" or "live").
        llm_enabled: Whether to enable LLM explanations.

    Returns:
        Tuple of (manifest_path, digest).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate or collect market events
    start_ts = int(time.time() * 1000)
    duration_ms = duration_s * 1000

    if source == "synthetic":
        logger.info(f"Generating synthetic events for {symbols} over {duration_s}s")
        generator = SyntheticMarketEventGenerator(
            symbols=symbols,
            cadence_ms=cadence_ms,
        )
        market_events = list(generator.generate(start_ts, duration_ms))
    else:
        # Live mode - placeholder for future implementation
        raise NotImplementedError("Live recording not implemented. Use --source synthetic.")

    # Sort events by timestamp for determinism
    market_events.sort(key=lambda e: (e.ts, e.recv_ts))
    logger.info(f"Recorded {len(market_events)} market events")

    # Process through real pipeline (shared with run_replay.py)
    try:
        from scripts.run_replay import ReplayPipeline
    except ModuleNotFoundError:
        from run_replay import ReplayPipeline  # type: ignore[import-not-found,no-redef]

    pipeline = ReplayPipeline()
    rank_events = pipeline.replay(market_events)

    # Write output files
    market_events_file = out_dir / "market_events.jsonl"
    rank_events_file = out_dir / "expected_rank_events.jsonl"
    manifest_file = out_dir / "manifest.json"

    write_jsonl(market_events_file, market_events)
    write_jsonl(rank_events_file, rank_events)

    # Compute checksums and digest
    market_sha256 = compute_file_sha256(market_events_file)
    rank_sha256 = compute_file_sha256(rank_events_file)
    digest = compute_rank_events_digest(rank_events)

    # Get time range
    time_range_ms = [market_events[0].ts, market_events[-1].ts] if market_events else [0, 0]

    # Build manifest
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source": source,
        "symbols": symbols,
        "duration_s": duration_s,
        "cadence_ms": cadence_ms,
        "llm_enabled": llm_enabled,
        "sha256": {
            "market_events.jsonl": market_sha256,
            "expected_rank_events.jsonl": rank_sha256,
        },
        "replay": {
            "rank_event_stream_digest": digest,
        },
        "stats": {
            "total_market_events": len(market_events),
            "total_rank_events": len(rank_events),
            "symbols": symbols,
            "time_range_ms": time_range_ms,
        },
    }

    # Write manifest
    with manifest_file.open("wb") as f:
        f.write(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    logger.info(f"Manifest written to {manifest_file}")
    logger.info(f"  market_events.jsonl sha256: {market_sha256}")
    logger.info(f"  expected_rank_events.jsonl sha256: {rank_sha256}")
    logger.info(f"  rank_event_stream_digest: {digest}")

    return manifest_file, digest


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record market events and expected rank events for replay testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--duration-s",
        type=int,
        required=True,
        help="Recording duration in seconds",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for fixture files",
    )
    parser.add_argument(
        "--cadence-ms",
        type=int,
        default=100,
        help="Milliseconds between events in synthetic mode (default: 100)",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "live"],
        default="synthetic",
        help="Data source (default: synthetic)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM explanations (default: off)",
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

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    try:
        manifest_path, digest = run_record(
            symbols=symbols,
            duration_s=args.duration_s,
            out_dir=args.out_dir,
            cadence_ms=args.cadence_ms,
            source=args.source,
            llm_enabled=args.llm,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("RECORD SUMMARY")
        print("=" * 60)
        print(f"Output:         {args.out_dir}")
        print(f"Symbols:        {', '.join(symbols)}")
        print(f"Duration:       {args.duration_s}s")
        print(f"Source:         {args.source}")
        print(f"LLM:            {'enabled' if args.llm else 'disabled'}")
        print(f"Digest:         {digest}")
        print(f"Manifest:       {manifest_path}")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.exception(f"Recording failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
