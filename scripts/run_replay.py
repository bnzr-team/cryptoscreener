#!/usr/bin/env python3
"""
Replay harness for CryptoScreener-X.

Replays recorded market events through the pipeline and verifies
deterministic output by comparing RankEvent digests.

Usage:
    python -m scripts.run_replay --fixture tests/fixtures/sample_run/

The replay harness:
1. Loads MarketEvents from fixture
2. Processes them through a minimal pipeline (stub for now)
3. Emits RankEvents
4. Computes digest of RankEvent stream
5. Compares against expected digest (if provided)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from collections.abc import Iterator

from cryptoscreener.contracts import (
    MarketEvent,
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


def load_market_events(filepath: Path) -> list[MarketEvent]:
    """
    Load MarketEvents from JSONL file.

    Args:
        filepath: Path to market_events.jsonl

    Returns:
        List of MarketEvent objects sorted by timestamp.
    """
    events: list[MarketEvent] = []
    with filepath.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
                event = MarketEvent.model_validate(data)
                events.append(event)
            except Exception as e:
                logger.error(f"Failed to parse line {line_num}: {e}")
                raise

    # Sort by timestamp for deterministic processing
    events.sort(key=lambda e: (e.ts, e.recv_ts))
    logger.info(f"Loaded {len(events)} market events from {filepath}")
    return events


def load_expected_rank_events(filepath: Path) -> list[RankEvent]:
    """
    Load expected RankEvents from JSONL file.

    Args:
        filepath: Path to expected_rank_events.jsonl

    Returns:
        List of RankEvent objects.
    """
    events: list[RankEvent] = []
    with filepath.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
                event = RankEvent.model_validate(data)
                events.append(event)
            except Exception as e:
                logger.error(f"Failed to parse line {line_num}: {e}")
                raise

    logger.info(f"Loaded {len(events)} expected rank events from {filepath}")
    return events


class MinimalReplayPipeline:
    """
    Minimal replay pipeline for testing.

    This is a stub implementation that demonstrates the replay harness structure.
    The actual feature engine, ML inference, and ranker will be implemented later.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize pipeline with deterministic seed."""
        self.seed = seed
        self._symbol_state: dict[str, dict[str, float]] = {}
        self._rank_counter = 0

    def process_event(self, event: MarketEvent) -> Iterator[RankEvent]:
        """
        Process a single market event and yield any resulting RankEvents.

        This is a stub that generates deterministic events based on simple rules.
        Real implementation will use feature engine + ML inference + ranker.

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

        # Simple deterministic logic for stub
        if event.type.value == "trade":
            state["trade_count"] += 1
            state["last_ts"] = event.ts

            # Deterministic score based on trade count (stub logic)
            base_score = min(0.5 + state["trade_count"] * 0.1, 0.95)
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
                    score=state["score"],
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
                    score=min(state["score"] + 0.1, 0.95),
                    payload=RankEventPayload(prediction={"status": "TRADEABLE"}, llm_text=""),
                )

    def replay(self, events: list[MarketEvent]) -> list[RankEvent]:
        """
        Replay all market events and collect RankEvents.

        Args:
            events: List of MarketEvent objects (should be sorted by ts).

        Returns:
            List of RankEvent objects emitted during replay.
        """
        rank_events: list[RankEvent] = []

        for event in events:
            for rank_event in self.process_event(event):
                rank_events.append(rank_event)
                logger.debug(
                    f"RankEvent: {rank_event.event.value} {rank_event.symbol} "
                    f"rank={rank_event.rank} score={rank_event.score}"
                )

        logger.info(f"Replay complete: {len(rank_events)} rank events emitted")
        return rank_events


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_replay(
    fixture_path: Path,
    verify_expected: bool = True,
) -> tuple[list[RankEvent], str, bool]:
    """
    Run replay on a fixture directory.

    Args:
        fixture_path: Path to fixture directory.
        verify_expected: Whether to verify against expected output.

    Returns:
        Tuple of (rank_events, digest, passed).
    """
    market_events_file = fixture_path / "market_events.jsonl"
    expected_events_file = fixture_path / "expected_rank_events.jsonl"

    if not market_events_file.exists():
        raise FileNotFoundError(f"Market events file not found: {market_events_file}")

    # Log fixture checksums for all files
    manifest_file = fixture_path / "manifest.json"
    logger.info(f"Fixture: {fixture_path}")
    logger.info(f"  market_events.jsonl sha256: {compute_file_sha256(market_events_file)}")
    if expected_events_file.exists():
        logger.info(
            f"  expected_rank_events.jsonl sha256: {compute_file_sha256(expected_events_file)}"
        )
    if manifest_file.exists():
        logger.info(f"  manifest.json sha256: {compute_file_sha256(manifest_file)}")

    # Load and replay
    market_events = load_market_events(market_events_file)
    pipeline = MinimalReplayPipeline(seed=42)
    rank_events = pipeline.replay(market_events)

    # Compute digest
    digest = compute_rank_events_digest(rank_events)
    logger.info(f"RankEvent stream digest: {digest}")
    logger.info(f"  Total RankEvents: {len(rank_events)}")

    # Log individual events
    for i, evt in enumerate(rank_events):
        logger.info(f"  [{i}] {evt.event.value} {evt.symbol} rank={evt.rank} score={evt.score}")

    # Verify against expected (if requested and file exists)
    passed = True
    if verify_expected and expected_events_file.exists():
        expected_events = load_expected_rank_events(expected_events_file)
        expected_digest = compute_rank_events_digest(expected_events)
        logger.info(f"Expected digest: {expected_digest}")

        if digest == expected_digest:
            logger.info("✓ DETERMINISM CHECK PASSED: digests match")
        else:
            logger.error("✗ DETERMINISM CHECK FAILED: digests do not match")
            logger.error(f"  Expected: {expected_digest}")
            logger.error(f"  Got:      {digest}")
            passed = False

            # Show differences
            if len(rank_events) != len(expected_events):
                logger.error(
                    f"  Event count mismatch: expected {len(expected_events)}, got {len(rank_events)}"
                )

    return rank_events, digest, passed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Replay market events and verify determinism.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="Path to fixture directory containing market_events.jsonl",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification against expected output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write RankEvents to this file (JSONL format)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        rank_events, digest, passed = run_replay(
            fixture_path=args.fixture,
            verify_expected=not args.no_verify,
        )

        # Write output if requested
        if args.output:
            with args.output.open("w") as f:
                for evt in rank_events:
                    f.write(evt.to_json().decode() + "\n")
            logger.info(f"Wrote {len(rank_events)} events to {args.output}")

        # Print summary
        print("\n" + "=" * 60)
        print("REPLAY SUMMARY")
        print("=" * 60)
        print(f"Fixture:        {args.fixture}")
        print(f"RankEvents:     {len(rank_events)}")
        print(f"Digest:         {digest}")
        print(f"Determinism:    {'PASSED' if passed else 'FAILED'}")
        print("=" * 60)

        return 0 if passed else 1

    except Exception as e:
        logger.exception(f"Replay failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
