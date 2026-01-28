#!/usr/bin/env python3
"""
Replay harness for CryptoScreener-X.

Replays recorded market events through the real pipeline and verifies
deterministic output by comparing RankEvent digests.

Usage:
    python -m scripts.run_replay --fixture tests/fixtures/sample_run/

The replay harness:
1. Loads MarketEvents from fixture
2. Processes them through the real pipeline (FeatureEngine → BaselineRunner → Ranker → Alerter)
3. Emits RankEvents
4. Computes digest of RankEvent stream
5. Compares against expected digest (if provided)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path

import orjson

from cryptoscreener.alerting.alerter import Alerter, AlerterConfig
from cryptoscreener.contracts import (
    MarketEvent,
    RankEvent,
    compute_rank_events_digest,
)
from cryptoscreener.features.engine import FeatureEngine, FeatureEngineConfig
from cryptoscreener.model_runner import BaselineRunner, ModelRunnerConfig
from cryptoscreener.ranker.ranker import Ranker, RankerConfig

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


class ReplayPipeline:
    """
    Replay pipeline using real components.

    Mirrors the live pipeline (run_live.py LivePipeline) processing path:
    FeatureEngine → BaselineRunner → Ranker → Alerter.
    """

    def __init__(self, snapshot_cadence_ms: int = 1000) -> None:
        """Initialize pipeline with real components."""
        self._feature_engine = FeatureEngine(
            config=FeatureEngineConfig(
                snapshot_cadence_ms=snapshot_cadence_ms,
                max_symbols=500,
            )
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
                llm_enabled=False,
            )
        )
        self._snapshot_cadence_ms = snapshot_cadence_ms

    async def _replay_async(self, events: list[MarketEvent]) -> list[RankEvent]:
        """Async replay core — processes events through real components."""
        rank_events: list[RankEvent] = []
        last_process_ts = 0

        for event in events:
            # Feed event to feature engine (async)
            await self._feature_engine.process_event(event)

            # Check cadence boundary
            if event.ts - last_process_ts >= self._snapshot_cadence_ms:
                # Emit snapshots for all tracked symbols
                snapshots = await self._feature_engine.emit_snapshots(event.ts)

                # Predict on each snapshot
                predictions = {}
                for snapshot in snapshots:
                    prediction = self._model_runner.predict(snapshot)
                    predictions[snapshot.symbol] = prediction

                if predictions:
                    # Ranker pass
                    new_rank_events = self._ranker.update(predictions, event.ts)
                    rank_events.extend(new_rank_events)

                    # Alerter pass — process each prediction
                    for symbol, prediction in predictions.items():
                        state = self._ranker.get_state(symbol)
                        rank = max(0, state.rank) if state else 0
                        score = state.score if state else 0.0
                        alert_events = self._alerter.process_prediction(
                            prediction=prediction,
                            ts=event.ts,
                            rank=rank,
                            score=score,
                        )
                        rank_events.extend(alert_events)

                last_process_ts = event.ts

        return rank_events

    def replay(self, events: list[MarketEvent]) -> list[RankEvent]:
        """
        Replay all market events through real pipeline.

        Args:
            events: List of MarketEvent objects (should be sorted by ts).

        Returns:
            List of RankEvent objects emitted during replay.
        """
        rank_events = asyncio.run(self._replay_async(events))

        for rank_event in rank_events:
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
    pipeline = ReplayPipeline()
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
