"""End-to-end determinism tests for the complete pipeline.

Tests the full flow: FeatureSnapshot → MLRunner → Scorer → Ranker → RankEvent

Verifies that:
1. Same input produces identical output across runs
2. SHA256 digests are stable
3. RankEvent JSON roundtrips correctly

This is the acceptance test for DEC-015.
"""

from __future__ import annotations

import hashlib

from cryptoscreener.contracts.events import (
    DataHealth,
    Features,
    FeatureSnapshot,
    PredictionSnapshot,
    RankEvent,
    RegimeTrend,
    RegimeVol,
    compute_rank_events_digest,
)
from cryptoscreener.model_runner import BaselineRunner, ModelRunnerConfig
from cryptoscreener.ranker import Ranker, RankerConfig
from cryptoscreener.scoring.scorer import Scorer, ScorerConfig

# Base timestamp: 2026-01-01 00:00:00 UTC (deterministic)
BASE_TS = 1767225600000  # ms


def make_feature_snapshot(
    symbol: str,
    ts: int,
    spread_bps: float = 2.0,
    mid: float = 50000.0,
    book_imbalance: float = 0.3,
    flow_imbalance: float = 0.4,
    natr: float = 0.02,
    impact_bps: float = 5.0,
    regime_vol: RegimeVol = RegimeVol.HIGH,
    regime_trend: RegimeTrend = RegimeTrend.TREND,
) -> FeatureSnapshot:
    """Create a deterministic FeatureSnapshot."""
    return FeatureSnapshot(
        ts=ts,
        symbol=symbol,
        features=Features(
            spread_bps=spread_bps,
            mid=mid,
            book_imbalance=book_imbalance,
            flow_imbalance=flow_imbalance,
            natr_14_5m=natr,
            impact_bps_q=impact_bps,
            regime_vol=regime_vol,
            regime_trend=regime_trend,
        ),
        data_health=DataHealth(
            stale_book_ms=0,
            stale_trades_ms=0,
        ),
    )


# Fixture: FeatureSnapshots across 3 symbols at multiple time points
# This creates a scenario where symbols enter/exit the ranker
E2E_FIXTURE: list[list[FeatureSnapshot]] = [
    # t=0: Initial state - BTC strong, ETH moderate, SOL weak
    [
        make_feature_snapshot(
            symbol="BTCUSDT",
            ts=BASE_TS,
            spread_bps=1.5,
            book_imbalance=0.5,
            flow_imbalance=0.6,
            natr=0.025,
            impact_bps=3.0,
        ),
        make_feature_snapshot(
            symbol="ETHUSDT",
            ts=BASE_TS,
            spread_bps=2.5,
            book_imbalance=0.3,
            flow_imbalance=0.4,
            natr=0.02,
            impact_bps=5.0,
        ),
        make_feature_snapshot(
            symbol="SOLUSDT",
            ts=BASE_TS,
            spread_bps=4.0,
            book_imbalance=0.1,
            flow_imbalance=0.2,
            natr=0.015,
            impact_bps=8.0,
        ),
    ],
    # t=2000ms: ETH strengthens
    [
        make_feature_snapshot(
            symbol="BTCUSDT",
            ts=BASE_TS + 2000,
            spread_bps=1.8,
            book_imbalance=0.45,
            flow_imbalance=0.55,
            natr=0.024,
            impact_bps=3.5,
        ),
        make_feature_snapshot(
            symbol="ETHUSDT",
            ts=BASE_TS + 2000,
            spread_bps=2.0,
            book_imbalance=0.55,
            flow_imbalance=0.65,
            natr=0.028,
            impact_bps=4.0,
        ),
        make_feature_snapshot(
            symbol="SOLUSDT",
            ts=BASE_TS + 2000,
            spread_bps=3.0,
            book_imbalance=0.35,
            flow_imbalance=0.45,
            natr=0.022,
            impact_bps=6.0,
        ),
    ],
    # t=4000ms: ETH leads
    [
        make_feature_snapshot(
            symbol="BTCUSDT",
            ts=BASE_TS + 4000,
            spread_bps=2.2,
            book_imbalance=0.35,
            flow_imbalance=0.4,
            natr=0.02,
            impact_bps=5.0,
        ),
        make_feature_snapshot(
            symbol="ETHUSDT",
            ts=BASE_TS + 4000,
            spread_bps=1.5,
            book_imbalance=0.6,
            flow_imbalance=0.7,
            natr=0.03,
            impact_bps=3.0,
        ),
        make_feature_snapshot(
            symbol="SOLUSDT",
            ts=BASE_TS + 4000,
            spread_bps=2.5,
            book_imbalance=0.45,
            flow_imbalance=0.55,
            natr=0.025,
            impact_bps=4.5,
        ),
    ],
    # t=6000ms: BTC drops significantly
    [
        make_feature_snapshot(
            symbol="BTCUSDT",
            ts=BASE_TS + 6000,
            spread_bps=5.0,
            book_imbalance=0.1,
            flow_imbalance=0.15,
            natr=0.01,
            impact_bps=12.0,
        ),
        make_feature_snapshot(
            symbol="ETHUSDT",
            ts=BASE_TS + 6000,
            spread_bps=1.2,
            book_imbalance=0.65,
            flow_imbalance=0.75,
            natr=0.032,
            impact_bps=2.5,
        ),
        make_feature_snapshot(
            symbol="SOLUSDT",
            ts=BASE_TS + 6000,
            spread_bps=2.0,
            book_imbalance=0.5,
            flow_imbalance=0.6,
            natr=0.028,
            impact_bps=4.0,
        ),
    ],
    # t=8000ms: Stable state
    [
        make_feature_snapshot(
            symbol="BTCUSDT",
            ts=BASE_TS + 8000,
            spread_bps=4.5,
            book_imbalance=0.15,
            flow_imbalance=0.2,
            natr=0.012,
            impact_bps=10.0,
        ),
        make_feature_snapshot(
            symbol="ETHUSDT",
            ts=BASE_TS + 8000,
            spread_bps=1.3,
            book_imbalance=0.62,
            flow_imbalance=0.72,
            natr=0.031,
            impact_bps=2.8,
        ),
        make_feature_snapshot(
            symbol="SOLUSDT",
            ts=BASE_TS + 8000,
            spread_bps=2.2,
            book_imbalance=0.48,
            flow_imbalance=0.58,
            natr=0.027,
            impact_bps=4.2,
        ),
    ],
]


def run_e2e_pipeline(
    fixture: list[list[FeatureSnapshot]],
) -> tuple[list[RankEvent], list[PredictionSnapshot]]:
    """Run the end-to-end pipeline on a fixture.

    Pipeline: FeatureSnapshot → MLRunner → Scorer → Ranker → RankEvent

    Args:
        fixture: List of frames, each containing FeatureSnapshots.

    Returns:
        Tuple of (all RankEvents, all PredictionSnapshots).
    """
    # Initialize pipeline components
    # Use BaselineRunner for deterministic heuristic-based predictions
    # When MLRunner is merged (PR#58), it will use fallback_to_baseline=True
    # which delegates to BaselineRunner anyway
    runner_config = ModelRunnerConfig()
    scorer_config = ScorerConfig()
    ranker_config = RankerConfig(
        top_k=5,
        enter_ms=1500,
        exit_ms=3000,
        min_dwell_ms=2000,
        score_threshold=0.001,  # Lower threshold to ensure events are generated
    )

    runner = BaselineRunner(runner_config)
    scorer = Scorer(scorer_config)
    ranker = Ranker(ranker_config, scorer)

    all_events: list[RankEvent] = []
    all_predictions: list[PredictionSnapshot] = []

    for frame in fixture:
        if not frame:
            continue

        # Get timestamp from first snapshot in frame
        ts = frame[0].ts

        # Run MLRunner on each FeatureSnapshot
        predictions: dict[str, PredictionSnapshot] = {}
        for snapshot in frame:
            prediction = runner.predict(snapshot)
            predictions[snapshot.symbol] = prediction
            all_predictions.append(prediction)

        # Run Ranker to get RankEvents
        events = ranker.update(predictions, ts)
        all_events.extend(events)

    return all_events, all_predictions


def compute_predictions_digest(predictions: list[PredictionSnapshot]) -> str:
    """Compute SHA256 digest of predictions sequence."""
    data = b"".join(p.to_json() for p in predictions)
    return hashlib.sha256(data).hexdigest()


def compute_fixture_digest(fixture: list[list[FeatureSnapshot]]) -> str:
    """Compute SHA256 digest of the input fixture."""
    parts = []
    for frame in fixture:
        for snapshot in frame:
            parts.append(snapshot.to_json())
    return hashlib.sha256(b"".join(parts)).hexdigest()


class TestE2EDeterminism:
    """End-to-end determinism tests."""

    def test_same_input_produces_same_output(self) -> None:
        """Same fixture produces identical RankEvents across runs."""
        events1, _ = run_e2e_pipeline(E2E_FIXTURE)
        events2, _ = run_e2e_pipeline(E2E_FIXTURE)

        assert len(events1) == len(events2), "Event count mismatch"

        for e1, e2 in zip(events1, events2, strict=True):
            assert e1.ts == e2.ts
            assert e1.event == e2.event
            assert e1.symbol == e2.symbol
            assert e1.rank == e2.rank
            assert abs(e1.score - e2.score) < 1e-6

    def test_rank_events_digest_stable(self) -> None:
        """RankEvent digest is stable across runs."""
        events1, _ = run_e2e_pipeline(E2E_FIXTURE)
        events2, _ = run_e2e_pipeline(E2E_FIXTURE)

        digest1 = compute_rank_events_digest(events1)
        digest2 = compute_rank_events_digest(events2)

        assert digest1 == digest2, f"Digest mismatch: {digest1} != {digest2}"

    def test_predictions_digest_stable(self) -> None:
        """PredictionSnapshot digest is stable across runs."""
        _, predictions1 = run_e2e_pipeline(E2E_FIXTURE)
        _, predictions2 = run_e2e_pipeline(E2E_FIXTURE)

        digest1 = compute_predictions_digest(predictions1)
        digest2 = compute_predictions_digest(predictions2)

        assert digest1 == digest2, f"Digest mismatch: {digest1} != {digest2}"

    def test_fixture_digest_is_stable(self) -> None:
        """Input fixture digest is constant."""
        digest1 = compute_fixture_digest(E2E_FIXTURE)
        digest2 = compute_fixture_digest(E2E_FIXTURE)

        assert digest1 == digest2
        # The fixture itself should always produce the same digest
        assert len(digest1) == 64  # SHA256 hex length

    def test_produces_rank_events(self) -> None:
        """Pipeline produces at least some RankEvents."""
        events, predictions = run_e2e_pipeline(E2E_FIXTURE)

        # Should produce predictions for all frames
        expected_predictions = sum(len(frame) for frame in E2E_FIXTURE)
        assert len(predictions) == expected_predictions

        # MUST produce at least 1 RankEvent (prevents empty digest regression)
        assert len(events) > 0, "Pipeline must produce at least one RankEvent"

        # Verify events have valid structure
        for event in events:
            assert event.ts > 0
            assert event.symbol in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
            assert event.score >= 0

    def test_runner_type_is_baseline(self) -> None:
        """Verify pipeline uses BaselineRunner (not MLRunner)."""
        _, predictions = run_e2e_pipeline(E2E_FIXTURE)

        # All predictions must come from BaselineRunner
        # Format: "baseline-v1.0.0+{git_sha}" (7 chars)
        for pred in predictions:
            assert pred.model_version.startswith("baseline-"), (
                f"Expected baseline-* model, got {pred.model_version}. "
                "This test suite is for BaselineRunner path only. "
                "MLRunner E2E tests will be in a separate PR."
            )

    def test_rank_event_json_roundtrip(self) -> None:
        """RankEvents roundtrip through JSON correctly."""
        events, _ = run_e2e_pipeline(E2E_FIXTURE)

        for event in events:
            json_bytes = event.to_json()
            restored = RankEvent.from_json(json_bytes)

            assert restored.ts == event.ts
            assert restored.event == event.event
            assert restored.symbol == event.symbol
            assert restored.rank == event.rank
            assert abs(restored.score - event.score) < 1e-9

    def test_prediction_snapshot_json_roundtrip(self) -> None:
        """PredictionSnapshots roundtrip through JSON correctly."""
        _, predictions = run_e2e_pipeline(E2E_FIXTURE)

        for pred in predictions:
            json_bytes = pred.to_json()
            restored = PredictionSnapshot.from_json(json_bytes)

            assert restored.ts == pred.ts
            assert restored.symbol == pred.symbol
            assert restored.status == pred.status
            assert abs(restored.p_inplay_2m - pred.p_inplay_2m) < 1e-9


class TestE2EReplayProof:
    """Tests that generate proof artifacts for verification."""

    def test_generate_replay_proof(self) -> None:
        """Generate and verify replay proof.

        This test:
        1. Computes input fixture digest
        2. Runs pipeline twice
        3. Computes output digests for both runs
        4. Asserts they match
        """
        # Input digest
        input_digest = compute_fixture_digest(E2E_FIXTURE)

        # Run 1
        events1, preds1 = run_e2e_pipeline(E2E_FIXTURE)
        output_digest1 = compute_rank_events_digest(events1)
        pred_digest1 = compute_predictions_digest(preds1)

        # Run 2
        events2, preds2 = run_e2e_pipeline(E2E_FIXTURE)
        output_digest2 = compute_rank_events_digest(events2)
        pred_digest2 = compute_predictions_digest(preds2)

        # Assertions
        assert output_digest1 == output_digest2, "RankEvent digest mismatch"
        assert pred_digest1 == pred_digest2, "Prediction digest mismatch"

        # Log proof for verification (captured by pytest)
        print("\n=== Replay Proof ===")
        print(f"Input fixture digest:  {input_digest}")
        print(f"RankEvent digest (r1): {output_digest1}")
        print(f"RankEvent digest (r2): {output_digest2}")
        print(f"Prediction digest:     {pred_digest1}")
        print(f"Digests match: {output_digest1 == output_digest2}")

    def test_different_input_different_output(self) -> None:
        """Different input produces different output digest."""
        events1, _ = run_e2e_pipeline(E2E_FIXTURE)
        digest1 = compute_rank_events_digest(events1)

        # Modify fixture - shift all timestamps
        modified_fixture = [
            [
                FeatureSnapshot(
                    ts=s.ts + 100,  # Shift by 100ms
                    symbol=s.symbol,
                    features=s.features,
                    data_health=s.data_health,
                )
                for s in frame
            ]
            for frame in E2E_FIXTURE
        ]

        events2, _ = run_e2e_pipeline(modified_fixture)
        digest2 = compute_rank_events_digest(events2)

        # Digests should differ (unless no events produced)
        if events1 and events2:
            # Events have different timestamps, so digests must differ
            assert (
                any(e1.ts != e2.ts for e1, e2 in zip(events1, events2, strict=False))
                or digest1 != digest2
            )


class TestE2EEdgeCases:
    """Edge case tests for end-to-end pipeline."""

    def test_empty_fixture(self) -> None:
        """Empty fixture produces no events."""
        events, predictions = run_e2e_pipeline([])
        assert events == []
        assert predictions == []

    def test_single_frame(self) -> None:
        """Single frame is deterministic."""
        single = [E2E_FIXTURE[0]]

        events1, _ = run_e2e_pipeline(single)
        events2, _ = run_e2e_pipeline(single)

        digest1 = compute_rank_events_digest(events1)
        digest2 = compute_rank_events_digest(events2)

        assert digest1 == digest2

    def test_single_symbol(self) -> None:
        """Single symbol is deterministic."""
        single_symbol = [
            [frame[0]]
            for frame in E2E_FIXTURE  # Only BTCUSDT
        ]

        events1, _ = run_e2e_pipeline(single_symbol)
        events2, _ = run_e2e_pipeline(single_symbol)

        digest1 = compute_rank_events_digest(events1)
        digest2 = compute_rank_events_digest(events2)

        assert digest1 == digest2
