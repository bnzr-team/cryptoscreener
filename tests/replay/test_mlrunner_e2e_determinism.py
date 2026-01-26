"""MLRunner End-to-end determinism tests.

Tests the full MLRunner flow: FeatureSnapshot → MLRunner → Scorer → Ranker → RankEvent

This is the acceptance test for DEC-019: MLRunner E2E Acceptance as CI Gate.

MLRunner Determinism Contract:
1. DEV mode with no model → falls back to BaselineRunner (deterministic)
2. PROD mode with no model → returns DATA_ISSUE (deterministic)
3. Both paths produce stable digests across multiple runs

This test suite validates that the MLRunner path produces identical output
given identical input, regardless of run order or invocation count.
"""

from __future__ import annotations

import hashlib

import pytest

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
from cryptoscreener.model_runner import (
    InferenceStrictness,
    MLRunner,
    MLRunnerConfig,
)
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
MLRUNNER_FIXTURE: list[list[FeatureSnapshot]] = [
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


def run_mlrunner_pipeline_dev(
    fixture: list[list[FeatureSnapshot]],
) -> tuple[list[RankEvent], list[PredictionSnapshot]]:
    """Run the MLRunner pipeline in DEV mode (fallback to baseline).

    Pipeline: FeatureSnapshot → MLRunner(DEV) → Scorer → Ranker → RankEvent

    In DEV mode without model artifacts, MLRunner falls back to BaselineRunner,
    which is fully deterministic.

    Args:
        fixture: List of frames, each containing FeatureSnapshots.

    Returns:
        Tuple of (all RankEvents, all PredictionSnapshots).
    """
    # Initialize MLRunner in DEV mode (default)
    # No model path → falls back to BaselineRunner
    runner_config = MLRunnerConfig(
        strictness=InferenceStrictness.DEV,
        fallback_to_baseline=True,
    )
    scorer_config = ScorerConfig()
    ranker_config = RankerConfig(
        top_k=5,
        enter_ms=1500,
        exit_ms=3000,
        min_dwell_ms=2000,
        score_threshold=0.001,
    )

    runner = MLRunner(runner_config)
    scorer = Scorer(scorer_config)
    ranker = Ranker(ranker_config, scorer)

    # Verify we're using fallback
    assert runner.is_using_fallback, (
        "MLRunner should fall back to baseline in DEV mode without model"
    )

    all_events: list[RankEvent] = []
    all_predictions: list[PredictionSnapshot] = []

    for frame in fixture:
        if not frame:
            continue

        ts = frame[0].ts
        predictions: dict[str, PredictionSnapshot] = {}

        for snapshot in frame:
            prediction = runner.predict(snapshot)
            predictions[snapshot.symbol] = prediction
            all_predictions.append(prediction)

        events = ranker.update(predictions, ts)
        all_events.extend(events)

    return all_events, all_predictions


def run_mlrunner_pipeline_prod(
    fixture: list[list[FeatureSnapshot]],
) -> tuple[list[RankEvent], list[PredictionSnapshot]]:
    """Run the MLRunner pipeline in PROD mode (no fallback).

    Pipeline: FeatureSnapshot → MLRunner(PROD) → Scorer → Ranker → RankEvent

    In PROD mode without model artifacts, MLRunner returns DATA_ISSUE for all
    predictions (fail-safe behavior per DEC-017).

    Args:
        fixture: List of frames, each containing FeatureSnapshots.

    Returns:
        Tuple of (all RankEvents, all PredictionSnapshots).
    """
    # Initialize MLRunner in PROD mode
    # No model path → artifact error, returns DATA_ISSUE
    runner_config = MLRunnerConfig(
        strictness=InferenceStrictness.PROD,
    )
    scorer_config = ScorerConfig()
    ranker_config = RankerConfig(
        top_k=5,
        enter_ms=1500,
        exit_ms=3000,
        min_dwell_ms=2000,
        score_threshold=0.001,
    )

    runner = MLRunner(runner_config)
    scorer = Scorer(scorer_config)
    ranker = Ranker(ranker_config, scorer)

    # Verify we have artifact error (PROD mode)
    assert runner.has_artifact_error, (
        "MLRunner should have artifact error in PROD mode without model"
    )
    assert runner.artifact_error_code == "RC_MODEL_UNAVAILABLE"

    all_events: list[RankEvent] = []
    all_predictions: list[PredictionSnapshot] = []

    for frame in fixture:
        if not frame:
            continue

        ts = frame[0].ts
        predictions: dict[str, PredictionSnapshot] = {}

        for snapshot in frame:
            prediction = runner.predict(snapshot)
            predictions[snapshot.symbol] = prediction
            all_predictions.append(prediction)

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


class TestMLRunnerDevModeDeterminism:
    """MLRunner DEV mode determinism tests (fallback to baseline)."""

    def test_same_input_produces_same_output(self) -> None:
        """Same fixture produces identical RankEvents across runs."""
        events1, _ = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)
        events2, _ = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)

        assert len(events1) == len(events2), "Event count mismatch"

        for e1, e2 in zip(events1, events2, strict=True):
            assert e1.ts == e2.ts
            assert e1.event == e2.event
            assert e1.symbol == e2.symbol
            assert e1.rank == e2.rank
            assert abs(e1.score - e2.score) < 1e-6

    def test_rank_events_digest_stable(self) -> None:
        """RankEvent digest is stable across runs."""
        events1, _ = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)
        events2, _ = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)

        digest1 = compute_rank_events_digest(events1)
        digest2 = compute_rank_events_digest(events2)

        assert digest1 == digest2, f"Digest mismatch: {digest1} != {digest2}"

    def test_predictions_digest_stable(self) -> None:
        """PredictionSnapshot digest is stable across runs."""
        _, predictions1 = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)
        _, predictions2 = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)

        digest1 = compute_predictions_digest(predictions1)
        digest2 = compute_predictions_digest(predictions2)

        assert digest1 == digest2, f"Digest mismatch: {digest1} != {digest2}"

    def test_produces_rank_events(self) -> None:
        """Pipeline produces at least some RankEvents."""
        events, predictions = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)

        expected_predictions = sum(len(frame) for frame in MLRUNNER_FIXTURE)
        assert len(predictions) == expected_predictions

        # MUST produce at least 1 RankEvent (prevents empty digest regression)
        assert len(events) > 0, "Pipeline must produce at least one RankEvent"

        for event in events:
            assert event.ts > 0
            assert event.symbol in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
            assert event.score >= 0

    def test_runner_uses_fallback(self) -> None:
        """Verify MLRunner uses fallback to baseline in DEV mode."""
        config = MLRunnerConfig(
            strictness=InferenceStrictness.DEV,
            fallback_to_baseline=True,
        )
        runner = MLRunner(config)

        assert runner.is_using_fallback, "Should use fallback without model"
        assert runner.strictness == InferenceStrictness.DEV
        assert not runner.has_artifact_error  # DEV mode doesn't track as error


class TestMLRunnerProdModeDeterminism:
    """MLRunner PROD mode determinism tests (DATA_ISSUE on missing artifacts)."""

    def test_same_input_produces_same_output(self) -> None:
        """Same fixture produces identical results across runs."""
        events1, preds1 = run_mlrunner_pipeline_prod(MLRUNNER_FIXTURE)
        events2, preds2 = run_mlrunner_pipeline_prod(MLRUNNER_FIXTURE)

        # In PROD mode without model, all predictions are DATA_ISSUE
        # So ranker produces no TRADEABLE events
        assert len(events1) == len(events2), "Event count mismatch"
        assert len(preds1) == len(preds2), "Prediction count mismatch"

    def test_predictions_digest_stable(self) -> None:
        """PredictionSnapshot digest is stable in PROD mode."""
        _, predictions1 = run_mlrunner_pipeline_prod(MLRUNNER_FIXTURE)
        _, predictions2 = run_mlrunner_pipeline_prod(MLRUNNER_FIXTURE)

        digest1 = compute_predictions_digest(predictions1)
        digest2 = compute_predictions_digest(predictions2)

        assert digest1 == digest2, f"Digest mismatch: {digest1} != {digest2}"

    def test_all_predictions_are_data_issue(self) -> None:
        """All predictions in PROD mode without model are DATA_ISSUE."""
        _, predictions = run_mlrunner_pipeline_prod(MLRUNNER_FIXTURE)

        from cryptoscreener.contracts.events import PredictionStatus

        for pred in predictions:
            assert pred.status == PredictionStatus.DATA_ISSUE
            assert any(r.code == "RC_MODEL_UNAVAILABLE" for r in pred.reasons)

    def test_runner_has_artifact_error(self) -> None:
        """Verify MLRunner has artifact error in PROD mode without model."""
        config = MLRunnerConfig(
            strictness=InferenceStrictness.PROD,
        )
        runner = MLRunner(config)

        assert runner.has_artifact_error, "Should have artifact error in PROD mode"
        assert runner.artifact_error_code == "RC_MODEL_UNAVAILABLE"
        assert not runner.is_using_fallback  # PROD mode never falls back


class TestMLRunnerE2EReplayProof:
    """Tests that generate proof artifacts for MLRunner verification."""

    def test_generate_dev_mode_replay_proof(self) -> None:
        """Generate and verify replay proof for DEV mode.

        This test:
        1. Computes input fixture digest
        2. Runs DEV mode pipeline twice
        3. Computes output digests for both runs
        4. Asserts they match
        """
        # Input digest
        input_digest = compute_fixture_digest(MLRUNNER_FIXTURE)

        # Run 1
        events1, preds1 = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)
        output_digest1 = compute_rank_events_digest(events1)
        pred_digest1 = compute_predictions_digest(preds1)

        # Run 2
        events2, preds2 = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)
        output_digest2 = compute_rank_events_digest(events2)
        pred_digest2 = compute_predictions_digest(preds2)

        # Assertions
        assert output_digest1 == output_digest2, "RankEvent digest mismatch"
        assert pred_digest1 == pred_digest2, "Prediction digest mismatch"

        # Log proof for verification (captured by pytest -s)
        print("\n=== MLRunner DEV Mode Replay Proof ===")
        print(f"Input fixture digest:  {input_digest}")
        print(f"RankEvent digest (r1): {output_digest1}")
        print(f"RankEvent digest (r2): {output_digest2}")
        print(f"Prediction digest:     {pred_digest1}")
        print(f"Digests match: {output_digest1 == output_digest2}")

    def test_generate_prod_mode_replay_proof(self) -> None:
        """Generate and verify replay proof for PROD mode.

        In PROD mode without model, all predictions are DATA_ISSUE.
        This is still deterministic behavior.
        """
        # Input digest
        input_digest = compute_fixture_digest(MLRUNNER_FIXTURE)

        # Run 1
        _events1, preds1 = run_mlrunner_pipeline_prod(MLRUNNER_FIXTURE)
        pred_digest1 = compute_predictions_digest(preds1)

        # Run 2
        _events2, preds2 = run_mlrunner_pipeline_prod(MLRUNNER_FIXTURE)
        pred_digest2 = compute_predictions_digest(preds2)

        # Assertions
        assert pred_digest1 == pred_digest2, "Prediction digest mismatch"

        # Log proof for verification
        print("\n=== MLRunner PROD Mode Replay Proof ===")
        print(f"Input fixture digest:  {input_digest}")
        print(f"Prediction digest (r1): {pred_digest1}")
        print(f"Prediction digest (r2): {pred_digest2}")
        print(f"Digests match: {pred_digest1 == pred_digest2}")
        print("Note: PROD mode without model returns DATA_ISSUE for all predictions")

    def test_rank_event_json_roundtrip(self) -> None:
        """RankEvents roundtrip through JSON correctly."""
        events, _ = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)

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
        _, predictions = run_mlrunner_pipeline_dev(MLRUNNER_FIXTURE)

        for pred in predictions:
            json_bytes = pred.to_json()
            restored = PredictionSnapshot.from_json(json_bytes)

            assert restored.ts == pred.ts
            assert restored.symbol == pred.symbol
            assert restored.status == pred.status
            assert abs(restored.p_inplay_2m - pred.p_inplay_2m) < 1e-9


class TestMLRunnerE2EEdgeCases:
    """Edge case tests for MLRunner E2E pipeline."""

    def test_empty_fixture(self) -> None:
        """Empty fixture produces no events."""
        events, predictions = run_mlrunner_pipeline_dev([])
        assert events == []
        assert predictions == []

    def test_single_frame(self) -> None:
        """Single frame is deterministic."""
        single = [MLRUNNER_FIXTURE[0]]

        events1, _ = run_mlrunner_pipeline_dev(single)
        events2, _ = run_mlrunner_pipeline_dev(single)

        digest1 = compute_rank_events_digest(events1)
        digest2 = compute_rank_events_digest(events2)

        assert digest1 == digest2

    def test_single_symbol(self) -> None:
        """Single symbol is deterministic."""
        single_symbol = [
            [frame[0]]
            for frame in MLRUNNER_FIXTURE  # Only BTCUSDT
        ]

        events1, _ = run_mlrunner_pipeline_dev(single_symbol)
        events2, _ = run_mlrunner_pipeline_dev(single_symbol)

        digest1 = compute_rank_events_digest(events1)
        digest2 = compute_rank_events_digest(events2)

        assert digest1 == digest2

    def test_fixture_digest_is_stable(self) -> None:
        """Input fixture digest is constant."""
        digest1 = compute_fixture_digest(MLRUNNER_FIXTURE)
        digest2 = compute_fixture_digest(MLRUNNER_FIXTURE)

        assert digest1 == digest2
        assert len(digest1) == 64  # SHA256 hex length


# ============================================================================
# Real Model Inference Tests (with actual model + calibration artifacts)
# ============================================================================

# Check if numpy is available for real model inference tests
try:
    import numpy as np  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def run_mlrunner_pipeline_with_model(
    fixture: list[list[FeatureSnapshot]],
) -> tuple[list[RankEvent], list[PredictionSnapshot]]:
    """Run the MLRunner pipeline with real model and calibration.

    Pipeline: FeatureSnapshot → MLRunner(with model) → Scorer → Ranker → RankEvent

    Uses actual model.pkl and calibration.json fixtures.

    Args:
        fixture: List of frames, each containing FeatureSnapshots.

    Returns:
        Tuple of (all RankEvents, all PredictionSnapshots).
    """
    from tests.fixtures.mlrunner_model import (
        CALIBRATION_PATH,
        CALIBRATION_SHA256,
        get_model_path,
        get_model_sha256,
    )

    model_path = get_model_path()
    model_sha256 = get_model_sha256()

    # Initialize MLRunner with real model and calibration
    runner_config = MLRunnerConfig(
        strictness=InferenceStrictness.DEV,
        model_path=model_path,
        model_sha256=model_sha256,
        calibration_path=CALIBRATION_PATH,
        calibration_sha256=CALIBRATION_SHA256,
        require_calibration=True,
        fallback_to_baseline=False,  # Don't fallback - use real model
        model_version=f"test-v1.0.0+{model_sha256[:7]}",  # Match fixture model
    )
    scorer_config = ScorerConfig()
    # Use faster timing for test fixture (2000ms between frames)
    # to ensure SYMBOL_ENTER events are generated
    ranker_config = RankerConfig(
        top_k=5,
        enter_ms=500,
        exit_ms=1000,
        min_dwell_ms=500,
        score_threshold=0.001,
    )

    runner = MLRunner(runner_config)
    scorer = Scorer(scorer_config)
    ranker = Ranker(ranker_config, scorer)

    # Verify model and calibration are loaded (not fallback)
    assert not runner.is_using_fallback, "MLRunner should use real model, not fallback"
    assert runner.has_calibration, "MLRunner should have calibration loaded"

    all_events: list[RankEvent] = []
    all_predictions: list[PredictionSnapshot] = []

    for frame in fixture:
        if not frame:
            continue

        ts = frame[0].ts
        predictions: dict[str, PredictionSnapshot] = {}

        for snapshot in frame:
            prediction = runner.predict(snapshot)
            predictions[snapshot.symbol] = prediction
            all_predictions.append(prediction)

        events = ranker.update(predictions, ts)
        all_events.extend(events)

    return all_events, all_predictions


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy required for real model inference")
class TestMLRunnerRealModelDeterminism:
    """MLRunner determinism tests with real model and calibration artifacts.

    These tests verify that MLRunner produces identical output when using
    actual model inference (not fallback to baseline).

    Note: These tests require numpy for scikit-learn style inference.
    """

    def test_model_fixture_exists(self) -> None:
        """Verify model and calibration fixtures exist (generated on-the-fly)."""
        from tests.fixtures.mlrunner_model import (
            CALIBRATION_PATH,
            get_model_path,
        )

        model_path = get_model_path()
        assert model_path.exists(), f"Model fixture not found: {model_path}"
        assert CALIBRATION_PATH.exists(), f"Calibration fixture not found: {CALIBRATION_PATH}"

    def test_model_loads_successfully(self) -> None:
        """MLRunner can load the model and calibration artifacts."""
        from tests.fixtures.mlrunner_model import (
            CALIBRATION_PATH,
            CALIBRATION_SHA256,
            get_model_path,
            get_model_sha256,
        )

        model_path = get_model_path()
        model_sha256 = get_model_sha256()

        config = MLRunnerConfig(
            model_path=model_path,
            model_sha256=model_sha256,
            calibration_path=CALIBRATION_PATH,
            calibration_sha256=CALIBRATION_SHA256,
            require_calibration=True,
            fallback_to_baseline=False,
        )
        runner = MLRunner(config)

        assert not runner.is_using_fallback
        assert runner.has_calibration
        assert "p_inplay_30s" in runner.calibration_heads
        assert "p_inplay_2m" in runner.calibration_heads

    def test_same_input_produces_same_output(self) -> None:
        """Same fixture produces identical predictions with real model."""
        _events1, preds1 = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)
        _events2, preds2 = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)

        assert len(preds1) == len(preds2), "Prediction count mismatch"

        for p1, p2 in zip(preds1, preds2, strict=True):
            assert p1.ts == p2.ts
            assert p1.symbol == p2.symbol
            assert abs(p1.p_inplay_30s - p2.p_inplay_30s) < 1e-6
            assert abs(p1.p_inplay_2m - p2.p_inplay_2m) < 1e-6
            assert abs(p1.p_inplay_5m - p2.p_inplay_5m) < 1e-6
            assert abs(p1.p_toxic - p2.p_toxic) < 1e-6
            assert p1.status == p2.status

    def test_predictions_digest_stable(self) -> None:
        """Prediction digest is stable with real model inference."""
        _, predictions1 = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)
        _, predictions2 = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)

        digest1 = compute_predictions_digest(predictions1)
        digest2 = compute_predictions_digest(predictions2)

        assert digest1 == digest2, f"Digest mismatch: {digest1} != {digest2}"

    def test_rank_events_digest_stable(self) -> None:
        """RankEvent digest is stable with real model inference."""
        events1, _ = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)
        events2, _ = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)

        digest1 = compute_rank_events_digest(events1)
        digest2 = compute_rank_events_digest(events2)

        assert digest1 == digest2, f"Digest mismatch: {digest1} != {digest2}"

    def test_produces_non_zero_probabilities(self) -> None:
        """Real model produces non-zero probabilities (not DATA_ISSUE)."""
        _, predictions = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)

        from cryptoscreener.contracts.events import PredictionStatus

        # At least some predictions should have non-zero probs
        has_non_zero = any(
            p.p_inplay_2m > 0.01 and p.status != PredictionStatus.DATA_ISSUE for p in predictions
        )
        assert has_non_zero, "Model should produce non-zero probabilities"

    def test_calibration_applied(self) -> None:
        """Predictions have calibration applied (not baseline heuristics)."""
        _, predictions = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)

        # All predictions should have our fixture calibration version
        for pred in predictions:
            assert "abc1234" in pred.calibration_version, (
                f"Expected calibration from fixture, got {pred.calibration_version}"
            )

        # At least some predictions should have calibration adjustment reason
        # (RC_CALIBRATION_ADJ is added when calibration differs from raw probs)
        has_cal_adj_count = sum(
            1 for p in predictions if any(r.code == "RC_CALIBRATION_ADJ" for r in p.reasons)
        )
        assert has_cal_adj_count > 0, "At least some predictions should show calibration effect"

    def test_model_version_reflects_artifact(self) -> None:
        """Model version reflects test fixture, not baseline."""
        from tests.fixtures.mlrunner_model import get_model_sha256

        model_sha256 = get_model_sha256()
        _, predictions = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)

        expected_prefix = "test-v1.0.0+"
        for pred in predictions:
            assert pred.model_version.startswith(expected_prefix), (
                f"Expected model_version starting with '{expected_prefix}', "
                f"got '{pred.model_version}'"
            )
            # Verify SHA256 fragment is in version
            assert model_sha256[:7] in pred.model_version, (
                f"Model version should contain SHA256 fragment, got {pred.model_version}"
            )

    def test_produces_rank_events(self) -> None:
        """Real model produces non-empty RankEvents (not empty digest)."""
        events, _ = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)

        # Must produce at least 1 RankEvent (prevents empty digest)
        assert len(events) > 0, "Pipeline must produce at least one RankEvent"

        # Verify events have valid structure
        for event in events:
            assert event.ts > 0
            assert event.symbol in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
            assert event.score >= 0

    def test_generate_real_model_replay_proof(self) -> None:
        """Generate and verify replay proof with real model.

        This test:
        1. Computes input fixture digest
        2. Runs pipeline with real model twice
        3. Computes output digests for both runs
        4. Asserts they match
        """
        from tests.fixtures.mlrunner_model import (
            CALIBRATION_SHA256,
            get_model_sha256,
        )

        model_sha256 = get_model_sha256()

        # Input digest
        input_digest = compute_fixture_digest(MLRUNNER_FIXTURE)

        # Run 1
        events1, preds1 = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)
        output_digest1 = compute_rank_events_digest(events1)
        pred_digest1 = compute_predictions_digest(preds1)

        # Run 2
        events2, preds2 = run_mlrunner_pipeline_with_model(MLRUNNER_FIXTURE)
        output_digest2 = compute_rank_events_digest(events2)
        pred_digest2 = compute_predictions_digest(preds2)

        # Assertions
        assert output_digest1 == output_digest2, "RankEvent digest mismatch"
        assert pred_digest1 == pred_digest2, "Prediction digest mismatch"

        # Log proof for verification
        print("\n=== MLRunner Real Model Replay Proof ===")
        print(f"Model artifact SHA256:       {model_sha256[:16]}...")
        print(f"Calibration artifact SHA256: {CALIBRATION_SHA256[:16]}...")
        print(f"Input fixture digest:        {input_digest}")
        print(f"RankEvent digest (r1):       {output_digest1}")
        print(f"RankEvent digest (r2):       {output_digest2}")
        print(f"Prediction digest (r1):      {pred_digest1}")
        print(f"Prediction digest (r2):      {pred_digest2}")
        print(f"Digests match: {output_digest1 == output_digest2}")
