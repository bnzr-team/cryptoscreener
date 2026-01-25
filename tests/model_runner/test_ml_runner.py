"""Tests for ML model runner with calibration."""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

import pytest

from cryptoscreener.calibration import CalibrationArtifact, PlattCalibrator
from cryptoscreener.calibration.artifact import CalibrationMetadata
from cryptoscreener.contracts.events import (
    DataHealth,
    Features,
    FeatureSnapshot,
    PredictionStatus,
    ReasonCode,
    RegimeTrend,
    RegimeVol,
)
from cryptoscreener.model_runner import (
    ArtifactIntegrityError,
    BaselineRunner,
    CalibrationArtifactError,
    MLRunner,
    MLRunnerConfig,
)
from cryptoscreener.model_runner.ml_runner import _compute_file_sha256


class MockSklearnModel:
    """Mock sklearn-style model for testing (must be module-level for pickle)."""

    def predict_proba(self, X):
        # Use lists instead of numpy arrays for portability
        # sklearn models accept list-of-lists format
        return [
            [[0.3, 0.7]],  # p_inplay_30s
            [[0.4, 0.6]],  # p_inplay_2m
            [[0.35, 0.65]],  # p_inplay_5m
            [[0.1, 0.1]],  # p_toxic
        ]


def make_feature_snapshot(
    symbol: str = "BTCUSDT",
    ts: int = 1000000000000,
    spread_bps: float = 2.0,
    mid: float = 50000.0,
    book_imbalance: float = 0.3,
    flow_imbalance: float = 0.4,
    natr: float = 0.02,
    impact_bps: float = 5.0,
    regime_vol: RegimeVol = RegimeVol.HIGH,
    regime_trend: RegimeTrend = RegimeTrend.TREND,
    stale_book_ms: int = 0,
    stale_trades_ms: int = 0,
) -> FeatureSnapshot:
    """Create a test FeatureSnapshot."""
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
            stale_book_ms=stale_book_ms,
            stale_trades_ms=stale_trades_ms,
        ),
    )


def make_calibration_artifact(
    heads: list[str] | None = None,
) -> CalibrationArtifact:
    """Create a test calibration artifact."""
    heads = heads or ["p_inplay_30s", "p_inplay_2m", "p_inplay_5m", "p_toxic"]

    calibrators = {head: PlattCalibrator(a=1.2, b=-0.1, head_name=head) for head in heads}

    metadata = CalibrationMetadata(
        schema_version="1.0.0",
        git_sha="abc123",
        config_hash="config123",
        data_hash="data456",
        calibration_timestamp="2024-01-01T00:00:00Z",
        method="platt",
        heads=heads,
        n_samples=1000,
    )

    return CalibrationArtifact(calibrators=calibrators, metadata=metadata)


def save_calibration_to_temp(artifact: CalibrationArtifact) -> Path:
    """Save calibration artifact to temp file."""
    import orjson

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as f:
        f.write(orjson.dumps(artifact.to_dict(), option=orjson.OPT_INDENT_2))
        return Path(f.name)


class TestMLRunnerFallback:
    """Tests for MLRunner fallback behavior."""

    def test_fallback_when_no_model(self) -> None:
        """MLRunner should fall back to baseline when no model provided."""
        config = MLRunnerConfig(
            model_path=None,
            calibration_path=None,
            fallback_to_baseline=True,
        )
        runner = MLRunner(config)

        assert runner.is_using_fallback is True
        assert runner.has_calibration is False

    def test_fallback_when_model_not_found(self) -> None:
        """MLRunner should fall back when model file doesn't exist."""
        config = MLRunnerConfig(
            model_path=Path("/nonexistent/model.pkl"),
            calibration_path=None,
            fallback_to_baseline=True,
        )
        runner = MLRunner(config)

        assert runner.is_using_fallback is True

    def test_fallback_produces_valid_prediction(self) -> None:
        """Fallback runner should produce valid PredictionSnapshot."""
        config = MLRunnerConfig(fallback_to_baseline=True)
        runner = MLRunner(config)
        snapshot = make_feature_snapshot()

        prediction = runner.predict(snapshot)

        assert prediction.symbol == "BTCUSDT"
        assert 0 <= prediction.p_inplay_30s <= 1
        assert 0 <= prediction.p_inplay_2m <= 1
        assert 0 <= prediction.p_inplay_5m <= 1
        assert 0 <= prediction.p_toxic <= 1
        assert prediction.status in PredictionStatus

    def test_fallback_matches_baseline_runner(self) -> None:
        """Fallback should produce same output as BaselineRunner."""
        config = MLRunnerConfig(fallback_to_baseline=True)
        ml_runner = MLRunner(config)
        baseline_runner = BaselineRunner()

        snapshot = make_feature_snapshot()

        ml_prediction = ml_runner.predict(snapshot)
        baseline_prediction = baseline_runner.predict(snapshot)

        # Same probabilities
        assert ml_prediction.p_inplay_30s == baseline_prediction.p_inplay_30s
        assert ml_prediction.p_inplay_2m == baseline_prediction.p_inplay_2m
        assert ml_prediction.p_inplay_5m == baseline_prediction.p_inplay_5m
        assert ml_prediction.p_toxic == baseline_prediction.p_toxic

        # Same status
        assert ml_prediction.status == baseline_prediction.status


class TestMLRunnerCalibration:
    """Tests for MLRunner calibration integration."""

    def test_loads_calibration_artifact(self) -> None:
        """MLRunner should load calibration artifact."""
        artifact = make_calibration_artifact()
        cal_path = save_calibration_to_temp(artifact)

        try:
            config = MLRunnerConfig(
                calibration_path=cal_path,
                require_calibration=True,
                fallback_to_baseline=True,
            )
            runner = MLRunner(config)

            assert runner.has_calibration is True
            assert "p_inplay_30s" in runner.calibration_heads
            assert "p_inplay_2m" in runner.calibration_heads
            assert "p_toxic" in runner.calibration_heads
        finally:
            cal_path.unlink()

    def test_calibration_not_required_allows_missing(self) -> None:
        """Should not raise when calibration missing and not required."""
        config = MLRunnerConfig(
            calibration_path=Path("/nonexistent/cal.json"),
            require_calibration=False,
            fallback_to_baseline=True,
        )
        # Should not raise
        runner = MLRunner(config)
        assert runner.has_calibration is False

    def test_calibration_required_raises_when_missing(self) -> None:
        """Should raise CalibrationArtifactError when required but missing."""
        config = MLRunnerConfig(
            calibration_path=Path("/nonexistent/cal.json"),
            require_calibration=True,
            fallback_to_baseline=True,
        )

        with pytest.raises(CalibrationArtifactError):
            MLRunner(config)

    def test_calibration_version_updated_from_artifact(self) -> None:
        """calibration_version should reflect loaded artifact."""
        artifact = make_calibration_artifact()
        cal_path = save_calibration_to_temp(artifact)

        try:
            config = MLRunnerConfig(
                calibration_path=cal_path,
                require_calibration=True,
                fallback_to_baseline=True,
            )
            runner = MLRunner(config)

            # Should include schema version and git sha
            assert "1.0.0" in runner.calibration_version
            assert "abc123" in runner.calibration_version
        finally:
            cal_path.unlink()


class TestMLRunnerDataIssues:
    """Tests for MLRunner data issue handling."""

    def test_stale_book_returns_data_issue(self) -> None:
        """Should return DATA_ISSUE status for stale book data."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))
        snapshot = make_feature_snapshot(stale_book_ms=10000)

        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE
        assert any(r.code == "RC_DATA_STALE" for r in prediction.reasons)

    def test_stale_trades_returns_data_issue(self) -> None:
        """Should return DATA_ISSUE status for stale trade data."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))
        snapshot = make_feature_snapshot(stale_trades_ms=35000)

        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE


class TestMLRunnerDeterminism:
    """Tests for MLRunner determinism (replay-safe)."""

    def test_same_input_same_output(self) -> None:
        """Same input should produce identical output."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))
        snapshot = make_feature_snapshot()

        pred1 = runner.predict(snapshot)
        pred2 = runner.predict(snapshot)

        # All fields should match
        assert pred1.to_json() == pred2.to_json()

    def test_determinism_across_runner_instances(self) -> None:
        """Different runner instances should produce same output."""
        config = MLRunnerConfig(fallback_to_baseline=True)
        runner1 = MLRunner(config)
        runner2 = MLRunner(config)

        snapshot = make_feature_snapshot()

        pred1 = runner1.predict(snapshot)
        pred2 = runner2.predict(snapshot)

        assert pred1.to_json() == pred2.to_json()

    def test_batch_matches_sequential(self) -> None:
        """Batch prediction should match sequential predictions."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))
        snapshots = [
            make_feature_snapshot(symbol="BTCUSDT", ts=1000),
            make_feature_snapshot(symbol="ETHUSDT", ts=1000),
            make_feature_snapshot(symbol="SOLUSDT", ts=1000),
        ]

        batch_results = runner.predict_batch(snapshots)
        sequential_results = [runner.predict(s) for s in snapshots]

        for batch, seq in zip(batch_results, sequential_results, strict=True):
            assert batch.to_json() == seq.to_json()


class TestMLRunnerGates:
    """Tests for PRD critical gate enforcement."""

    def test_spread_gate_blocks_tradeable(self) -> None:
        """Spread exceeding max should block TRADEABLE."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))
        # High spread
        snapshot = make_feature_snapshot(spread_bps=15.0)

        prediction = runner.predict(snapshot)

        # Should be WATCH, not TRADEABLE (gate failed)
        if prediction.status not in (PredictionStatus.DEAD, PredictionStatus.TRAP):
            assert prediction.status == PredictionStatus.WATCH
            assert any(r.code == "RC_GATE_SPREAD_FAIL" for r in prediction.reasons)

    def test_impact_gate_blocks_tradeable(self) -> None:
        """Impact exceeding max should block TRADEABLE."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))
        # High impact
        snapshot = make_feature_snapshot(impact_bps=25.0)

        prediction = runner.predict(snapshot)

        # Should be WATCH, not TRADEABLE (gate failed)
        if prediction.status not in (PredictionStatus.DEAD, PredictionStatus.TRAP):
            assert prediction.status == PredictionStatus.WATCH
            assert any(r.code == "RC_GATE_IMPACT_FAIL" for r in prediction.reasons)


class TestMLRunnerMetrics:
    """Tests for MLRunner metrics tracking."""

    def test_tracks_predictions_made(self) -> None:
        """Should track number of predictions made."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))
        snapshot = make_feature_snapshot()

        runner.predict(snapshot)
        runner.predict(snapshot)

        assert runner.metrics.predictions_made == 2

    def test_tracks_per_symbol(self) -> None:
        """Should track predictions per symbol."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))

        runner.predict(make_feature_snapshot(symbol="BTCUSDT"))
        runner.predict(make_feature_snapshot(symbol="BTCUSDT"))
        runner.predict(make_feature_snapshot(symbol="ETHUSDT"))

        assert runner.metrics.predictions_per_symbol["BTCUSDT"] == 2
        assert runner.metrics.predictions_per_symbol["ETHUSDT"] == 1

    def test_reset_clears_metrics(self) -> None:
        """reset_metrics should clear all counters."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))

        runner.predict(make_feature_snapshot())
        assert runner.metrics.predictions_made == 1

        runner.reset_metrics()
        assert runner.metrics.predictions_made == 0


class TestMLRunnerReplayDeterminism:
    """Comprehensive replay determinism tests.

    CRITICAL: Same input data + same config → same output events.
    This is required for reproducible backtests.
    """

    def test_digest_stable_across_runs(self) -> None:
        """Runner digest should be stable for same config."""
        config = MLRunnerConfig(
            model_version="ml-v1.0.0+abc1234",
            calibration_version="cal-v1.0.0+def5678",
            fallback_to_baseline=True,
        )
        runner1 = MLRunner(config)
        runner2 = MLRunner(config)

        assert runner1.compute_digest() == runner2.compute_digest()

    def test_prediction_json_deterministic(self) -> None:
        """Prediction JSON should be byte-for-byte identical."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))
        snapshot = make_feature_snapshot(
            symbol="BTCUSDT",
            ts=1704067200000,
            spread_bps=2.5,
            mid=45000.0,
            book_imbalance=0.35,
            flow_imbalance=0.42,
        )

        pred1 = runner.predict(snapshot)
        pred2 = runner.predict(snapshot)

        # JSON bytes must match
        json1 = pred1.to_json()
        json2 = pred2.to_json()
        assert json1 == json2

        # Hash must match
        hash1 = hashlib.sha256(json1).hexdigest()
        hash2 = hashlib.sha256(json2).hexdigest()
        assert hash1 == hash2

    def test_multi_symbol_replay_determinism(self) -> None:
        """Multi-symbol batch should produce deterministic hashes."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))

        snapshots = [
            make_feature_snapshot(symbol="BTCUSDT", ts=1000, spread_bps=2.0),
            make_feature_snapshot(symbol="ETHUSDT", ts=1000, spread_bps=3.0),
            make_feature_snapshot(symbol="SOLUSDT", ts=1000, spread_bps=4.0),
            make_feature_snapshot(symbol="BNBUSDT", ts=1000, spread_bps=2.5),
        ]

        # Run 1
        preds1 = runner.predict_batch(snapshots)
        hash1 = hashlib.sha256(b"".join(p.to_json() for p in preds1)).hexdigest()

        # Run 2
        preds2 = runner.predict_batch(snapshots)
        hash2 = hashlib.sha256(b"".join(p.to_json() for p in preds2)).hexdigest()

        assert hash1 == hash2, "Replay determinism violated"


class TestMLRunnerArtifactIntegrity:
    """Tests for artifact hash verification."""

    def test_calibration_hash_match_succeeds(self) -> None:
        """Should load calibration when hash matches."""
        artifact = make_calibration_artifact()
        cal_path = save_calibration_to_temp(artifact)

        try:
            # Compute actual hash
            actual_hash = _compute_file_sha256(cal_path)

            config = MLRunnerConfig(
                calibration_path=cal_path,
                calibration_sha256=actual_hash,
                require_calibration=True,
                fallback_to_baseline=True,
            )
            runner = MLRunner(config)

            assert runner.has_calibration is True
        finally:
            cal_path.unlink()

    def test_calibration_hash_mismatch_raises(self) -> None:
        """Should raise CalibrationArtifactError when hash mismatches."""
        artifact = make_calibration_artifact()
        cal_path = save_calibration_to_temp(artifact)

        try:
            wrong_hash = "0" * 64  # Clearly wrong hash

            config = MLRunnerConfig(
                calibration_path=cal_path,
                calibration_sha256=wrong_hash,
                require_calibration=True,
                fallback_to_baseline=True,
            )

            with pytest.raises(CalibrationArtifactError) as exc_info:
                MLRunner(config)

            assert "integrity" in str(exc_info.value).lower()
        finally:
            cal_path.unlink()

    def test_calibration_hash_mismatch_fallback_no_calibration(self) -> None:
        """Should proceed without calibration when hash mismatches and not required."""
        artifact = make_calibration_artifact()
        cal_path = save_calibration_to_temp(artifact)

        try:
            wrong_hash = "0" * 64

            config = MLRunnerConfig(
                calibration_path=cal_path,
                calibration_sha256=wrong_hash,
                require_calibration=False,  # Not required
                fallback_to_baseline=True,
            )

            runner = MLRunner(config)
            assert runner.has_calibration is False  # Calibration skipped due to hash mismatch
        finally:
            cal_path.unlink()

    def test_model_hash_mismatch_raises_without_fallback(self) -> None:
        """Should raise ArtifactIntegrityError when model hash mismatches."""
        import pickle

        # Create a temp model file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump({"dummy": "model"}, f)
            model_path = Path(f.name)

        try:
            wrong_hash = "0" * 64

            config = MLRunnerConfig(
                model_path=model_path,
                model_sha256=wrong_hash,
                fallback_to_baseline=False,  # No fallback
            )

            with pytest.raises(ArtifactIntegrityError) as exc_info:
                MLRunner(config)

            assert "mismatch" in str(exc_info.value).lower()
        finally:
            model_path.unlink()

    def test_model_hash_mismatch_falls_back(self) -> None:
        """Should fall back to baseline when model hash mismatches."""
        import pickle

        # Create a temp model file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump({"dummy": "model"}, f)
            model_path = Path(f.name)

        try:
            wrong_hash = "0" * 64

            config = MLRunnerConfig(
                model_path=model_path,
                model_sha256=wrong_hash,
                fallback_to_baseline=True,  # Allow fallback
            )

            runner = MLRunner(config)
            assert runner.is_using_fallback is True

            # Should still produce valid predictions via fallback
            snapshot = make_feature_snapshot()
            prediction = runner.predict(snapshot)
            assert 0 <= prediction.p_inplay_2m <= 1
        finally:
            model_path.unlink()

    def test_model_hash_match_succeeds(self) -> None:
        """Should load model when hash matches."""
        pytest.importorskip("numpy")

        import pickle

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump(MockSklearnModel(), f)
            model_path = Path(f.name)

        try:
            actual_hash = _compute_file_sha256(model_path)

            config = MLRunnerConfig(
                model_path=model_path,
                model_sha256=actual_hash,
                fallback_to_baseline=False,
                require_calibration=False,
            )

            runner = MLRunner(config)
            assert runner.is_using_fallback is False

            # Should produce valid predictions from model
            snapshot = make_feature_snapshot()
            prediction = runner.predict(snapshot)
            assert 0 <= prediction.p_inplay_2m <= 1
        finally:
            model_path.unlink()

    def test_hash_verification_case_insensitive(self) -> None:
        """Hash comparison should be case-insensitive."""
        artifact = make_calibration_artifact()
        cal_path = save_calibration_to_temp(artifact)

        try:
            actual_hash = _compute_file_sha256(cal_path)
            # Use uppercase
            upper_hash = actual_hash.upper()

            config = MLRunnerConfig(
                calibration_path=cal_path,
                calibration_sha256=upper_hash,
                require_calibration=True,
                fallback_to_baseline=True,
            )
            runner = MLRunner(config)

            assert runner.has_calibration is True
        finally:
            cal_path.unlink()


class TestEvidenceNoDigitsPolicy:
    """Tests enforcing LLM-friendly evidence strings (no digits).

    POLICY: ReasonCode.evidence must NOT contain digits (0-9).
    Numbers belong in the 'value' field; evidence is for LLM consumption.

    This prevents the LLM from extracting new numbers from evidence strings,
    enforcing the "no-new-numbers" policy for downstream consumers.
    """

    DIGIT_PATTERN = re.compile(r"\d")

    def _assert_no_digits_in_evidence(self, reasons: list[ReasonCode]) -> None:
        """Assert that no evidence string contains digits."""
        for reason in reasons:
            if self.DIGIT_PATTERN.search(reason.evidence):
                pytest.fail(
                    f"Evidence string contains digits (violates no-digits policy): "
                    f"code={reason.code}, evidence='{reason.evidence}'"
                )

    def test_baseline_runner_evidence_no_digits(self) -> None:
        """BaselineRunner evidence strings must not contain digits."""
        from cryptoscreener.model_runner.baseline import BaselineRunner

        runner = BaselineRunner()

        # Test various scenarios that produce different ReasonCodes
        test_cases = [
            # Normal case with flow imbalance
            make_feature_snapshot(flow_imbalance=0.5, book_imbalance=0.4),
            # High spread (gate failure)
            make_feature_snapshot(spread_bps=15.0),
            # High impact (gate failure)
            make_feature_snapshot(impact_bps=25.0),
            # Stale data
            make_feature_snapshot(stale_book_ms=10000),
            # Tight spread
            make_feature_snapshot(spread_bps=1.0),
            # Wide spread
            make_feature_snapshot(spread_bps=12.0),
            # High volatility regime
            make_feature_snapshot(regime_vol=RegimeVol.HIGH),
        ]

        for snapshot in test_cases:
            prediction = runner.predict(snapshot)
            self._assert_no_digits_in_evidence(prediction.reasons)

    def test_ml_runner_fallback_evidence_no_digits(self) -> None:
        """MLRunner (fallback mode) evidence strings must not contain digits."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))

        test_cases = [
            make_feature_snapshot(flow_imbalance=0.5, book_imbalance=0.4),
            make_feature_snapshot(spread_bps=15.0),
            make_feature_snapshot(impact_bps=25.0),
            make_feature_snapshot(stale_book_ms=10000),
        ]

        for snapshot in test_cases:
            prediction = runner.predict(snapshot)
            self._assert_no_digits_in_evidence(prediction.reasons)

    def test_data_issue_evidence_no_digits(self) -> None:
        """DATA_ISSUE prediction evidence must not contain digits."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))

        # Stale book data
        snapshot = make_feature_snapshot(stale_book_ms=10000)
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE
        self._assert_no_digits_in_evidence(prediction.reasons)

    def test_gate_failure_evidence_no_digits(self) -> None:
        """Gate failure evidence must not contain digits."""
        runner = MLRunner(MLRunnerConfig(fallback_to_baseline=True))

        # Spread gate failure
        snapshot = make_feature_snapshot(spread_bps=15.0)
        prediction = runner.predict(snapshot)

        gate_reasons = [r for r in prediction.reasons if "GATE" in r.code]
        assert len(gate_reasons) > 0, "Expected gate failure reasons"
        self._assert_no_digits_in_evidence(gate_reasons)


class TestProdModeStrictness:
    """Tests for PROD mode strictness (DEC-017).

    In PROD mode:
    - No fallback to baseline when model missing
    - No fallback when calibration missing
    - No exceptions raised (returns DATA_ISSUE instead)
    - Never returns TRADEABLE without valid model+calibration
    """

    def test_prod_mode_no_model_returns_data_issue(self) -> None:
        """PROD mode: missing model → DATA_ISSUE with RC_MODEL_UNAVAILABLE."""
        from cryptoscreener.model_runner import InferenceStrictness

        config = MLRunnerConfig(
            strictness=InferenceStrictness.PROD,
            # No model_path → model missing
        )
        runner = MLRunner(config)

        # Should have artifact error flag with specific code
        assert runner.has_artifact_error
        assert runner.artifact_error_code == "RC_MODEL_UNAVAILABLE"
        assert runner.artifact_error_reason is not None
        assert "Model path not configured" in runner.artifact_error_reason

        # Should NOT be using fallback (PROD doesn't fallback)
        assert not runner.is_using_fallback

        # predict() should return DATA_ISSUE with correct reason code
        snapshot = make_feature_snapshot()
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE
        assert len(prediction.reasons) == 1
        assert prediction.reasons[0].code == "RC_MODEL_UNAVAILABLE"

    def test_prod_mode_no_calibration_returns_data_issue(self, tmp_path: Path) -> None:
        """PROD mode: missing calibration → DATA_ISSUE with RC_CALIBRATION_MISSING."""
        from cryptoscreener.model_runner import InferenceStrictness

        # Create a valid model file
        model_path = tmp_path / "model.pkl"
        import pickle

        with model_path.open("wb") as f:
            pickle.dump(MockSklearnModel(), f)

        config = MLRunnerConfig(
            model_path=model_path,
            strictness=InferenceStrictness.PROD,
            # No calibration_path → calibration missing
        )
        runner = MLRunner(config)

        # Should have artifact error with specific code
        assert runner.has_artifact_error
        assert runner.artifact_error_code == "RC_CALIBRATION_MISSING"
        assert runner.artifact_error_reason is not None
        assert "Calibration path not configured" in runner.artifact_error_reason

        # predict() should return DATA_ISSUE with correct reason code
        snapshot = make_feature_snapshot()
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE
        assert prediction.reasons[0].code == "RC_CALIBRATION_MISSING"

    def test_prod_mode_hash_mismatch_returns_data_issue(self, tmp_path: Path) -> None:
        """PROD mode: model hash mismatch → DATA_ISSUE with RC_ARTIFACT_INTEGRITY_FAIL."""
        from cryptoscreener.model_runner import InferenceStrictness

        # Create a valid model file
        model_path = tmp_path / "model.pkl"
        import pickle

        with model_path.open("wb") as f:
            pickle.dump(MockSklearnModel(), f)

        config = MLRunnerConfig(
            model_path=model_path,
            model_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            strictness=InferenceStrictness.PROD,
        )
        runner = MLRunner(config)

        # Should have artifact error with specific code
        assert runner.has_artifact_error
        assert runner.artifact_error_code == "RC_ARTIFACT_INTEGRITY_FAIL"
        assert runner.artifact_error_reason is not None
        assert "hash mismatch" in runner.artifact_error_reason.lower()

        # predict() should return DATA_ISSUE with correct reason code
        snapshot = make_feature_snapshot()
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE
        assert prediction.reasons[0].code == "RC_ARTIFACT_INTEGRITY_FAIL"

    def test_dev_mode_fallback_works(self) -> None:
        """DEV mode: missing model → fallback to baseline (default behavior)."""
        from cryptoscreener.model_runner import InferenceStrictness

        config = MLRunnerConfig(
            strictness=InferenceStrictness.DEV,
            fallback_to_baseline=True,
            # No model_path → fallback
        )
        runner = MLRunner(config)

        # Should be using fallback (DEV allows it)
        assert runner.is_using_fallback
        assert not runner.has_artifact_error

        # predict() should work via baseline
        snapshot = make_feature_snapshot()
        prediction = runner.predict(snapshot)

        # Should get a valid prediction (not DATA_ISSUE)
        assert prediction.status in (
            PredictionStatus.TRADEABLE,
            PredictionStatus.WATCH,
            PredictionStatus.DEAD,
            PredictionStatus.TRAP,
        )

    def test_dev_mode_missing_calibration_allowed(self, tmp_path: Path) -> None:
        """DEV mode: missing calibration with require_calibration=False works."""
        from cryptoscreener.model_runner import InferenceStrictness

        # Create a valid model file
        model_path = tmp_path / "model.pkl"
        import pickle

        with model_path.open("wb") as f:
            pickle.dump(MockSklearnModel(), f)

        config = MLRunnerConfig(
            model_path=model_path,
            strictness=InferenceStrictness.DEV,
            require_calibration=False,
            # No calibration_path but not required
        )
        runner = MLRunner(config)

        # Should not have error (model loaded, calibration optional)
        assert not runner.has_artifact_error
        assert not runner.has_calibration
        assert not runner.is_using_fallback  # Has model, not falling back
        assert runner.strictness == InferenceStrictness.DEV

    def test_strictness_property(self) -> None:
        """strictness property returns configured value."""
        from cryptoscreener.model_runner import InferenceStrictness

        config_dev = MLRunnerConfig(strictness=InferenceStrictness.DEV)
        config_prod = MLRunnerConfig(strictness=InferenceStrictness.PROD)

        runner_dev = MLRunner(config_dev)
        runner_prod = MLRunner(config_prod)

        assert runner_dev.strictness == InferenceStrictness.DEV
        assert runner_prod.strictness == InferenceStrictness.PROD

    def test_prod_mode_calibration_hash_mismatch(self, tmp_path: Path) -> None:
        """PROD mode: calibration hash mismatch → DATA_ISSUE with RC_ARTIFACT_INTEGRITY_FAIL."""
        from cryptoscreener.model_runner import InferenceStrictness

        # Create valid model and calibration files
        model_path = tmp_path / "model.pkl"
        calibration_path = tmp_path / "calibration.json"
        import json
        import pickle

        with model_path.open("wb") as f:
            pickle.dump(MockSklearnModel(), f)

        # Create minimal calibration artifact
        calibration_data = {
            "calibrators": {"p_inplay_2m": {"a": 1.0, "b": 0.0}},
            "metadata": {
                "schema_version": "1.0.0",
                "git_sha": "abc1234",
                "config_hash": "xyz",
                "data_hash": "123",
            },
        }
        with calibration_path.open("w") as f:
            json.dump(calibration_data, f)

        config = MLRunnerConfig(
            model_path=model_path,
            calibration_path=calibration_path,
            calibration_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            strictness=InferenceStrictness.PROD,
        )
        runner = MLRunner(config)

        # Should have artifact error with integrity code
        assert runner.has_artifact_error
        assert runner.artifact_error_code == "RC_ARTIFACT_INTEGRITY_FAIL"
        assert runner.artifact_error_reason is not None
        assert "hash mismatch" in runner.artifact_error_reason.lower()

        # predict() should return DATA_ISSUE with correct reason code
        snapshot = make_feature_snapshot()
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE
        assert prediction.reasons[0].code == "RC_ARTIFACT_INTEGRITY_FAIL"


class TestDataFreshnessThresholds:
    """Tests for configurable data freshness thresholds (DEC-017).

    Per DATA_FRESHNESS_RULES.md SSOT:
    - Default book threshold: 1000ms
    - Default trades threshold: 2000ms
    """

    def test_default_thresholds_match_ssot(self) -> None:
        """Default thresholds match DATA_FRESHNESS_RULES.md SSOT."""
        from cryptoscreener.model_runner import ModelRunnerConfig

        config = ModelRunnerConfig()
        assert config.stale_book_max_ms == 1000  # SSOT: 1000ms
        assert config.stale_trades_max_ms == 2000  # SSOT: 2000ms

    def test_custom_book_threshold(self) -> None:
        """Custom book stale threshold is honored."""
        from cryptoscreener.model_runner import ModelRunnerConfig

        # With lenient threshold (5000ms) - should not trigger DATA_ISSUE
        config = ModelRunnerConfig(stale_book_max_ms=5000)
        runner = BaselineRunner(config)

        snapshot = make_feature_snapshot(stale_book_ms=3000)  # 3s < 5s
        prediction = runner.predict(snapshot)

        assert prediction.status != PredictionStatus.DATA_ISSUE

    def test_custom_trades_threshold(self) -> None:
        """Custom trades stale threshold is honored."""
        from cryptoscreener.model_runner import ModelRunnerConfig

        # With lenient threshold (30000ms) - should not trigger DATA_ISSUE
        config = ModelRunnerConfig(stale_trades_max_ms=30000)
        runner = BaselineRunner(config)

        snapshot = make_feature_snapshot(stale_trades_ms=15000)  # 15s < 30s
        prediction = runner.predict(snapshot)

        assert prediction.status != PredictionStatus.DATA_ISSUE

    def test_strict_book_threshold_triggers_data_issue(self) -> None:
        """Book exceeding strict threshold triggers DATA_ISSUE."""
        from cryptoscreener.model_runner import ModelRunnerConfig

        config = ModelRunnerConfig(stale_book_max_ms=1000)  # SSOT default
        runner = BaselineRunner(config)

        snapshot = make_feature_snapshot(stale_book_ms=1500)  # 1.5s > 1s
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE

    def test_strict_trades_threshold_triggers_data_issue(self) -> None:
        """Trades exceeding strict threshold triggers DATA_ISSUE."""
        from cryptoscreener.model_runner import ModelRunnerConfig

        config = ModelRunnerConfig(stale_trades_max_ms=2000)  # SSOT default
        runner = BaselineRunner(config)

        snapshot = make_feature_snapshot(stale_trades_ms=2500)  # 2.5s > 2s
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE
