"""Tests for ML model runner with calibration."""

from __future__ import annotations

import hashlib
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
    RegimeTrend,
    RegimeVol,
)
from cryptoscreener.model_runner import (
    BaselineRunner,
    CalibrationArtifactError,
    MLRunner,
    MLRunnerConfig,
)


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
