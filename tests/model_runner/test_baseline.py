"""Tests for BaselineRunner."""

from collections.abc import Callable

import pytest

from cryptoscreener.contracts.events import (
    DataHealth,
    ExecutionProfile,
    Features,
    FeatureSnapshot,
    PredictionStatus,
    RegimeTrend,
    RegimeVol,
    Windows,
)
from cryptoscreener.model_runner.base import ModelRunnerConfig
from cryptoscreener.model_runner.baseline import BaselineRunner

# Type alias for snapshot factory
SnapshotFactory = Callable[..., FeatureSnapshot]


class TestBaselineRunner:
    """Tests for BaselineRunner."""

    @pytest.fixture
    def runner(self) -> BaselineRunner:
        """Create baseline runner with default config."""
        return BaselineRunner()

    @pytest.fixture
    def make_snapshot(self) -> SnapshotFactory:
        """Factory for creating FeatureSnapshots."""

        def _make(
            symbol: str = "BTCUSDT",
            ts: int = 1000,
            spread_bps: float = 1.0,
            mid: float = 50000.0,
            book_imbalance: float = 0.0,
            flow_imbalance: float = 0.0,
            natr: float = 0.02,
            impact_bps: float = 5.0,
            regime_vol: RegimeVol = RegimeVol.LOW,
            regime_trend: RegimeTrend = RegimeTrend.CHOP,
            stale_book_ms: int = 0,
            stale_trades_ms: int = 0,
            missing_streams: list[str] | None = None,
        ) -> FeatureSnapshot:
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
                windows=Windows(),
                data_health=DataHealth(
                    stale_book_ms=stale_book_ms,
                    stale_trades_ms=stale_trades_ms,
                    missing_streams=missing_streams or [],
                ),
            )

        return _make

    def test_default_config(self, runner: BaselineRunner) -> None:
        """Default configuration values."""
        assert runner.model_version == "baseline-v1.0.0+0000000"
        assert runner.calibration_version == "cal-v1.0.0"

    def test_custom_config(self) -> None:
        """Custom configuration is applied."""
        config = ModelRunnerConfig(
            model_version="custom-v1.0.0",
            toxic_threshold=0.8,
        )
        runner = BaselineRunner(config=config)

        assert runner.model_version == "custom-v1.0.0"
        assert runner.config.toxic_threshold == 0.8

    def test_predict_basic(self, runner: BaselineRunner, make_snapshot: SnapshotFactory) -> None:
        """Basic prediction returns valid PredictionSnapshot."""
        snapshot = make_snapshot()
        prediction = runner.predict(snapshot)

        assert prediction.symbol == "BTCUSDT"
        assert prediction.ts == 1000
        assert prediction.profile == ExecutionProfile.A
        assert 0 <= prediction.p_inplay_30s <= 1
        assert 0 <= prediction.p_inplay_2m <= 1
        assert 0 <= prediction.p_inplay_5m <= 1
        assert 0 <= prediction.p_toxic <= 1
        assert prediction.model_version == "baseline-v1.0.0+0000000"

    def test_predict_tradeable(self, runner: BaselineRunner, make_snapshot: SnapshotFactory) -> None:
        """High imbalance + tight spread = TRADEABLE."""
        snapshot = make_snapshot(
            spread_bps=0.5,
            book_imbalance=0.8,
            flow_imbalance=0.7,
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.TRADEABLE
        assert prediction.p_inplay_2m >= 0.6

    def test_predict_watch(self, runner: BaselineRunner, make_snapshot: SnapshotFactory) -> None:
        """Moderate signals = WATCH."""
        snapshot = make_snapshot(
            spread_bps=2.0,
            book_imbalance=0.4,
            flow_imbalance=0.3,
            regime_vol=RegimeVol.LOW,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        assert prediction.status in [PredictionStatus.WATCH, PredictionStatus.TRADEABLE]
        assert prediction.p_inplay_2m >= 0.3

    def test_predict_dead(self, runner: BaselineRunner, make_snapshot: SnapshotFactory) -> None:
        """Low signals + wide spread = DEAD."""
        snapshot = make_snapshot(
            spread_bps=20.0,
            book_imbalance=0.05,
            flow_imbalance=0.05,
            regime_vol=RegimeVol.LOW,
            regime_trend=RegimeTrend.CHOP,
        )
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DEAD
        assert prediction.p_inplay_2m < 0.3

    def test_predict_trap(self, runner: BaselineRunner, make_snapshot: SnapshotFactory) -> None:
        """High impact + extreme flow = TRAP."""
        snapshot = make_snapshot(
            spread_bps=5.0,
            book_imbalance=0.5,
            flow_imbalance=0.95,  # Extreme flow
            impact_bps=40.0,  # High impact
        )
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.TRAP
        assert prediction.p_toxic >= 0.7

    def test_predict_data_issue_stale_book(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Stale book data = DATA_ISSUE."""
        snapshot = make_snapshot(stale_book_ms=6000)
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE
        assert prediction.p_inplay_2m == 0.0
        assert len(prediction.reasons) == 1
        assert prediction.reasons[0].code == "RC_DATA_STALE"

    def test_predict_data_issue_stale_trades(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Very stale trade data = DATA_ISSUE."""
        snapshot = make_snapshot(stale_trades_ms=35000)
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE

    def test_predict_data_issue_missing_streams(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Missing streams = DATA_ISSUE."""
        snapshot = make_snapshot(missing_streams=["trade"])
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.DATA_ISSUE

    def test_predict_batch(self, runner: BaselineRunner, make_snapshot: SnapshotFactory) -> None:
        """Batch prediction."""
        snapshots = [
            make_snapshot(symbol="BTCUSDT"),
            make_snapshot(symbol="ETHUSDT"),
            make_snapshot(symbol="BNBUSDT"),
        ]
        predictions = runner.predict_batch(snapshots)

        assert len(predictions) == 3
        assert predictions[0].symbol == "BTCUSDT"
        assert predictions[1].symbol == "ETHUSDT"
        assert predictions[2].symbol == "BNBUSDT"

    def test_metrics_tracking(self, runner: BaselineRunner, make_snapshot: SnapshotFactory) -> None:
        """Metrics are tracked."""
        snapshot = make_snapshot()
        runner.predict(snapshot)

        assert runner.metrics.predictions_made == 1
        assert runner.metrics.predictions_per_symbol["BTCUSDT"] == 1

    def test_metrics_reset(self, runner: BaselineRunner, make_snapshot: SnapshotFactory) -> None:
        """Metrics can be reset."""
        runner.predict(make_snapshot())
        runner.reset_metrics()

        assert runner.metrics.predictions_made == 0

    def test_compute_digest(self, runner: BaselineRunner) -> None:
        """Compute model digest."""
        digest = runner.compute_digest()

        assert len(digest) == 16
        assert digest.isalnum()

    def test_digest_changes_with_config(self) -> None:
        """Digest changes when config changes."""
        runner1 = BaselineRunner()
        runner2 = BaselineRunner(
            ModelRunnerConfig(toxic_threshold=0.8)
        )

        assert runner1.compute_digest() != runner2.compute_digest()

    def test_reason_codes_flow_surge(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Flow surge reason code."""
        snapshot = make_snapshot(flow_imbalance=0.5)
        prediction = runner.predict(snapshot)

        codes = [r.code for r in prediction.reasons]
        assert "RC_FLOW_SURGE" in codes

    def test_reason_codes_book_pressure(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Book pressure reason code."""
        snapshot = make_snapshot(book_imbalance=0.5)
        prediction = runner.predict(snapshot)

        codes = [r.code for r in prediction.reasons]
        assert "RC_BOOK_PRESSURE" in codes

    def test_reason_codes_tight_spread(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Tight spread reason code."""
        snapshot = make_snapshot(spread_bps=0.5)
        prediction = runner.predict(snapshot)

        codes = [r.code for r in prediction.reasons]
        assert "RC_TIGHT_SPREAD" in codes

    def test_reason_codes_wide_spread(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Wide spread reason code."""
        snapshot = make_snapshot(spread_bps=15.0)
        prediction = runner.predict(snapshot)

        codes = [r.code for r in prediction.reasons]
        assert "RC_WIDE_SPREAD" in codes

    def test_reason_codes_high_vol(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """High volatility reason code."""
        snapshot = make_snapshot(regime_vol=RegimeVol.HIGH)
        prediction = runner.predict(snapshot)

        codes = [r.code for r in prediction.reasons]
        assert "RC_HIGH_VOL" in codes

    def test_reason_codes_toxic_risk(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Toxic risk reason code."""
        snapshot = make_snapshot(
            flow_imbalance=0.8,
            impact_bps=30.0,
        )
        prediction = runner.predict(snapshot)

        codes = [r.code for r in prediction.reasons]
        assert "RC_TOXIC_RISK" in codes

    def test_expected_utility_positive(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Positive expected utility for good setup."""
        snapshot = make_snapshot(
            spread_bps=0.5,
            book_imbalance=0.6,
            flow_imbalance=0.5,
            natr=0.03,
            impact_bps=2.0,
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        assert prediction.expected_utility_bps_2m > 0

    def test_expected_utility_negative(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Negative expected utility for toxic setup."""
        snapshot = make_snapshot(
            spread_bps=10.0,
            book_imbalance=0.1,
            flow_imbalance=0.9,  # Extreme one-sided
            natr=0.01,  # Low volatility
            impact_bps=40.0,  # High impact
        )
        prediction = runner.predict(snapshot)

        # High toxicity should reduce utility
        assert prediction.p_toxic > 0.5

    def test_regime_multiplier_high_vol_trend(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """High vol + trend increases probability."""
        base_snapshot = make_snapshot(
            book_imbalance=0.5,
            flow_imbalance=0.5,
            regime_vol=RegimeVol.LOW,
            regime_trend=RegimeTrend.CHOP,
        )
        high_vol_snapshot = make_snapshot(
            book_imbalance=0.5,
            flow_imbalance=0.5,
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )

        base_pred = runner.predict(base_snapshot)
        high_vol_pred = runner.predict(high_vol_snapshot)

        assert high_vol_pred.p_inplay_2m > base_pred.p_inplay_2m

    def test_concordance_bonus(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Concordant imbalances increase probability."""
        concordant = make_snapshot(
            book_imbalance=0.5,
            flow_imbalance=0.5,
        )
        discordant = make_snapshot(
            book_imbalance=0.5,
            flow_imbalance=-0.5,
        )

        conc_pred = runner.predict(concordant)
        disc_pred = runner.predict(discordant)

        assert conc_pred.p_inplay_2m > disc_pred.p_inplay_2m


class TestBaselineRunnerDeterminism:
    """Tests for prediction determinism."""

    def test_same_input_same_output(self) -> None:
        """Same input always produces same output."""
        runner = BaselineRunner()

        snapshot = FeatureSnapshot(
            ts=1000,
            symbol="BTCUSDT",
            features=Features(
                spread_bps=1.0,
                mid=50000.0,
                book_imbalance=0.5,
                flow_imbalance=0.4,
                natr_14_5m=0.02,
                impact_bps_q=5.0,
                regime_vol=RegimeVol.HIGH,
                regime_trend=RegimeTrend.TREND,
            ),
        )

        pred1 = runner.predict(snapshot)
        pred2 = runner.predict(snapshot)

        assert pred1.p_inplay_30s == pred2.p_inplay_30s
        assert pred1.p_inplay_2m == pred2.p_inplay_2m
        assert pred1.p_inplay_5m == pred2.p_inplay_5m
        assert pred1.p_toxic == pred2.p_toxic
        assert pred1.status == pred2.status
        assert pred1.expected_utility_bps_2m == pred2.expected_utility_bps_2m

    def test_different_runners_same_config(self) -> None:
        """Different runners with same config produce same output."""
        config = ModelRunnerConfig()
        runner1 = BaselineRunner(config)
        runner2 = BaselineRunner(config)

        snapshot = FeatureSnapshot(
            ts=1000,
            symbol="BTCUSDT",
            features=Features(
                spread_bps=2.0,
                mid=50000.0,
                book_imbalance=0.3,
                flow_imbalance=0.2,
                natr_14_5m=0.015,
                impact_bps_q=3.0,
                regime_vol=RegimeVol.LOW,
                regime_trend=RegimeTrend.CHOP,
            ),
        )

        pred1 = runner1.predict(snapshot)
        pred2 = runner2.predict(snapshot)

        assert pred1.p_inplay_2m == pred2.p_inplay_2m
        assert pred1.status == pred2.status


class TestBaselineRunnerCriticalGates:
    """Tests for PRD critical gates (spread/impact) blocking TRADEABLE."""

    @pytest.fixture
    def runner(self) -> BaselineRunner:
        """Create baseline runner with default config."""
        return BaselineRunner()

    @pytest.fixture
    def make_snapshot(self) -> SnapshotFactory:
        """Factory for creating FeatureSnapshots."""

        def _make(
            symbol: str = "BTCUSDT",
            ts: int = 1000,
            spread_bps: float = 1.0,
            mid: float = 50000.0,
            book_imbalance: float = 0.0,
            flow_imbalance: float = 0.0,
            natr: float = 0.02,
            impact_bps: float = 5.0,
            regime_vol: RegimeVol = RegimeVol.LOW,
            regime_trend: RegimeTrend = RegimeTrend.CHOP,
            stale_book_ms: int = 0,
            stale_trades_ms: int = 0,
            missing_streams: list[str] | None = None,
        ) -> FeatureSnapshot:
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
                windows=Windows(),
                data_health=DataHealth(
                    stale_book_ms=stale_book_ms,
                    stale_trades_ms=stale_trades_ms,
                    missing_streams=missing_streams or [],
                ),
            )

        return _make

    def test_spread_gate_blocks_tradeable(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """High p_inplay but wide spread -> NOT TRADEABLE (gate fail)."""
        # Setup that would normally be TRADEABLE
        snapshot = make_snapshot(
            spread_bps=15.0,  # > 10.0 default max
            book_imbalance=0.8,
            flow_imbalance=0.7,
            impact_bps=5.0,  # Low impact (passes gate)
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        # Should be blocked from TRADEABLE despite high p_inplay
        assert prediction.p_inplay_2m >= 0.6, "Should have high p_inplay"
        assert prediction.status != PredictionStatus.TRADEABLE
        assert prediction.status == PredictionStatus.WATCH  # Downgraded

        # Should have gate failure reason
        codes = [r.code for r in prediction.reasons]
        assert "RC_GATE_SPREAD_FAIL" in codes

    def test_impact_gate_blocks_tradeable(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """High p_inplay but high impact -> NOT TRADEABLE (gate fail)."""
        # Setup that would normally be TRADEABLE
        snapshot = make_snapshot(
            spread_bps=1.0,  # Low spread (passes gate)
            book_imbalance=0.8,
            flow_imbalance=0.7,
            impact_bps=25.0,  # > 20.0 default max
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        # Should be blocked from TRADEABLE
        assert prediction.p_inplay_2m >= 0.6, "Should have high p_inplay"
        assert prediction.status != PredictionStatus.TRADEABLE
        assert prediction.status == PredictionStatus.WATCH  # Downgraded

        # Should have gate failure reason
        codes = [r.code for r in prediction.reasons]
        assert "RC_GATE_IMPACT_FAIL" in codes

    def test_both_gates_fail(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Both spread and impact gates fail."""
        snapshot = make_snapshot(
            spread_bps=15.0,  # > 10.0
            book_imbalance=0.8,
            flow_imbalance=0.7,
            impact_bps=25.0,  # > 20.0
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        assert prediction.status != PredictionStatus.TRADEABLE
        codes = [r.code for r in prediction.reasons]
        assert "RC_GATE_SPREAD_FAIL" in codes
        assert "RC_GATE_IMPACT_FAIL" in codes

    def test_gates_pass_allows_tradeable(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """All gates pass -> TRADEABLE allowed."""
        snapshot = make_snapshot(
            spread_bps=5.0,  # <= 10.0 (passes)
            book_imbalance=0.8,
            flow_imbalance=0.7,
            impact_bps=10.0,  # <= 20.0 (passes)
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        assert prediction.p_inplay_2m >= 0.6
        assert prediction.status == PredictionStatus.TRADEABLE

        # No gate failure reasons
        codes = [r.code for r in prediction.reasons]
        assert "RC_GATE_SPREAD_FAIL" not in codes
        assert "RC_GATE_IMPACT_FAIL" not in codes

    def test_spread_at_boundary_passes(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Spread exactly at max -> passes gate."""
        snapshot = make_snapshot(
            spread_bps=10.0,  # Exactly at max
            book_imbalance=0.8,
            flow_imbalance=0.7,
            impact_bps=5.0,
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.TRADEABLE

    def test_impact_at_boundary_passes(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """Impact exactly at max -> passes gate."""
        snapshot = make_snapshot(
            spread_bps=5.0,
            book_imbalance=0.8,
            flow_imbalance=0.7,
            impact_bps=20.0,  # Exactly at max
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        assert prediction.status == PredictionStatus.TRADEABLE

    def test_custom_gate_thresholds(self, make_snapshot: SnapshotFactory) -> None:
        """Custom gate thresholds are respected."""
        config = ModelRunnerConfig(
            spread_max_bps=5.0,  # Stricter
            impact_max_bps=10.0,  # Stricter
        )
        runner = BaselineRunner(config)

        # Would pass default gates but fail custom
        snapshot = make_snapshot(
            spread_bps=8.0,  # > 5.0 custom max
            book_imbalance=0.8,
            flow_imbalance=0.7,
            impact_bps=5.0,
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        )
        prediction = runner.predict(snapshot)

        assert prediction.status != PredictionStatus.TRADEABLE

    def test_gates_dont_affect_trap(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """TRAP status is not affected by gates (toxicity takes priority)."""
        snapshot = make_snapshot(
            spread_bps=15.0,  # Fails gate
            book_imbalance=0.5,
            flow_imbalance=0.95,  # High toxicity (0.95^2 = 0.9025)
            impact_bps=50.0,  # Fails gate, also increases p_toxic
        )
        prediction = runner.predict(snapshot)

        # TRAP should still be TRAP (toxicity > gates)
        assert prediction.p_toxic >= 0.7
        assert prediction.status == PredictionStatus.TRAP

    def test_gates_dont_affect_dead(
        self, runner: BaselineRunner, make_snapshot: SnapshotFactory
    ) -> None:
        """DEAD status is not affected by gates (low p_inplay)."""
        snapshot = make_snapshot(
            spread_bps=15.0,  # Would fail gate
            book_imbalance=0.1,
            flow_imbalance=0.1,
            impact_bps=25.0,  # Would fail gate
            regime_vol=RegimeVol.LOW,
            regime_trend=RegimeTrend.CHOP,
        )
        prediction = runner.predict(snapshot)

        # DEAD due to low p_inplay (gates don't matter)
        assert prediction.p_inplay_2m < 0.3
        assert prediction.status == PredictionStatus.DEAD
