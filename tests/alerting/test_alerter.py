"""Tests for alerter module."""

from cryptoscreener.alerting.alerter import (
    Alerter,
    AlerterConfig,
    AlerterMetrics,
    SymbolAlertState,
)
from cryptoscreener.contracts.events import (
    DataHealth,
    ExecutionProfile,
    PredictionSnapshot,
    PredictionStatus,
    RankEventType,
)


def make_prediction(
    symbol: str = "BTCUSDT",
    status: PredictionStatus = PredictionStatus.TRADEABLE,
    p_inplay_2m: float = 0.7,
    p_toxic: float = 0.1,
) -> PredictionSnapshot:
    """Create a test prediction."""
    return PredictionSnapshot(
        ts=1000,
        symbol=symbol,
        profile=ExecutionProfile.A,
        p_inplay_30s=p_inplay_2m * 0.8,
        p_inplay_2m=p_inplay_2m,
        p_inplay_5m=p_inplay_2m * 1.1,
        expected_utility_bps_2m=10.0,
        p_toxic=p_toxic,
        status=status,
        reasons=[],
        model_version="baseline-1.0",
        calibration_version="cal-1.0",
        data_health=DataHealth(
            stale_book_ms=0,
            stale_trades_ms=0,
            missing_streams=[],
        ),
    )


class TestAlerterConfig:
    """Tests for AlerterConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AlerterConfig()
        assert config.cooldown_ms == 120_000
        assert config.stable_ms == 2000
        assert config.max_alerts_per_min == 30
        assert config.data_issue_threshold_ms == 10_000

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AlerterConfig(
            cooldown_ms=60_000,
            stable_ms=1000,
            max_alerts_per_min=10,
            data_issue_threshold_ms=5000,
        )
        assert config.cooldown_ms == 60_000
        assert config.stable_ms == 1000
        assert config.max_alerts_per_min == 10
        assert config.data_issue_threshold_ms == 5000


class TestSymbolAlertState:
    """Tests for SymbolAlertState."""

    def test_default_state(self) -> None:
        """Test default state values."""
        state = SymbolAlertState()
        assert state.last_status is None
        assert state.status_since_ts == 0
        assert state.last_alert_ts == {}
        assert state.data_issue_since_ts == 0


class TestAlerterMetrics:
    """Tests for AlerterMetrics."""

    def test_default_metrics(self) -> None:
        """Test default metrics values."""
        metrics = AlerterMetrics()
        assert metrics.alerts_generated == 0
        assert metrics.alerts_suppressed_cooldown == 0
        assert metrics.alerts_suppressed_stability == 0
        assert metrics.alerts_suppressed_rate_limit == 0

    def test_reset(self) -> None:
        """Test metrics reset."""
        metrics = AlerterMetrics()
        metrics.alerts_generated = 5
        metrics.alerts_suppressed_cooldown = 3
        metrics.alerts_suppressed_stability = 2
        metrics.alerts_suppressed_rate_limit = 1

        metrics.reset()

        assert metrics.alerts_generated == 0
        assert metrics.alerts_suppressed_cooldown == 0
        assert metrics.alerts_suppressed_stability == 0
        assert metrics.alerts_suppressed_rate_limit == 0


class TestAlerterBasic:
    """Basic alerter tests."""

    def test_init_default_config(self) -> None:
        """Test alerter initialization with default config."""
        alerter = Alerter()
        assert alerter.config.cooldown_ms == 120_000

    def test_init_custom_config(self) -> None:
        """Test alerter initialization with custom config."""
        config = AlerterConfig(cooldown_ms=60_000)
        alerter = Alerter(config)
        assert alerter.config.cooldown_ms == 60_000

    def test_reset(self) -> None:
        """Test alerter reset."""
        config = AlerterConfig(stable_ms=100)  # Short stability for test
        alerter = Alerter(config)
        prediction = make_prediction()

        # Generate an alert (need two calls - first sets status, second fires)
        alerter.process_prediction(prediction, ts=1000)
        alerter.process_prediction(prediction, ts=2000)
        assert alerter.metrics.alerts_generated > 0

        # Reset
        alerter.reset()
        assert alerter.metrics.alerts_generated == 0


class TestStabilityRequirement:
    """Tests for stability requirement (hysteresis)."""

    def test_no_alert_without_stability(self) -> None:
        """Test that alerts are suppressed if status not stable long enough."""
        config = AlerterConfig(stable_ms=2000)
        alerter = Alerter(config)

        prediction = make_prediction(status=PredictionStatus.TRADEABLE)

        # First observation - sets status_since_ts
        events = alerter.process_prediction(prediction, ts=1000)
        assert len(events) == 0  # Not stable yet

        # Not enough time passed
        events = alerter.process_prediction(prediction, ts=1500)
        assert len(events) == 0
        assert alerter.metrics.alerts_suppressed_stability >= 1

    def test_alert_after_stability(self) -> None:
        """Test that alert fires after status is stable long enough."""
        config = AlerterConfig(stable_ms=2000, cooldown_ms=1000)
        alerter = Alerter(config)

        prediction = make_prediction(status=PredictionStatus.TRADEABLE)

        # First observation
        alerter.process_prediction(prediction, ts=1000)

        # After stable_ms
        events = alerter.process_prediction(prediction, ts=3001)
        assert len(events) == 1
        assert events[0].event == RankEventType.ALERT_TRADABLE

    def test_stability_resets_on_status_change(self) -> None:
        """Test that stability counter resets when status changes."""
        config = AlerterConfig(stable_ms=2000)
        alerter = Alerter(config)

        # Start with TRADEABLE
        pred1 = make_prediction(status=PredictionStatus.TRADEABLE)
        alerter.process_prediction(pred1, ts=1000)
        alerter.process_prediction(pred1, ts=2500)  # Almost stable

        # Change to WATCH
        pred2 = make_prediction(status=PredictionStatus.WATCH)
        alerter.process_prediction(pred2, ts=2600)

        # Back to TRADEABLE - stability should reset
        pred3 = make_prediction(status=PredictionStatus.TRADEABLE)
        events = alerter.process_prediction(pred3, ts=2700)
        assert len(events) == 0  # Not stable yet (just changed)


class TestCooldown:
    """Tests for cooldown per symbol per event type."""

    def test_cooldown_blocks_repeat_alert(self) -> None:
        """Test that cooldown prevents repeated alerts."""
        config = AlerterConfig(cooldown_ms=10_000, stable_ms=100)
        alerter = Alerter(config)

        prediction = make_prediction(status=PredictionStatus.TRADEABLE)

        # First alert
        alerter.process_prediction(prediction, ts=1000)
        events1 = alerter.process_prediction(prediction, ts=2000)
        assert len(events1) == 1

        # Second attempt within cooldown
        events2 = alerter.process_prediction(prediction, ts=5000)
        assert len(events2) == 0
        assert alerter.metrics.alerts_suppressed_cooldown >= 1

    def test_cooldown_expires(self) -> None:
        """Test that cooldown expires after cooldown_ms."""
        config = AlerterConfig(cooldown_ms=10_000, stable_ms=100)
        alerter = Alerter(config)

        prediction = make_prediction(status=PredictionStatus.TRADEABLE)

        # First alert
        alerter.process_prediction(prediction, ts=1000)
        events1 = alerter.process_prediction(prediction, ts=2000)
        assert len(events1) == 1

        # After cooldown expires
        events2 = alerter.process_prediction(prediction, ts=15000)
        assert len(events2) == 1

    def test_cooldown_per_symbol(self) -> None:
        """Test that cooldown is per-symbol."""
        config = AlerterConfig(cooldown_ms=10_000, stable_ms=100)
        alerter = Alerter(config)

        pred_btc = make_prediction(symbol="BTCUSDT", status=PredictionStatus.TRADEABLE)
        pred_eth = make_prediction(symbol="ETHUSDT", status=PredictionStatus.TRADEABLE)

        # Alert for BTC
        alerter.process_prediction(pred_btc, ts=1000)
        events_btc = alerter.process_prediction(pred_btc, ts=2000)
        assert len(events_btc) == 1

        # Alert for ETH should not be blocked
        alerter.process_prediction(pred_eth, ts=2100)
        events_eth = alerter.process_prediction(pred_eth, ts=3100)
        assert len(events_eth) == 1

    def test_cooldown_per_event_type(self) -> None:
        """Test that cooldown is per-event-type."""
        config = AlerterConfig(cooldown_ms=10_000, stable_ms=100)
        alerter = Alerter(config)

        # TRADEABLE alert
        pred1 = make_prediction(status=PredictionStatus.TRADEABLE)
        alerter.process_prediction(pred1, ts=1000)
        events1 = alerter.process_prediction(pred1, ts=2000)
        assert len(events1) == 1
        assert events1[0].event == RankEventType.ALERT_TRADABLE

        # Change to TRAP - different event type, should not be blocked
        pred2 = make_prediction(status=PredictionStatus.TRAP, p_toxic=0.8)
        alerter.process_prediction(pred2, ts=2100)
        events2 = alerter.process_prediction(pred2, ts=4200)
        assert len(events2) == 1
        assert events2[0].event == RankEventType.ALERT_TRAP


class TestRateLimiting:
    """Tests for global rate limiting."""

    def test_rate_limit_blocks_excessive_alerts(self) -> None:
        """Test that rate limit blocks alerts when threshold reached."""
        config = AlerterConfig(max_alerts_per_min=5, stable_ms=100, cooldown_ms=100)
        alerter = Alerter(config)

        # Generate max_alerts_per_min alerts
        for i in range(5):
            symbol = f"SYM{i}USDT"
            pred = make_prediction(symbol=symbol, status=PredictionStatus.TRADEABLE)
            alerter.process_prediction(pred, ts=1000 + i * 200)
            alerter.process_prediction(pred, ts=1200 + i * 200)

        assert alerter.metrics.alerts_generated == 5

        # Next alert should be blocked
        pred_extra = make_prediction(symbol="EXTRAUSDT", status=PredictionStatus.TRADEABLE)
        alerter.process_prediction(pred_extra, ts=3000)
        events = alerter.process_prediction(pred_extra, ts=3200)
        assert len(events) == 0
        assert alerter.metrics.alerts_suppressed_rate_limit >= 1

    def test_rate_limit_window_slides(self) -> None:
        """Test that rate limit window slides (old alerts expire)."""
        config = AlerterConfig(max_alerts_per_min=3, stable_ms=100, cooldown_ms=100)
        alerter = Alerter(config)

        # Generate 3 alerts at t=1000
        for i in range(3):
            symbol = f"SYM{i}USDT"
            pred = make_prediction(symbol=symbol, status=PredictionStatus.TRADEABLE)
            alerter.process_prediction(pred, ts=1000)
            alerter.process_prediction(pred, ts=1200)

        # After 1 minute, old alerts expire
        pred_new = make_prediction(symbol="NEWUSDT", status=PredictionStatus.TRADEABLE)
        alerter.process_prediction(pred_new, ts=62000)
        events = alerter.process_prediction(pred_new, ts=63000)
        assert len(events) == 1


class TestTradeableAlert:
    """Tests for TRADEABLE alerts."""

    def test_tradeable_alert_generated(self) -> None:
        """Test that TRADEABLE status generates ALERT_TRADABLE."""
        config = AlerterConfig(stable_ms=100)
        alerter = Alerter(config)

        prediction = make_prediction(status=PredictionStatus.TRADEABLE)
        alerter.process_prediction(prediction, ts=1000)
        events = alerter.process_prediction(prediction, ts=2000)

        assert len(events) == 1
        assert events[0].event == RankEventType.ALERT_TRADABLE
        assert events[0].symbol == "BTCUSDT"

    def test_watch_status_no_alert(self) -> None:
        """Test that WATCH status does not generate alert."""
        alerter = Alerter()
        prediction = make_prediction(status=PredictionStatus.WATCH)

        events = alerter.process_prediction(prediction, ts=1000)
        assert len(events) == 0


class TestTrapAlert:
    """Tests for TRAP alerts."""

    def test_trap_alert_generated(self) -> None:
        """Test that TRAP status generates ALERT_TRAP."""
        config = AlerterConfig(stable_ms=100)
        alerter = Alerter(config)

        prediction = make_prediction(status=PredictionStatus.TRAP, p_toxic=0.8)
        alerter.process_prediction(prediction, ts=1000)
        events = alerter.process_prediction(prediction, ts=2000)

        assert len(events) == 1
        assert events[0].event == RankEventType.ALERT_TRAP


class TestDataIssueAlert:
    """Tests for DATA_ISSUE alerts."""

    def test_data_issue_alert_after_threshold(self) -> None:
        """Test that DATA_ISSUE generates alert after threshold."""
        config = AlerterConfig(data_issue_threshold_ms=5000, cooldown_ms=1000)
        alerter = Alerter(config)

        prediction = make_prediction(status=PredictionStatus.DATA_ISSUE)

        # First observation - starts timer
        events1 = alerter.process_prediction(prediction, ts=1000)
        assert len(events1) == 0

        # Before threshold
        events2 = alerter.process_prediction(prediction, ts=3000)
        assert len(events2) == 0

        # After threshold
        events3 = alerter.process_prediction(prediction, ts=7000)
        assert len(events3) == 1
        assert events3[0].event == RankEventType.DATA_ISSUE

    def test_data_issue_timer_resets_on_recovery(self) -> None:
        """Test that data issue timer resets when status recovers."""
        config = AlerterConfig(data_issue_threshold_ms=5000)
        alerter = Alerter(config)

        # Start with data issue
        pred_issue = make_prediction(status=PredictionStatus.DATA_ISSUE)
        alerter.process_prediction(pred_issue, ts=1000)
        alerter.process_prediction(pred_issue, ts=3000)

        # Recover
        pred_ok = make_prediction(status=PredictionStatus.WATCH)
        alerter.process_prediction(pred_ok, ts=4000)

        # New data issue - timer should restart
        pred_issue2 = make_prediction(status=PredictionStatus.DATA_ISSUE)
        events = alerter.process_prediction(pred_issue2, ts=5000)
        assert len(events) == 0  # Timer just started


class TestAlertPayload:
    """Tests for alert event payloads."""

    def test_alert_contains_prediction(self) -> None:
        """Test that alert event contains prediction in payload."""
        config = AlerterConfig(stable_ms=100)
        alerter = Alerter(config)

        prediction = make_prediction(status=PredictionStatus.TRADEABLE)
        alerter.process_prediction(prediction, ts=1000)
        events = alerter.process_prediction(prediction, ts=2000)

        assert len(events) == 1
        assert events[0].payload is not None
        assert events[0].payload.prediction is not None
        assert events[0].payload.prediction["symbol"] == "BTCUSDT"

    def test_alert_contains_rank_and_score(self) -> None:
        """Test that alert event contains rank and score."""
        config = AlerterConfig(stable_ms=100)
        alerter = Alerter(config)

        prediction = make_prediction(status=PredictionStatus.TRADEABLE)
        alerter.process_prediction(prediction, ts=1000)
        events = alerter.process_prediction(prediction, ts=2000, rank=5, score=0.75)

        assert len(events) == 1
        assert events[0].rank == 5
        assert events[0].score == 0.75


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self) -> None:
        """Test that same input produces same output."""
        config = AlerterConfig(stable_ms=100)

        # Run 1
        alerter1 = Alerter(config)
        pred = make_prediction(status=PredictionStatus.TRADEABLE)
        alerter1.process_prediction(pred, ts=1000)
        events1 = alerter1.process_prediction(pred, ts=2000)

        # Run 2
        alerter2 = Alerter(config)
        alerter2.process_prediction(pred, ts=1000)
        events2 = alerter2.process_prediction(pred, ts=2000)

        # Should be identical
        assert len(events1) == len(events2)
        assert events1[0].event == events2[0].event
        assert events1[0].symbol == events2[0].symbol
        assert events1[0].ts == events2[0].ts

    def test_metrics_consistency(self) -> None:
        """Test that metrics are consistent across runs."""
        config = AlerterConfig(stable_ms=100, cooldown_ms=1000)

        predictions = [
            make_prediction(symbol="BTCUSDT", status=PredictionStatus.TRADEABLE),
            make_prediction(symbol="ETHUSDT", status=PredictionStatus.TRAP, p_toxic=0.8),
        ]

        # Run 1
        alerter1 = Alerter(config)
        for i, pred in enumerate(predictions):
            alerter1.process_prediction(pred, ts=1000 + i * 200)
            alerter1.process_prediction(pred, ts=1200 + i * 200)

        # Run 2
        alerter2 = Alerter(config)
        for i, pred in enumerate(predictions):
            alerter2.process_prediction(pred, ts=1000 + i * 200)
            alerter2.process_prediction(pred, ts=1200 + i * 200)

        # Metrics should match
        assert alerter1.metrics.alerts_generated == alerter2.metrics.alerts_generated
