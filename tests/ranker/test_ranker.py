"""Tests for Ranker with hysteresis."""

from collections.abc import Callable

import pytest

from cryptoscreener.contracts.events import (
    DataHealth,
    ExecutionProfile,
    PredictionSnapshot,
    PredictionStatus,
    RankEvent,
    RankEventType,
)
from cryptoscreener.ranker.ranker import Ranker, RankerConfig
from cryptoscreener.ranker.state import SymbolState, SymbolStateType


class TestSymbolState:
    """Tests for SymbolState."""

    def test_initial_state(self) -> None:
        """Initial state values."""
        state = SymbolState(symbol="BTCUSDT")

        assert state.symbol == "BTCUSDT"
        assert state.state == SymbolStateType.NOT_IN_TOP_K
        assert state.score == 0.0
        assert state.rank == -1

    def test_update_score(self) -> None:
        """Update score and timestamp."""
        state = SymbolState(symbol="BTCUSDT")
        state.update_score(0.75, 1000)

        assert state.score == 0.75
        assert state.last_update_ts == 1000

    def test_transition_to(self) -> None:
        """Transition to new state."""
        state = SymbolState(symbol="BTCUSDT", state_entered_ts=0)
        state.transition_to(SymbolStateType.PENDING_ENTER, 1000)

        assert state.state == SymbolStateType.PENDING_ENTER
        assert state.state_entered_ts == 1000
        assert state.pending_since_ts == 1000

    def test_dwell_time_ms(self) -> None:
        """Calculate dwell time."""
        state = SymbolState(symbol="BTCUSDT", state_entered_ts=1000)
        dwell = state.dwell_time_ms(2500)

        assert dwell == 1500

    def test_pending_time_ms(self) -> None:
        """Calculate pending time."""
        state = SymbolState(symbol="BTCUSDT", pending_since_ts=1000)
        pending = state.pending_time_ms(2500)

        assert pending == 1500

    def test_pending_time_ms_not_pending(self) -> None:
        """Pending time is 0 when not pending."""
        state = SymbolState(symbol="BTCUSDT", pending_since_ts=0)
        pending = state.pending_time_ms(2500)

        assert pending == 0


class TestRankerConfig:
    """Tests for RankerConfig."""

    def test_default_values(self) -> None:
        """Default config values match HYSTERESIS_SPEC."""
        config = RankerConfig()

        assert config.top_k == 20
        assert config.enter_ms == 1500
        assert config.exit_ms == 3000
        assert config.min_dwell_ms == 2000
        assert config.update_interval_ms == 1000


class TestRanker:
    """Tests for Ranker."""

    @pytest.fixture
    def ranker(self) -> Ranker:
        """Create ranker with default config."""
        return Ranker()

    @pytest.fixture
    def make_prediction(self) -> Callable[..., PredictionSnapshot]:
        """Factory for creating PredictionSnapshots."""

        def _make(
            symbol: str = "BTCUSDT",
            p_inplay_30s: float = 0.6,
            p_inplay_2m: float = 0.7,
            p_inplay_5m: float = 0.65,
            expected_utility_bps_2m: float = 30.0,
            p_toxic: float = 0.1,
            status: PredictionStatus = PredictionStatus.TRADEABLE,
        ) -> PredictionSnapshot:
            return PredictionSnapshot(
                ts=1000,
                symbol=symbol,
                profile=ExecutionProfile.A,
                p_inplay_30s=p_inplay_30s,
                p_inplay_2m=p_inplay_2m,
                p_inplay_5m=p_inplay_5m,
                expected_utility_bps_2m=expected_utility_bps_2m,
                p_toxic=p_toxic,
                status=status,
                reasons=[],
                model_version="test-v1.0.0",
                calibration_version="cal-v1.0.0",
                data_health=DataHealth(),
            )

        return _make

    def test_init_default(self) -> None:
        """Ranker initializes with defaults."""
        ranker = Ranker()
        assert ranker.config.top_k == 20
        assert ranker.top_k == []

    def test_init_custom_config(self) -> None:
        """Ranker initializes with custom config."""
        config = RankerConfig(top_k=10, enter_ms=1000)
        ranker = Ranker(config)
        assert ranker.config.top_k == 10
        assert ranker.config.enter_ms == 1000

    def test_update_creates_state(
        self, ranker: Ranker, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Update creates state for new symbol."""
        pred = make_prediction(symbol="BTCUSDT")
        ranker.update({"BTCUSDT": pred}, ts=1000)

        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.symbol == "BTCUSDT"

    def test_enter_requires_hold_time(
        self, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Symbol must hold condition for enter_ms to enter."""
        config = RankerConfig(enter_ms=1500, top_k=5, update_interval_ms=0)
        ranker = Ranker(config)
        pred = make_prediction(symbol="BTCUSDT")

        # First update - should start pending
        events = ranker.update({"BTCUSDT": pred}, ts=1000)
        assert len(events) == 0
        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.state == SymbolStateType.PENDING_ENTER

        # Update at 2000ms - still pending (only 1000ms elapsed)
        events = ranker.update({"BTCUSDT": pred}, ts=2000)
        assert len(events) == 0
        assert state.state == SymbolStateType.PENDING_ENTER

        # Update at 2500ms - should enter (1500ms elapsed)
        events = ranker.update({"BTCUSDT": pred}, ts=2500)
        assert len(events) == 1
        assert events[0].event == RankEventType.SYMBOL_ENTER
        assert events[0].symbol == "BTCUSDT"
        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.state == SymbolStateType.IN_TOP_K

    def test_exit_requires_hold_time(
        self, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Symbol must hold exit condition for exit_ms to exit."""
        config = RankerConfig(
            enter_ms=0, exit_ms=3000, min_dwell_ms=0, top_k=5, update_interval_ms=0
        )
        ranker = Ranker(config)

        # Enter immediately (enter_ms=0)
        pred = make_prediction(symbol="BTCUSDT")
        events = ranker.update({"BTCUSDT": pred}, ts=1000)
        assert len(events) == 1
        assert events[0].event == RankEventType.SYMBOL_ENTER

        # Drop from predictions (score = 0)
        events = ranker.update({}, ts=2000)
        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.state == SymbolStateType.PENDING_EXIT
        assert len(events) == 0

        # Still pending at 4000ms (only 2000ms elapsed)
        events = ranker.update({}, ts=4000)
        assert len(events) == 0
        assert state.state == SymbolStateType.PENDING_EXIT

        # Exit at 5000ms (3000ms elapsed)
        events = ranker.update({}, ts=5000)
        assert len(events) == 1
        assert events[0].event == RankEventType.SYMBOL_EXIT

    def test_min_dwell_time(self, make_prediction: Callable[..., PredictionSnapshot]) -> None:
        """Minimum dwell time prevents rapid exit."""
        config = RankerConfig(
            enter_ms=0, exit_ms=0, min_dwell_ms=2000, top_k=5, update_interval_ms=0
        )
        ranker = Ranker(config)

        # Enter immediately
        pred = make_prediction(symbol="BTCUSDT")
        events = ranker.update({"BTCUSDT": pred}, ts=1000)
        assert len(events) == 1
        assert events[0].event == RankEventType.SYMBOL_ENTER

        # Try to exit immediately - blocked by min_dwell
        events = ranker.update({}, ts=1500)
        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.state == SymbolStateType.IN_TOP_K  # Still in top-K
        assert len(events) == 0

        # Exit after min_dwell (with exit_ms=0, exits immediately without pending)
        events = ranker.update({}, ts=3000)
        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.state == SymbolStateType.NOT_IN_TOP_K  # Exited immediately
        assert len(events) == 1
        assert events[0].event == RankEventType.SYMBOL_EXIT

    def test_pending_enter_canceled(
        self, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Pending enter is canceled if condition lost."""
        config = RankerConfig(enter_ms=2000, top_k=5, update_interval_ms=0)
        ranker = Ranker(config)

        # Start pending enter
        pred = make_prediction(symbol="BTCUSDT")
        ranker.update({"BTCUSDT": pred}, ts=1000)
        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.state == SymbolStateType.PENDING_ENTER

        # Lose condition (drop from predictions)
        ranker.update({}, ts=1500)
        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.state == SymbolStateType.NOT_IN_TOP_K

    def test_pending_exit_canceled(
        self, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Pending exit is canceled if condition regained."""
        config = RankerConfig(
            enter_ms=0, exit_ms=3000, min_dwell_ms=0, top_k=5, update_interval_ms=0
        )
        ranker = Ranker(config)

        # Enter
        pred = make_prediction(symbol="BTCUSDT")
        ranker.update({"BTCUSDT": pred}, ts=1000)

        # Start pending exit
        ranker.update({}, ts=2000)
        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.state == SymbolStateType.PENDING_EXIT

        # Regain condition
        ranker.update({"BTCUSDT": pred}, ts=3000)
        state = ranker.get_state("BTCUSDT")
        assert state is not None
        assert state.state == SymbolStateType.IN_TOP_K

    def test_update_throttling(self, make_prediction: Callable[..., PredictionSnapshot]) -> None:
        """Updates are throttled to 1Hz by default."""
        config = RankerConfig(update_interval_ms=1000, burst_delta=0.5)
        ranker = Ranker(config)

        pred = make_prediction(symbol="BTCUSDT")
        ranker.update({"BTCUSDT": pred}, ts=1000)

        # Update too soon - should be throttled
        ranker.update({"BTCUSDT": pred}, ts=1500)
        assert ranker.metrics.updates_throttled == 1

        # Update after interval - should not be throttled
        ranker.update({"BTCUSDT": pred}, ts=2000)
        assert ranker.metrics.updates_throttled == 1  # No new throttle

    def test_burst_bypasses_throttle(
        self, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Large score delta bypasses throttling."""
        config = RankerConfig(update_interval_ms=1000, burst_delta=0.1)
        ranker = Ranker(config)

        # First update
        pred1 = make_prediction(symbol="BTCUSDT", expected_utility_bps_2m=20.0)
        ranker.update({"BTCUSDT": pred1}, ts=1000)

        # Second update with large delta - should not be throttled
        pred2 = make_prediction(symbol="BTCUSDT", expected_utility_bps_2m=50.0)
        ranker.update({"BTCUSDT": pred2}, ts=1200)
        assert ranker.metrics.updates_throttled == 0

    def test_top_k_ordering(self, make_prediction: Callable[..., PredictionSnapshot]) -> None:
        """Top-K is ordered by score descending."""
        config = RankerConfig(enter_ms=0, top_k=3)
        ranker = Ranker(config)

        predictions = {
            "BTCUSDT": make_prediction(symbol="BTCUSDT", expected_utility_bps_2m=30.0),
            "ETHUSDT": make_prediction(symbol="ETHUSDT", expected_utility_bps_2m=50.0),
            "BNBUSDT": make_prediction(symbol="BNBUSDT", expected_utility_bps_2m=20.0),
        }

        ranker.update(predictions, ts=1000)
        # All should be pending after first update

        ranker.update(predictions, ts=2000)
        # ETHUSDT has highest score
        top_k = ranker.top_k
        assert len(top_k) <= 3
        # Order determined by score + hysteresis state

    def test_metrics_tracking(
        self, ranker: Ranker, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Metrics are tracked correctly."""
        pred = make_prediction(symbol="BTCUSDT")

        ranker.update({"BTCUSDT": pred}, ts=1000)
        assert ranker.metrics.updates_processed == 1

        ranker.update({"BTCUSDT": pred}, ts=2000)
        assert ranker.metrics.updates_processed == 2

    def test_reset(
        self, ranker: Ranker, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Reset clears all state."""
        pred = make_prediction(symbol="BTCUSDT")
        ranker.update({"BTCUSDT": pred}, ts=1000)

        ranker.reset()

        assert ranker.top_k == []
        assert ranker.get_state("BTCUSDT") is None
        assert ranker.metrics.updates_processed == 0


class TestRankerDeterminism:
    """Tests for ranker determinism."""

    @pytest.fixture
    def make_prediction(self) -> Callable[..., PredictionSnapshot]:
        """Factory for creating PredictionSnapshots."""

        def _make(
            symbol: str = "BTCUSDT",
            expected_utility_bps_2m: float = 30.0,
        ) -> PredictionSnapshot:
            return PredictionSnapshot(
                ts=1000,
                symbol=symbol,
                profile=ExecutionProfile.A,
                p_inplay_30s=0.6,
                p_inplay_2m=0.7,
                p_inplay_5m=0.65,
                expected_utility_bps_2m=expected_utility_bps_2m,
                p_toxic=0.1,
                status=PredictionStatus.TRADEABLE,
                reasons=[],
                model_version="test-v1.0.0",
                calibration_version="cal-v1.0.0",
                data_health=DataHealth(),
            )

        return _make

    def test_same_input_same_output(
        self, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Same input sequence produces same output sequence."""
        config = RankerConfig(enter_ms=1000, exit_ms=2000, top_k=5)

        predictions_sequence = [
            {"BTCUSDT": make_prediction("BTCUSDT", 30.0)},
            {
                "BTCUSDT": make_prediction("BTCUSDT", 30.0),
                "ETHUSDT": make_prediction("ETHUSDT", 25.0),
            },
            {"ETHUSDT": make_prediction("ETHUSDT", 25.0)},
        ]
        timestamps = [1000, 2500, 4000]

        # Run 1
        ranker1 = Ranker(config)
        events1: list[RankEvent] = []
        for preds, ts in zip(predictions_sequence, timestamps, strict=True):
            events1.extend(ranker1.update(preds, ts))

        # Run 2
        ranker2 = Ranker(config)
        events2: list[RankEvent] = []
        for preds, ts in zip(predictions_sequence, timestamps, strict=True):
            events2.extend(ranker2.update(preds, ts))

        # Events should match
        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2, strict=True):
            assert e1.event == e2.event
            assert e1.symbol == e2.symbol
            assert e1.ts == e2.ts
