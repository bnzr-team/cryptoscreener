"""
Tests for delivery deduplication and anti-spam (DEC-039).
"""

from __future__ import annotations

import time
from unittest.mock import patch

from cryptoscreener.contracts import RankEvent, RankEventPayload, RankEventType
from cryptoscreener.delivery.config import DedupeConfig
from cryptoscreener.delivery.dedupe import DeliveryDeduplicator


def make_event(
    symbol: str = "BTCUSDT",
    event_type: RankEventType = RankEventType.ALERT_TRADABLE,
    status: str = "TRADEABLE",
    ts: int = 1706400000000,
    rank: int = 0,
) -> RankEvent:
    """Helper to create test events."""
    return RankEvent(
        ts=ts,
        event=event_type,
        symbol=symbol,
        rank=rank,
        score=0.9,
        payload=RankEventPayload(prediction={"status": status}, llm_text=""),
    )


class TestDeliveryDeduplicator:
    """Tests for DeliveryDeduplicator."""

    def test_first_event_always_passes(self) -> None:
        """First event for a symbol should always pass."""
        config = DedupeConfig(per_symbol_cooldown_s=120.0)
        deduper = DeliveryDeduplicator(config)

        event = make_event()
        assert deduper.should_deliver(event) is True

    def test_cooldown_blocks_same_symbol_event_type(self) -> None:
        """Same symbol+event_type within cooldown should be blocked."""
        config = DedupeConfig(per_symbol_cooldown_s=120.0)
        deduper = DeliveryDeduplicator(config)

        event1 = make_event(symbol="BTCUSDT", event_type=RankEventType.ALERT_TRADABLE)
        event2 = make_event(symbol="BTCUSDT", event_type=RankEventType.ALERT_TRADABLE)

        assert deduper.should_deliver(event1) is True
        assert deduper.should_deliver(event2) is False  # blocked by cooldown

        assert deduper.metrics.suppressed_cooldown == 1

    def test_different_symbol_passes(self) -> None:
        """Different symbol should pass even if same event type."""
        config = DedupeConfig(per_symbol_cooldown_s=120.0)
        deduper = DeliveryDeduplicator(config)

        event1 = make_event(symbol="BTCUSDT", event_type=RankEventType.ALERT_TRADABLE)
        event2 = make_event(symbol="ETHUSDT", event_type=RankEventType.ALERT_TRADABLE)

        assert deduper.should_deliver(event1) is True
        assert deduper.should_deliver(event2) is True

    def test_different_event_type_passes(self) -> None:
        """Different event type should pass for same symbol."""
        config = DedupeConfig(per_symbol_cooldown_s=120.0, status_transition_only=False)
        deduper = DeliveryDeduplicator(config)

        event1 = make_event(symbol="BTCUSDT", event_type=RankEventType.ALERT_TRADABLE)
        event2 = make_event(symbol="BTCUSDT", event_type=RankEventType.SYMBOL_ENTER)

        assert deduper.should_deliver(event1) is True
        assert deduper.should_deliver(event2) is True

    def test_cooldown_expires(self) -> None:
        """Event should pass after cooldown expires."""
        config = DedupeConfig(
            per_symbol_cooldown_s=0.1,  # 100ms for fast test
            status_transition_only=False,
        )
        deduper = DeliveryDeduplicator(config)

        event1 = make_event()
        event2 = make_event()

        assert deduper.should_deliver(event1) is True
        assert deduper.should_deliver(event2) is False

        # Wait for cooldown
        time.sleep(0.15)

        event3 = make_event()
        assert deduper.should_deliver(event3) is True

    def test_global_rate_limit(self) -> None:
        """Global rate limit should block excess events."""
        config = DedupeConfig(
            per_symbol_cooldown_s=0,  # No cooldown
            global_max_per_minute=3,
            status_transition_only=False,
        )
        deduper = DeliveryDeduplicator(config)

        # First 3 should pass
        for i in range(3):
            event = make_event(symbol=f"SYM{i}USDT")
            assert deduper.should_deliver(event) is True

        # 4th should be blocked by global limit
        event = make_event(symbol="SYM3USDT")
        assert deduper.should_deliver(event) is False
        assert deduper.metrics.suppressed_rate_limit == 1

    def test_global_rate_limit_window_slides(self) -> None:
        """Global rate limit window should slide (old events expire)."""
        config = DedupeConfig(
            per_symbol_cooldown_s=0,
            global_max_per_minute=2,
            status_transition_only=False,
        )
        deduper = DeliveryDeduplicator(config)

        # Mock time to simulate window sliding
        base_time = time.time()

        with patch("time.time", return_value=base_time):
            assert deduper.should_deliver(make_event(symbol="SYM1USDT")) is True
            assert deduper.should_deliver(make_event(symbol="SYM2USDT")) is True
            assert deduper.should_deliver(make_event(symbol="SYM3USDT")) is False

        # Move time forward past 60s window
        with patch("time.time", return_value=base_time + 61):
            assert deduper.should_deliver(make_event(symbol="SYM4USDT")) is True

    def test_status_transition_only_blocks_same_status(self) -> None:
        """Status transition mode should block events with same status."""
        config = DedupeConfig(
            per_symbol_cooldown_s=0,
            status_transition_only=True,
        )
        deduper = DeliveryDeduplicator(config)

        event1 = make_event(symbol="BTCUSDT", status="TRADEABLE")
        event2 = make_event(symbol="BTCUSDT", status="TRADEABLE")

        assert deduper.should_deliver(event1) is True
        assert deduper.should_deliver(event2) is False  # same status

        assert deduper.metrics.suppressed_duplicate == 1

    def test_status_transition_allows_different_status(self) -> None:
        """Status transition mode should allow events with different status."""
        config = DedupeConfig(
            per_symbol_cooldown_s=0,
            status_transition_only=True,
        )
        deduper = DeliveryDeduplicator(config)

        event1 = make_event(symbol="BTCUSDT", status="TRADEABLE")
        event2 = make_event(symbol="BTCUSDT", status="TRAP")

        assert deduper.should_deliver(event1) is True
        assert deduper.should_deliver(event2) is True

    def test_filter_batch(self) -> None:
        """filter_batch should return only events that pass."""
        config = DedupeConfig(
            per_symbol_cooldown_s=120.0,
            status_transition_only=False,
        )
        deduper = DeliveryDeduplicator(config)

        events = [
            make_event(symbol="BTCUSDT"),
            make_event(symbol="BTCUSDT"),  # blocked by cooldown
            make_event(symbol="ETHUSDT"),
        ]

        filtered = deduper.filter_batch(events)
        assert len(filtered) == 2
        assert filtered[0].symbol == "BTCUSDT"
        assert filtered[1].symbol == "ETHUSDT"

    def test_reset_clears_state(self) -> None:
        """reset() should clear all state."""
        config = DedupeConfig(per_symbol_cooldown_s=120.0)
        deduper = DeliveryDeduplicator(config)

        event = make_event()
        assert deduper.should_deliver(event) is True
        assert deduper.should_deliver(event) is False

        deduper.reset()

        assert deduper.should_deliver(event) is True
        assert deduper.metrics.total_received == 1

    def test_get_stable_key(self) -> None:
        """get_stable_key should produce consistent keys."""
        config = DedupeConfig()
        deduper = DeliveryDeduplicator(config)

        event = make_event(symbol="BTCUSDT", status="TRADEABLE", rank=2)

        key = deduper.get_stable_key(event)
        assert "BTCUSDT" in key
        assert "ALERT_TRADABLE" in key
        assert "TRADEABLE" in key
        assert "top5" in key

    def test_get_stable_key_rank_buckets(self) -> None:
        """get_stable_key should bucket ranks correctly."""
        config = DedupeConfig()
        deduper = DeliveryDeduplicator(config)

        event_top5 = make_event(rank=3)
        assert "top5" in deduper.get_stable_key(event_top5)

        event_top10 = make_event(rank=7)
        assert "top10" in deduper.get_stable_key(event_top10)

        event_other = make_event(rank=15)
        assert "other" in deduper.get_stable_key(event_other)

    def test_metrics_tracking(self) -> None:
        """Metrics should track all events correctly."""
        config = DedupeConfig(
            per_symbol_cooldown_s=120.0,
            global_max_per_minute=100,
            status_transition_only=False,
        )
        deduper = DeliveryDeduplicator(config)

        events = [
            make_event(symbol="BTCUSDT"),
            make_event(symbol="BTCUSDT"),  # cooldown
            make_event(symbol="ETHUSDT"),
        ]

        for event in events:
            deduper.should_deliver(event)

        m = deduper.metrics
        assert m.total_received == 3
        assert m.total_passed == 2
        assert m.suppressed_cooldown == 1
        assert m.suppressed_rate_limit == 0
        assert m.suppressed_duplicate == 0
