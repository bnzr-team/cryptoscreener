"""
Tests for delivery router (DEC-039).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptoscreener.contracts import RankEvent, RankEventPayload, RankEventType
from cryptoscreener.delivery.config import (
    DedupeConfig,
    DeliveryConfig,
    SinkConfig,
    TelegramSinkConfig,
    WebhookSinkConfig,
)
from cryptoscreener.delivery.router import DeliveryRouter
from cryptoscreener.delivery.sinks.base import DeliveryResult


def make_event(
    symbol: str = "BTCUSDT",
    event_type: RankEventType = RankEventType.ALERT_TRADABLE,
    ts: int = 1706400000000,
) -> RankEvent:
    """Helper to create test events."""
    return RankEvent(
        ts=ts,
        event=event_type,
        symbol=symbol,
        rank=0,
        score=0.9,
        payload=RankEventPayload(
            prediction={"status": "TRADEABLE"},
            llm_text="",
        ),
    )


class TestDeliveryRouter:
    """Tests for DeliveryRouter."""

    def test_init_no_sinks_enabled(self) -> None:
        """Router initializes with warning when no sinks enabled."""
        config = DeliveryConfig(enabled=True)
        router = DeliveryRouter(config)

        assert len(router._sinks) == 0

    def test_init_with_telegram_sink(self) -> None:
        """Router initializes Telegram sink when enabled."""
        config = DeliveryConfig(
            enabled=True,
            sinks=SinkConfig(
                telegram=TelegramSinkConfig(
                    enabled=True,
                    bot_token="test_token",
                    chat_id="12345",
                )
            ),
        )
        router = DeliveryRouter(config)

        assert len(router._sinks) == 1
        assert router._sinks[0].sink_type == "telegram"

    @pytest.mark.asyncio
    async def test_publish_disabled_returns_empty(self) -> None:
        """Publish returns empty when delivery is disabled."""
        config = DeliveryConfig(enabled=False)
        router = DeliveryRouter(config)

        events = [make_event()]
        results = await router.publish(events)

        assert results == []

    @pytest.mark.asyncio
    async def test_publish_empty_events_returns_empty(self) -> None:
        """Publish returns empty for empty event list."""
        config = DeliveryConfig(enabled=True)
        router = DeliveryRouter(config)

        results = await router.publish([])

        assert results == []

    @pytest.mark.asyncio
    async def test_publish_deduplicates_events(self) -> None:
        """Publish applies deduplication."""
        config = DeliveryConfig(
            enabled=True,
            dedupe=DedupeConfig(
                per_symbol_cooldown_s=120.0,
                status_transition_only=False,
            ),
        )
        router = DeliveryRouter(config)

        events = [
            make_event(symbol="BTCUSDT"),
            make_event(symbol="BTCUSDT"),  # Will be deduplicated
        ]

        # Mock formatter to track calls
        router._formatter.format_batch = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        await router.publish(events)

        # Only one event should pass deduplication
        assert router._deduplicator.metrics.total_passed == 1
        assert router._deduplicator.metrics.suppressed_cooldown == 1

        await router.close()

    @pytest.mark.asyncio
    async def test_publish_dry_run_mode(self) -> None:
        """Dry run mode logs but doesn't send."""
        config = DeliveryConfig(
            enabled=True,
            dry_run=True,
            sinks=SinkConfig(
                webhook=WebhookSinkConfig(
                    enabled=True,
                    url="https://example.com/webhook",
                )
            ),
        )
        router = DeliveryRouter(config)

        # In dry run, we shouldn't actually send
        for sink in router._sinks:
            sink.send = AsyncMock(return_value=DeliveryResult(success=True, sink_name="test"))  # type: ignore[method-assign]

        events = [make_event()]
        results = await router.publish(events)

        # Should return dry_run results
        assert len(results) == len(router._sinks)
        for result in results:
            assert result.sink_name == "dry_run"

        # Sinks should not have been called
        for sink in router._sinks:
            sink.send.assert_not_called()  # type: ignore[attr-defined]

        await router.close()

    @pytest.mark.asyncio
    async def test_publish_sends_to_all_sinks(self) -> None:
        """Publish sends to all enabled sinks."""
        config = DeliveryConfig(
            enabled=True,
            sinks=SinkConfig(
                telegram=TelegramSinkConfig(
                    enabled=True,
                    bot_token="test_token",
                    chat_id="12345",
                ),
                webhook=WebhookSinkConfig(
                    enabled=True,
                    url="https://example.com/webhook",
                ),
            ),
        )
        router = DeliveryRouter(config)

        # Mock all sinks
        for sink in router._sinks:
            sink.send = AsyncMock(  # type: ignore[method-assign]
                return_value=DeliveryResult(success=True, sink_name=sink.name)
            )

        events = [make_event()]
        results = await router.publish(events)

        # Should have results from both sinks
        assert len(results) == 2

        # All sinks should have been called
        for sink in router._sinks:
            sink.send.assert_called_once()  # type: ignore[attr-defined]

        await router.close()

    @pytest.mark.asyncio
    async def test_publish_tracks_metrics(self) -> None:
        """Publish tracks delivery metrics."""
        config = DeliveryConfig(
            enabled=True,
            sinks=SinkConfig(
                webhook=WebhookSinkConfig(
                    enabled=True,
                    url="https://example.com/webhook",
                ),
            ),
        )
        router = DeliveryRouter(config)

        router._sinks[0].send = AsyncMock(  # type: ignore[method-assign]
            return_value=DeliveryResult(success=True, sink_name="webhook:custom")
        )

        events = [make_event(symbol="BTCUSDT"), make_event(symbol="ETHUSDT")]
        await router.publish(events)

        m = router.metrics
        assert m.total_received == 2
        assert m.total_delivered == 1
        assert m.sink_successes["webhook:custom"] == 1

        await router.close()

    @pytest.mark.asyncio
    async def test_publish_handles_sink_failure(self) -> None:
        """Publish handles sink failures gracefully."""
        config = DeliveryConfig(
            enabled=True,
            sinks=SinkConfig(
                webhook=WebhookSinkConfig(
                    enabled=True,
                    url="https://example.com/webhook",
                ),
            ),
        )
        router = DeliveryRouter(config)

        router._sinks[0].send = AsyncMock(  # type: ignore[method-assign]
            return_value=DeliveryResult(
                success=False,
                sink_name="webhook:custom",
                error="Connection failed",
            )
        )

        events = [make_event()]
        results = await router.publish(events)

        assert len(results) == 1
        assert results[0].success is False

        m = router.metrics
        assert m.total_failed == 1
        assert m.sink_failures["webhook:custom"] == 1

        await router.close()

    @pytest.mark.asyncio
    async def test_publish_handles_sink_exception(self) -> None:
        """Publish handles sink exceptions gracefully."""
        config = DeliveryConfig(
            enabled=True,
            sinks=SinkConfig(
                webhook=WebhookSinkConfig(
                    enabled=True,
                    url="https://example.com/webhook",
                ),
            ),
        )
        router = DeliveryRouter(config)

        router._sinks[0].send = AsyncMock(side_effect=Exception("Unexpected error"))  # type: ignore[method-assign]

        events = [make_event()]
        results = await router.publish(events)

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
        assert "Unexpected error" in results[0].error

        await router.close()

    @pytest.mark.asyncio
    async def test_publish_one_convenience(self) -> None:
        """publish_one is convenience for single event."""
        config = DeliveryConfig(enabled=True)
        router = DeliveryRouter(config)

        router.publish = AsyncMock(return_value=[])  # type: ignore[method-assign]

        event = make_event()
        await router.publish_one(event)

        router.publish.assert_called_once_with([event])

        await router.close()

    @pytest.mark.asyncio
    async def test_close_closes_all_sinks(self) -> None:
        """Close closes all sinks."""
        config = DeliveryConfig(
            enabled=True,
            sinks=SinkConfig(
                telegram=TelegramSinkConfig(
                    enabled=True,
                    bot_token="test_token",
                    chat_id="12345",
                ),
                webhook=WebhookSinkConfig(
                    enabled=True,
                    url="https://example.com/webhook",
                ),
            ),
        )
        router = DeliveryRouter(config)

        for sink in router._sinks:
            sink.close = AsyncMock()  # type: ignore[method-assign]

        await router.close()

        for sink in router._sinks:
            sink.close.assert_called_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_publish_after_close_ignored(self) -> None:
        """Publish after close is ignored."""
        config = DeliveryConfig(enabled=True)
        router = DeliveryRouter(config)

        await router.close()

        events = [make_event()]
        results = await router.publish(events)

        assert results == []

    def test_reset_metrics(self) -> None:
        """reset_metrics clears all metrics."""
        config = DeliveryConfig(enabled=True)
        router = DeliveryRouter(config)

        router._metrics.total_received = 10
        router._metrics.total_delivered = 5

        router.reset_metrics()

        assert router._metrics.total_received == 0
        assert router._metrics.total_delivered == 0

    def test_dedupe_metrics_property(self) -> None:
        """dedupe_metrics returns deduplicator metrics."""
        config = DeliveryConfig(enabled=True)
        router = DeliveryRouter(config)

        dm = router.dedupe_metrics
        assert "received" in dm
        assert "passed" in dm
        assert "suppressed_cooldown" in dm
