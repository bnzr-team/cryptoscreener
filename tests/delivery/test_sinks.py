"""
Tests for delivery sinks (DEC-039).

Uses aiohttp test utilities to mock HTTP responses.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptoscreener.delivery.config import (
    SlackSinkConfig,
    TelegramSinkConfig,
    WebhookSinkConfig,
)
from cryptoscreener.delivery.formatter import FormattedMessage
from cryptoscreener.delivery.sinks.slack import SlackSink
from cryptoscreener.delivery.sinks.telegram import TelegramSink
from cryptoscreener.delivery.sinks.webhook import WebhookSink


@pytest.fixture
def sample_message() -> FormattedMessage:
    """Create a sample formatted message for testing."""
    return FormattedMessage(
        text="Test alert message",
        html="<b>Test</b> alert message",
        markdown="*Test* alert message",
    )


class TestTelegramSink:
    """Tests for Telegram sink."""

    @pytest.mark.asyncio
    async def test_send_disabled(self, sample_message: FormattedMessage) -> None:
        """Send returns failure when sink is disabled."""
        config = TelegramSinkConfig(enabled=False)
        sink = TelegramSink(config)

        result = await sink.send(sample_message)

        assert result.success is False
        assert result.error is not None
        assert "not enabled" in result.error.lower()
        await sink.close()

    @pytest.mark.asyncio
    async def test_send_success(self, sample_message: FormattedMessage) -> None:
        """Send succeeds with 200 response."""
        config = TelegramSinkConfig(
            enabled=True,
            bot_token="test_token",
            chat_id="12345",
        )
        sink = TelegramSink(config)

        # Mock the aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        sink._session = mock_session

        result = await sink.send(sample_message)

        assert result.success is True
        assert result.status_code == 200
        mock_session.post.assert_called_once()

        await sink.close()

    @pytest.mark.asyncio
    async def test_send_rate_limited(self, sample_message: FormattedMessage) -> None:
        """Send handles 429 rate limit response."""
        config = TelegramSinkConfig(
            enabled=True,
            bot_token="test_token",
            chat_id="12345",
            max_retries=0,  # Don't retry
        )
        sink = TelegramSink(config)

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.json = AsyncMock(return_value={"parameters": {"retry_after": 30}})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        sink._session = mock_session

        result = await sink.send(sample_message)

        assert result.success is False
        assert result.status_code == 429
        assert result.retry_after_s == 30

        await sink.close()

    @pytest.mark.asyncio
    async def test_name_includes_chat_id(self) -> None:
        """Sink name includes chat_id for identification."""
        config = TelegramSinkConfig(
            enabled=True,
            bot_token="test_token",
            chat_id="12345",
        )
        sink = TelegramSink(config)

        assert "12345" in sink.name
        assert sink.sink_type == "telegram"

        await sink.close()


class TestSlackSink:
    """Tests for Slack sink."""

    @pytest.mark.asyncio
    async def test_send_disabled(self, sample_message: FormattedMessage) -> None:
        """Send returns failure when sink is disabled."""
        config = SlackSinkConfig(enabled=False)
        sink = SlackSink(config)

        result = await sink.send(sample_message)

        assert result.success is False
        assert result.error is not None
        assert "not enabled" in result.error.lower()
        await sink.close()

    @pytest.mark.asyncio
    async def test_send_success(self, sample_message: FormattedMessage) -> None:
        """Send succeeds with 200 response."""
        config = SlackSinkConfig(
            enabled=True,
            webhook_url="https://hooks.slack.com/test",
        )
        sink = SlackSink(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        sink._session = mock_session

        result = await sink.send(sample_message)

        assert result.success is True
        assert result.status_code == 200

        await sink.close()

    @pytest.mark.asyncio
    async def test_uses_markdown_format(self, sample_message: FormattedMessage) -> None:
        """Slack sink uses markdown format from message."""
        config = SlackSinkConfig(
            enabled=True,
            webhook_url="https://hooks.slack.com/test",
        )
        sink = SlackSink(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        sink._session = mock_session

        await sink.send(sample_message)

        # Check the payload sent
        call_args = mock_session.post.call_args
        payload = call_args.kwargs.get("json")
        assert payload is not None
        assert payload["text"] == sample_message.markdown
        assert payload["mrkdwn"] is True

        await sink.close()

    @pytest.mark.asyncio
    async def test_name_does_not_expose_url(self) -> None:
        """Sink name should not expose webhook URL."""
        config = SlackSinkConfig(
            enabled=True,
            webhook_url="https://hooks.slack.com/services/SECRET/TOKEN",
        )
        sink = SlackSink(config)

        assert "SECRET" not in sink.name
        assert "TOKEN" not in sink.name
        assert sink.sink_type == "slack"

        await sink.close()


class TestWebhookSink:
    """Tests for generic webhook sink."""

    @pytest.mark.asyncio
    async def test_send_disabled(self, sample_message: FormattedMessage) -> None:
        """Send returns failure when sink is disabled."""
        config = WebhookSinkConfig(enabled=False)
        sink = WebhookSink(config)

        result = await sink.send(sample_message)

        assert result.success is False
        assert result.error is not None
        assert "not enabled" in result.error.lower()
        await sink.close()

    @pytest.mark.asyncio
    async def test_send_success(self, sample_message: FormattedMessage) -> None:
        """Send succeeds with 2xx response."""
        config = WebhookSinkConfig(
            enabled=True,
            url="https://webhook.example.com/alerts",
        )
        sink = WebhookSink(config)

        mock_response = AsyncMock()
        mock_response.status = 201  # Accept any 2xx
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        sink._session = mock_session

        result = await sink.send(sample_message)

        assert result.success is True
        assert result.status_code == 201

        await sink.close()

    @pytest.mark.asyncio
    async def test_includes_all_formats(self, sample_message: FormattedMessage) -> None:
        """Webhook payload includes all format variants."""
        config = WebhookSinkConfig(
            enabled=True,
            url="https://webhook.example.com/alerts",
        )
        sink = WebhookSink(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        sink._session = mock_session

        await sink.send(sample_message)

        call_args = mock_session.post.call_args
        payload = call_args.kwargs.get("json")
        assert payload["text"] == sample_message.text
        assert payload["html"] == sample_message.html
        assert payload["markdown"] == sample_message.markdown

        await sink.close()

    @pytest.mark.asyncio
    async def test_custom_headers(self, sample_message: FormattedMessage) -> None:
        """Custom headers are included in request."""
        config = WebhookSinkConfig(
            enabled=True,
            url="https://webhook.example.com/alerts",
            headers={"X-Custom-Header": "test-value", "Authorization": "Bearer token"},
        )
        sink = WebhookSink(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        sink._session = mock_session

        await sink.send(sample_message)

        call_args = mock_session.post.call_args
        headers = call_args.kwargs.get("headers")
        assert headers["X-Custom-Header"] == "test-value"
        assert headers["Authorization"] == "Bearer token"

        await sink.close()

    @pytest.mark.asyncio
    async def test_name_does_not_expose_url(self) -> None:
        """Sink name should not expose webhook URL."""
        config = WebhookSinkConfig(
            enabled=True,
            url="https://webhook.example.com/secret/path",
        )
        sink = WebhookSink(config)

        assert "secret" not in sink.name
        assert sink.sink_type == "webhook"

        await sink.close()


class TestSinkClose:
    """Tests for sink cleanup."""

    @pytest.mark.asyncio
    async def test_telegram_close_idempotent(self) -> None:
        """Telegram sink close is idempotent."""
        config = TelegramSinkConfig(enabled=True, bot_token="test", chat_id="123")
        sink = TelegramSink(config)

        await sink.close()
        await sink.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_slack_close_idempotent(self) -> None:
        """Slack sink close is idempotent."""
        config = SlackSinkConfig(enabled=True, webhook_url="https://test")
        sink = SlackSink(config)

        await sink.close()
        await sink.close()

    @pytest.mark.asyncio
    async def test_webhook_close_idempotent(self) -> None:
        """Webhook sink close is idempotent."""
        config = WebhookSinkConfig(enabled=True, url="https://test")
        sink = WebhookSink(config)

        await sink.close()
        await sink.close()
