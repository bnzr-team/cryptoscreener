"""
Slack sink (DEC-039).

Delivers messages via Slack Incoming Webhooks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import aiohttp

from cryptoscreener.delivery.sinks.base import DeliveryResult, DeliverySink

if TYPE_CHECKING:
    from cryptoscreener.delivery.config import SlackSinkConfig
    from cryptoscreener.delivery.formatter import FormattedMessage

logger = logging.getLogger(__name__)


class SlackSink(DeliverySink):
    """
    Slack delivery sink using Incoming Webhooks.

    Uses mrkdwn format for rich text formatting.
    """

    def __init__(self, config: SlackSinkConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        # Don't expose webhook URL in name
        return "slack:webhook"

    @property
    def sink_type(self) -> Literal["telegram", "slack", "webhook"]:
        return "slack"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, message: FormattedMessage) -> DeliveryResult:
        """Send message to Slack."""
        if not self._config.enabled:
            return DeliveryResult(
                success=False,
                sink_name=self.name,
                error="Slack sink not enabled",
            )

        # Use markdown format for Slack (mrkdwn)
        payload = {
            "text": message.markdown,
            "mrkdwn": True,
        }

        for attempt in range(self._config.max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(self._config.webhook_url, json=payload) as resp:
                    status = resp.status

                    if status == 200:
                        return DeliveryResult(
                            success=True,
                            sink_name=self.name,
                            status_code=status,
                        )

                    # Rate limited
                    if status == 429:
                        retry_after = resp.headers.get("Retry-After", "60")
                        try:
                            retry_after_s = float(retry_after)
                        except ValueError:
                            retry_after_s = 60.0
                        logger.warning(
                            "Slack rate limited",
                            extra={"retry_after": retry_after_s, "attempt": attempt},
                        )
                        if attempt < self._config.max_retries:
                            import asyncio

                            await asyncio.sleep(min(retry_after_s, 5))
                            continue
                        return DeliveryResult(
                            success=False,
                            sink_name=self.name,
                            error=f"Rate limited (retry_after={retry_after_s})",
                            status_code=status,
                            retry_after_s=retry_after_s,
                        )

                    # Other errors
                    error_text = await resp.text()
                    logger.error(
                        "Slack send failed",
                        extra={"status": status, "error": error_text, "attempt": attempt},
                    )
                    if attempt < self._config.max_retries:
                        import asyncio

                        await asyncio.sleep(1)
                        continue
                    return DeliveryResult(
                        success=False,
                        sink_name=self.name,
                        error=f"HTTP {status}: {error_text[:200]}",
                        status_code=status,
                    )

            except aiohttp.ClientError as e:
                logger.error(
                    "Slack connection error",
                    extra={"error": str(e), "attempt": attempt},
                )
                if attempt < self._config.max_retries:
                    import asyncio

                    await asyncio.sleep(1)
                    continue
                return DeliveryResult(
                    success=False,
                    sink_name=self.name,
                    error=f"Connection error: {e}",
                )

        return DeliveryResult(
            success=False,
            sink_name=self.name,
            error="Max retries exceeded",
        )

    async def close(self) -> None:
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
