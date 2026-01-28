"""
Webhook sink (DEC-039).

Delivers messages via generic HTTP webhook.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import aiohttp

from cryptoscreener.delivery.sinks.base import DeliveryResult, DeliverySink

if TYPE_CHECKING:
    from cryptoscreener.contracts import RankEvent
    from cryptoscreener.delivery.config import WebhookSinkConfig
    from cryptoscreener.delivery.formatter import FormattedMessage

logger = logging.getLogger(__name__)


class WebhookSink(DeliverySink):
    """
    Generic webhook delivery sink.

    Sends RankEvent as JSON to configured URL.
    Also includes formatted text in payload.
    """

    def __init__(self, config: WebhookSinkConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        # Don't expose webhook URL in name
        return "webhook:custom"

    @property
    def sink_type(self) -> Literal["telegram", "slack", "webhook"]:
        return "webhook"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, message: FormattedMessage) -> DeliveryResult:
        """Send message to webhook."""
        if not self._config.enabled:
            return DeliveryResult(
                success=False,
                sink_name=self.name,
                error="Webhook sink not enabled",
            )

        # Webhook payload includes all format variants
        payload = {
            "text": message.text,
            "html": message.html,
            "markdown": message.markdown,
        }

        headers = {"Content-Type": "application/json"}
        headers.update(self._config.headers)

        for attempt in range(self._config.max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(
                    self._config.url, json=payload, headers=headers
                ) as resp:
                    status = resp.status

                    if 200 <= status < 300:
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
                            "Webhook rate limited",
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
                        "Webhook send failed",
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
                    "Webhook connection error",
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

    async def send_raw(self, event: RankEvent) -> DeliveryResult:
        """
        Send raw RankEvent JSON to webhook.

        Alternative to formatted message - sends full event data.
        """
        if not self._config.enabled:
            return DeliveryResult(
                success=False,
                sink_name=self.name,
                error="Webhook sink not enabled",
            )

        # Send raw RankEvent JSON
        payload = event.model_dump(mode="json")

        headers = {"Content-Type": "application/json"}
        headers.update(self._config.headers)

        try:
            session = await self._get_session()
            async with session.post(
                self._config.url, json=payload, headers=headers
            ) as resp:
                status = resp.status

                if 200 <= status < 300:
                    return DeliveryResult(
                        success=True,
                        sink_name=self.name,
                        status_code=status,
                    )

                error_text = await resp.text()
                return DeliveryResult(
                    success=False,
                    sink_name=self.name,
                    error=f"HTTP {status}: {error_text[:200]}",
                    status_code=status,
                )

        except aiohttp.ClientError as e:
            return DeliveryResult(
                success=False,
                sink_name=self.name,
                error=f"Connection error: {e}",
            )

    async def close(self) -> None:
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
