"""
Telegram sink (DEC-039).

Delivers messages via Telegram Bot API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import aiohttp

from cryptoscreener.delivery.sinks.base import DeliveryResult, DeliverySink

if TYPE_CHECKING:
    from cryptoscreener.delivery.config import TelegramSinkConfig
    from cryptoscreener.delivery.formatter import FormattedMessage

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramSink(DeliverySink):
    """
    Telegram delivery sink using Bot API.

    Uses sendMessage endpoint with HTML parse mode by default.
    """

    def __init__(self, config: TelegramSinkConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return f"telegram:{self._config.chat_id}"

    @property
    def sink_type(self) -> Literal["telegram", "slack", "webhook"]:
        return "telegram"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, message: FormattedMessage) -> DeliveryResult:
        """Send message to Telegram."""
        if not self._config.enabled:
            return DeliveryResult(
                success=False,
                sink_name=self.name,
                error="Telegram sink not enabled",
            )

        # Select message format based on parse_mode
        if self._config.parse_mode == "HTML":
            text = message.html
        elif self._config.parse_mode in ("Markdown", "MarkdownV2"):
            # Use plain text for Markdown modes as our escaping is for Slack mrkdwn
            text = message.text
        else:
            text = message.text

        url = f"{TELEGRAM_API_BASE}/bot{self._config.bot_token}/sendMessage"
        payload = {
            "chat_id": self._config.chat_id,
            "text": text,
            "parse_mode": self._config.parse_mode,
            "disable_web_page_preview": True,
        }

        for attempt in range(self._config.max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload) as resp:
                    status = resp.status

                    if status == 200:
                        return DeliveryResult(
                            success=True,
                            sink_name=self.name,
                            status_code=status,
                        )

                    # Rate limited
                    if status == 429:
                        data = await resp.json()
                        retry_after = data.get("parameters", {}).get("retry_after", 60)
                        logger.warning(
                            "Telegram rate limited",
                            extra={"retry_after": retry_after, "attempt": attempt},
                        )
                        if attempt < self._config.max_retries:
                            import asyncio

                            await asyncio.sleep(min(retry_after, 5))
                            continue
                        return DeliveryResult(
                            success=False,
                            sink_name=self.name,
                            error=f"Rate limited (retry_after={retry_after})",
                            status_code=status,
                            retry_after_s=retry_after,
                        )

                    # Other errors
                    error_text = await resp.text()
                    logger.error(
                        "Telegram send failed",
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
                    "Telegram connection error",
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

        # Should not reach here, but safety return
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
