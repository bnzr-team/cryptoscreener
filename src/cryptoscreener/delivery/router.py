"""
Delivery router (DEC-039).

Routes RankEvents to configured sinks with deduplication and formatting.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cryptoscreener.delivery.config import DeliveryConfig  # noqa: TC001
from cryptoscreener.delivery.dedupe import DeliveryDeduplicator
from cryptoscreener.delivery.formatter import FormattedMessage, RankEventFormatter
from cryptoscreener.delivery.sinks.base import DeliveryResult, DeliverySink
from cryptoscreener.delivery.sinks.slack import SlackSink
from cryptoscreener.delivery.sinks.telegram import TelegramSink
from cryptoscreener.delivery.sinks.webhook import WebhookSink

if TYPE_CHECKING:
    from cryptoscreener.contracts import RankEvent

logger = logging.getLogger(__name__)


@dataclass
class DeliveryMetrics:
    """Metrics for delivery operations."""

    total_received: int = 0
    total_delivered: int = 0
    total_failed: int = 0
    total_suppressed: int = 0
    sink_successes: dict[str, int] = field(default_factory=dict)
    sink_failures: dict[str, int] = field(default_factory=dict)


class DeliveryRouter:
    """
    Routes RankEvents to configured sinks.

    Flow:
    1. Events pass through deduplicator (anti-spam)
    2. Passing events are formatted
    3. Formatted messages sent to all enabled sinks
    4. Results aggregated and logged

    Thread-safe for use from async event loop.
    """

    def __init__(self, config: DeliveryConfig) -> None:
        self._config = config
        self._deduplicator = DeliveryDeduplicator(config.dedupe)
        self._formatter = RankEventFormatter(
            include_llm_text=config.include_llm_text,
            include_grafana_link=config.include_grafana_link,
            grafana_base_url=config.grafana_base_url,
        )
        self._sinks: list[DeliverySink] = []
        self._metrics = DeliveryMetrics()
        self._closed = False

        # Initialize enabled sinks
        self._init_sinks()

    def _init_sinks(self) -> None:
        """Initialize configured sinks."""
        sinks_config = self._config.sinks

        if sinks_config.telegram.enabled:
            self._sinks.append(TelegramSink(sinks_config.telegram))
            logger.info("Telegram sink enabled")

        if sinks_config.slack.enabled:
            self._sinks.append(SlackSink(sinks_config.slack))
            logger.info("Slack sink enabled")

        if sinks_config.webhook.enabled:
            self._sinks.append(WebhookSink(sinks_config.webhook))
            logger.info("Webhook sink enabled")

        if not self._sinks:
            logger.warning("No delivery sinks enabled")

    @property
    def metrics(self) -> DeliveryMetrics:
        """Get current delivery metrics."""
        return self._metrics

    @property
    def dedupe_metrics(self) -> dict[str, int]:
        """Get deduplication metrics."""
        dm = self._deduplicator.metrics
        return {
            "received": dm.total_received,
            "passed": dm.total_passed,
            "suppressed_cooldown": dm.suppressed_cooldown,
            "suppressed_rate_limit": dm.suppressed_rate_limit,
            "suppressed_duplicate": dm.suppressed_duplicate,
        }

    async def publish(self, events: list[RankEvent]) -> list[DeliveryResult]:
        """
        Publish RankEvents to all configured sinks.

        Events are deduplicated, formatted, and sent to all enabled sinks.

        Args:
            events: List of RankEvents to deliver

        Returns:
            List of DeliveryResults (one per sink per event batch)
        """
        if self._closed:
            logger.warning("DeliveryRouter is closed, ignoring publish")
            return []

        if not self._config.enabled:
            return []

        if not events:
            return []

        self._metrics.total_received += len(events)

        # Apply deduplication
        filtered = self._deduplicator.filter_batch(events)
        suppressed = len(events) - len(filtered)
        self._metrics.total_suppressed += suppressed

        if suppressed > 0:
            logger.debug(
                "Suppressed events",
                extra={"suppressed": suppressed, "passed": len(filtered)},
            )

        if not filtered:
            return []

        # Format messages
        message = self._formatter.format_batch(filtered)

        # Dry run mode
        if self._config.dry_run:
            logger.info(
                "Dry run delivery",
                extra={"events": len(filtered), "text": message.text[:200]},
            )
            return [
                DeliveryResult(success=True, sink_name="dry_run")
                for _ in self._sinks
            ]

        # Send to all sinks concurrently
        results = await asyncio.gather(
            *[self._send_to_sink(sink, message) for sink in self._sinks],
            return_exceptions=True,
        )

        # Process results
        final_results: list[DeliveryResult] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Sink delivery exception", extra={"error": str(result)})
                final_results.append(
                    DeliveryResult(
                        success=False,
                        sink_name="unknown",
                        error=str(result),
                    )
                )
                self._metrics.total_failed += 1
            elif isinstance(result, DeliveryResult):
                final_results.append(result)
                if result.success:
                    self._metrics.total_delivered += 1
                    self._metrics.sink_successes[result.sink_name] = (
                        self._metrics.sink_successes.get(result.sink_name, 0) + 1
                    )
                else:
                    self._metrics.total_failed += 1
                    self._metrics.sink_failures[result.sink_name] = (
                        self._metrics.sink_failures.get(result.sink_name, 0) + 1
                    )

        return final_results

    async def publish_one(self, event: RankEvent) -> list[DeliveryResult]:
        """
        Publish a single RankEvent.

        Convenience method for single event delivery.
        """
        return await self.publish([event])

    async def _send_to_sink(
        self, sink: DeliverySink, message: FormattedMessage
    ) -> DeliveryResult:
        """Send message to a single sink with error handling."""
        try:
            return await sink.send(message)
        except Exception as e:
            logger.error(
                "Sink send error",
                extra={"sink": sink.name, "error": str(e)},
            )
            return DeliveryResult(
                success=False,
                sink_name=sink.name,
                error=str(e),
            )

    async def close(self) -> None:
        """Close all sinks and release resources."""
        self._closed = True
        for sink in self._sinks:
            try:
                await sink.close()
            except Exception as e:
                logger.error(
                    "Error closing sink",
                    extra={"sink": sink.name, "error": str(e)},
                )
        self._sinks.clear()

    def reset_metrics(self) -> None:
        """Reset all metrics (for testing)."""
        self._metrics = DeliveryMetrics()
        self._deduplicator.reset()
