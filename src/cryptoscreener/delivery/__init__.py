"""
RankEvent Delivery Module (DEC-039).

Delivers RankEvents to configured sinks (Telegram, Slack, Webhook)
with anti-spam, deduplication, and deterministic formatting.
"""

from __future__ import annotations

from cryptoscreener.delivery.config import DeliveryConfig, SinkConfig
from cryptoscreener.delivery.dedupe import DeliveryDeduplicator
from cryptoscreener.delivery.formatter import RankEventFormatter
from cryptoscreener.delivery.router import DeliveryRouter

__all__ = [
    "DeliveryConfig",
    "DeliveryDeduplicator",
    "DeliveryRouter",
    "RankEventFormatter",
    "SinkConfig",
]
