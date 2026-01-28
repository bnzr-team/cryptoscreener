"""
Delivery sinks (DEC-039).

Sink implementations for RankEvent delivery.
"""

from __future__ import annotations

from cryptoscreener.delivery.sinks.base import DeliveryResult, DeliverySink
from cryptoscreener.delivery.sinks.slack import SlackSink
from cryptoscreener.delivery.sinks.telegram import TelegramSink
from cryptoscreener.delivery.sinks.webhook import WebhookSink

__all__ = [
    "DeliveryResult",
    "DeliverySink",
    "SlackSink",
    "TelegramSink",
    "WebhookSink",
]
