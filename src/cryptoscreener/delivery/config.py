"""
Delivery configuration (DEC-039).

Configuration for RankEvent delivery sinks and anti-spam controls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

# Redacted env vars for logging (DEC-034 secrets strategy)
DELIVERY_REDACTED_ENV_VARS = frozenset({
    "TELEGRAM_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "DELIVERY_WEBHOOK_URL",
})


@dataclass
class TelegramSinkConfig:
    """Telegram sink configuration."""

    enabled: bool = False
    bot_token: str = ""  # From TELEGRAM_BOT_TOKEN env var
    chat_id: str = ""  # Target chat/channel ID
    parse_mode: Literal["HTML", "Markdown", "MarkdownV2"] = "HTML"
    timeout_s: float = 10.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.enabled:
            if not self.bot_token:
                self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            if not self.chat_id:
                self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if self.enabled and not self.bot_token:
                raise ValueError("TELEGRAM_BOT_TOKEN required when Telegram sink enabled")
            if self.enabled and not self.chat_id:
                raise ValueError("TELEGRAM_CHAT_ID required when Telegram sink enabled")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s}")


@dataclass
class SlackSinkConfig:
    """Slack sink configuration."""

    enabled: bool = False
    webhook_url: str = ""  # From SLACK_WEBHOOK_URL env var
    timeout_s: float = 10.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.enabled:
            if not self.webhook_url:
                self.webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
            if self.enabled and not self.webhook_url:
                raise ValueError("SLACK_WEBHOOK_URL required when Slack sink enabled")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s}")


@dataclass
class WebhookSinkConfig:
    """Generic webhook sink configuration."""

    enabled: bool = False
    url: str = ""  # From DELIVERY_WEBHOOK_URL env var
    timeout_s: float = 10.0
    max_retries: int = 2
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enabled:
            if not self.url:
                self.url = os.environ.get("DELIVERY_WEBHOOK_URL", "")
            if self.enabled and not self.url:
                raise ValueError("DELIVERY_WEBHOOK_URL required when Webhook sink enabled")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s}")


@dataclass
class DedupeConfig:
    """Deduplication and anti-spam configuration."""

    # Per-symbol cooldown: don't send same symbol+event_type within N seconds
    per_symbol_cooldown_s: float = 120.0

    # Global rate limit: max N alerts per minute across all symbols
    global_max_per_minute: int = 30

    # Status transition only: only send on status changes (if status available)
    status_transition_only: bool = True

    def __post_init__(self) -> None:
        if self.per_symbol_cooldown_s < 0:
            raise ValueError(f"per_symbol_cooldown_s must be >= 0, got {self.per_symbol_cooldown_s}")
        if self.global_max_per_minute < 1:
            raise ValueError(f"global_max_per_minute must be >= 1, got {self.global_max_per_minute}")


@dataclass
class SinkConfig:
    """Combined sink configuration."""

    telegram: TelegramSinkConfig = field(default_factory=TelegramSinkConfig)
    slack: SlackSinkConfig = field(default_factory=SlackSinkConfig)
    webhook: WebhookSinkConfig = field(default_factory=WebhookSinkConfig)

    def any_enabled(self) -> bool:
        """Check if any sink is enabled."""
        return self.telegram.enabled or self.slack.enabled or self.webhook.enabled


@dataclass
class DeliveryConfig:
    """Main delivery configuration."""

    enabled: bool = False
    sinks: SinkConfig = field(default_factory=SinkConfig)
    dedupe: DedupeConfig = field(default_factory=DedupeConfig)

    # Formatting
    include_llm_text: bool = True  # Include LLM explanation if available
    include_grafana_link: bool = False  # Include Grafana dashboard link
    grafana_base_url: str = ""  # Base URL for Grafana dashboards

    # Dry run mode: log but don't send
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.include_grafana_link and not self.grafana_base_url:
            # Optional: warn but don't fail
            pass
