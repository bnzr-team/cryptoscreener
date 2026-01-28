"""
RankEvent formatter (DEC-039).

Deterministic template-based formatting for RankEvent delivery.
LLM text included if available, but formatting never depends on LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptoscreener.contracts import RankEvent

# Event type to emoji/icon mapping
EVENT_TYPE_ICONS = {
    "SYMBOL_ENTER": "\u2934\ufe0f",  # up arrow
    "SYMBOL_EXIT": "\u2935\ufe0f",  # down arrow
    "ALERT_TRADABLE": "\u2705",  # green check
    "ALERT_TRAP": "\u26a0\ufe0f",  # warning
    "DATA_ISSUE": "\u274c",  # red X
}

# Event type to human label
EVENT_TYPE_LABELS = {
    "SYMBOL_ENTER": "Entered Ranking",
    "SYMBOL_EXIT": "Exited Ranking",
    "ALERT_TRADABLE": "Tradeable",
    "ALERT_TRAP": "Trap - Avoid",
    "DATA_ISSUE": "Data Issue",
}


@dataclass
class FormattedMessage:
    """Formatted message ready for delivery."""

    text: str  # Plain text version
    html: str  # HTML version (for Telegram)
    markdown: str  # Markdown version (for Slack)


class RankEventFormatter:
    """
    Deterministic formatter for RankEvent objects.

    Produces consistent, template-based output regardless of LLM availability.
    LLM text is included if present, but never required.
    """

    def __init__(
        self,
        include_llm_text: bool = True,
        include_grafana_link: bool = False,
        grafana_base_url: str = "",
    ) -> None:
        self._include_llm_text = include_llm_text
        self._include_grafana_link = include_grafana_link
        self._grafana_base_url = grafana_base_url.rstrip("/")

    def format(self, event: RankEvent) -> FormattedMessage:
        """Format RankEvent into delivery message."""
        icon = EVENT_TYPE_ICONS.get(event.event.value, "\u2139\ufe0f")
        label = EVENT_TYPE_LABELS.get(event.event.value, event.event.value)

        # Format timestamp
        ts_dt = datetime.fromtimestamp(event.ts / 1000, tz=UTC)
        ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Extract prediction status if available
        prediction = event.payload.prediction or {}
        status = prediction.get("status", "")
        p_inplay = prediction.get("p_inplay_2m")
        p_toxic = prediction.get("p_toxic")

        # Build plain text
        lines = [
            f"{icon} {event.symbol} - {label}",
            f"Time: {ts_str}",
            f"Rank: #{event.rank + 1} | Score: {event.score:.4f}",
        ]

        if status:
            lines.append(f"Status: {status}")

        if p_inplay is not None:
            lines.append(f"p_inplay: {p_inplay:.2%}")

        if p_toxic is not None:
            lines.append(f"p_toxic: {p_toxic:.2%}")

        # Include LLM text if available and enabled
        llm_text = event.payload.llm_text
        if self._include_llm_text and llm_text:
            lines.append("")
            lines.append(llm_text)

        # Include Grafana link if enabled
        if self._include_grafana_link and self._grafana_base_url:
            grafana_url = f"{self._grafana_base_url}/d/cryptoscreener-overview"
            lines.append("")
            lines.append(f"Dashboard: {grafana_url}")

        text = "\n".join(lines)

        # Build HTML version
        html_lines = [
            f"{icon} <b>{self._escape_html(event.symbol)}</b> - {self._escape_html(label)}",
            f"<i>Time:</i> {ts_str}",
            f"<i>Rank:</i> #{event.rank + 1} | <i>Score:</i> {event.score:.4f}",
        ]

        if status:
            html_lines.append(f"<i>Status:</i> {self._escape_html(status)}")

        if p_inplay is not None:
            html_lines.append(f"<i>p_inplay:</i> {p_inplay:.2%}")

        if p_toxic is not None:
            html_lines.append(f"<i>p_toxic:</i> {p_toxic:.2%}")

        if self._include_llm_text and llm_text:
            html_lines.append("")
            html_lines.append(self._escape_html(llm_text))

        if self._include_grafana_link and self._grafana_base_url:
            grafana_url = f"{self._grafana_base_url}/d/cryptoscreener-overview"
            html_lines.append("")
            html_lines.append(f'<a href="{grafana_url}">Dashboard</a>')

        html = "\n".join(html_lines)

        # Build Markdown version
        md_lines = [
            f"{icon} *{self._escape_markdown(event.symbol)}* - {self._escape_markdown(label)}",
            f"_Time:_ {ts_str}",
            f"_Rank:_ #{event.rank + 1} | _Score:_ {event.score:.4f}",
        ]

        if status:
            md_lines.append(f"_Status:_ {self._escape_markdown(status)}")

        if p_inplay is not None:
            md_lines.append(f"_p_inplay:_ {p_inplay:.2%}")

        if p_toxic is not None:
            md_lines.append(f"_p_toxic:_ {p_toxic:.2%}")

        if self._include_llm_text and llm_text:
            md_lines.append("")
            md_lines.append(self._escape_markdown(llm_text))

        if self._include_grafana_link and self._grafana_base_url:
            grafana_url = f"{self._grafana_base_url}/d/cryptoscreener-overview"
            md_lines.append("")
            md_lines.append(f"<{grafana_url}|Dashboard>")

        markdown = "\n".join(md_lines)

        return FormattedMessage(text=text, html=html, markdown=markdown)

    def format_batch(self, events: list[RankEvent]) -> FormattedMessage:
        """Format multiple RankEvents into a single message."""
        if not events:
            return FormattedMessage(text="", html="", markdown="")

        if len(events) == 1:
            return self.format(events[0])

        # Batch header
        ts_dt = datetime.fromtimestamp(events[0].ts / 1000, tz=UTC)
        ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        text_parts = [f"=== {len(events)} Alerts ({ts_str}) ===", ""]
        html_parts = [f"<b>=== {len(events)} Alerts ({ts_str}) ===</b>", ""]
        md_parts = [f"*=== {len(events)} Alerts ({ts_str}) ===*", ""]

        for event in events:
            formatted = self.format(event)
            text_parts.append(formatted.text)
            text_parts.append("")
            html_parts.append(formatted.html)
            html_parts.append("")
            md_parts.append(formatted.markdown)
            md_parts.append("")

        return FormattedMessage(
            text="\n".join(text_parts).rstrip(),
            html="\n".join(html_parts).rstrip(),
            markdown="\n".join(md_parts).rstrip(),
        )

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape Markdown special characters for Slack mrkdwn."""
        # Slack mrkdwn uses limited set of special chars
        for char in ["*", "_", "`", "~"]:
            text = text.replace(char, f"\\{char}")
        return text
