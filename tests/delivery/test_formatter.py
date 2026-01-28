"""
Tests for RankEvent formatter (DEC-039).
"""

from __future__ import annotations

from cryptoscreener.contracts import RankEvent, RankEventPayload, RankEventType
from cryptoscreener.delivery.formatter import RankEventFormatter


class TestRankEventFormatter:
    """Tests for deterministic RankEvent formatting."""

    def test_format_alert_tradable_basic(self) -> None:
        """Test formatting ALERT_TRADABLE event with minimal payload."""
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.ALERT_TRADABLE,
            symbol="BTCUSDT",
            rank=0,
            score=0.85,
            payload=RankEventPayload(prediction={}, llm_text=""),
        )

        formatter = RankEventFormatter()
        message = formatter.format(event)

        assert "BTCUSDT" in message.text
        assert "Tradeable" in message.text
        assert "#1" in message.text  # rank 0 -> #1
        assert "0.8500" in message.text  # score

    def test_format_with_prediction_payload(self) -> None:
        """Test formatting with full prediction payload."""
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.ALERT_TRADABLE,
            symbol="ETHUSDT",
            rank=2,
            score=0.75,
            payload=RankEventPayload(
                prediction={
                    "status": "TRADEABLE",
                    "p_inplay_2m": 0.83,
                    "p_toxic": 0.15,
                },
                llm_text="",
            ),
        )

        formatter = RankEventFormatter()
        message = formatter.format(event)

        assert "ETHUSDT" in message.text
        assert "Status: TRADEABLE" in message.text
        assert "p_inplay: 83" in message.text  # formatted as percentage
        assert "p_toxic: 15" in message.text

    def test_format_with_llm_text(self) -> None:
        """Test formatting includes LLM text when available."""
        llm_text = "BTC showing strong momentum with volume spike."
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.ALERT_TRADABLE,
            symbol="BTCUSDT",
            rank=0,
            score=0.9,
            payload=RankEventPayload(prediction={}, llm_text=llm_text),
        )

        formatter = RankEventFormatter(include_llm_text=True)
        message = formatter.format(event)

        assert llm_text in message.text

    def test_format_without_llm_text_when_disabled(self) -> None:
        """Test LLM text is excluded when include_llm_text=False."""
        llm_text = "Should not appear"
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.ALERT_TRADABLE,
            symbol="BTCUSDT",
            rank=0,
            score=0.9,
            payload=RankEventPayload(prediction={}, llm_text=llm_text),
        )

        formatter = RankEventFormatter(include_llm_text=False)
        message = formatter.format(event)

        assert llm_text not in message.text

    def test_format_with_grafana_link(self) -> None:
        """Test Grafana link is included when configured."""
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.ALERT_TRADABLE,
            symbol="BTCUSDT",
            rank=0,
            score=0.9,
            payload=RankEventPayload(prediction={}, llm_text=""),
        )

        formatter = RankEventFormatter(
            include_grafana_link=True,
            grafana_base_url="https://grafana.example.com",
        )
        message = formatter.format(event)

        assert "grafana.example.com" in message.text
        assert "Dashboard" in message.text

    def test_format_alert_trap(self) -> None:
        """Test formatting ALERT_TRAP event."""
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.ALERT_TRAP,
            symbol="XYZUSDT",
            rank=5,
            score=0.3,
            payload=RankEventPayload(prediction={"status": "TRAP"}, llm_text=""),
        )

        formatter = RankEventFormatter()
        message = formatter.format(event)

        assert "XYZUSDT" in message.text
        assert "Trap" in message.text or "Avoid" in message.text

    def test_format_symbol_enter(self) -> None:
        """Test formatting SYMBOL_ENTER event."""
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.SYMBOL_ENTER,
            symbol="SOLUSDT",
            rank=3,
            score=0.7,
            payload=RankEventPayload(prediction={}, llm_text=""),
        )

        formatter = RankEventFormatter()
        message = formatter.format(event)

        assert "SOLUSDT" in message.text
        assert "Entered" in message.text or "Ranking" in message.text

    def test_format_data_issue(self) -> None:
        """Test formatting DATA_ISSUE event."""
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.DATA_ISSUE,
            symbol="BADUSDT",
            rank=0,
            score=0.0,
            payload=RankEventPayload(prediction={"status": "DATA_ISSUE"}, llm_text=""),
        )

        formatter = RankEventFormatter()
        message = formatter.format(event)

        assert "BADUSDT" in message.text
        assert "Data Issue" in message.text or "DATA_ISSUE" in message.text

    def test_html_escaping(self) -> None:
        """Test HTML special characters are escaped."""
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.ALERT_TRADABLE,
            symbol="BTCUSDT",
            rank=0,
            score=0.9,
            payload=RankEventPayload(
                prediction={},
                llm_text="Test <script>alert('xss')</script> text",
            ),
        )

        formatter = RankEventFormatter()
        message = formatter.format(event)

        assert "<script>" not in message.html
        assert "&lt;script&gt;" in message.html

    def test_format_batch_empty(self) -> None:
        """Test formatting empty batch."""
        formatter = RankEventFormatter()
        message = formatter.format_batch([])

        assert message.text == ""
        assert message.html == ""
        assert message.markdown == ""

    def test_format_batch_single(self) -> None:
        """Test formatting single event batch delegates to format()."""
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.ALERT_TRADABLE,
            symbol="BTCUSDT",
            rank=0,
            score=0.9,
            payload=RankEventPayload(prediction={}, llm_text=""),
        )

        formatter = RankEventFormatter()
        single = formatter.format(event)
        batch = formatter.format_batch([event])

        assert single.text == batch.text

    def test_format_batch_multiple(self) -> None:
        """Test formatting multiple events in batch."""
        events = [
            RankEvent(
                ts=1706400000000,
                event=RankEventType.ALERT_TRADABLE,
                symbol="BTCUSDT",
                rank=0,
                score=0.9,
                payload=RankEventPayload(prediction={}, llm_text=""),
            ),
            RankEvent(
                ts=1706400001000,
                event=RankEventType.ALERT_TRADABLE,
                symbol="ETHUSDT",
                rank=1,
                score=0.8,
                payload=RankEventPayload(prediction={}, llm_text=""),
            ),
        ]

        formatter = RankEventFormatter()
        message = formatter.format_batch(events)

        assert "2 Alerts" in message.text
        assert "BTCUSDT" in message.text
        assert "ETHUSDT" in message.text

    def test_deterministic_output(self) -> None:
        """Test formatter produces identical output for same input."""
        event = RankEvent(
            ts=1706400000000,
            event=RankEventType.ALERT_TRADABLE,
            symbol="BTCUSDT",
            rank=0,
            score=0.85123456,
            payload=RankEventPayload(
                prediction={"p_inplay_2m": 0.75, "p_toxic": 0.2},
                llm_text="Test explanation",
            ),
        )

        formatter = RankEventFormatter()
        output1 = formatter.format(event)
        output2 = formatter.format(event)

        assert output1.text == output2.text
        assert output1.html == output2.html
        assert output1.markdown == output2.markdown
