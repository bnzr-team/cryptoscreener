"""Tests for RouterMetrics."""

from cryptoscreener.stream_router.metrics import RouterMetrics


class TestRouterMetrics:
    """Tests for RouterMetrics."""

    def test_initial_state(self) -> None:
        """Initial state has zero counts."""
        metrics = RouterMetrics()

        assert metrics.events_received == 0
        assert metrics.events_routed == 0
        assert metrics.events_dropped == 0
        assert metrics.stale_events == 0
        assert metrics.late_events == 0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.drop_rate == 0.0

    def test_record_event_basic(self) -> None:
        """Record basic event updates counts."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=50)

        assert metrics.events_received == 1
        assert metrics.events_routed == 1
        assert metrics.events_dropped == 0
        assert metrics.events_per_symbol["BTCUSDT"] == 1

    def test_record_event_dropped(self) -> None:
        """Dropped events increment dropped count."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=50, dropped=True)

        assert metrics.events_received == 1
        assert metrics.events_routed == 0
        assert metrics.events_dropped == 1

    def test_record_event_stale(self) -> None:
        """Stale events increment stale count."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=50, stale=True)

        assert metrics.stale_events == 1
        assert metrics.events_routed == 1  # Still routed

    def test_record_event_late(self) -> None:
        """Late events increment late count."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=50, late=True)

        assert metrics.late_events == 1
        assert metrics.events_routed == 1  # Still routed

    def test_latency_tracking(self) -> None:
        """Latency is tracked correctly."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=100)
        metrics.record_event("BTCUSDT", latency_ms=200)
        metrics.record_event("BTCUSDT", latency_ms=300)

        assert metrics.latency_samples == 3
        assert metrics.total_latency_ms == 600
        assert metrics.avg_latency_ms == 200.0
        assert metrics.max_latency_ms == 300

    def test_multiple_symbols(self) -> None:
        """Per-symbol tracking works."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=50)
        metrics.record_event("BTCUSDT", latency_ms=50)
        metrics.record_event("ETHUSDT", latency_ms=50)

        assert metrics.events_per_symbol["BTCUSDT"] == 2
        assert metrics.events_per_symbol["ETHUSDT"] == 1
        assert metrics.events_received == 3

    def test_record_unknown_symbol(self) -> None:
        """Unknown symbols are tracked and dropped."""
        metrics = RouterMetrics()

        metrics.record_unknown_symbol("UNKNOWN")

        assert metrics.unknown_symbols == 1
        assert metrics.events_dropped == 1
        assert metrics.events_received == 1

    def test_drop_rate(self) -> None:
        """Drop rate calculation."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=50)  # routed
        metrics.record_event("BTCUSDT", latency_ms=50)  # routed
        metrics.record_event("BTCUSDT", latency_ms=50, dropped=True)  # dropped
        metrics.record_event("BTCUSDT", latency_ms=50, dropped=True)  # dropped

        assert metrics.drop_rate == 0.5

    def test_stale_rate(self) -> None:
        """Stale rate calculation."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=50)  # fresh
        metrics.record_event("BTCUSDT", latency_ms=50, stale=True)  # stale
        metrics.record_event("BTCUSDT", latency_ms=50, stale=True)  # stale
        metrics.record_event("BTCUSDT", latency_ms=50)  # fresh

        assert metrics.stale_rate == 0.5

    def test_reset(self) -> None:
        """Reset clears all metrics."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=100)
        metrics.record_event("ETHUSDT", latency_ms=200, stale=True)

        metrics.reset()

        assert metrics.events_received == 0
        assert metrics.events_routed == 0
        assert metrics.stale_events == 0
        assert metrics.avg_latency_ms == 0.0
        assert len(metrics.events_per_symbol) == 0

    def test_to_dict(self) -> None:
        """Conversion to dictionary."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=100)
        metrics.record_event("ETHUSDT", latency_ms=200)

        d = metrics.to_dict()

        assert d["events_received"] == 2
        assert d["events_routed"] == 2
        assert d["avg_latency_ms"] == 150.0
        assert d["symbol_count"] == 2

    def test_merge(self) -> None:
        """Merge two metrics objects."""
        m1 = RouterMetrics()
        m1.record_event("BTCUSDT", latency_ms=100)
        m1.record_event("BTCUSDT", latency_ms=200)

        m2 = RouterMetrics()
        m2.record_event("ETHUSDT", latency_ms=50)
        m2.record_event("BTCUSDT", latency_ms=300)

        m1.merge(m2)

        assert m1.events_received == 4
        assert m1.events_routed == 4
        assert m1.events_per_symbol["BTCUSDT"] == 3
        assert m1.events_per_symbol["ETHUSDT"] == 1
        assert m1.max_latency_ms == 300

    def test_zero_latency_ignored(self) -> None:
        """Zero latency doesn't update max."""
        metrics = RouterMetrics()

        metrics.record_event("BTCUSDT", latency_ms=0)

        assert metrics.latency_samples == 0
        assert metrics.max_latency_ms == 0
