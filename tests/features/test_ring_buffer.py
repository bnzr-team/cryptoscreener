"""Tests for RingBuffer."""

from cryptoscreener.features.ring_buffer import RingBuffer, TimestampedValue


class TestRingBuffer:
    """Tests for RingBuffer."""

    def test_push_and_get_values(self) -> None:
        """Push values and retrieve them."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        buf.push(100, 1)
        buf.push(200, 2)
        buf.push(300, 3)

        values = buf.get_values(300)
        assert values == [1, 2, 3]

    def test_window_eviction(self) -> None:
        """Values older than window are evicted."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        buf.push(100, 1)
        buf.push(500, 2)
        buf.push(1100, 3)  # Now ts=100 is older than 1000ms

        values = buf.get_values(1100)
        assert values == [2, 3]
        assert 1 not in values

    def test_eviction_on_get(self) -> None:
        """Eviction happens when getting values with newer timestamp."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        buf.push(100, 1)
        buf.push(200, 2)
        buf.push(300, 3)

        # All values are in window at ts=300
        assert len(buf.get_values(300)) == 3

        # At ts=1200, ts=100 and ts=200 are out of window
        values = buf.get_values(1200)
        assert values == [3]

    def test_empty_buffer(self) -> None:
        """Empty buffer returns empty list."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        assert buf.get_values(1000) == []
        assert buf.empty
        assert len(buf) == 0

    def test_len(self) -> None:
        """Length reflects current buffer size."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        assert len(buf) == 0

        buf.push(100, 1)
        assert len(buf) == 1

        buf.push(200, 2)
        assert len(buf) == 2

        # Evict: cutoff = 1200 - 1000 = 200, both ts=100 and ts=200 evicted
        buf.push(1200, 3)
        assert len(buf) == 1  # ts=100 and ts=200 evicted

    def test_clear(self) -> None:
        """Clear removes all values."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        buf.push(100, 1)
        buf.push(200, 2)
        assert len(buf) == 2

        buf.clear()
        assert len(buf) == 0
        assert buf.empty

    def test_oldest_newest_ts(self) -> None:
        """Get oldest and newest timestamps."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        assert buf.oldest_ts() is None
        assert buf.newest_ts() is None

        buf.push(100, 1)
        buf.push(200, 2)
        buf.push(300, 3)

        assert buf.oldest_ts() == 100
        assert buf.newest_ts() == 300

    def test_max_size_limit(self) -> None:
        """Buffer respects max_size limit."""
        buf: RingBuffer[int] = RingBuffer(window_ms=100000, max_size=5)

        for i in range(10):
            buf.push(i * 100, i)

        # Only last 5 values kept
        assert len(buf) == 5
        values = buf.get_values(900)
        assert values == [5, 6, 7, 8, 9]

    def test_get_window(self) -> None:
        """Get window returns TimestampedValue objects."""
        buf: RingBuffer[str] = RingBuffer(window_ms=1000)

        buf.push(100, "a")
        buf.push(200, "b")

        window = buf.get_window(200)
        assert len(window) == 2
        assert all(isinstance(item, TimestampedValue) for item in window)
        assert window[0].ts == 100
        assert window[0].value == "a"
        assert window[1].ts == 200
        assert window[1].value == "b"

    def test_generic_types(self) -> None:
        """Buffer works with different types."""
        # Float buffer
        float_buf: RingBuffer[float] = RingBuffer(window_ms=1000)
        float_buf.push(100, 1.5)
        assert float_buf.get_values(100) == [1.5]

        # Dict buffer
        dict_buf: RingBuffer[dict[str, int]] = RingBuffer(window_ms=1000)
        dict_buf.push(100, {"x": 1})
        assert dict_buf.get_values(100) == [{"x": 1}]

    def test_boundary_conditions(self) -> None:
        """Test exact boundary of window."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        buf.push(0, 1)
        buf.push(500, 2)
        buf.push(999, 3)

        # At ts=1000, value at ts=0 is exactly 1000ms old (should be kept)
        values = buf.get_values(999)
        assert len(values) == 3

        # At ts=1001, value at ts=0 is 1001ms old (should be evicted)
        values = buf.get_values(1001)
        assert values == [2, 3]

    def test_push_order_preserved(self) -> None:
        """Values are returned in push order."""
        buf: RingBuffer[str] = RingBuffer(window_ms=10000)

        buf.push(100, "first")
        buf.push(200, "second")
        buf.push(300, "third")

        values = buf.get_values(300)
        assert values == ["first", "second", "third"]


class TestRingBufferEdgeCases:
    """Edge case tests for RingBuffer."""

    def test_zero_window(self) -> None:
        """Zero window means all values are immediately stale."""
        buf: RingBuffer[int] = RingBuffer(window_ms=0)

        buf.push(100, 1)
        # At same timestamp, cutoff = 100, ts=100 <= 100, so evicted
        assert buf.get_values(100) == []
        # At next timestamp, also evicted
        assert buf.get_values(101) == []

    def test_large_timestamp_gap(self) -> None:
        """Large gap evicts all previous values."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        buf.push(100, 1)
        buf.push(200, 2)

        # Jump far into future
        values = buf.get_values(1000000)
        assert values == []

    def test_same_timestamp_multiple_values(self) -> None:
        """Multiple values at same timestamp are all kept."""
        buf: RingBuffer[int] = RingBuffer(window_ms=1000)

        buf.push(100, 1)
        buf.push(100, 2)
        buf.push(100, 3)

        values = buf.get_values(100)
        assert values == [1, 2, 3]
