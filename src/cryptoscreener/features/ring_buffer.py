"""Ring buffer implementation for time-based rolling windows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class TimestampedValue(Generic[T]):
    """A value with an associated timestamp."""

    ts: int  # milliseconds
    value: T


@dataclass
class RingBuffer(Generic[T]):
    """
    Time-based ring buffer for rolling window aggregations.

    Stores values with timestamps and automatically evicts values
    older than the specified window duration.

    Attributes:
        window_ms: Window duration in milliseconds.
        max_size: Maximum number of elements (prevents unbounded growth).
    """

    window_ms: int
    max_size: int = 10000
    _data: deque[TimestampedValue[T]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        """Initialize the internal deque with max size."""
        self._data = deque(maxlen=self.max_size)

    def push(self, ts: int, value: T) -> None:
        """
        Add a value with timestamp to the buffer.

        Args:
            ts: Timestamp in milliseconds.
            value: The value to store.
        """
        self._data.append(TimestampedValue(ts=ts, value=value))
        self._evict(ts)

    def _evict(self, current_ts: int) -> None:
        """Remove values older than window_ms from current timestamp."""
        cutoff = current_ts - self.window_ms
        while self._data and self._data[0].ts <= cutoff:
            self._data.popleft()

    def get_window(self, current_ts: int) -> list[TimestampedValue[T]]:
        """
        Get all values within the window relative to current timestamp.

        Args:
            current_ts: Current timestamp in milliseconds.

        Returns:
            List of TimestampedValue objects within the window.
        """
        self._evict(current_ts)
        return list(self._data)

    def get_values(self, current_ts: int) -> list[T]:
        """
        Get just the values (without timestamps) within the window.

        Args:
            current_ts: Current timestamp in milliseconds.

        Returns:
            List of values within the window.
        """
        return [item.value for item in self.get_window(current_ts)]

    def __len__(self) -> int:
        """Return current number of elements in buffer."""
        return len(self._data)

    def clear(self) -> None:
        """Clear all values from the buffer."""
        self._data.clear()

    @property
    def empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._data) == 0

    def oldest_ts(self) -> int | None:
        """Get timestamp of oldest value, or None if empty."""
        if self._data:
            return self._data[0].ts
        return None

    def newest_ts(self) -> int | None:
        """Get timestamp of newest value, or None if empty."""
        if self._data:
            return self._data[-1].ts
        return None
