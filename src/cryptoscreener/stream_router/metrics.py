"""Metrics tracking for stream router."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RouterMetrics:
    """
    Metrics for stream router operations.

    Tracks event flow, latency, and error conditions.
    """

    # Event counts
    events_received: int = 0
    events_routed: int = 0
    events_dropped: int = 0

    # Latency tracking (in milliseconds)
    total_latency_ms: int = 0
    max_latency_ms: int = 0
    latency_samples: int = 0

    # Error conditions
    stale_events: int = 0  # Events older than threshold
    late_events: int = 0  # Events arriving out of order
    unknown_symbols: int = 0  # Events for untracked symbols

    # Per-symbol counts
    events_per_symbol: dict[str, int] = field(default_factory=dict)

    def record_event(
        self,
        symbol: str,
        latency_ms: int,
        *,
        stale: bool = False,
        late: bool = False,
        dropped: bool = False,
    ) -> None:
        """
        Record metrics for a single event.

        Args:
            symbol: Symbol of the event.
            latency_ms: Event latency in milliseconds.
            stale: Whether event was stale (older than threshold).
            late: Whether event arrived out of order.
            dropped: Whether event was dropped.
        """
        self.events_received += 1

        if dropped:
            self.events_dropped += 1
        else:
            self.events_routed += 1

        # Latency tracking
        if latency_ms > 0:
            self.total_latency_ms += latency_ms
            self.latency_samples += 1
            if latency_ms > self.max_latency_ms:
                self.max_latency_ms = latency_ms

        # Error conditions
        if stale:
            self.stale_events += 1
        if late:
            self.late_events += 1

        # Per-symbol tracking
        self.events_per_symbol[symbol] = self.events_per_symbol.get(symbol, 0) + 1

    def record_unknown_symbol(self, symbol: str) -> None:
        """Record an event for an unknown/untracked symbol."""
        self.unknown_symbols += 1
        self.events_dropped += 1
        self.events_received += 1

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.latency_samples == 0:
            return 0.0
        return self.total_latency_ms / self.latency_samples

    @property
    def drop_rate(self) -> float:
        """Calculate event drop rate (0.0 to 1.0)."""
        if self.events_received == 0:
            return 0.0
        return self.events_dropped / self.events_received

    @property
    def stale_rate(self) -> float:
        """Calculate stale event rate (0.0 to 1.0)."""
        if self.events_received == 0:
            return 0.0
        return self.stale_events / self.events_received

    def reset(self) -> None:
        """Reset all metrics to initial values."""
        self.events_received = 0
        self.events_routed = 0
        self.events_dropped = 0
        self.total_latency_ms = 0
        self.max_latency_ms = 0
        self.latency_samples = 0
        self.stale_events = 0
        self.late_events = 0
        self.unknown_symbols = 0
        self.events_per_symbol.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/export."""
        return {
            "events_received": self.events_received,
            "events_routed": self.events_routed,
            "events_dropped": self.events_dropped,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": self.max_latency_ms,
            "stale_events": self.stale_events,
            "late_events": self.late_events,
            "unknown_symbols": self.unknown_symbols,
            "drop_rate": round(self.drop_rate, 4),
            "stale_rate": round(self.stale_rate, 4),
            "symbol_count": len(self.events_per_symbol),
        }

    def merge(self, other: RouterMetrics) -> RouterMetrics:
        """
        Merge another RouterMetrics into this one.

        Args:
            other: RouterMetrics to merge.

        Returns:
            Self for chaining.
        """
        self.events_received += other.events_received
        self.events_routed += other.events_routed
        self.events_dropped += other.events_dropped
        self.total_latency_ms += other.total_latency_ms
        self.latency_samples += other.latency_samples
        self.max_latency_ms = max(self.max_latency_ms, other.max_latency_ms)
        self.stale_events += other.stale_events
        self.late_events += other.late_events
        self.unknown_symbols += other.unknown_symbols

        for symbol, count in other.events_per_symbol.items():
            self.events_per_symbol[symbol] = self.events_per_symbol.get(symbol, 0) + count

        return self
