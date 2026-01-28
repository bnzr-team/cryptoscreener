"""
Delivery deduplication and anti-spam (DEC-039).

Implements:
- Per-symbol cooldown
- Global rate limiting
- Deduplication by stable key
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptoscreener.contracts import RankEvent
    from cryptoscreener.delivery.config import DedupeConfig


@dataclass
class DedupeMetrics:
    """Metrics for deduplication decisions."""

    total_received: int = 0
    total_passed: int = 0
    suppressed_cooldown: int = 0
    suppressed_rate_limit: int = 0
    suppressed_duplicate: int = 0


@dataclass
class SymbolDedupeState:
    """Per-symbol deduplication state."""

    # Last delivery timestamp per event type
    last_delivery_ts: dict[str, float] = field(default_factory=dict)

    # Last status (for transition-only mode)
    last_status: str | None = None


class DeliveryDeduplicator:
    """
    Deduplication and rate limiting for RankEvent delivery.

    Anti-spam controls:
    1. Per-symbol cooldown: don't send same symbol+event_type within cooldown_s
    2. Global rate limit: max N deliveries per minute across all symbols
    3. Deduplication: skip if same stable key (symbol + event_type + status)

    Uses existing RankEvent fields only - no new fields invented.
    """

    def __init__(self, config: DedupeConfig) -> None:
        self._config = config
        self._symbol_state: dict[str, SymbolDedupeState] = {}
        self._global_timestamps: deque[float] = deque()  # Recent delivery timestamps
        self._metrics = DedupeMetrics()

    @property
    def metrics(self) -> DedupeMetrics:
        """Get current deduplication metrics."""
        return self._metrics

    def should_deliver(self, event: RankEvent) -> bool:
        """
        Check if event should be delivered.

        Returns True if event passes all anti-spam gates.
        """
        self._metrics.total_received += 1
        now = time.time()

        # Get or create symbol state
        if event.symbol not in self._symbol_state:
            self._symbol_state[event.symbol] = SymbolDedupeState()
        state = self._symbol_state[event.symbol]

        event_type = event.event.value

        # Check 1: Global rate limit
        if not self._check_global_rate_limit(now):
            self._metrics.suppressed_rate_limit += 1
            return False

        # Check 2: Per-symbol cooldown
        if not self._check_symbol_cooldown(state, event_type, now):
            self._metrics.suppressed_cooldown += 1
            return False

        # Check 3: Status transition (if enabled)
        if self._config.status_transition_only:
            prediction = event.payload.prediction or {}
            current_status = prediction.get("status", "")
            if current_status and state.last_status == current_status:
                # Same status, not a transition
                self._metrics.suppressed_duplicate += 1
                return False
            # Update last status
            if current_status:
                state.last_status = current_status

        # All checks passed - record delivery
        self._record_delivery(state, event_type, now)
        self._metrics.total_passed += 1
        return True

    def filter_batch(self, events: list[RankEvent]) -> list[RankEvent]:
        """
        Filter a batch of events, returning only those that should be delivered.
        """
        return [e for e in events if self.should_deliver(e)]

    def reset(self) -> None:
        """Reset all state (for testing)."""
        self._symbol_state.clear()
        self._global_timestamps.clear()
        self._metrics = DedupeMetrics()

    def _check_global_rate_limit(self, now: float) -> bool:
        """Check if within global rate limit."""
        # Remove timestamps older than 60 seconds
        cutoff = now - 60.0
        while self._global_timestamps and self._global_timestamps[0] < cutoff:
            self._global_timestamps.popleft()

        return len(self._global_timestamps) < self._config.global_max_per_minute

    def _check_symbol_cooldown(
        self, state: SymbolDedupeState, event_type: str, now: float
    ) -> bool:
        """Check if symbol+event_type is within cooldown."""
        last_ts = state.last_delivery_ts.get(event_type)
        if last_ts is None:
            return True

        elapsed = now - last_ts
        return elapsed >= self._config.per_symbol_cooldown_s

    def _record_delivery(
        self, state: SymbolDedupeState, event_type: str, now: float
    ) -> None:
        """Record a successful delivery."""
        state.last_delivery_ts[event_type] = now
        self._global_timestamps.append(now)

    def get_stable_key(self, event: RankEvent) -> str:
        """
        Generate stable deduplication key for an event.

        Uses only existing RankEvent fields:
        - symbol
        - event type
        - status (from prediction payload)
        - rank bucket (0-4, 5-9, 10+)
        """
        prediction = event.payload.prediction or {}
        status = prediction.get("status", "UNKNOWN")

        # Bucket rank into groups
        if event.rank < 5:
            rank_bucket = "top5"
        elif event.rank < 10:
            rank_bucket = "top10"
        else:
            rank_bucket = "other"

        return f"{event.symbol}:{event.event.value}:{status}:{rank_bucket}"
