"""Stream router for routing MarketEvents to FeatureEngine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cryptoscreener.contracts.events import MarketEvent
from cryptoscreener.stream_router.metrics import RouterMetrics

if TYPE_CHECKING:
    from cryptoscreener.features.engine import FeatureEngine

logger = logging.getLogger(__name__)


@dataclass
class StreamRouterConfig:
    """Configuration for StreamRouter."""

    # Staleness threshold in milliseconds
    stale_threshold_ms: int = 5000

    # Maximum latency before event is considered late
    late_threshold_ms: int = 1000

    # Whether to drop stale events
    drop_stale: bool = False

    # Whether to track per-symbol metrics
    track_per_symbol: bool = True

    # Symbol filter (None means accept all)
    symbol_filter: set[str] | None = None


EventCallback = Callable[[MarketEvent], None]


class StreamRouter:
    """
    Routes MarketEvents from connectors to FeatureEngine.

    Responsibilities:
    - Validate and filter incoming events
    - Detect stale/late events
    - Track routing metrics
    - Forward events to FeatureEngine

    Usage:
        router = StreamRouter(feature_engine)
        await router.route(event)  # Route single event
        # or
        router.on_event(callback)  # Register callback for routed events
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        config: StreamRouterConfig | None = None,
    ) -> None:
        """
        Initialize stream router.

        Args:
            feature_engine: FeatureEngine to route events to.
            config: Router configuration. Uses defaults if not provided.
        """
        self._engine = feature_engine
        self._config = config or StreamRouterConfig()
        self._metrics = RouterMetrics()
        self._callbacks: list[EventCallback] = []

        # Track last timestamp per symbol for late detection
        self._last_ts: dict[str, int] = {}

    @property
    def config(self) -> StreamRouterConfig:
        """Get router configuration."""
        return self._config

    @property
    def metrics(self) -> RouterMetrics:
        """Get router metrics."""
        return self._metrics

    @property
    def feature_engine(self) -> FeatureEngine:
        """Get the feature engine."""
        return self._engine

    def on_event(self, callback: EventCallback) -> None:
        """
        Register a callback for routed events.

        Callbacks are invoked after successful routing.

        Args:
            callback: Function to call with each routed event.
        """
        self._callbacks.append(callback)

    def _get_current_ts(self) -> int:
        """Get current timestamp in milliseconds."""
        import time

        return int(time.time() * 1000)

    def _check_symbol_filter(self, symbol: str) -> bool:
        """Check if symbol passes filter."""
        if self._config.symbol_filter is None:
            return True
        return symbol in self._config.symbol_filter

    def _check_stale(self, event: MarketEvent, current_ts: int) -> bool:
        """Check if event is stale."""
        age = current_ts - event.ts
        return age > self._config.stale_threshold_ms

    def _check_late(self, event: MarketEvent) -> bool:
        """Check if event is late (out of order for its symbol)."""
        symbol = event.symbol
        last_ts = self._last_ts.get(symbol, 0)
        return event.ts < last_ts

    def _compute_latency(self, event: MarketEvent) -> int:
        """Compute event latency (recv_ts - ts)."""
        return max(0, event.recv_ts - event.ts)

    async def route(self, event: MarketEvent) -> bool:
        """
        Route a single event to the feature engine.

        Args:
            event: MarketEvent to route.

        Returns:
            True if event was routed, False if dropped.
        """
        symbol = event.symbol
        current_ts = self._get_current_ts()
        latency_ms = self._compute_latency(event)

        # Check symbol filter
        if not self._check_symbol_filter(symbol):
            self._metrics.record_unknown_symbol(symbol)
            return False

        # Check staleness
        is_stale = self._check_stale(event, current_ts)
        if is_stale and self._config.drop_stale:
            self._metrics.record_event(symbol, latency_ms, stale=True, dropped=True)
            logger.debug("Dropped stale event for %s (age: %dms)", symbol, current_ts - event.ts)
            return False

        # Check late
        is_late = self._check_late(event)

        # Record metrics
        self._metrics.record_event(symbol, latency_ms, stale=is_stale, late=is_late)

        # Update last timestamp for this symbol
        if event.ts > self._last_ts.get(symbol, 0):
            self._last_ts[symbol] = event.ts

        # Route to feature engine
        await self._engine.process_event(event)

        # Invoke callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.exception("Callback error for %s: %s", symbol, e)

        return True

    async def route_batch(self, events: list[MarketEvent]) -> int:
        """
        Route a batch of events.

        Args:
            events: List of MarketEvents to route.

        Returns:
            Number of successfully routed events.
        """
        routed = 0
        for event in events:
            if await self.route(event):
                routed += 1
        return routed

    def set_symbol_filter(self, symbols: set[str] | None) -> None:
        """
        Set the symbol filter.

        Args:
            symbols: Set of symbols to accept, or None to accept all.
        """
        self._config.symbol_filter = symbols
        if symbols:
            logger.info("Symbol filter set: %d symbols", len(symbols))
        else:
            logger.info("Symbol filter cleared (accepting all)")

    def add_symbols(self, symbols: set[str]) -> None:
        """
        Add symbols to the filter.

        Args:
            symbols: Symbols to add.
        """
        if self._config.symbol_filter is None:
            self._config.symbol_filter = set()
        self._config.symbol_filter.update(symbols)

    def remove_symbols(self, symbols: set[str]) -> None:
        """
        Remove symbols from the filter.

        Args:
            symbols: Symbols to remove.
        """
        if self._config.symbol_filter is not None:
            self._config.symbol_filter -= symbols

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics.reset()

    def reset_timestamps(self) -> None:
        """Reset timestamp tracking (for late detection)."""
        self._last_ts.clear()

    def get_symbol_stats(self, symbol: str) -> dict[str, int]:
        """
        Get statistics for a specific symbol.

        Args:
            symbol: Symbol to get stats for.

        Returns:
            Dictionary with event count and last timestamp.
        """
        return {
            "event_count": self._metrics.events_per_symbol.get(symbol, 0),
            "last_ts": self._last_ts.get(symbol, 0),
        }
