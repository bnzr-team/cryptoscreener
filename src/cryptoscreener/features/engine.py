"""Feature engine for computing FeatureSnapshot from MarketEvents."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from cryptoscreener.contracts.events import FeatureSnapshot, MarketEvent
from cryptoscreener.features.symbol_state import SymbolState

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineConfig:
    """Configuration for FeatureEngine."""

    # Snapshot emission cadence in milliseconds
    snapshot_cadence_ms: int = 1000

    # Maximum symbols to track (prevents unbounded growth)
    max_symbols: int = 500

    # Staleness thresholds for data health
    stale_book_threshold_ms: int = 5000
    stale_trades_threshold_ms: int = 10000


SnapshotCallback = Callable[[FeatureSnapshot], Any]


class FeatureEngine:
    """
    Feature engine that processes MarketEvents and emits FeatureSnapshots.

    Responsibilities:
    - Maintain per-symbol state using ring buffers
    - Compute features at configurable cadence
    - Emit FeatureSnapshot via callback or async iterator

    Usage:
        engine = FeatureEngine()
        engine.on_snapshot(callback)  # Register callback
        await engine.process_event(event)  # Process incoming events

        # Or use as async iterator
        async for snapshot in engine.snapshots():
            process(snapshot)
    """

    def __init__(self, config: FeatureEngineConfig | None = None) -> None:
        """
        Initialize feature engine.

        Args:
            config: Engine configuration. Uses defaults if not provided.
        """
        self._config = config or FeatureEngineConfig()
        self._states: dict[str, SymbolState] = {}
        self._callbacks: list[SnapshotCallback] = []
        self._snapshot_queue: asyncio.Queue[FeatureSnapshot] = asyncio.Queue()
        self._running = False
        self._emission_task: asyncio.Task[None] | None = None
        self._last_emission_ts: dict[str, int] = {}

    @property
    def config(self) -> FeatureEngineConfig:
        """Get engine configuration."""
        return self._config

    @property
    def symbols(self) -> list[str]:
        """Get list of tracked symbols."""
        return list(self._states.keys())

    @property
    def running(self) -> bool:
        """Check if emission loop is running."""
        return self._running

    def on_snapshot(self, callback: SnapshotCallback) -> None:
        """
        Register a callback for snapshot emission.

        Args:
            callback: Function to call with each FeatureSnapshot.
        """
        self._callbacks.append(callback)

    def get_state(self, symbol: str) -> SymbolState | None:
        """
        Get state for a symbol.

        Args:
            symbol: Symbol to look up.

        Returns:
            SymbolState if exists, None otherwise.
        """
        return self._states.get(symbol.upper())

    def _get_or_create_state(self, symbol: str) -> SymbolState:
        """Get or create state for a symbol."""
        symbol = symbol.upper()
        if symbol not in self._states:
            if len(self._states) >= self._config.max_symbols:
                logger.warning(
                    "Max symbols reached (%d), ignoring new symbol: %s",
                    self._config.max_symbols,
                    symbol,
                )
                raise ValueError(f"Max symbols ({self._config.max_symbols}) reached")
            self._states[symbol] = SymbolState(symbol=symbol)
            self._last_emission_ts[symbol] = 0
        return self._states[symbol]

    async def process_event(self, event: MarketEvent) -> None:
        """
        Process an incoming market event.

        Updates the internal state for the event's symbol.

        Args:
            event: MarketEvent to process.
        """
        try:
            state = self._get_or_create_state(event.symbol)
            state.process_event(event)
        except ValueError:
            # Max symbols reached, ignore
            pass

    def compute_snapshot(self, symbol: str, ts: int) -> FeatureSnapshot | None:
        """
        Compute a FeatureSnapshot for a symbol at given timestamp.

        Args:
            symbol: Symbol to compute snapshot for.
            ts: Current timestamp in milliseconds.

        Returns:
            FeatureSnapshot if symbol exists, None otherwise.
        """
        state = self._states.get(symbol.upper())
        if not state:
            return None

        return FeatureSnapshot(
            ts=ts,
            symbol=symbol.upper(),
            features=state.compute_features(ts),
            windows=state.compute_windows(ts),
            data_health=state.compute_data_health(ts),
        )

    async def _emit_snapshot(self, snapshot: FeatureSnapshot) -> None:
        """Emit snapshot to callbacks and queue."""
        # Queue for async iteration
        try:
            self._snapshot_queue.put_nowait(snapshot)
        except asyncio.QueueFull:
            logger.warning("Snapshot queue full, dropping snapshot for %s", snapshot.symbol)

        # Callbacks
        for callback in self._callbacks:
            try:
                result = callback(snapshot)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.exception("Callback error for %s: %s", snapshot.symbol, e)

    async def emit_snapshots(self, current_ts: int) -> list[FeatureSnapshot]:
        """
        Emit snapshots for all symbols that are due.

        Args:
            current_ts: Current timestamp in milliseconds.

        Returns:
            List of emitted snapshots.
        """
        emitted: list[FeatureSnapshot] = []

        for symbol in list(self._states.keys()):
            last_ts = self._last_emission_ts.get(symbol, 0)

            if current_ts - last_ts >= self._config.snapshot_cadence_ms:
                snapshot = self.compute_snapshot(symbol, current_ts)
                if snapshot:
                    await self._emit_snapshot(snapshot)
                    emitted.append(snapshot)
                    self._last_emission_ts[symbol] = current_ts

        return emitted

    async def snapshots(self) -> AsyncIterator[FeatureSnapshot]:
        """
        Async iterator for emitted snapshots.

        Yields:
            FeatureSnapshot objects as they are emitted.
        """
        while True:
            snapshot = await self._snapshot_queue.get()
            yield snapshot

    async def start(self, get_current_ts: Callable[[], int] | None = None) -> None:
        """
        Start the emission loop.

        Args:
            get_current_ts: Function to get current timestamp.
                           Defaults to using last event timestamp.
        """
        if self._running:
            return

        self._running = True

        async def emission_loop() -> None:
            import time

            while self._running:
                ts = get_current_ts() if get_current_ts else int(time.time() * 1000)
                await self.emit_snapshots(ts)
                await asyncio.sleep(self._config.snapshot_cadence_ms / 1000)

        self._emission_task = asyncio.create_task(emission_loop())
        logger.info("Feature engine started with cadence %dms", self._config.snapshot_cadence_ms)

    async def stop(self) -> None:
        """Stop the emission loop."""
        import contextlib

        self._running = False
        if self._emission_task:
            self._emission_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._emission_task
            self._emission_task = None
        logger.info("Feature engine stopped")

    def clear(self) -> None:
        """Clear all state."""
        self._states.clear()
        self._last_emission_ts.clear()

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from tracking.

        Args:
            symbol: Symbol to remove.

        Returns:
            True if symbol was removed, False if not found.
        """
        symbol = symbol.upper()
        if symbol in self._states:
            del self._states[symbol]
            self._last_emission_ts.pop(symbol, None)
            return True
        return False
