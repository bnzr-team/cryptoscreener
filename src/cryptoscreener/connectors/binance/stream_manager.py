"""
Stream manager for Binance WebSocket sharding.

Per BINANCE_LIMITS.md:
- Shard symbols across multiple WS connections
- Keep subscription count under 70-80% of max
- Batch subscribe messages and throttle operations
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from cryptoscreener.connectors.backoff import CircuitBreaker, ReconnectLimiter, RestGovernor
from cryptoscreener.connectors.binance.rest_client import BinanceRestClient
from cryptoscreener.connectors.binance.shard import WebSocketShard
from cryptoscreener.connectors.binance.types import (
    ConnectionState,
    ConnectorConfig,
    ConnectorMetrics,
    RawMessage,
    ShardMetrics,
    StreamSubscription,
    StreamType,
    SymbolInfo,
)
from cryptoscreener.contracts.events import MarketEvent, MarketEventType

logger = logging.getLogger(__name__)

# Type alias for event callback
EventCallback = Callable[[MarketEvent], Coroutine[Any, Any, None]]


class BinanceStreamManager:
    """
    Manages multiple WebSocket shards for Binance market data.

    Responsibilities:
    - Bootstrap with exchangeInfo
    - Create and manage shards
    - Distribute subscriptions across shards
    - Convert raw messages to MarketEvents
    - Provide async iterator interface for events
    """

    def __init__(
        self,
        config: ConnectorConfig | None = None,
        on_event: EventCallback | None = None,
        *,
        reconnect_limiter: ReconnectLimiter | None = None,
        time_fn: Callable[[], int] | None = None,
    ) -> None:
        """
        Initialize the stream manager.

        Args:
            config: Connector configuration.
            on_event: Optional callback for MarketEvents.
            reconnect_limiter: Global reconnect limiter (DEC-023b).
            time_fn: Time provider for deterministic testing (DEC-023b).
        """
        self._config = config or ConnectorConfig()
        self._on_event = on_event

        self._circuit_breaker = CircuitBreaker()
        self._governor: RestGovernor | None = None  # DEC-026: stored for metrics access
        self._rest_client = BinanceRestClient(
            config=self._config,
            circuit_breaker=self._circuit_breaker,
            governor=self._governor,
        )

        # DEC-023b: Global reconnect limiter shared across all shards
        if reconnect_limiter is not None:
            self._reconnect_limiter = reconnect_limiter
        else:
            self._reconnect_limiter = ReconnectLimiter(_time_fn=time_fn)

        # DEC-023b: Time provider for deterministic testing
        self._time_fn = time_fn

        self._shards: dict[int, WebSocketShard] = {}
        self._next_shard_id = 0
        self._subscriptions: dict[str, StreamSubscription] = {}
        self._stream_to_shard: dict[str, int] = {}

        self._symbols: dict[str, SymbolInfo] = {}
        self._running = False

        self._event_queue: asyncio.Queue[MarketEvent] = asyncio.Queue()

    async def start(self) -> None:
        """Start the stream manager."""
        if self._running:
            return

        logger.info("Starting BinanceStreamManager")
        self._running = True

    async def stop(self) -> None:
        """Stop all shards and cleanup."""
        if not self._running:
            return

        logger.info("Stopping BinanceStreamManager")
        self._running = False

        # Disconnect all shards
        for shard in self._shards.values():
            await shard.disconnect()
        self._shards.clear()

        # Close REST client
        await self._rest_client.close()

        logger.info("BinanceStreamManager stopped")

    async def bootstrap(self) -> list[SymbolInfo]:
        """
        Bootstrap by fetching tradeable symbols from Binance.

        Returns:
            List of tradeable SymbolInfo (unsorted).
        """
        logger.info("Bootstrapping from Binance exchangeInfo")

        symbols = await self._rest_client.get_tradeable_symbols()
        self._symbols = {s.symbol: s for s in symbols}

        logger.info(
            "Bootstrap complete",
            extra={"symbols": len(self._symbols)},
        )

        return list(self._symbols.values())

    async def get_top_symbols_by_volume(self, top_n: int) -> list[str]:
        """
        Get top N symbols sorted by 24h quote volume.

        This makes two REST calls:
        1. exchangeInfo to get tradeable symbols
        2. 24hr tickers to get volumes

        Args:
            top_n: Number of top symbols to return.

        Returns:
            List of symbol names sorted by 24h volume (descending).
        """
        # Ensure we have symbol info
        if not self._symbols:
            await self.bootstrap()

        # Fetch 24h volumes
        volumes = await self._rest_client.get_24h_tickers()

        # Filter to tradeable symbols and sort by volume
        tradeable_volumes = [(symbol, volumes.get(symbol, 0.0)) for symbol in self._symbols]
        tradeable_volumes.sort(key=lambda x: x[1], reverse=True)

        top_symbols = [symbol for symbol, _ in tradeable_volumes[:top_n]]

        logger.info(
            "Top symbols by volume",
            extra={"top_n": top_n, "top_3": top_symbols[:3]},
        )

        return top_symbols

    def get_symbol_info(self, symbol: str) -> SymbolInfo | None:
        """Get symbol info by symbol name."""
        return self._symbols.get(symbol.upper())

    async def subscribe(
        self,
        symbols: list[str],
        stream_types: list[StreamType] | None = None,
    ) -> int:
        """
        Subscribe to market data for symbols.

        Args:
            symbols: List of symbol names.
            stream_types: Types of streams to subscribe (default: TRADE, BOOK_TICKER).

        Returns:
            Number of subscriptions added.
        """
        if stream_types is None:
            stream_types = [StreamType.TRADE, StreamType.BOOK_TICKER]

        # Build subscriptions
        subs_to_add: list[StreamSubscription] = []
        for symbol in symbols:
            symbol_upper = symbol.upper()
            for stream_type in stream_types:
                sub = StreamSubscription(
                    symbol=symbol_upper,
                    stream_type=stream_type,
                )
                stream_name = sub.to_stream_name()
                if stream_name not in self._subscriptions:
                    subs_to_add.append(sub)

        if not subs_to_add:
            return 0

        # Distribute across shards
        added = 0
        for sub in subs_to_add:
            shard = await self._get_or_create_shard_for_subscription()
            if shard is None:
                logger.warning("Cannot create more shards, dropping subscription")
                break

            result = await shard.add_subscriptions([sub])
            if result:
                stream_name = sub.to_stream_name()
                self._subscriptions[stream_name] = sub
                self._stream_to_shard[stream_name] = shard.shard_id
                added += 1

        logger.info(
            "Subscriptions added",
            extra={"added": added, "total": len(self._subscriptions)},
        )

        return added

    async def unsubscribe(
        self,
        symbols: list[str],
        stream_types: list[StreamType] | None = None,
    ) -> int:
        """
        Unsubscribe from market data for symbols.

        Args:
            symbols: List of symbol names.
            stream_types: Types of streams to unsubscribe (default: all types).

        Returns:
            Number of subscriptions removed.
        """
        if stream_types is None:
            stream_types = list(StreamType)

        removed = 0
        for symbol in symbols:
            symbol_upper = symbol.upper()
            for stream_type in stream_types:
                sub = StreamSubscription(
                    symbol=symbol_upper,
                    stream_type=stream_type,
                )
                stream_name = sub.to_stream_name()

                if stream_name not in self._subscriptions:
                    continue

                shard_id = self._stream_to_shard.get(stream_name)
                if shard_id is not None and shard_id in self._shards:
                    shard = self._shards[shard_id]
                    await shard.remove_subscriptions([sub])

                del self._subscriptions[stream_name]
                self._stream_to_shard.pop(stream_name, None)
                removed += 1

        logger.info(
            "Subscriptions removed",
            extra={"removed": removed, "total": len(self._subscriptions)},
        )

        return removed

    async def _get_or_create_shard_for_subscription(self) -> WebSocketShard | None:
        """
        Get existing shard with capacity or create a new one.

        DEC-023b: Passes global ReconnectLimiter to new shards.

        Returns:
            WebSocketShard if available, None if max shards reached.
        """
        # Find shard with capacity
        for shard in self._shards.values():
            if shard.can_add_streams:
                return shard

        # Create new shard if under limit
        if len(self._shards) >= self._config.max_shards:
            return None

        shard_id = self._next_shard_id
        self._next_shard_id += 1

        # DEC-023b: Pass global reconnect_limiter to shard
        # MessageThrottler is per-shard so shard creates its own
        shard = WebSocketShard(
            shard_id=shard_id,
            config=self._config,
            circuit_breaker=self._circuit_breaker,
            on_message=self._handle_raw_message,
            on_state_change=self._handle_shard_state_change,
            reconnect_limiter=self._reconnect_limiter,
            time_fn=self._time_fn,
        )

        self._shards[shard_id] = shard

        # Connect the shard
        try:
            await shard.connect()
        except Exception as e:
            logger.error(
                "Failed to connect new shard",
                extra={"shard_id": shard_id, "error": str(e)},
            )
            del self._shards[shard_id]
            return None

        return shard

    async def _handle_raw_message(self, raw: RawMessage) -> None:
        """
        Convert raw WebSocket message to MarketEvent.

        Args:
            raw: Raw message from WebSocket shard.
        """
        try:
            event = self._parse_raw_message(raw)
            if event:
                # Put in queue for async iteration
                await self._event_queue.put(event)

                # Call callback if provided
                if self._on_event:
                    await self._on_event(event)

        except Exception as e:
            logger.warning(
                "Failed to parse message",
                extra={"shard_id": raw.shard_id, "error": str(e)},
            )

    def _parse_raw_message(self, raw: RawMessage) -> MarketEvent | None:
        """
        Parse raw WebSocket data into MarketEvent.

        Args:
            raw: Raw message data.

        Returns:
            MarketEvent if parseable, None otherwise.
        """
        data = raw.data

        # Determine event type from data structure
        event_type: MarketEventType | None = None
        symbol: str | None = None
        ts: int = 0

        # aggTrade: {"e":"aggTrade","E":ts,"s":"BTCUSDT",...}
        if data.get("e") == "aggTrade":
            event_type = MarketEventType.TRADE
            symbol = data.get("s")
            ts = data.get("E", 0)

        # bookTicker: {"e":"bookTicker","u":id,"s":"BTCUSDT",...}
        elif data.get("e") == "bookTicker":
            event_type = MarketEventType.BOOK
            symbol = data.get("s")
            ts = data.get("E", int(time.time() * 1000))

        # depth: {"e":"depthUpdate",...}
        elif data.get("e") == "depthUpdate":
            event_type = MarketEventType.BOOK
            symbol = data.get("s")
            ts = data.get("E", 0)

        # kline: {"e":"kline","E":ts,"s":"BTCUSDT","k":{...}}
        elif data.get("e") == "kline":
            event_type = MarketEventType.KLINE
            symbol = data.get("s")
            ts = data.get("E", 0)

        # markPrice: {"e":"markPriceUpdate",...}
        elif data.get("e") == "markPriceUpdate":
            event_type = MarketEventType.MARK
            symbol = data.get("s")
            ts = data.get("E", 0)

        # forceOrder: {"e":"forceOrder",...}
        elif data.get("e") == "forceOrder":
            event_type = MarketEventType.FUNDING
            symbol = data.get("o", {}).get("s")
            ts = data.get("E", 0)

        if event_type is None or symbol is None:
            return None

        return MarketEvent(
            ts=ts,
            source="binance_usdm",
            symbol=symbol,
            type=event_type,
            payload=data,
            recv_ts=raw.recv_ts,
        )

    def _handle_shard_state_change(
        self,
        shard_id: int,
        state: ConnectionState,
    ) -> None:
        """Handle shard state change notification."""
        logger.debug(
            "Shard state changed",
            extra={"shard_id": shard_id, "state": state.value},
        )

    async def events(self) -> AsyncIterator[MarketEvent]:
        """
        Async iterator for MarketEvents.

        Yields:
            MarketEvent as they are received.
        """
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                yield event
            except TimeoutError:
                continue

    def get_metrics(self) -> ConnectorMetrics:
        """
        Get current connector metrics.

        DEC-023b: Includes aggregated throttle/limiter metrics.

        Returns:
            ConnectorMetrics with aggregated stats.
        """
        shard_metrics: list[ShardMetrics] = []
        total_messages = 0
        total_streams = 0
        active_shards = 0
        total_reconnects_denied = 0
        total_messages_delayed = 0
        # DEC-025: Aggregate new metrics
        total_disconnects = 0
        total_reconnect_attempts = 0

        for shard in self._shards.values():
            metrics = shard.get_metrics()
            shard_metrics.append(metrics)
            total_messages += metrics.messages_received
            total_streams += metrics.stream_count
            # DEC-023b: Aggregate throttle metrics
            total_reconnects_denied += metrics.reconnect_denied
            total_messages_delayed += metrics.messages_delayed
            # DEC-025: Aggregate WS storm metrics
            total_disconnects += metrics.total_disconnects
            total_reconnect_attempts += metrics.reconnect_attempts
            if metrics.state == ConnectionState.CONNECTED:
                active_shards += 1

        return ConnectorMetrics(
            total_streams=total_streams,
            total_messages=total_messages,
            active_shards=active_shards,
            shard_metrics=shard_metrics,
            circuit_breaker_open=not self._circuit_breaker.can_execute(),
            # DEC-023b: Limiter/throttler metrics
            reconnect_limiter_in_cooldown=bool(self._reconnect_limiter.get_status()["in_cooldown"]),
            total_reconnects_denied=total_reconnects_denied,
            total_messages_delayed=total_messages_delayed,
            # DEC-025: WS storm alert metrics
            total_disconnects=total_disconnects,
            total_reconnect_attempts=total_reconnect_attempts,
        )

    async def force_disconnect(self) -> None:
        """
        DEC-027: Force-close all shard WS connections (simulates network drop).

        Closes the raw WebSocket without going through graceful disconnect(),
        so the shard's receive loop detects the close and triggers auto-reconnect
        via its existing backoff/limiter path.
        """
        for shard in self._shards.values():
            if shard._ws and not shard._ws.closed:
                await shard._ws.close()

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Read-only access to circuit breaker (DEC-026: metrics wiring)."""
        return self._circuit_breaker

    @property
    def governor(self) -> RestGovernor | None:
        """Read-only access to REST governor, if configured (DEC-026: metrics wiring)."""
        return self._governor
