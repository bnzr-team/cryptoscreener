"""
Types and configuration for Binance USD-M Futures WebSocket connector.

Per BINANCE_LIMITS.md:
- Max 1024 streams per connection (we use 800 for headroom)
- Max 10 incoming messages per second per connection
- Use combined streams with sharding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StreamType(str, Enum):
    """Binance WebSocket stream types."""

    TRADE = "trade"  # aggTrade stream
    BOOK_TICKER = "bookTicker"  # Best bid/ask
    DEPTH = "depth"  # Order book depth
    KLINE = "kline"  # Candlestick/kline
    MARK_PRICE = "markPrice"  # Mark price
    FORCE_ORDER = "forceOrder"  # Liquidation orders


class ConnectionState(str, Enum):
    """WebSocket connection state."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class StreamSubscription:
    """
    A stream subscription request.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        stream_type: Type of market data stream.
        interval: Interval for kline streams (e.g., "1m", "5m").
    """

    symbol: str
    stream_type: StreamType
    interval: str | None = None

    def to_stream_name(self) -> str:
        """
        Convert to Binance stream name format.

        Examples:
            - btcusdt@aggTrade
            - btcusdt@bookTicker
            - btcusdt@kline_1m
            - btcusdt@depth@100ms
        """
        symbol_lower = self.symbol.lower()

        if self.stream_type == StreamType.TRADE:
            return f"{symbol_lower}@aggTrade"
        if self.stream_type == StreamType.BOOK_TICKER:
            return f"{symbol_lower}@bookTicker"
        if self.stream_type == StreamType.DEPTH:
            return f"{symbol_lower}@depth@100ms"
        if self.stream_type == StreamType.KLINE:
            interval = self.interval or "1m"
            return f"{symbol_lower}@kline_{interval}"
        if self.stream_type == StreamType.MARK_PRICE:
            return f"{symbol_lower}@markPrice@1s"
        if self.stream_type == StreamType.FORCE_ORDER:
            return f"{symbol_lower}@forceOrder"

        return f"{symbol_lower}@{self.stream_type.value}"


@dataclass
class ShardConfig:
    """
    Configuration for a WebSocket shard (single connection).

    Per BINANCE_LIMITS.md:
    - Max 1024 streams per connection
    - We use 800 for headroom (~78% utilization)
    """

    max_streams: int = 800  # Headroom under 1024 limit
    max_messages_per_second: int = 10  # Per connection limit
    ping_interval_ms: int = 30000  # 30s ping interval
    ping_timeout_ms: int = 10000  # 10s ping timeout
    subscribe_batch_size: int = 100  # Batch subscribe operations


@dataclass
class ConnectorConfig:
    """
    Configuration for the Binance WebSocket connector.

    Attributes:
        base_ws_url: WebSocket base URL for USD-M Futures.
        base_rest_url: REST API base URL for bootstrap.
        shard_config: Configuration for each shard.
        max_shards: Maximum number of concurrent shards.
        enable_combined_streams: Use combined stream endpoint.
    """

    base_ws_url: str = "wss://fstream.binance.com"
    base_rest_url: str = "https://fapi.binance.com"
    shard_config: ShardConfig = field(default_factory=ShardConfig)
    max_shards: int = 10  # Max 10 concurrent connections
    enable_combined_streams: bool = True
    request_timeout_ms: int = 10000  # REST request timeout


@dataclass
class RawMessage:
    """
    Raw message received from WebSocket.

    Attributes:
        data: Parsed JSON data from message.
        recv_ts: Local receive timestamp (ms).
        shard_id: ID of the shard that received this message.
    """

    data: dict[str, Any]
    recv_ts: int
    shard_id: int


@dataclass
class ExchangeInfo:
    """
    Exchange information from REST bootstrap.

    Attributes:
        symbols: List of symbol metadata.
        server_time: Server timestamp (ms).
        rate_limits: Rate limit information.
    """

    symbols: list[dict[str, Any]]
    server_time: int
    rate_limits: list[dict[str, Any]]


@dataclass
class SymbolInfo:
    """
    Parsed symbol information.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        base_asset: Base asset (e.g., "BTC").
        quote_asset: Quote asset (e.g., "USDT").
        price_precision: Price decimal precision.
        quantity_precision: Quantity decimal precision.
        contract_type: Contract type (e.g., "PERPETUAL").
        status: Symbol trading status.
    """

    symbol: str
    base_asset: str
    quote_asset: str
    price_precision: int
    quantity_precision: int
    contract_type: str
    status: str

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> SymbolInfo:
        """Parse from raw exchangeInfo response."""
        return cls(
            symbol=data["symbol"],
            base_asset=data["baseAsset"],
            quote_asset=data["quoteAsset"],
            price_precision=data["pricePrecision"],
            quantity_precision=data["quantityPrecision"],
            contract_type=data.get("contractType", "PERPETUAL"),
            status=data["status"],
        )


@dataclass
class ShardMetrics:
    """
    Metrics for a single shard.

    Attributes:
        shard_id: Shard identifier.
        stream_count: Number of subscribed streams.
        messages_received: Total messages received.
        messages_per_second: Current message rate.
        last_message_ts: Timestamp of last message.
        reconnect_count: Number of reconnections.
        reconnect_denied: Number of reconnects blocked by limiter (DEC-023b).
        messages_delayed: Number of messages delayed by throttler (DEC-023b).
        state: Current connection state.
    """

    shard_id: int
    stream_count: int = 0
    messages_received: int = 0
    messages_per_second: float = 0.0
    last_message_ts: int = 0
    reconnect_count: int = 0
    reconnect_denied: int = 0  # DEC-023b: blocked by ReconnectLimiter
    messages_delayed: int = 0  # DEC-023b: delayed by MessageThrottler
    state: ConnectionState = ConnectionState.DISCONNECTED


@dataclass
class ConnectorMetrics:
    """
    Aggregated metrics for the connector.

    Attributes:
        total_streams: Total streams across all shards.
        total_messages: Total messages received.
        active_shards: Number of active shards.
        shard_metrics: Per-shard metrics.
        circuit_breaker_open: Whether circuit breaker is open.
        reconnect_limiter_in_cooldown: Whether global reconnect limiter is in cooldown (DEC-023b).
        total_reconnects_denied: Total reconnects denied across all shards (DEC-023b).
        total_messages_delayed: Total messages delayed across all shards (DEC-023b).
        last_error: Last error message if any.
    """

    total_streams: int = 0
    total_messages: int = 0
    active_shards: int = 0
    shard_metrics: list[ShardMetrics] = field(default_factory=list)
    circuit_breaker_open: bool = False
    reconnect_limiter_in_cooldown: bool = False  # DEC-023b
    total_reconnects_denied: int = 0  # DEC-023b
    total_messages_delayed: int = 0  # DEC-023b
    last_error: str | None = None
