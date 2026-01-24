"""
Binance USD-M Futures WebSocket connector.

Per BINANCE_LIMITS.md and CLAUDE.md requirements:
- WS market streams are the primary data source
- Sharding across multiple connections (≤800 streams each)
- Exponential backoff + jitter for reconnects
- Circuit breaker on repeated disconnects
- REST only for bootstrap (exchangeInfo)
"""

from cryptoscreener.connectors.binance.rest_client import BinanceRestClient
from cryptoscreener.connectors.binance.shard import WebSocketShard
from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager
from cryptoscreener.connectors.binance.types import (
    ConnectionState,
    ConnectorConfig,
    ConnectorMetrics,
    ExchangeInfo,
    RawMessage,
    ShardConfig,
    ShardMetrics,
    StreamSubscription,
    StreamType,
    SymbolInfo,
)

__all__ = [
    "BinanceRestClient",
    "BinanceStreamManager",
    "ConnectionState",
    "ConnectorConfig",
    "ConnectorMetrics",
    "ExchangeInfo",
    "RawMessage",
    "ShardConfig",
    "ShardMetrics",
    "StreamSubscription",
    "StreamType",
    "SymbolInfo",
    "WebSocketShard",
]
