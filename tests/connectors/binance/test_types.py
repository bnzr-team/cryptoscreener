"""Tests for Binance connector types."""

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


class TestStreamSubscription:
    """Tests for StreamSubscription."""

    def test_trade_stream_name(self) -> None:
        """aggTrade stream format."""
        sub = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.TRADE)
        assert sub.to_stream_name() == "btcusdt@aggTrade"

    def test_book_ticker_stream_name(self) -> None:
        """bookTicker stream format."""
        sub = StreamSubscription(symbol="ETHUSDT", stream_type=StreamType.BOOK_TICKER)
        assert sub.to_stream_name() == "ethusdt@bookTicker"

    def test_depth_stream_name(self) -> None:
        """depth stream format."""
        sub = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.DEPTH)
        assert sub.to_stream_name() == "btcusdt@depth@100ms"

    def test_kline_stream_name_default_interval(self) -> None:
        """kline stream with default 1m interval."""
        sub = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.KLINE)
        assert sub.to_stream_name() == "btcusdt@kline_1m"

    def test_kline_stream_name_custom_interval(self) -> None:
        """kline stream with custom interval."""
        sub = StreamSubscription(
            symbol="BTCUSDT",
            stream_type=StreamType.KLINE,
            interval="5m",
        )
        assert sub.to_stream_name() == "btcusdt@kline_5m"

    def test_mark_price_stream_name(self) -> None:
        """markPrice stream format."""
        sub = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.MARK_PRICE)
        assert sub.to_stream_name() == "btcusdt@markPrice@1s"

    def test_force_order_stream_name(self) -> None:
        """forceOrder stream format."""
        sub = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.FORCE_ORDER)
        assert sub.to_stream_name() == "btcusdt@forceOrder"

    def test_symbol_lowercased(self) -> None:
        """Symbol is lowercased in stream name."""
        sub = StreamSubscription(symbol="BtCuSdT", stream_type=StreamType.TRADE)
        assert sub.to_stream_name() == "btcusdt@aggTrade"

    def test_subscription_hashable(self) -> None:
        """Subscriptions can be used in sets."""
        sub1 = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.TRADE)
        sub2 = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.TRADE)
        sub3 = StreamSubscription(symbol="ETHUSDT", stream_type=StreamType.TRADE)

        s = {sub1, sub2, sub3}
        assert len(s) == 2


class TestShardConfig:
    """Tests for ShardConfig."""

    def test_default_values(self) -> None:
        """Default configuration values."""
        config = ShardConfig()
        assert config.max_streams == 800
        assert config.max_messages_per_second == 10
        assert config.ping_interval_ms == 30000
        assert config.ping_timeout_ms == 10000
        assert config.subscribe_batch_size == 100


class TestConnectorConfig:
    """Tests for ConnectorConfig."""

    def test_default_urls(self) -> None:
        """Default Binance URLs."""
        config = ConnectorConfig()
        assert config.base_ws_url == "wss://fstream.binance.com"
        assert config.base_rest_url == "https://fapi.binance.com"

    def test_default_shard_config(self) -> None:
        """Default shard config is created."""
        config = ConnectorConfig()
        assert config.shard_config.max_streams == 800


class TestSymbolInfo:
    """Tests for SymbolInfo."""

    def test_from_raw(self) -> None:
        """Parse from raw exchangeInfo response."""
        raw = {
            "symbol": "BTCUSDT",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "pricePrecision": 2,
            "quantityPrecision": 3,
            "contractType": "PERPETUAL",
            "status": "TRADING",
        }

        info = SymbolInfo.from_raw(raw)

        assert info.symbol == "BTCUSDT"
        assert info.base_asset == "BTC"
        assert info.quote_asset == "USDT"
        assert info.price_precision == 2
        assert info.quantity_precision == 3
        assert info.contract_type == "PERPETUAL"
        assert info.status == "TRADING"

    def test_from_raw_missing_contract_type(self) -> None:
        """Default contract type when missing."""
        raw = {
            "symbol": "BTCUSDT",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "pricePrecision": 2,
            "quantityPrecision": 3,
            "status": "TRADING",
        }

        info = SymbolInfo.from_raw(raw)
        assert info.contract_type == "PERPETUAL"


class TestRawMessage:
    """Tests for RawMessage."""

    def test_raw_message_fields(self) -> None:
        """RawMessage stores all fields."""
        msg = RawMessage(
            data={"e": "aggTrade", "s": "BTCUSDT"},
            recv_ts=1234567890,
            shard_id=0,
        )

        assert msg.data["e"] == "aggTrade"
        assert msg.recv_ts == 1234567890
        assert msg.shard_id == 0


class TestShardMetrics:
    """Tests for ShardMetrics."""

    def test_default_values(self) -> None:
        """Default metrics values."""
        metrics = ShardMetrics(shard_id=0)

        assert metrics.shard_id == 0
        assert metrics.stream_count == 0
        assert metrics.messages_received == 0
        assert metrics.messages_per_second == 0.0
        assert metrics.last_message_ts == 0
        assert metrics.reconnect_count == 0
        assert metrics.state == ConnectionState.DISCONNECTED


class TestConnectorMetrics:
    """Tests for ConnectorMetrics."""

    def test_default_values(self) -> None:
        """Default connector metrics."""
        metrics = ConnectorMetrics()

        assert metrics.total_streams == 0
        assert metrics.total_messages == 0
        assert metrics.active_shards == 0
        assert metrics.shard_metrics == []
        assert metrics.circuit_breaker_open is False
        assert metrics.last_error is None


class TestExchangeInfo:
    """Tests for ExchangeInfo."""

    def test_exchange_info_fields(self) -> None:
        """ExchangeInfo stores all fields."""
        info = ExchangeInfo(
            symbols=[{"symbol": "BTCUSDT"}],
            server_time=1234567890,
            rate_limits=[{"type": "REQUEST_WEIGHT"}],
        )

        assert len(info.symbols) == 1
        assert info.server_time == 1234567890
        assert len(info.rate_limits) == 1


class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_all_states(self) -> None:
        """All connection states are defined."""
        states = [
            ConnectionState.DISCONNECTED,
            ConnectionState.CONNECTING,
            ConnectionState.CONNECTED,
            ConnectionState.RECONNECTING,
            ConnectionState.CLOSING,
            ConnectionState.CLOSED,
        ]
        assert len(states) == 6

    def test_state_values(self) -> None:
        """State values are strings."""
        assert ConnectionState.CONNECTED.value == "CONNECTED"
        assert ConnectionState.DISCONNECTED.value == "DISCONNECTED"
