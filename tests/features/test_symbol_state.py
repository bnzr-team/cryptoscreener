"""Tests for SymbolState."""

import pytest

from cryptoscreener.contracts.events import (
    MarketEvent,
    MarketEventType,
    RegimeTrend,
    RegimeVol,
)
from cryptoscreener.features.symbol_state import BookData, SymbolState


class TestSymbolState:
    """Tests for SymbolState."""

    @pytest.fixture
    def state(self) -> SymbolState:
        """Create a fresh SymbolState."""
        return SymbolState(symbol="BTCUSDT")

    def test_initial_state(self, state: SymbolState) -> None:
        """Initial state has no data."""
        assert state.symbol == "BTCUSDT"
        assert state.last_book is None
        assert state.last_book_ts == 0
        assert state.last_trade_ts == 0

    def test_update_book(self, state: SymbolState) -> None:
        """Book update stores last book state."""
        event = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="BTCUSDT",
            type=MarketEventType.BOOK,
            payload={"b": "50000.0", "B": "1.5", "a": "50001.0", "A": "2.0"},
            recv_ts=1001,
        )

        state.update_book(event)

        assert state.last_book is not None
        assert state.last_book.bid_price == 50000.0
        assert state.last_book.bid_qty == 1.5
        assert state.last_book.ask_price == 50001.0
        assert state.last_book.ask_qty == 2.0
        assert state.last_book_ts == 1000

    def test_update_trade(self, state: SymbolState) -> None:
        """Trade update populates ring buffers."""
        event = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="BTCUSDT",
            type=MarketEventType.TRADE,
            payload={"p": "50000.0", "q": "0.5", "m": True},
            recv_ts=1001,
        )

        state.update_trade(event)

        assert state.last_trade_ts == 1000
        assert len(state.trades_1s) == 1
        assert len(state.trades_10s) == 1
        assert len(state.prices_5m) == 1

    def test_compute_spread_bps(self, state: SymbolState) -> None:
        """Spread calculation in basis points."""
        state.last_book = BookData(
            bid_price=50000.0,
            bid_qty=1.0,
            ask_price=50010.0,
            ask_qty=1.0,
            ts=1000,
        )

        spread = state.compute_spread_bps()
        # Mid = 50005, spread = 10, bps = 10/50005 * 10000 ≈ 2.0
        assert 1.9 < spread < 2.1

    def test_compute_spread_no_book(self, state: SymbolState) -> None:
        """Spread is 0 when no book data."""
        assert state.compute_spread_bps() == 0.0

    def test_compute_mid(self, state: SymbolState) -> None:
        """Mid price calculation."""
        state.last_book = BookData(
            bid_price=50000.0,
            bid_qty=1.0,
            ask_price=50010.0,
            ask_qty=1.0,
            ts=1000,
        )

        assert state.compute_mid() == 50005.0

    def test_compute_book_imbalance(self, state: SymbolState) -> None:
        """Book imbalance calculation."""
        # More bid qty = positive imbalance
        state.last_book = BookData(
            bid_price=50000.0,
            bid_qty=10.0,
            ask_price=50010.0,
            ask_qty=5.0,
            ts=1000,
        )

        imbalance = state.compute_book_imbalance()
        # (10 - 5) / 15 = 0.333...
        assert 0.33 < imbalance < 0.34

        # More ask qty = negative imbalance
        state.last_book = BookData(
            bid_price=50000.0,
            bid_qty=5.0,
            ask_price=50010.0,
            ask_qty=10.0,
            ts=1000,
        )

        imbalance = state.compute_book_imbalance()
        assert -0.34 < imbalance < -0.33

    def test_compute_flow_imbalance(self, state: SymbolState) -> None:
        """Flow imbalance from trade data."""
        # Add buy trades (is_buyer_maker=False means buyer was taker)
        for i in range(5):
            event = MarketEvent(
                ts=1000 + i * 10,
                source="binance_usdm",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"p": "50000.0", "q": "1.0", "m": False},
                recv_ts=1000 + i * 10,
            )
            state.update_trade(event)

        # All buys = positive flow imbalance
        imbalance = state.compute_flow_imbalance(1050)
        assert imbalance == 1.0

        # Add sell trades
        for i in range(5):
            event = MarketEvent(
                ts=1100 + i * 10,
                source="binance_usdm",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"p": "50000.0", "q": "1.0", "m": True},
                recv_ts=1100 + i * 10,
            )
            state.update_trade(event)

        # Equal buys and sells = zero imbalance
        imbalance = state.compute_flow_imbalance(1150)
        assert imbalance == 0.0

    def test_compute_natr(self, state: SymbolState) -> None:
        """NATR calculation from price history."""
        # Add prices with 1% range
        prices = [50000, 50100, 50200, 50300, 50400, 50500]
        for i, price in enumerate(prices):
            state.prices_5m.push(1000 + i * 1000, float(price))

        natr = state.compute_natr(6000)
        # Range = 500, mean ≈ 50250, NATR ≈ 0.00995
        assert 0.009 < natr < 0.011

    def test_compute_natr_empty(self, state: SymbolState) -> None:
        """NATR is 0 with insufficient data."""
        assert state.compute_natr(1000) == 0.0

    def test_compute_regime_vol(self, state: SymbolState) -> None:
        """Volatility regime classification."""
        # Low volatility (small price range)
        for i in range(10):
            state.prices_5m.push(1000 + i * 1000, 50000.0 + i * 0.1)

        assert state.compute_regime_vol(10000) == RegimeVol.LOW

        # High volatility (large price range)
        state.prices_5m.clear()
        for i in range(10):
            state.prices_5m.push(1000 + i * 1000, 50000.0 + i * 100)

        assert state.compute_regime_vol(10000) == RegimeVol.HIGH

    def test_compute_data_health(self, state: SymbolState) -> None:
        """Data health metrics."""
        state.last_book_ts = 1000
        state.last_trade_ts = 1000

        health = state.compute_data_health(2000)
        assert health.stale_book_ms == 1000
        assert health.stale_trades_ms == 1000
        assert health.missing_streams == []

        # Stale data
        health = state.compute_data_health(20000)
        assert "book" in health.missing_streams
        assert "trades" in health.missing_streams

    def test_compute_features(self, state: SymbolState) -> None:
        """Full feature computation."""
        # Set up book
        state.last_book = BookData(
            bid_price=50000.0,
            bid_qty=10.0,
            ask_price=50010.0,
            ask_qty=10.0,
            ts=1000,
        )
        state.last_book_ts = 1000

        # Add trades
        for i in range(10):
            event = MarketEvent(
                ts=1000 + i * 100,
                source="binance_usdm",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"p": str(50000 + i), "q": "1.0", "m": i % 2 == 0},
                recv_ts=1000 + i * 100,
            )
            state.update_trade(event)

        features = state.compute_features(2000)

        assert features.mid == 50005.0
        assert features.spread_bps > 0
        assert -1 <= features.book_imbalance <= 1
        assert -1 <= features.flow_imbalance <= 1
        assert features.natr_14_5m >= 0
        assert features.regime_vol in [RegimeVol.LOW, RegimeVol.HIGH]
        assert features.regime_trend in [RegimeTrend.TREND, RegimeTrend.CHOP]

    def test_compute_windows(self, state: SymbolState) -> None:
        """Window aggregate computation."""
        # Add trades
        for i in range(20):
            event = MarketEvent(
                ts=i * 100,
                source="binance_usdm",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"p": "50000.0", "q": "1.0", "m": False},
                recv_ts=i * 100,
            )
            state.update_trade(event)

        windows = state.compute_windows(2000)

        assert "trade_count" in windows.w1s
        assert "volume" in windows.w10s
        assert "vwap" in windows.w60s

    def test_process_event_trade(self, state: SymbolState) -> None:
        """Process event routes to correct handler."""
        event = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="BTCUSDT",
            type=MarketEventType.TRADE,
            payload={"p": "50000.0", "q": "1.0", "m": False},
            recv_ts=1001,
        )

        state.process_event(event)
        assert state.last_trade_ts == 1000

    def test_process_event_book(self, state: SymbolState) -> None:
        """Process event routes to correct handler."""
        event = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="BTCUSDT",
            type=MarketEventType.BOOK,
            payload={"b": "50000.0", "B": "1.0", "a": "50001.0", "A": "1.0"},
            recv_ts=1001,
        )

        state.process_event(event)
        assert state.last_book is not None


class TestSymbolStateEdgeCases:
    """Edge case tests for SymbolState."""

    def test_zero_quantities(self) -> None:
        """Handle zero quantities gracefully."""
        state = SymbolState(symbol="BTCUSDT")
        state.last_book = BookData(
            bid_price=50000.0,
            bid_qty=0.0,
            ask_price=50010.0,
            ask_qty=0.0,
            ts=1000,
        )

        assert state.compute_book_imbalance() == 0.0

    def test_zero_mid_price(self) -> None:
        """Handle zero mid price."""
        state = SymbolState(symbol="BTCUSDT")
        state.last_book = BookData(
            bid_price=0.0,
            bid_qty=1.0,
            ask_price=0.0,
            ask_qty=1.0,
            ts=1000,
        )

        assert state.compute_spread_bps() == 0.0
        # compute_features handles zero mid by defaulting to 1.0
        features = state.compute_features(1000)
        assert features.mid == 1.0

    def test_empty_trades_flow_imbalance(self) -> None:
        """Flow imbalance is 0 with no trades."""
        state = SymbolState(symbol="BTCUSDT")
        assert state.compute_flow_imbalance(1000) == 0.0
