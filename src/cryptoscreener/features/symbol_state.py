"""Per-symbol state tracking for feature computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cryptoscreener.contracts.events import (
    DataHealth,
    Features,
    MarketEvent,
    MarketEventType,
    RegimeTrend,
    RegimeVol,
    Windows,
)
from cryptoscreener.features.ring_buffer import RingBuffer


@dataclass
class TradeData:
    """Parsed trade data from MarketEvent payload."""

    price: float
    quantity: float
    is_buyer_maker: bool
    ts: int


@dataclass
class BookData:
    """Parsed order book data from MarketEvent payload."""

    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    ts: int


@dataclass
class SymbolState:
    """
    Maintains state for a single symbol.

    Tracks:
    - Ring buffers for trades at multiple windows (1s, 10s, 60s, 5m)
    - Last known book state
    - Last update timestamps for staleness detection
    """

    symbol: str

    # Ring buffers for different windows
    trades_1s: RingBuffer[TradeData] = field(
        default_factory=lambda: RingBuffer[TradeData](window_ms=1000)
    )
    trades_10s: RingBuffer[TradeData] = field(
        default_factory=lambda: RingBuffer[TradeData](window_ms=10000)
    )
    trades_60s: RingBuffer[TradeData] = field(
        default_factory=lambda: RingBuffer[TradeData](window_ms=60000)
    )
    trades_5m: RingBuffer[TradeData] = field(
        default_factory=lambda: RingBuffer[TradeData](window_ms=300000)
    )

    # Price history for ATR calculation
    prices_5m: RingBuffer[float] = field(
        default_factory=lambda: RingBuffer[float](window_ms=300000)
    )

    # Last known book state
    last_book: BookData | None = None
    last_book_ts: int = 0
    last_trade_ts: int = 0

    def update_trade(self, event: MarketEvent) -> None:
        """
        Update state with a trade event.

        Args:
            event: MarketEvent of type TRADE.
        """
        payload = event.payload
        trade = TradeData(
            price=float(payload.get("p", payload.get("price", 0))),
            quantity=float(payload.get("q", payload.get("qty", 0))),
            is_buyer_maker=bool(payload.get("m", payload.get("is_buyer_maker", False))),
            ts=event.ts,
        )

        # Push to all trade buffers
        self.trades_1s.push(event.ts, trade)
        self.trades_10s.push(event.ts, trade)
        self.trades_60s.push(event.ts, trade)
        self.trades_5m.push(event.ts, trade)

        # Track price for ATR
        self.prices_5m.push(event.ts, trade.price)

        self.last_trade_ts = event.ts

    def update_book(self, event: MarketEvent) -> None:
        """
        Update state with a book ticker event.

        Args:
            event: MarketEvent of type BOOK.
        """
        payload = event.payload
        self.last_book = BookData(
            bid_price=float(payload.get("b", payload.get("bid_price", 0))),
            bid_qty=float(payload.get("B", payload.get("bid_qty", 0))),
            ask_price=float(payload.get("a", payload.get("ask_price", 0))),
            ask_qty=float(payload.get("A", payload.get("ask_qty", 0))),
            ts=event.ts,
        )
        self.last_book_ts = event.ts

    def process_event(self, event: MarketEvent) -> None:
        """
        Process any market event and update state accordingly.

        Args:
            event: MarketEvent to process.
        """
        if event.type == MarketEventType.TRADE:
            self.update_trade(event)
        elif event.type == MarketEventType.BOOK:
            self.update_book(event)
        # Other event types (KLINE, MARK, etc.) can be added here

    def compute_spread_bps(self) -> float:
        """Compute spread in basis points."""
        if not self.last_book:
            return 0.0

        mid = (self.last_book.bid_price + self.last_book.ask_price) / 2
        if mid <= 0:
            return 0.0

        spread = self.last_book.ask_price - self.last_book.bid_price
        return (spread / mid) * 10000

    def compute_mid(self) -> float:
        """Compute mid price."""
        if not self.last_book:
            return 0.0
        return (self.last_book.bid_price + self.last_book.ask_price) / 2

    def compute_book_imbalance(self) -> float:
        """
        Compute order book imbalance.

        Returns value in [-1, 1] where:
        - Positive = more bid pressure
        - Negative = more ask pressure
        """
        if not self.last_book:
            return 0.0

        total = self.last_book.bid_qty + self.last_book.ask_qty
        if total <= 0:
            return 0.0

        return (self.last_book.bid_qty - self.last_book.ask_qty) / total

    def compute_flow_imbalance(self, current_ts: int, window_ms: int = 10000) -> float:
        """
        Compute trade flow imbalance over a window.

        Returns value in [-1, 1] where:
        - Positive = more buyer aggression (sells hitting bid)
        - Negative = more seller aggression (buys hitting ask)

        Note: is_buyer_maker=True means taker was selling (hit the bid).
        """
        if window_ms == 1000:
            trades = self.trades_1s.get_values(current_ts)
        elif window_ms == 10000:
            trades = self.trades_10s.get_values(current_ts)
        elif window_ms == 60000:
            trades = self.trades_60s.get_values(current_ts)
        else:
            trades = self.trades_5m.get_values(current_ts)

        if not trades:
            return 0.0

        buy_volume = sum(t.quantity for t in trades if not t.is_buyer_maker)
        sell_volume = sum(t.quantity for t in trades if t.is_buyer_maker)
        total = buy_volume + sell_volume

        if total <= 0:
            return 0.0

        return (buy_volume - sell_volume) / total

    def compute_natr(self, current_ts: int) -> float:
        """
        Compute Normalized ATR over 5m window.

        Simplified: uses price range / mean price.
        """
        prices = self.prices_5m.get_values(current_ts)
        if len(prices) < 2:
            return 0.0

        price_min = min(prices)
        price_max = max(prices)
        price_mean = sum(prices) / len(prices)

        if price_mean <= 0:
            return 0.0

        return (price_max - price_min) / price_mean

    def compute_impact_bps(self, current_ts: int, ref_qty: float = 1.0) -> float:
        """
        Estimate price impact in bps for reference quantity.

        Simplified: uses spread + volume-weighted adjustment.
        """
        spread_bps = self.compute_spread_bps()

        # Get recent volume for scaling
        trades = self.trades_10s.get_values(current_ts)
        if not trades:
            return spread_bps

        avg_trade_size = sum(t.quantity for t in trades) / len(trades)
        if avg_trade_size <= 0:
            return spread_bps

        # Impact scales with size relative to average
        size_ratio = ref_qty / avg_trade_size
        return spread_bps * (1 + size_ratio * 0.1)

    def compute_regime_vol(self, current_ts: int) -> RegimeVol:
        """Classify volatility regime based on NATR."""
        natr = self.compute_natr(current_ts)
        # Threshold: 0.5% is high volatility
        return RegimeVol.HIGH if natr > 0.005 else RegimeVol.LOW

    def compute_regime_trend(self, current_ts: int) -> RegimeTrend:
        """
        Classify trend regime.

        Uses price direction consistency over the window.
        """
        prices = self.prices_5m.get_values(current_ts)
        if len(prices) < 10:
            return RegimeTrend.CHOP

        # Count directional changes
        changes = 0
        for i in range(1, len(prices)):
            if i >= 2 and (prices[i] - prices[i - 1]) * (prices[i - 1] - prices[i - 2]) < 0:
                changes += 1

        # If less than 30% of moves are reversals, it's trending
        reversal_ratio = changes / (len(prices) - 1)
        return RegimeTrend.TREND if reversal_ratio < 0.3 else RegimeTrend.CHOP

    def compute_window_aggregates(self, current_ts: int, window_ms: int) -> dict[str, Any]:
        """Compute aggregate features for a specific window."""
        if window_ms == 1000:
            trades = self.trades_1s.get_values(current_ts)
        elif window_ms == 10000:
            trades = self.trades_10s.get_values(current_ts)
        elif window_ms == 60000:
            trades = self.trades_60s.get_values(current_ts)
        else:
            trades = self.trades_5m.get_values(current_ts)

        if not trades:
            return {
                "trade_count": 0,
                "volume": 0.0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "vwap": 0.0,
            }

        total_volume = sum(t.quantity for t in trades)
        buy_volume = sum(t.quantity for t in trades if not t.is_buyer_maker)
        sell_volume = sum(t.quantity for t in trades if t.is_buyer_maker)

        vwap = 0.0
        if total_volume > 0:
            vwap = sum(t.price * t.quantity for t in trades) / total_volume

        return {
            "trade_count": len(trades),
            "volume": total_volume,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "vwap": vwap,
        }

    def compute_data_health(self, current_ts: int) -> DataHealth:
        """Compute data health metrics."""
        stale_book_ms = current_ts - self.last_book_ts if self.last_book_ts > 0 else 0
        stale_trades_ms = current_ts - self.last_trade_ts if self.last_trade_ts > 0 else 0

        missing_streams: list[str] = []
        if stale_book_ms > 5000:
            missing_streams.append("book")
        if stale_trades_ms > 10000:
            missing_streams.append("trades")

        return DataHealth(
            stale_book_ms=stale_book_ms,
            stale_trades_ms=stale_trades_ms,
            missing_streams=missing_streams,
        )

    def compute_features(self, current_ts: int) -> Features:
        """Compute all core features at current timestamp."""
        mid = self.compute_mid()
        if mid <= 0:
            mid = 1.0  # Prevent division by zero

        return Features(
            spread_bps=self.compute_spread_bps(),
            mid=mid,
            book_imbalance=self.compute_book_imbalance(),
            flow_imbalance=self.compute_flow_imbalance(current_ts),
            natr_14_5m=self.compute_natr(current_ts),
            impact_bps_q=self.compute_impact_bps(current_ts),
            regime_vol=self.compute_regime_vol(current_ts),
            regime_trend=self.compute_regime_trend(current_ts),
        )

    def compute_windows(self, current_ts: int) -> Windows:
        """Compute rolling window aggregates."""
        return Windows(
            w1s=self.compute_window_aggregates(current_ts, 1000),
            w10s=self.compute_window_aggregates(current_ts, 10000),
            w60s=self.compute_window_aggregates(current_ts, 60000),
        )
