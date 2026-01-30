"""Core trading simulator.

Deterministic offline simulation of trading strategy on market events.
Produces SimArtifacts with fills, positions, and metrics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from cryptoscreener.trading.contracts import (
    FillEvent,
    MarginType,
    OrderAck,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    PositionSnapshot,
    SessionState,
    SessionStateEnum,
    TimeInForce,
)
from cryptoscreener.trading.sim.artifacts import SimArtifacts, build_artifacts
from cryptoscreener.trading.sim.config import FillModel, SimConfig
from cryptoscreener.trading.sim.fill_model import (
    CrossFillModel,
    FillResult,
    MarketTick,
    OrderState,
)
from cryptoscreener.trading.sim.metrics import SimResult, compute_metrics


@dataclass
class PendingOrder:
    """A pending order in the simulator."""

    order_id: int
    client_order_id: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    placed_ts: int
    order_type: OrderType = OrderType.LIMIT


@dataclass
class SimulatorState:
    """Internal state of the simulator."""

    # Position tracking
    position_qty: Decimal = Decimal("0")
    position_side: PositionSide = PositionSide.FLAT
    entry_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Order tracking
    pending_orders: dict[int, PendingOrder] = field(default_factory=dict)
    next_order_id: int = 1
    next_trade_id: int = 1

    # Market state
    last_bid: Decimal = Decimal("0")
    last_ask: Decimal = Decimal("0")
    last_trade_price: Decimal = Decimal("0")
    last_book_ts: int = 0
    last_trade_ts: int = 0

    # Session
    session_id: str = ""
    session_state: SessionStateEnum = SessionStateEnum.INITIALIZING
    orders_placed: int = 0

    # Outputs
    fills: list[FillEvent] = field(default_factory=list)
    orders: list[OrderAck] = field(default_factory=list)
    positions: list[PositionSnapshot] = field(default_factory=list)
    session_states: list[SessionState] = field(default_factory=list)


# Strategy callback type
StrategyCallback = Callable[
    [SimulatorState, Decimal, Decimal, int],  # state, bid, ask, ts
    list[tuple[OrderSide, Decimal, Decimal]],  # List of (side, price, qty) orders to place
]


def simple_mm_strategy(
    state: SimulatorState,
    bid: Decimal,
    ask: Decimal,
    _ts: int,
    spread_bps: Decimal = Decimal("10"),
    order_qty: Decimal = Decimal("0.001"),
    max_position: Decimal = Decimal("0.01"),
) -> list[tuple[OrderSide, Decimal, Decimal]]:
    """Simple market-making strategy for testing.

    Places limit orders on both sides with a configurable spread.
    Respects max position limits. Prioritizes closing positions
    to complete round trips.

    Args:
        state: Current simulator state.
        bid: Current best bid.
        ask: Current best ask.
        _ts: Current timestamp (unused in this strategy).
        spread_bps: Spread to add in basis points.
        order_qty: Order quantity.
        max_position: Maximum position size.

    Returns:
        List of orders to place: (side, price, quantity).
    """
    orders: list[tuple[OrderSide, Decimal, Decimal]] = []

    if bid <= 0 or ask <= 0:
        return orders

    mid = (bid + ask) / 2
    spread_frac = spread_bps / Decimal("10000")

    # Calculate our prices
    our_bid = mid * (1 - spread_frac)
    our_ask = mid * (1 + spread_frac)

    current_pos = state.position_qty

    # Strategy: If we have a position, prioritize closing it
    # If flat, place orders on both sides to enter
    if current_pos > 0:
        # Long position: place SELL to close
        if len(state.pending_orders) == 0:
            orders.append((OrderSide.SELL, our_ask, min(order_qty, current_pos)))
    elif current_pos < 0:
        # Short position: place BUY to close
        if len(state.pending_orders) == 0:
            orders.append((OrderSide.BUY, our_bid, min(order_qty, abs(current_pos))))
    else:
        # Flat: place both sides to enter
        if len(state.pending_orders) < 2:
            if current_pos + order_qty <= max_position:
                orders.append((OrderSide.BUY, our_bid, order_qty))
            if current_pos - order_qty >= -max_position:
                orders.append((OrderSide.SELL, our_ask, order_qty))

    return orders


class Simulator:
    """Deterministic trading simulator.

    Processes market events, applies strategy, matches orders, tracks PnL.
    """

    def __init__(
        self,
        config: SimConfig,
        strategy: StrategyCallback | None = None,
    ) -> None:
        """Initialize simulator.

        Args:
            config: Simulation configuration.
            strategy: Strategy callback. Defaults to simple_mm_strategy.
        """
        self.config = config
        self.strategy = strategy or simple_mm_strategy

        # Select fill model
        if config.fill_model == FillModel.CROSS:
            self.fill_model = CrossFillModel()
        else:
            raise NotImplementedError(f"Fill model {config.fill_model} not implemented (Phase 2)")

        self._state = SimulatorState()

    def run(self, events: list[dict[str, Any]]) -> SimArtifacts:
        """Run simulation on market events.

        Args:
            events: List of market events (dicts with ts, type, payload, symbol).

        Returns:
            SimArtifacts with all outputs and deterministic SHA256.
        """
        if not events:
            return self._build_empty_artifacts()

        # Initialize session with deterministic session_id
        self._state = SimulatorState()
        # Use hash of config + first event timestamp for deterministic session_id
        config_hash = hashlib.md5(
            f"{self.config.symbol}:{events[0]['ts']}".encode(), usedforsecurity=False
        ).hexdigest()[:12]
        self._state.session_id = f"sim_{config_hash}"
        self._transition_state(SessionStateEnum.READY, events[0]["ts"], "session_init")

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: (e["ts"], e.get("recv_ts", e["ts"])))

        start_ts = sorted_events[0]["ts"]
        end_ts = sorted_events[-1]["ts"]

        # Transition to active
        self._transition_state(SessionStateEnum.ACTIVE, start_ts, "simulation_start")

        # Process each event
        for event in sorted_events:
            self._process_event(event)

            # Check kill switch
            if self._check_kill_switch():
                self._transition_state(
                    SessionStateEnum.KILLED,
                    event["ts"],
                    f"max_loss_exceeded:{self.config.max_session_loss}",
                )
                break

        # Final state if not killed
        if self._state.session_state != SessionStateEnum.KILLED:
            self._transition_state(SessionStateEnum.STOPPED, end_ts, "simulation_complete")

        # Compute metrics
        metrics = compute_metrics(
            fills=self._state.fills,
            positions=self._state.positions,
            orders_placed=self._state.orders_placed,
            session_start_ts=start_ts,
            session_end_ts=end_ts,
        )

        # Build artifacts
        return build_artifacts(
            config=self.config,
            fills=self._state.fills,
            orders=self._state.orders,
            positions=self._state.positions,
            session_states=self._state.session_states,
            metrics=metrics,
        )

    def _process_event(self, event: dict[str, Any]) -> None:
        """Process a single market event."""
        symbol = event.get("symbol", "")
        if symbol != self.config.symbol:
            return  # Skip events for other symbols

        event_type = event.get("type", "")
        payload = event.get("payload", {})
        ts = event["ts"]

        if event_type == "trade":
            self._process_trade(ts, payload)
        elif event_type == "book":
            self._process_book(ts, payload)

        # Run strategy and place orders
        if self._state.last_bid > 0 and self._state.last_ask > 0:
            self._run_strategy(ts)

        # Record position snapshot
        self._record_position(ts)

    def _process_trade(self, ts: int, payload: dict[str, Any]) -> None:
        """Process a trade event."""
        price_str = payload.get("price", "0")
        self._state.last_trade_price = Decimal(str(price_str))
        self._state.last_trade_ts = ts

        # Check fills against pending orders
        tick = MarketTick(
            ts=ts,
            trade_price=self._state.last_trade_price,
        )
        self._check_fills(tick, ts)

    def _process_book(self, ts: int, payload: dict[str, Any]) -> None:
        """Process a book update event."""
        bid_str = payload.get("bid", "0")
        ask_str = payload.get("ask", "0")

        self._state.last_bid = Decimal(str(bid_str))
        self._state.last_ask = Decimal(str(ask_str))
        self._state.last_book_ts = ts

        # Update unrealized PnL
        self._update_unrealized_pnl()

    def _run_strategy(self, ts: int) -> None:
        """Run strategy and place orders."""
        # Check for stale quotes
        if ts - self._state.last_book_ts > self.config.stale_quote_ms:
            # Cancel all pending orders
            self._cancel_all_orders(ts, "stale_quotes")
            return

        # Get strategy orders
        new_orders = self.strategy(
            self._state,
            self._state.last_bid,
            self._state.last_ask,
            ts,
        )

        # Place new orders
        for side, price, qty in new_orders:
            self._place_order(side, price, qty, ts)

    def _place_order(
        self,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
        ts: int,
    ) -> None:
        """Place a new order."""
        order_id = self._state.next_order_id
        self._state.next_order_id += 1
        client_order_id = f"coid_{self._state.session_id}_{order_id:06d}"

        pending = PendingOrder(
            order_id=order_id,
            client_order_id=client_order_id,
            side=side,
            price=price,
            quantity=quantity,
            placed_ts=ts,
        )
        self._state.pending_orders[order_id] = pending
        self._state.orders_placed += 1

        # Record order ack
        ack = OrderAck(
            session_id=self._state.session_id,
            ts=ts,
            client_order_id=client_order_id,
            order_id=order_id,
            symbol=self.config.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            quantity=quantity,
            price=price,
            executed_qty=Decimal("0"),
            avg_price=Decimal("0"),
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
        )
        self._state.orders.append(ack)

    def _check_fills(self, tick: MarketTick, ts: int) -> None:
        """Check if any pending orders should be filled."""
        filled_order_ids: list[int] = []

        for order_id, pending in self._state.pending_orders.items():
            order_state = OrderState(
                side=pending.side,
                price=pending.price,
                quantity=pending.quantity,
                placed_ts=pending.placed_ts,
            )

            result: FillResult = self.fill_model.check_fill(order_state, tick)

            if result.filled and result.fill_price is not None and result.fill_qty is not None:
                self._execute_fill(pending, result, ts)
                filled_order_ids.append(order_id)

        # Remove filled orders
        for order_id in filled_order_ids:
            del self._state.pending_orders[order_id]

    def _execute_fill(
        self,
        order: PendingOrder,
        result: FillResult,
        ts: int,
    ) -> None:
        """Execute a fill and update position."""
        if result.fill_price is None or result.fill_qty is None:
            return

        fill_price = result.fill_price
        fill_qty = result.fill_qty

        # Calculate commission (maker for limit orders)
        notional = fill_price * fill_qty
        commission = notional * self.config.maker_fee_frac

        # Calculate realized PnL
        realized_pnl = Decimal("0")
        old_qty = self._state.position_qty

        if order.side == OrderSide.BUY:
            new_qty = old_qty + fill_qty
            if old_qty < 0:
                # Closing short position
                closed_qty = min(fill_qty, abs(old_qty))
                realized_pnl = closed_qty * (self._state.entry_price - fill_price)
        else:  # SELL
            new_qty = old_qty - fill_qty
            if old_qty > 0:
                # Closing long position
                closed_qty = min(fill_qty, old_qty)
                realized_pnl = closed_qty * (fill_price - self._state.entry_price)

        # Update position
        self._state.position_qty = new_qty
        self._state.realized_pnl += realized_pnl

        # Update entry price for new position
        if new_qty != 0:  # noqa: SIM102 - nested structure more readable here
            if (old_qty >= 0 and new_qty > old_qty) or (old_qty <= 0 and new_qty < old_qty):
                # Adding to position - weighted average
                if old_qty == 0:
                    self._state.entry_price = fill_price
                else:
                    total_notional = abs(old_qty) * self._state.entry_price + fill_qty * fill_price
                    self._state.entry_price = total_notional / abs(new_qty)

        # Update position side
        if new_qty > 0:
            self._state.position_side = PositionSide.LONG
        elif new_qty < 0:
            self._state.position_side = PositionSide.SHORT
        else:
            self._state.position_side = PositionSide.FLAT
            self._state.entry_price = Decimal("0")

        # Record fill event
        trade_id = self._state.next_trade_id
        self._state.next_trade_id += 1

        fill_event = FillEvent(
            session_id=self._state.session_id,
            ts=ts,
            symbol=self.config.symbol,
            order_id=order.order_id,
            trade_id=trade_id,
            client_order_id=order.client_order_id,
            side=order.side,
            fill_qty=fill_qty,
            fill_price=fill_price,
            commission=commission,
            commission_asset="USDT",
            realized_pnl=realized_pnl,
            is_maker=True,
            position_side=self._state.position_side,
        )
        self._state.fills.append(fill_event)

        # Record order update (filled)
        filled_ack = OrderAck(
            session_id=self._state.session_id,
            ts=ts,
            client_order_id=order.client_order_id,
            order_id=order.order_id,
            symbol=self.config.symbol,
            side=order.side,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            quantity=order.quantity,
            price=order.price,
            executed_qty=fill_qty,
            avg_price=fill_price,
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
        )
        self._state.orders.append(filled_ack)

    def _cancel_all_orders(self, ts: int, reason: str) -> None:
        """Cancel all pending orders."""
        for order_id, pending in list(self._state.pending_orders.items()):
            cancel_ack = OrderAck(
                session_id=self._state.session_id,
                ts=ts,
                client_order_id=pending.client_order_id,
                order_id=order_id,
                symbol=self.config.symbol,
                side=pending.side,
                order_type=OrderType.LIMIT,
                status=OrderStatus.CANCELED,
                quantity=pending.quantity,
                price=pending.price,
                executed_qty=Decimal("0"),
                avg_price=Decimal("0"),
                time_in_force=TimeInForce.GTC,
                reduce_only=False,
                error_code=None,
                error_msg=reason,
            )
            self._state.orders.append(cancel_ack)

        self._state.pending_orders.clear()

    def _update_unrealized_pnl(self) -> None:
        """Update unrealized PnL based on current market price."""
        if self._state.position_qty == 0:
            self._state.unrealized_pnl = Decimal("0")
            return

        mid = (self._state.last_bid + self._state.last_ask) / 2
        if mid <= 0:
            return

        qty = self._state.position_qty
        if qty > 0:
            self._state.unrealized_pnl = qty * (mid - self._state.entry_price)
        else:
            self._state.unrealized_pnl = abs(qty) * (self._state.entry_price - mid)

    def _record_position(self, ts: int) -> None:
        """Record current position snapshot."""
        mid = (
            (self._state.last_bid + self._state.last_ask) / 2
            if self._state.last_bid > 0
            else Decimal("0")
        )

        # Determine position side for snapshot
        if self._state.position_qty > 0:
            side = PositionSide.LONG
        elif self._state.position_qty < 0:
            side = PositionSide.SHORT
        else:
            side = PositionSide.FLAT

        pos = PositionSnapshot(
            session_id=self._state.session_id,
            ts=ts,
            symbol=self.config.symbol,
            side=side,
            quantity=abs(self._state.position_qty),
            entry_price=self._state.entry_price,
            mark_price=mid,
            unrealized_pnl=self._state.unrealized_pnl,
            realized_pnl_session=self._state.realized_pnl,
            leverage=1,
            liquidation_price=Decimal("0"),
            margin_type=MarginType.CROSS,
        )
        self._state.positions.append(pos)

    def _transition_state(self, new_state: SessionStateEnum, ts: int, reason: str) -> None:
        """Transition session state."""
        prev_state = self._state.session_state
        self._state.session_state = new_state

        state_event = SessionState(
            session_id=self._state.session_id,
            ts=ts,
            state=new_state,
            prev_state=prev_state,
            reason=reason,
            symbols_active=[self.config.symbol],
            open_orders_count=len(self._state.pending_orders),
            positions_count=1 if self._state.position_qty != 0 else 0,
            daily_pnl=self._state.realized_pnl + self._state.unrealized_pnl,
            daily_trades=len(self._state.fills),
            risk_utilization=abs(self._state.position_qty) / self.config.max_position_qty
            if self.config.max_position_qty > 0
            else Decimal("0"),
        )
        self._state.session_states.append(state_event)

    def _check_kill_switch(self) -> bool:
        """Check if kill switch should be triggered."""
        total_pnl = self._state.realized_pnl + self._state.unrealized_pnl
        return total_pnl < -self.config.max_session_loss

    def _build_empty_artifacts(self) -> SimArtifacts:
        """Build empty artifacts for empty input."""
        empty_metrics = SimResult(
            net_pnl=Decimal("0"),
            total_commissions=Decimal("0"),
            max_drawdown=Decimal("0"),
            total_fills=0,
            round_trips=0,
            win_rate=Decimal("0"),
            profit_factor=Decimal("0"),
            max_position=Decimal("0"),
            avg_session_duration_min=Decimal("0"),
            fill_rate=Decimal("0"),
            adverse_selection_rate=Decimal("0"),
        )
        return build_artifacts(
            config=self.config,
            fills=[],
            orders=[],
            positions=[],
            session_states=[],
            metrics=empty_metrics,
        )
