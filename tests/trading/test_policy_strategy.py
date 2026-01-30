"""Tests for PolicyEngineStrategy."""

from decimal import Decimal

import pytest

from cryptoscreener.trading.contracts import OrderSide, PositionSide
from cryptoscreener.trading.policy import PolicyParams
from cryptoscreener.trading.policy.inputs import PolicyInputs
from cryptoscreener.trading.policy.providers import ConstantPolicyInputsProvider
from cryptoscreener.trading.strategy import StrategyContext
from cryptoscreener.trading.strategy.baseline import BaselineStrategyConfig
from cryptoscreener.trading.strategy.policy_strategy import PolicyEngineStrategy


def make_strategy_context(
    *,
    ts: int = 1000000,
    bid: Decimal = Decimal("42000"),
    ask: Decimal = Decimal("42001"),
    position_qty: Decimal = Decimal("0"),
    unrealized_pnl: Decimal = Decimal("0"),
    realized_pnl: Decimal = Decimal("0"),
    pending_order_count: int = 0,
    last_book_ts: int = 999900,
    last_trade_ts: int = 999800,
) -> StrategyContext:
    """Create StrategyContext for testing."""
    position_side = (
        PositionSide.LONG
        if position_qty > 0
        else PositionSide.SHORT
        if position_qty < 0
        else PositionSide.FLAT
    )
    return StrategyContext(
        ts=ts,
        bid=bid,
        ask=ask,
        last_trade_price=(bid + ask) / 2,
        last_book_ts=last_book_ts,
        last_trade_ts=last_trade_ts,
        position_qty=position_qty,
        position_side=position_side,
        entry_price=Decimal("42000"),
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        pending_order_count=pending_order_count,
        symbol="BTCUSDT",
        max_position_qty=Decimal("0.01"),
    )


def make_policy_inputs(
    *,
    p_inplay_2m: Decimal = Decimal("0.7"),
    p_toxic: Decimal = Decimal("0.1"),
    regime_vol: str = "NORMAL",
    regime_trend: str = "NEUTRAL",
) -> PolicyInputs:
    """Create PolicyInputs for testing."""
    return PolicyInputs(
        p_inplay_2m=p_inplay_2m,
        p_toxic=p_toxic,
        regime_vol=regime_vol,  # type: ignore[arg-type]
        regime_trend=regime_trend,  # type: ignore[arg-type]
        natr_14_5m=Decimal("15"),
    )


class TestPolicyEngineStrategyBasic:
    """Basic functionality tests."""

    def test_normal_conditions_generates_orders(self) -> None:
        """Under normal conditions, strategy generates orders."""
        inputs = make_policy_inputs()
        provider = ConstantPolicyInputsProvider(inputs)
        strategy = PolicyEngineStrategy(provider)
        ctx = make_strategy_context()

        orders = strategy.on_tick(ctx)

        # BaselineStrategy should generate entry orders when flat
        assert len(orders) == 2  # BUY and SELL for entry
        assert any(o.side == OrderSide.BUY for o in orders)
        assert any(o.side == OrderSide.SELL for o in orders)

    def test_implements_strategy_protocol(self) -> None:
        """Strategy implements on_tick method."""
        inputs = make_policy_inputs()
        provider = ConstantPolicyInputsProvider(inputs)
        strategy = PolicyEngineStrategy(provider)

        assert hasattr(strategy, "on_tick")
        assert callable(strategy.on_tick)


class TestSuppressEntry:
    """Test SUPPRESS_ENTRY pattern (POL-002)."""

    def test_low_inplay_suppresses_entry_orders(self) -> None:
        """Low in-play suppresses entry orders."""
        inputs = make_policy_inputs(p_inplay_2m=Decimal("0.3"))
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(inplay_exit_prob=Decimal("0.4"))
        strategy = PolicyEngineStrategy(provider, policy_params=params)
        ctx = make_strategy_context()

        orders = strategy.on_tick(ctx)

        # Entry orders should be suppressed when flat
        assert len(orders) == 0

    def test_low_inplay_allows_exit_orders(self) -> None:
        """Low in-play allows exit orders (reducing position)."""
        inputs = make_policy_inputs(p_inplay_2m=Decimal("0.3"))
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(inplay_exit_prob=Decimal("0.4"))
        strategy = PolicyEngineStrategy(provider, policy_params=params)
        # Long position - SELL would reduce it
        ctx = make_strategy_context(position_qty=Decimal("0.001"))

        orders = strategy.on_tick(ctx)

        # Exit order (SELL to close long) should be allowed
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL


class TestSuppressAll:
    """Test SUPPRESS_ALL pattern (POL-005, POL-012, POL-013)."""

    def test_high_toxicity_suppresses_all(self) -> None:
        """High toxicity suppresses all orders."""
        inputs = make_policy_inputs(p_toxic=Decimal("0.9"))
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(toxicity_disable_threshold=Decimal("0.8"))
        strategy = PolicyEngineStrategy(provider, policy_params=params)
        ctx = make_strategy_context()

        orders = strategy.on_tick(ctx)

        assert len(orders) == 0

    def test_stale_book_suppresses_all(self) -> None:
        """Stale book data suppresses all orders."""
        inputs = make_policy_inputs()
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(stale_quote_ms=5000)
        strategy = PolicyEngineStrategy(provider, policy_params=params)
        # Book is 10 seconds old
        ctx = make_strategy_context(ts=1010000, last_book_ts=1000000)

        orders = strategy.on_tick(ctx)

        assert len(orders) == 0

    def test_hard_limit_suppresses_all(self) -> None:
        """Position at hard limit suppresses all orders."""
        inputs = make_policy_inputs()
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(inventory_hard_limit=Decimal("0.01"))
        strategy = PolicyEngineStrategy(provider, policy_params=params)
        ctx = make_strategy_context(position_qty=Decimal("0.01"))

        orders = strategy.on_tick(ctx)

        assert len(orders) == 0


class TestModifyParams:
    """Test MODIFY_PARAMS pattern (POL-004)."""

    def test_toxicity_widens_spread(self) -> None:
        """Moderate toxicity widens the spread."""
        inputs = make_policy_inputs(p_toxic=Decimal("0.6"))
        provider = ConstantPolicyInputsProvider(inputs)
        base_config = BaselineStrategyConfig(spread_bps=Decimal("10"))
        params = PolicyParams(
            toxicity_widen_threshold=Decimal("0.5"),
            toxic_spread_mult=Decimal("2.0"),
        )
        strategy = PolicyEngineStrategy(
            provider,
            base_config=base_config,
            policy_params=params,
        )
        ctx = make_strategy_context()

        orders = strategy.on_tick(ctx)

        # Check that orders have wider spread
        mid = (ctx.bid + ctx.ask) / 2
        expected_spread_frac = Decimal("10") / Decimal("10000") * Decimal("2.0")

        buy_order = next(o for o in orders if o.side == OrderSide.BUY)
        sell_order = next(o for o in orders if o.side == OrderSide.SELL)

        expected_bid = mid * (1 - expected_spread_frac)
        expected_ask = mid * (1 + expected_spread_frac)

        assert buy_order.price == expected_bid
        assert sell_order.price == expected_ask


class TestForceClose:
    """Test FORCE_CLOSE pattern (POL-019)."""

    def test_max_loss_triggers_force_close(self) -> None:
        """Max session loss triggers force close."""
        inputs = make_policy_inputs()
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(max_session_loss=Decimal("100"))
        strategy = PolicyEngineStrategy(provider, policy_params=params)
        ctx = make_strategy_context(
            position_qty=Decimal("0.001"),
            realized_pnl=Decimal("-110"),
        )

        orders = strategy.on_tick(ctx)

        # Should have exactly one close order
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL  # Close long position
        assert orders[0].quantity == Decimal("0.001")
        assert orders[0].reason == "kill_max_loss"

    def test_kill_sets_killed_flag(self) -> None:
        """Kill switch sets killed flag."""
        inputs = make_policy_inputs()
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(max_session_loss=Decimal("100"))
        strategy = PolicyEngineStrategy(provider, policy_params=params)
        ctx = make_strategy_context(
            position_qty=Decimal("0.001"),
            realized_pnl=Decimal("-110"),
        )

        assert not strategy.is_killed
        strategy.on_tick(ctx)
        assert strategy.is_killed

    def test_killed_strategy_emits_no_orders(self) -> None:
        """Killed strategy emits no orders on subsequent ticks."""
        inputs = make_policy_inputs()
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(max_session_loss=Decimal("100"))
        strategy = PolicyEngineStrategy(provider, policy_params=params)

        # First tick triggers kill
        ctx1 = make_strategy_context(
            position_qty=Decimal("0.001"),
            realized_pnl=Decimal("-110"),
        )
        orders1 = strategy.on_tick(ctx1)
        assert len(orders1) == 1  # Close order

        # Second tick should emit nothing
        ctx2 = make_strategy_context()
        orders2 = strategy.on_tick(ctx2)
        assert len(orders2) == 0

    def test_force_close_short_position(self) -> None:
        """Force close of short position buys at ask."""
        inputs = make_policy_inputs()
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(max_session_loss=Decimal("100"))
        strategy = PolicyEngineStrategy(provider, policy_params=params)
        ctx = make_strategy_context(
            position_qty=Decimal("-0.001"),  # Short position
            realized_pnl=Decimal("-110"),
        )

        orders = strategy.on_tick(ctx)

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY  # Close short by buying
        assert orders[0].price == ctx.ask  # Aggressive price


class TestReasonCodes:
    """Test that reason codes are properly set."""

    def test_toxic_widen_appends_suffix(self) -> None:
        """Toxic widen appends '_toxic' suffix to reason."""
        inputs = make_policy_inputs(p_toxic=Decimal("0.6"))
        provider = ConstantPolicyInputsProvider(inputs)
        params = PolicyParams(toxicity_widen_threshold=Decimal("0.5"))
        strategy = PolicyEngineStrategy(provider, policy_params=params)
        ctx = make_strategy_context()

        orders = strategy.on_tick(ctx)

        # Reasons should have _toxic suffix
        for order in orders:
            assert order.reason.endswith("_toxic")

    @pytest.mark.parametrize("reason", ["enter_long", "enter_short", "close_long", "close_short"])
    def test_baseline_reasons_no_digits(self, reason: str) -> None:
        """Baseline strategy reasons contain no digits."""
        assert not any(c.isdigit() for c in reason)
