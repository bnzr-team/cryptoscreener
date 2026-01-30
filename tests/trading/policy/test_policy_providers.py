"""Tests for PolicyInputsProviders."""

from decimal import Decimal

from cryptoscreener.trading.contracts import PositionSide
from cryptoscreener.trading.policy.inputs import PolicyInputs
from cryptoscreener.trading.policy.providers import (
    ConstantPolicyInputsProvider,
    FixturePolicyInputsProvider,
)
from cryptoscreener.trading.strategy import StrategyContext


def make_strategy_context() -> StrategyContext:
    """Create a minimal StrategyContext for testing."""
    return StrategyContext(
        ts=1000000,
        bid=Decimal("42000"),
        ask=Decimal("42001"),
        last_trade_price=Decimal("42000.50"),
        last_book_ts=999900,
        last_trade_ts=999800,
        position_qty=Decimal("0"),
        position_side=PositionSide.FLAT,
        entry_price=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        pending_order_count=0,
        symbol="BTCUSDT",
        max_position_qty=Decimal("0.01"),
    )


class TestConstantPolicyInputsProvider:
    """Tests for ConstantPolicyInputsProvider."""

    def test_returns_constant_inputs(self) -> None:
        """Provider returns the same inputs regardless of context."""
        inputs = PolicyInputs(
            p_inplay_2m=Decimal("0.7"),
            p_toxic=Decimal("0.2"),
            regime_vol="NORMAL",
            regime_trend="UP",
            natr_14_5m=Decimal("15"),
        )
        provider = ConstantPolicyInputsProvider(inputs)
        ctx = make_strategy_context()

        result = provider.get_inputs(ctx)

        assert result == inputs
        assert result.p_inplay_2m == Decimal("0.7")
        assert result.p_toxic == Decimal("0.2")

    def test_multiple_calls_same_result(self) -> None:
        """Multiple calls return identical inputs."""
        inputs = PolicyInputs(
            p_inplay_2m=Decimal("0.5"),
            p_toxic=Decimal("0.1"),
            regime_vol="LOW",
            regime_trend="NEUTRAL",
            natr_14_5m=Decimal("10"),
        )
        provider = ConstantPolicyInputsProvider(inputs)
        ctx = make_strategy_context()

        result1 = provider.get_inputs(ctx)
        result2 = provider.get_inputs(ctx)

        assert result1 == result2


class TestFixturePolicyInputsProvider:
    """Tests for FixturePolicyInputsProvider."""

    def test_monotonic_up_fixture(self) -> None:
        """monotonic_up fixture has low in-play, UP trend."""
        provider = FixturePolicyInputsProvider("monotonic_up")
        ctx = make_strategy_context()

        inputs = provider.get_inputs(ctx)

        assert inputs.p_inplay_2m == Decimal("0.3")
        assert inputs.p_toxic == Decimal("0.1")
        assert inputs.regime_vol == "NORMAL"
        assert inputs.regime_trend == "UP"

    def test_mean_reverting_fixture(self) -> None:
        """mean_reverting fixture has high in-play, NEUTRAL trend."""
        provider = FixturePolicyInputsProvider("mean_reverting_range")
        ctx = make_strategy_context()

        inputs = provider.get_inputs(ctx)

        assert inputs.p_inplay_2m == Decimal("0.8")
        assert inputs.p_toxic == Decimal("0.1")
        assert inputs.regime_trend == "NEUTRAL"

    def test_flash_crash_fixture(self) -> None:
        """flash_crash fixture has high toxicity, HIGH vol, DOWN trend."""
        provider = FixturePolicyInputsProvider("flash_crash")
        ctx = make_strategy_context()

        inputs = provider.get_inputs(ctx)

        assert inputs.p_inplay_2m == Decimal("0.2")
        assert inputs.p_toxic == Decimal("0.8")
        assert inputs.regime_vol == "HIGH"
        assert inputs.regime_trend == "DOWN"

    def test_ws_gap_fixture(self) -> None:
        """ws_gap fixture has medium in-play, low toxicity."""
        provider = FixturePolicyInputsProvider("ws_gap")
        ctx = make_strategy_context()

        inputs = provider.get_inputs(ctx)

        assert inputs.p_inplay_2m == Decimal("0.5")
        assert inputs.p_toxic == Decimal("0.2")
        assert inputs.regime_vol == "NORMAL"

    def test_unknown_fixture_uses_default(self) -> None:
        """Unknown fixture uses default inputs."""
        provider = FixturePolicyInputsProvider("unknown_fixture")
        ctx = make_strategy_context()

        inputs = provider.get_inputs(ctx)

        # Default values
        assert inputs.p_inplay_2m == Decimal("0.5")
        assert inputs.p_toxic == Decimal("0.2")
        assert inputs.regime_vol == "NORMAL"
        assert inputs.regime_trend == "NEUTRAL"

    def test_strips_jsonl_extension(self) -> None:
        """Provider strips .jsonl extension from fixture name."""
        provider = FixturePolicyInputsProvider("monotonic_up.jsonl")

        assert provider.fixture_name == "monotonic_up"

    def test_fixture_name_property(self) -> None:
        """fixture_name property returns stored name."""
        provider = FixturePolicyInputsProvider("flash_crash")

        assert provider.fixture_name == "flash_crash"
