"""Tests for PolicyEngine rule evaluation.

Tests MVP rules from DEC-044:
- POL-002: SUPPRESS_ENTRY on low in-play
- POL-004: MODIFY_PARAMS (spread widen) on toxicity
- POL-005: SUPPRESS_ALL on high toxicity
- POL-012: SUPPRESS_ALL on hard inventory limit
- POL-013: SUPPRESS_ALL on stale book data
- POL-019: FORCE_CLOSE + kill on max session loss
"""

from decimal import Decimal

import pytest

from cryptoscreener.trading.contracts import PositionSide
from cryptoscreener.trading.policy import (
    PolicyContext,
    PolicyEngine,
    PolicyParams,
    PolicyPattern,
)
from cryptoscreener.trading.policy.inputs import PolicyInputs


def make_policy_context(
    *,
    ts: int = 1000000,
    last_book_ts: int = 999900,
    last_trade_ts: int = 999800,
    bid: Decimal = Decimal("42000"),
    ask: Decimal = Decimal("42001"),
    position_qty: Decimal = Decimal("0"),
    unrealized_pnl: Decimal = Decimal("0"),
    realized_pnl: Decimal = Decimal("0"),
    max_position_qty: Decimal = Decimal("0.01"),
) -> PolicyContext:
    """Create a PolicyContext for testing."""
    mid = (bid + ask) / 2
    spread_bps = (ask - bid) / mid * Decimal("10000")
    position_side = (
        PositionSide.LONG
        if position_qty > 0
        else PositionSide.SHORT
        if position_qty < 0
        else PositionSide.FLAT
    )
    return PolicyContext(
        ts=ts,
        last_book_ts=last_book_ts,
        last_trade_ts=last_trade_ts,
        bid=bid,
        ask=ask,
        mid=mid,
        spread_bps=spread_bps,
        position_qty=position_qty,
        position_side=position_side,
        entry_price=Decimal("42000"),
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        symbol="BTCUSDT",
        max_position_qty=max_position_qty,
    )


def make_policy_inputs(
    *,
    p_inplay_2m: Decimal = Decimal("0.7"),
    p_toxic: Decimal = Decimal("0.1"),
    regime_vol: str = "NORMAL",
    regime_trend: str = "NEUTRAL",
    natr_14_5m: Decimal = Decimal("15"),
) -> PolicyInputs:
    """Create PolicyInputs for testing."""
    return PolicyInputs(
        p_inplay_2m=p_inplay_2m,
        p_toxic=p_toxic,
        regime_vol=regime_vol,  # type: ignore[arg-type]
        regime_trend=regime_trend,  # type: ignore[arg-type]
        natr_14_5m=natr_14_5m,
    )


class TestPolicyEngineDeterminism:
    """Test that PolicyEngine produces deterministic outputs."""

    def test_same_inputs_same_output(self) -> None:
        """Same context + inputs should produce identical output."""
        engine = PolicyEngine()
        ctx = make_policy_context()
        inputs = make_policy_inputs()

        output1 = engine.evaluate(ctx, inputs)
        output2 = engine.evaluate(ctx, inputs)

        assert output1 == output2

    def test_different_inputs_different_output(self) -> None:
        """Different inputs should produce different outputs."""
        engine = PolicyEngine()
        ctx = make_policy_context()

        # Normal inputs - no suppression
        inputs_normal = make_policy_inputs(p_inplay_2m=Decimal("0.7"))
        # Low in-play - should suppress entry
        inputs_low = make_policy_inputs(p_inplay_2m=Decimal("0.3"))

        output_normal = engine.evaluate(ctx, inputs_normal)
        output_low = engine.evaluate(ctx, inputs_low)

        assert output_normal != output_low
        assert PolicyPattern.SUPPRESS_ENTRY not in output_normal.patterns_active
        assert PolicyPattern.SUPPRESS_ENTRY in output_low.patterns_active


class TestPOL002LowInPlay:
    """Test POL-002: SUPPRESS_ENTRY on low in-play."""

    def test_low_inplay_suppresses_entry(self) -> None:
        """p_inplay < inplay_exit_prob should trigger SUPPRESS_ENTRY."""
        engine = PolicyEngine(PolicyParams(inplay_exit_prob=Decimal("0.4")))
        ctx = make_policy_context()
        inputs = make_policy_inputs(p_inplay_2m=Decimal("0.3"))

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.SUPPRESS_ENTRY in output.patterns_active
        assert "low_inplay_pause" in output.reason_codes

    def test_high_inplay_no_suppression(self) -> None:
        """p_inplay >= inplay_exit_prob should not trigger SUPPRESS_ENTRY."""
        engine = PolicyEngine(PolicyParams(inplay_exit_prob=Decimal("0.4")))
        ctx = make_policy_context()
        inputs = make_policy_inputs(p_inplay_2m=Decimal("0.6"))

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.SUPPRESS_ENTRY not in output.patterns_active
        assert "low_inplay_pause" not in output.reason_codes

    def test_boundary_inplay_no_suppression(self) -> None:
        """p_inplay == inplay_exit_prob should not trigger (only < triggers)."""
        engine = PolicyEngine(PolicyParams(inplay_exit_prob=Decimal("0.4")))
        ctx = make_policy_context()
        inputs = make_policy_inputs(p_inplay_2m=Decimal("0.4"))

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.SUPPRESS_ENTRY not in output.patterns_active


class TestPOL004ToxicWiden:
    """Test POL-004: MODIFY_PARAMS (spread widen) on toxicity."""

    def test_toxic_widens_spread(self) -> None:
        """p_toxic >= toxicity_widen_threshold triggers spread widening."""
        params = PolicyParams(
            toxicity_widen_threshold=Decimal("0.5"),
            toxic_spread_mult=Decimal("2.0"),
        )
        engine = PolicyEngine(params)
        ctx = make_policy_context()
        inputs = make_policy_inputs(p_toxic=Decimal("0.6"))

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.MODIFY_PARAMS in output.patterns_active
        assert "toxic_widen" in output.reason_codes
        assert output.has_spread_override
        assert output.spread_mult == Decimal("2.0")

    def test_low_toxic_no_widen(self) -> None:
        """p_toxic < toxicity_widen_threshold does not widen."""
        params = PolicyParams(toxicity_widen_threshold=Decimal("0.5"))
        engine = PolicyEngine(params)
        ctx = make_policy_context()
        inputs = make_policy_inputs(p_toxic=Decimal("0.3"))

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.MODIFY_PARAMS not in output.patterns_active
        assert not output.has_spread_override


class TestPOL005ToxicDisable:
    """Test POL-005: SUPPRESS_ALL on high toxicity."""

    def test_very_toxic_suppresses_all(self) -> None:
        """p_toxic >= toxicity_disable_threshold triggers SUPPRESS_ALL."""
        params = PolicyParams(toxicity_disable_threshold=Decimal("0.8"))
        engine = PolicyEngine(params)
        ctx = make_policy_context()
        inputs = make_policy_inputs(p_toxic=Decimal("0.85"))

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.SUPPRESS_ALL in output.patterns_active
        assert "toxic_disable" in output.reason_codes
        assert output.should_suppress_all

    def test_moderate_toxic_no_disable(self) -> None:
        """p_toxic < toxicity_disable_threshold does not disable."""
        params = PolicyParams(toxicity_disable_threshold=Decimal("0.8"))
        engine = PolicyEngine(params)
        ctx = make_policy_context()
        inputs = make_policy_inputs(p_toxic=Decimal("0.6"))

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.SUPPRESS_ALL not in output.patterns_active
        assert "toxic_disable" not in output.reason_codes


class TestPOL012HardLimit:
    """Test POL-012: SUPPRESS_ALL on hard inventory limit."""

    def test_hard_limit_blocks_entry(self) -> None:
        """Position at hard limit triggers SUPPRESS_ALL."""
        params = PolicyParams(inventory_hard_limit=Decimal("0.01"))
        engine = PolicyEngine(params)
        ctx = make_policy_context(
            position_qty=Decimal("0.01"),
            max_position_qty=Decimal("0.01"),
        )
        inputs = make_policy_inputs()

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.SUPPRESS_ALL in output.patterns_active
        assert "hard_limit_block" in output.reason_codes

    def test_below_hard_limit_allows_trading(self) -> None:
        """Position below hard limit does not suppress."""
        params = PolicyParams(inventory_hard_limit=Decimal("0.01"))
        engine = PolicyEngine(params)
        ctx = make_policy_context(
            position_qty=Decimal("0.005"),
            max_position_qty=Decimal("0.01"),
        )
        inputs = make_policy_inputs()

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.SUPPRESS_ALL not in output.patterns_active
        assert "hard_limit_block" not in output.reason_codes


class TestPOL013StaleBook:
    """Test POL-013: SUPPRESS_ALL on stale book data."""

    def test_stale_book_pauses_trading(self) -> None:
        """Book older than stale_quote_ms triggers SUPPRESS_ALL."""
        params = PolicyParams(stale_quote_ms=5000)
        engine = PolicyEngine(params)
        # Book is 6 seconds old
        ctx = make_policy_context(ts=1006000, last_book_ts=1000000)
        inputs = make_policy_inputs()

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.SUPPRESS_ALL in output.patterns_active
        assert "stale_book_pause" in output.reason_codes

    def test_fresh_book_allows_trading(self) -> None:
        """Fresh book does not pause."""
        params = PolicyParams(stale_quote_ms=5000)
        engine = PolicyEngine(params)
        # Book is 1 second old
        ctx = make_policy_context(ts=1001000, last_book_ts=1000000)
        inputs = make_policy_inputs()

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.SUPPRESS_ALL not in output.patterns_active
        assert "stale_book_pause" not in output.reason_codes


class TestPOL019KillSwitch:
    """Test POL-019: FORCE_CLOSE + kill on max session loss."""

    def test_max_loss_triggers_kill(self) -> None:
        """total_pnl < -max_session_loss triggers kill."""
        params = PolicyParams(max_session_loss=Decimal("100"))
        engine = PolicyEngine(params)
        ctx = make_policy_context(
            realized_pnl=Decimal("-80"),
            unrealized_pnl=Decimal("-30"),  # Total = -110
        )
        inputs = make_policy_inputs()

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.FORCE_CLOSE in output.patterns_active
        assert "kill_max_loss" in output.reason_codes
        assert output.force_close
        assert output.kill

    def test_within_loss_limit_no_kill(self) -> None:
        """total_pnl >= -max_session_loss does not kill."""
        params = PolicyParams(max_session_loss=Decimal("100"))
        engine = PolicyEngine(params)
        ctx = make_policy_context(
            realized_pnl=Decimal("-50"),
            unrealized_pnl=Decimal("-30"),  # Total = -80
        )
        inputs = make_policy_inputs()

        output = engine.evaluate(ctx, inputs)

        assert PolicyPattern.FORCE_CLOSE not in output.patterns_active
        assert not output.kill

    def test_kill_overrides_other_rules(self) -> None:
        """Kill switch takes priority over all other rules."""
        params = PolicyParams(max_session_loss=Decimal("100"))
        engine = PolicyEngine(params)
        ctx = make_policy_context(
            realized_pnl=Decimal("-150"),
            unrealized_pnl=Decimal("0"),
        )
        # Even with favorable inputs, kill should still trigger
        inputs = make_policy_inputs(p_inplay_2m=Decimal("0.9"), p_toxic=Decimal("0.0"))

        output = engine.evaluate(ctx, inputs)

        assert output.kill
        # Kill is the only pattern (early return)
        assert PolicyPattern.FORCE_CLOSE in output.patterns_active


class TestPolicyPriority:
    """Test that policy priority ordering is respected."""

    def test_kill_highest_priority(self) -> None:
        """Kill switch has highest priority."""
        params = PolicyParams(
            max_session_loss=Decimal("100"),
            toxicity_disable_threshold=Decimal("0.8"),
        )
        engine = PolicyEngine(params)
        # Both kill and toxic disable conditions met
        ctx = make_policy_context(realized_pnl=Decimal("-150"))
        inputs = make_policy_inputs(p_toxic=Decimal("0.9"))

        output = engine.evaluate(ctx, inputs)

        # Kill triggers, toxic_disable should NOT be in output (early return)
        assert output.kill
        assert "kill_max_loss" in output.reason_codes
        assert "toxic_disable" not in output.reason_codes

    def test_suppress_all_stops_lower_priority(self) -> None:
        """SUPPRESS_ALL prevents lower priority rules from adding patterns."""
        params = PolicyParams(
            toxicity_disable_threshold=Decimal("0.8"),
            toxicity_widen_threshold=Decimal("0.5"),
        )
        engine = PolicyEngine(params)
        ctx = make_policy_context()
        # p_toxic=0.9 triggers both disable (>=0.8) and widen (>=0.5)
        inputs = make_policy_inputs(p_toxic=Decimal("0.9"))

        output = engine.evaluate(ctx, inputs)

        # toxic_disable should be present
        assert "toxic_disable" in output.reason_codes
        # toxic_widen should NOT be present (SUPPRESS_ALL early returns)
        assert "toxic_widen" not in output.reason_codes


class TestPolicyContextProperties:
    """Test PolicyContext computed properties."""

    def test_total_pnl(self) -> None:
        """total_pnl is sum of realized and unrealized."""
        ctx = make_policy_context(
            realized_pnl=Decimal("50"),
            unrealized_pnl=Decimal("-20"),
        )
        assert ctx.total_pnl == Decimal("30")

    def test_book_age_ms(self) -> None:
        """book_age_ms is ts - last_book_ts."""
        ctx = make_policy_context(ts=1010000, last_book_ts=1000000)
        assert ctx.book_age_ms == 10000

    def test_inventory_ratio(self) -> None:
        """inventory_ratio is |position| / max_position."""
        ctx = make_policy_context(
            position_qty=Decimal("0.005"),
            max_position_qty=Decimal("0.01"),
        )
        assert ctx.inventory_ratio == Decimal("0.5")


class TestReasonCodesNoDigits:
    """Verify reason codes contain no digits per DEC-042."""

    @pytest.mark.parametrize(
        "scenario",
        [
            "low_inplay",
            "toxic_widen",
            "toxic_disable",
            "hard_limit",
            "stale_book",
            "kill",
        ],
    )
    def test_reason_codes_text_only(self, scenario: str) -> None:
        """All reason codes must be text-only (no digits)."""
        engine = PolicyEngine()

        if scenario == "low_inplay":
            ctx = make_policy_context()
            inputs = make_policy_inputs(p_inplay_2m=Decimal("0.2"))
        elif scenario == "toxic_widen":
            ctx = make_policy_context()
            inputs = make_policy_inputs(p_toxic=Decimal("0.6"))
        elif scenario == "toxic_disable":
            ctx = make_policy_context()
            inputs = make_policy_inputs(p_toxic=Decimal("0.9"))
        elif scenario == "hard_limit":
            ctx = make_policy_context(position_qty=Decimal("0.01"))
            inputs = make_policy_inputs()
        elif scenario == "stale_book":
            ctx = make_policy_context(ts=1010000, last_book_ts=1000000)
            inputs = make_policy_inputs()
        elif scenario == "kill":
            ctx = make_policy_context(realized_pnl=Decimal("-150"))
            inputs = make_policy_inputs()
        else:
            pytest.fail(f"Unknown scenario: {scenario}")

        output = engine.evaluate(ctx, inputs)

        for reason in output.reason_codes:
            assert not any(c.isdigit() for c in reason), f"Reason code '{reason}' contains digits"
