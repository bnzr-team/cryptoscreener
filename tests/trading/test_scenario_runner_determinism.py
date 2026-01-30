"""Scenario runner determinism tests.

Verifies:
- Two runs on same fixture produce identical combined_sha256
- Decisions are journaled correctly
- Strategy context is captured accurately
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cryptoscreener.trading.sim import ScenarioRunner, SimConfig
from cryptoscreener.trading.strategy import BaselineStrategy
from cryptoscreener.trading.strategy.baseline import BaselineStrategyConfig

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sim"


def load_fixture(name: str) -> list[dict[str, Any]]:
    """Load a sim fixture file."""
    filepath = FIXTURES_DIR / name
    events = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


class TestScenarioRunnerDeterminism:
    """Test that scenario runner produces deterministic output."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "monotonic_up.jsonl",
            "mean_reverting_range.jsonl",
            "flash_crash.jsonl",
            "ws_gap.jsonl",
        ],
    )
    def test_two_runs_same_combined_sha256(self, fixture_name: str) -> None:
        """Two runs on same fixture produce identical combined SHA256."""
        events = load_fixture(fixture_name)
        config = SimConfig(symbol="BTCUSDT")
        strategy = BaselineStrategy()

        # Run 1
        runner1 = ScenarioRunner(config, strategy)
        result1 = runner1.run(events)

        # Run 2
        runner2 = ScenarioRunner(config, strategy)
        result2 = runner2.run(events)

        # All hashes must match
        assert result1.combined_sha256 == result2.combined_sha256
        assert result1.decisions_sha256 == result2.decisions_sha256
        assert result1.artifacts_sha256 == result2.artifacts_sha256

        # All are 64-char hex
        assert len(result1.combined_sha256) == 64
        assert len(result1.decisions_sha256) == 64
        assert len(result1.artifacts_sha256) == 64

    def test_decisions_match_between_runs(self) -> None:
        """Decisions list is identical between runs."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")
        strategy = BaselineStrategy()

        runner1 = ScenarioRunner(config, strategy)
        result1 = runner1.run(events)

        runner2 = ScenarioRunner(config, strategy)
        result2 = runner2.run(events)

        assert len(result1.decisions) == len(result2.decisions)

        for d1, d2 in zip(result1.decisions, result2.decisions, strict=True):
            assert d1.ts == d2.ts
            assert d1.tick_seq == d2.tick_seq
            assert d1.bid == d2.bid
            assert d1.ask == d2.ask
            assert d1.position_qty == d2.position_qty
            assert len(d1.orders) == len(d2.orders)


class TestScenarioRunnerDecisions:
    """Test that decisions are captured correctly."""

    def test_decisions_have_sequential_tick_seq(self) -> None:
        """Decisions have sequential tick_seq starting from 0."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")
        strategy = BaselineStrategy()

        runner = ScenarioRunner(config, strategy)
        result = runner.run(events)

        for i, decision in enumerate(result.decisions):
            assert decision.tick_seq == i

    def test_decisions_capture_market_state(self) -> None:
        """Decisions capture bid/ask from market."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")
        strategy = BaselineStrategy()

        runner = ScenarioRunner(config, strategy)
        result = runner.run(events)

        # All decisions should have positive bid/ask (after book updates)
        has_market_data = False
        for decision in result.decisions:
            if decision.bid > 0 and decision.ask > 0:
                has_market_data = True
                # Mid should be average of bid and ask
                expected_mid = (decision.bid + decision.ask) / 2
                assert decision.mid == expected_mid

        assert has_market_data, "Should have some decisions with market data"

    def test_decisions_count_matches_events(self) -> None:
        """Number of decisions corresponds to processed events."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")
        strategy = BaselineStrategy()

        runner = ScenarioRunner(config, strategy)
        result = runner.run(events)

        # Should have one decision per event that triggers strategy
        # (trade or book events with valid quotes)
        assert len(result.decisions) > 0
        assert len(result.decisions) <= len(events)


class TestScenarioRunnerStrategy:
    """Test strategy integration with scenario runner."""

    def test_baseline_strategy_places_orders(self) -> None:
        """Baseline strategy places orders in mean-reverting market."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")
        strategy = BaselineStrategy(
            BaselineStrategyConfig(spread_bps=Decimal("1"))  # Tight spread
        )

        runner = ScenarioRunner(config, strategy)
        result = runner.run(events)

        # Should have some decisions with orders
        decisions_with_orders = [d for d in result.decisions if d.has_orders]
        assert len(decisions_with_orders) > 0

    def test_strategy_config_affects_orders(self) -> None:
        """Different strategy config produces different decisions."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")

        # Tight spread strategy
        strategy_tight = BaselineStrategy(
            BaselineStrategyConfig(spread_bps=Decimal("1"))
        )
        runner_tight = ScenarioRunner(config, strategy_tight)
        result_tight = runner_tight.run(events)

        # Wide spread strategy
        strategy_wide = BaselineStrategy(
            BaselineStrategyConfig(spread_bps=Decimal("100"))
        )
        runner_wide = ScenarioRunner(config, strategy_wide)
        result_wide = runner_wide.run(events)

        # Order prices should differ
        orders_tight = [
            o.price for d in result_tight.decisions for o in d.orders
        ]
        orders_wide = [
            o.price for d in result_wide.decisions for o in d.orders
        ]

        if orders_tight and orders_wide:
            # At least some prices should differ
            assert orders_tight != orders_wide


class TestScenarioRunnerArtifacts:
    """Test artifacts output from scenario runner."""

    def test_artifacts_match_simulator_output(self) -> None:
        """Artifacts from runner match direct simulator output."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")
        strategy = BaselineStrategy()

        runner = ScenarioRunner(config, strategy)
        result = runner.run(events)

        # Artifacts should have valid structure
        assert result.artifacts.sha256 == result.artifacts_sha256
        assert len(result.artifacts.session_states) > 0
        assert "net_pnl" in result.artifacts.metrics

    def test_empty_events_produces_valid_result(self) -> None:
        """Empty events produce valid (empty) result."""
        config = SimConfig(symbol="BTCUSDT")
        strategy = BaselineStrategy()

        runner = ScenarioRunner(config, strategy)
        result = runner.run([])

        # Should produce valid hashes
        assert len(result.combined_sha256) == 64
        assert len(result.decisions) == 0


class TestScenarioRunnerReplay:
    """Test replay verification capability."""

    def test_combined_hash_changes_with_strategy(self) -> None:
        """Combined hash changes when strategy behavior changes."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")

        # Strategy 1: small order qty
        strategy1 = BaselineStrategy(
            BaselineStrategyConfig(order_qty=Decimal("0.001"))
        )
        runner1 = ScenarioRunner(config, strategy1)
        result1 = runner1.run(events)

        # Strategy 2: larger order qty
        strategy2 = BaselineStrategy(
            BaselineStrategyConfig(order_qty=Decimal("0.01"))
        )
        runner2 = ScenarioRunner(config, strategy2)
        result2 = runner2.run(events)

        # Hashes should differ
        assert result1.combined_sha256 != result2.combined_sha256

    def test_combined_hash_changes_with_config(self) -> None:
        """Combined hash changes when sim config changes."""
        events = load_fixture("mean_reverting_range.jsonl")
        strategy = BaselineStrategy()

        # Config 1: low fee
        config1 = SimConfig(symbol="BTCUSDT", maker_fee_frac=Decimal("0.0001"))
        runner1 = ScenarioRunner(config1, strategy)
        result1 = runner1.run(events)

        # Config 2: higher fee
        config2 = SimConfig(symbol="BTCUSDT", maker_fee_frac=Decimal("0.001"))
        runner2 = ScenarioRunner(config2, strategy)
        result2 = runner2.run(events)

        # Hashes should differ (different PnL from fees)
        assert result1.artifacts_sha256 != result2.artifacts_sha256
