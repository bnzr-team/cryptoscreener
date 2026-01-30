"""Policy strategy determinism and A/B comparison tests.

Verifies:
- PolicyEngineStrategy produces deterministic output (same SHA256)
- BaselineStrategy produces deterministic output (same SHA256)
- Baseline vs Policy produces DIFFERENT digests (policy modifies behavior)
- Reason codes contain no digits (DEC-042 compliance)

DEC-044: PolicyEngine MVP tests.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cryptoscreener.trading.policy import PolicyParams
from cryptoscreener.trading.policy.providers import FixturePolicyInputsProvider
from cryptoscreener.trading.sim import ScenarioRunner, SimConfig
from cryptoscreener.trading.strategy import BaselineStrategy, PolicyEngineStrategy
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


class TestPolicyStrategyDeterminism:
    """Test that PolicyEngineStrategy produces deterministic output."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "monotonic_up.jsonl",
            "mean_reverting_range.jsonl",
            "flash_crash.jsonl",
            "ws_gap.jsonl",
        ],
    )
    def test_policy_two_runs_same_sha256(self, fixture_name: str) -> None:
        """Two runs with PolicyEngineStrategy produce identical SHA256."""
        events = load_fixture(fixture_name)
        config = SimConfig(symbol="BTCUSDT")
        base_config = BaselineStrategyConfig()
        policy_params = PolicyParams()

        # Run 1
        provider1 = FixturePolicyInputsProvider(fixture_name)
        strategy1 = PolicyEngineStrategy(
            provider1, base_config=base_config, policy_params=policy_params
        )
        runner1 = ScenarioRunner(config, strategy1)
        result1 = runner1.run(events)

        # Run 2
        provider2 = FixturePolicyInputsProvider(fixture_name)
        strategy2 = PolicyEngineStrategy(
            provider2, base_config=base_config, policy_params=policy_params
        )
        runner2 = ScenarioRunner(config, strategy2)
        result2 = runner2.run(events)

        # All hashes must match
        assert result1.combined_sha256 == result2.combined_sha256, (
            f"combined_sha256 mismatch: {result1.combined_sha256} != {result2.combined_sha256}"
        )
        assert result1.decisions_sha256 == result2.decisions_sha256
        assert result1.artifacts_sha256 == result2.artifacts_sha256

        # All are 64-char hex
        assert len(result1.combined_sha256) == 64
        assert len(result1.decisions_sha256) == 64
        assert len(result1.artifacts_sha256) == 64

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "monotonic_up.jsonl",
            "mean_reverting_range.jsonl",
            "flash_crash.jsonl",
            "ws_gap.jsonl",
        ],
    )
    def test_baseline_two_runs_same_sha256(self, fixture_name: str) -> None:
        """Two runs with BaselineStrategy produce identical SHA256."""
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


class TestPolicyVsBaselineComparison:
    """Test that Policy and Baseline produce different outputs."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            # These fixtures have policy inputs that trigger rules:
            # - monotonic_up: p_inplay=0.3 < 0.4 -> SUPPRESS_ENTRY
            # - flash_crash: p_toxic=0.8 >= 0.8 -> SUPPRESS_ALL
            "monotonic_up.jsonl",
            "flash_crash.jsonl",
        ],
    )
    def test_policy_vs_baseline_different_digests(self, fixture_name: str) -> None:
        """Policy and Baseline strategies produce different digests.

        This verifies the policy actually modifies behavior.
        Only tests fixtures where constant ML inputs trigger rules.
        """
        events = load_fixture(fixture_name)
        config = SimConfig(symbol="BTCUSDT")
        base_config = BaselineStrategyConfig()
        policy_params = PolicyParams()

        # Run with Baseline
        baseline_strategy = BaselineStrategy(base_config)
        runner_baseline = ScenarioRunner(config, baseline_strategy)
        result_baseline = runner_baseline.run(events)

        # Run with Policy
        provider = FixturePolicyInputsProvider(fixture_name)
        policy_strategy = PolicyEngineStrategy(
            provider, base_config=base_config, policy_params=policy_params
        )
        runner_policy = ScenarioRunner(config, policy_strategy)
        result_policy = runner_policy.run(events)

        # Digests SHOULD differ (policy modifies behavior)
        # At least decisions_sha256 should differ due to filtered orders
        assert result_baseline.combined_sha256 != result_policy.combined_sha256, (
            f"Expected different digests for {fixture_name}: "
            f"baseline={result_baseline.combined_sha256}, "
            f"policy={result_policy.combined_sha256}"
        )

    @pytest.mark.parametrize(
        "fixture_name",
        [
            # These fixtures have neutral policy inputs (no rules trigger):
            # - mean_reverting_range: p_inplay=0.8 > 0.4, p_toxic=0.1 < 0.5
            # - ws_gap: p_inplay=0.5 > 0.4, p_toxic=0.2 < 0.5
            # Staleness detection depends on actual time gaps in fixture data.
            "mean_reverting_range.jsonl",
            "ws_gap.jsonl",
        ],
    )
    def test_neutral_policy_same_as_baseline(self, fixture_name: str) -> None:
        """Neutral policy inputs produce same behavior as baseline.

        For fixtures with normal market conditions (high in-play, low toxic),
        PolicyEngineStrategy should behave identically to BaselineStrategy.
        """
        events = load_fixture(fixture_name)
        config = SimConfig(symbol="BTCUSDT")
        base_config = BaselineStrategyConfig()
        policy_params = PolicyParams()

        # Run with Baseline
        baseline_strategy = BaselineStrategy(base_config)
        runner_baseline = ScenarioRunner(config, baseline_strategy)
        result_baseline = runner_baseline.run(events)

        # Run with Policy
        provider = FixturePolicyInputsProvider(fixture_name)
        policy_strategy = PolicyEngineStrategy(
            provider, base_config=base_config, policy_params=policy_params
        )
        runner_policy = ScenarioRunner(config, policy_strategy)
        result_policy = runner_policy.run(events)

        # With neutral inputs, digests should match (no policy rules trigger)
        assert result_baseline.combined_sha256 == result_policy.combined_sha256, (
            f"Expected same digests for neutral fixture {fixture_name}"
        )


class TestReasonCodesNoDigits:
    """Test that all reason codes contain no digits (DEC-042)."""

    DIGIT_PATTERN = re.compile(r"\d")

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "monotonic_up.jsonl",
            "mean_reverting_range.jsonl",
            "flash_crash.jsonl",
            "ws_gap.jsonl",
        ],
    )
    def test_policy_reason_codes_no_digits(self, fixture_name: str) -> None:
        """All reason codes from PolicyEngineStrategy contain no digits."""
        events = load_fixture(fixture_name)
        config = SimConfig(symbol="BTCUSDT")
        base_config = BaselineStrategyConfig()
        policy_params = PolicyParams()

        provider = FixturePolicyInputsProvider(fixture_name)
        strategy = PolicyEngineStrategy(
            provider, base_config=base_config, policy_params=policy_params
        )
        runner = ScenarioRunner(config, strategy)
        result = runner.run(events)

        # Check all reason codes in decisions
        for decision in result.decisions:
            for order in decision.orders:
                reason = order.reason
                assert not self.DIGIT_PATTERN.search(reason), (
                    f"Reason code contains digits: '{reason}' "
                    f"in fixture {fixture_name}, tick {decision.tick_seq}"
                )


class TestPolicyActivation:
    """Test that specific policies are activated on expected fixtures."""

    def test_monotonic_up_suppresses_entry(self) -> None:
        """monotonic_up fixture triggers SUPPRESS_ENTRY (POL-002 low in-play)."""
        events = load_fixture("monotonic_up.jsonl")
        config = SimConfig(symbol="BTCUSDT")
        base_config = BaselineStrategyConfig()
        policy_params = PolicyParams()

        # Run with Baseline to get expected order count
        baseline_strategy = BaselineStrategy(base_config)
        runner_baseline = ScenarioRunner(config, baseline_strategy)
        result_baseline = runner_baseline.run(events)
        baseline_orders = sum(len(d.orders) for d in result_baseline.decisions)

        # Run with Policy
        provider = FixturePolicyInputsProvider("monotonic_up.jsonl")
        policy_strategy = PolicyEngineStrategy(
            provider, base_config=base_config, policy_params=policy_params
        )
        runner_policy = ScenarioRunner(config, policy_strategy)
        result_policy = runner_policy.run(events)
        policy_orders = sum(len(d.orders) for d in result_policy.decisions)

        # Policy should have fewer orders due to SUPPRESS_ENTRY
        # (monotonic_up has p_inplay=0.3 < inplay_exit_prob=0.4)
        assert policy_orders <= baseline_orders, (
            f"Expected policy to suppress entry orders: "
            f"baseline={baseline_orders}, policy={policy_orders}"
        )

    def test_flash_crash_widens_or_suppresses(self) -> None:
        """flash_crash fixture triggers toxic behavior (POL-004/005)."""
        events = load_fixture("flash_crash.jsonl")
        config = SimConfig(symbol="BTCUSDT")
        base_config = BaselineStrategyConfig(spread_bps=Decimal("10"))
        policy_params = PolicyParams()

        # Run with Policy
        provider = FixturePolicyInputsProvider("flash_crash.jsonl")
        policy_strategy = PolicyEngineStrategy(
            provider, base_config=base_config, policy_params=policy_params
        )
        runner_policy = ScenarioRunner(config, policy_strategy)
        result_policy = runner_policy.run(events)

        # flash_crash has p_toxic=0.8 >= toxicity_disable_threshold
        # So SUPPRESS_ALL should be active, meaning few or no orders
        policy_orders = sum(len(d.orders) for d in result_policy.decisions)

        # With toxic_disable_threshold=0.8 and p_toxic=0.8, should suppress all
        assert policy_orders == 0, (
            f"Expected flash_crash to suppress all orders due to high toxicity: "
            f"got {policy_orders} orders"
        )

    def test_ws_gap_neutral_policy_behavior(self) -> None:
        """ws_gap fixture with neutral policy inputs matches baseline.

        Note: ws_gap fixture has neutral policy inputs (p_inplay=0.5, p_toxic=0.2).
        Staleness detection (POL-013) requires actual time gaps exceeding stale_quote_ms
        in the book timestamps, which the constant stub inputs don't provide.
        With neutral inputs and no triggered rules, behavior matches baseline.
        """
        events = load_fixture("ws_gap.jsonl")
        config = SimConfig(symbol="BTCUSDT", stale_quote_ms=5000)
        base_config = BaselineStrategyConfig()
        policy_params = PolicyParams(stale_quote_ms=5000)

        # Run with Baseline
        baseline_strategy = BaselineStrategy(base_config)
        runner_baseline = ScenarioRunner(config, baseline_strategy)
        result_baseline = runner_baseline.run(events)

        # Run with Policy
        provider = FixturePolicyInputsProvider("ws_gap.jsonl")
        policy_strategy = PolicyEngineStrategy(
            provider, base_config=base_config, policy_params=policy_params
        )
        runner_policy = ScenarioRunner(config, policy_strategy)
        result_policy = runner_policy.run(events)

        # With neutral ML inputs and no staleness in fixture data,
        # policy behaves identically to baseline
        assert result_baseline.combined_sha256 == result_policy.combined_sha256


class TestPolicyMetrics:
    """Test that policy runs produce valid metrics."""

    def test_mean_reverting_produces_metrics(self) -> None:
        """Mean reverting scenario produces standard metrics."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")
        base_config = BaselineStrategyConfig()
        policy_params = PolicyParams()

        provider = FixturePolicyInputsProvider("mean_reverting_range.jsonl")
        strategy = PolicyEngineStrategy(
            provider, base_config=base_config, policy_params=policy_params
        )
        runner = ScenarioRunner(config, strategy)
        result = runner.run(events)

        # Check required metrics exist (per DEC-041 SimArtifacts)
        metrics = result.artifacts.metrics
        assert "net_pnl" in metrics
        assert "total_commissions" in metrics
        assert "total_fills" in metrics
        assert "max_drawdown" in metrics
        assert "round_trips" in metrics
        assert "win_rate" in metrics

    def test_flash_crash_bounded_loss(self) -> None:
        """Flash crash scenario has bounded loss."""
        events = load_fixture("flash_crash.jsonl")
        max_loss = Decimal("100")
        config = SimConfig(symbol="BTCUSDT", max_session_loss=max_loss)
        base_config = BaselineStrategyConfig()
        policy_params = PolicyParams(max_session_loss=max_loss)

        provider = FixturePolicyInputsProvider("flash_crash.jsonl")
        strategy = PolicyEngineStrategy(
            provider, base_config=base_config, policy_params=policy_params
        )
        runner = ScenarioRunner(config, strategy)
        result = runner.run(events)

        # Loss should be bounded
        net_pnl = Decimal(str(result.artifacts.metrics["net_pnl"]))
        assert net_pnl >= -max_loss, (
            f"Loss exceeded max_session_loss: net_pnl={net_pnl}, max_loss={max_loss}"
        )
