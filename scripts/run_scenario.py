#!/usr/bin/env python3
"""Run trading scenario with strategy plugin.

Usage:
    python scripts/run_scenario.py --events tests/fixtures/sim/mean_reverting_range.jsonl --symbol BTCUSDT

Outputs:
    decisions.jsonl - Strategy decisions per tick
    sim_artifacts.json - Full simulation artifacts
    scenario_digest.txt - Combined SHA256 digest

See DEC-042 for design rationale.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path


def main() -> int:
    """Run scenario."""
    parser = argparse.ArgumentParser(description="Run trading scenario with strategy")
    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help="Path to market events JSONL file",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--maker-fee",
        type=str,
        default="0.0002",
        help="Maker fee fraction (default: 0.0002 = 2bps)",
    )
    parser.add_argument(
        "--taker-fee",
        type=str,
        default="0.0004",
        help="Taker fee fraction (default: 0.0004 = 4bps)",
    )
    parser.add_argument(
        "--max-loss",
        type=str,
        default="100",
        help="Maximum session loss before kill switch (default: 100 USD)",
    )
    parser.add_argument(
        "--stale-ms",
        type=int,
        default=5000,
        help="Stale quote threshold in milliseconds (default: 5000)",
    )
    parser.add_argument(
        "--spread-bps",
        type=str,
        default="10",
        help="Strategy spread in basis points (default: 10)",
    )
    parser.add_argument(
        "--order-qty",
        type=str,
        default="0.001",
        help="Strategy order quantity (default: 0.001)",
    )
    parser.add_argument(
        "--max-position",
        type=str,
        default="0.01",
        help="Maximum position size (default: 0.01)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["baseline", "policy"],
        default="baseline",
        help="Strategy type: baseline (simple MM) or policy (with policy engine)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.events.exists():
        print(f"ERROR: Events file not found: {args.events}")
        return 1

    # Set output directory
    out_dir = args.out or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load events
    print(f"Loading events from {args.events}...")
    events = []
    with open(args.events) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    print(f"  Loaded {len(events)} events")

    # Import after arg parsing to fail fast on bad args
    from cryptoscreener.trading.policy import PolicyParams
    from cryptoscreener.trading.policy.providers import FixturePolicyInputsProvider
    from cryptoscreener.trading.sim import ScenarioRunner, SimConfig, write_scenario_outputs
    from cryptoscreener.trading.strategy import BaselineStrategy, PolicyEngineStrategy
    from cryptoscreener.trading.strategy.baseline import BaselineStrategyConfig

    # Create config
    config = SimConfig(
        symbol=args.symbol,
        maker_fee_frac=Decimal(args.maker_fee),
        taker_fee_frac=Decimal(args.taker_fee),
        max_session_loss=Decimal(args.max_loss),
        stale_quote_ms=args.stale_ms,
    )
    print(
        f"Config: symbol={config.symbol}, maker_fee={config.maker_fee_frac}, "
        f"max_loss={config.max_session_loss}"
    )

    # Create strategy
    strategy_config = BaselineStrategyConfig(
        spread_bps=Decimal(args.spread_bps),
        order_qty=Decimal(args.order_qty),
        max_position=Decimal(args.max_position),
    )

    strategy: BaselineStrategy | PolicyEngineStrategy
    if args.strategy == "baseline":
        strategy = BaselineStrategy(strategy_config)
        print(
            f"Strategy: baseline, spread_bps={strategy_config.spread_bps}, "
            f"order_qty={strategy_config.order_qty}"
        )
    else:
        # Policy strategy with fixture-based inputs
        fixture_name = args.events.stem  # e.g., "mean_reverting_range"
        inputs_provider = FixturePolicyInputsProvider(fixture_name)
        policy_params = PolicyParams(
            max_session_loss=Decimal(args.max_loss),
            stale_quote_ms=args.stale_ms,
        )
        strategy = PolicyEngineStrategy(
            inputs_provider,
            base_config=strategy_config,
            policy_params=policy_params,
        )
        print(
            f"Strategy: policy (fixture={fixture_name}), "
            f"spread_bps={strategy_config.spread_bps}, "
            f"order_qty={strategy_config.order_qty}"
        )

    # Run scenario
    print("Running scenario...")
    runner = ScenarioRunner(config, strategy)
    result = runner.run(events)

    # Write outputs
    decisions_path, artifacts_path = write_scenario_outputs(result, out_dir)
    print(f"  Decisions written to {decisions_path}")
    print(f"  Artifacts written to {artifacts_path}")

    # Write combined digest
    digest_path = out_dir / "scenario_digest.txt"
    with open(digest_path, "w") as f:
        f.write(f"decisions_sha256={result.decisions_sha256}\n")
        f.write(f"artifacts_sha256={result.artifacts_sha256}\n")
        f.write(f"combined_sha256={result.combined_sha256}\n")
    print(f"  Digest written to {digest_path}")

    # Print summary
    metrics = result.artifacts.metrics
    print("\n=== SCENARIO RESULTS ===")
    print(f"  Combined SHA256: {result.combined_sha256}")
    print(f"  Decisions SHA256: {result.decisions_sha256}")
    print(f"  Artifacts SHA256: {result.artifacts_sha256}")
    print(f"  Total Decisions: {len(result.decisions)}")
    print(f"  Decisions with Orders: {sum(1 for d in result.decisions if d.has_orders)}")
    print(f"  Net PnL: ${metrics['net_pnl']}")
    print(f"  Total Commissions: ${metrics['total_commissions']}")
    print(f"  Max Drawdown: ${metrics['max_drawdown']}")
    print(f"  Total Fills: {metrics['total_fills']}")
    print(f"  Round Trips: {metrics['round_trips']}")
    print(f"  Win Rate: {float(metrics['win_rate']):.1%}")
    print(f"  Final State: {result.artifacts.session_states[-1]['state']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
