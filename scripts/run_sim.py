#!/usr/bin/env python3
"""Run trading simulator on market events.

Usage:
    python scripts/run_sim.py --events tests/fixtures/sim/mean_reverting_range.jsonl --symbol BTCUSDT

Outputs:
    sim_artifacts.json - Full simulation artifacts
    sha256.txt - SHA256 digest of artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path


def main() -> int:
    """Run simulator."""
    parser = argparse.ArgumentParser(description="Run trading simulator")
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

    # Import simulator (after arg parsing to fail fast on bad args)
    from cryptoscreener.trading.sim import SimConfig, Simulator
    from cryptoscreener.trading.sim.artifacts import dump_artifacts_json

    # Create config
    config = SimConfig(
        symbol=args.symbol,
        maker_fee_frac=Decimal(args.maker_fee),
        taker_fee_frac=Decimal(args.taker_fee),
        max_session_loss=Decimal(args.max_loss),
        stale_quote_ms=args.stale_ms,
    )
    print(
        f"Config: symbol={config.symbol}, maker_fee={config.maker_fee_frac}, max_loss={config.max_session_loss}"
    )

    # Run simulation
    print("Running simulation...")
    sim = Simulator(config)
    artifacts = sim.run(events)

    # Write artifacts
    artifacts_path = out_dir / "sim_artifacts.json"
    with open(artifacts_path, "wb") as f:
        f.write(dump_artifacts_json(artifacts))
    print(f"  Artifacts written to {artifacts_path}")

    # Write SHA256
    sha256_path = out_dir / "sha256.txt"
    with open(sha256_path, "w") as f:
        f.write(f"{artifacts.sha256}\n")
    print(f"  SHA256 written to {sha256_path}")

    # Print summary
    metrics = artifacts.metrics
    print("\n=== SIMULATION RESULTS ===")
    print(f"  SHA256: {artifacts.sha256}")
    print(f"  Net PnL: ${metrics['net_pnl']}")
    print(f"  Total Commissions: ${metrics['total_commissions']}")
    print(f"  Max Drawdown: ${metrics['max_drawdown']}")
    print(f"  Total Fills: {metrics['total_fills']}")
    print(f"  Round Trips: {metrics['round_trips']}")
    print(f"  Win Rate: {float(metrics['win_rate']):.1%}")
    print(f"  Max Position: {metrics['max_position']}")
    print(f"  Session Duration: {float(metrics['avg_session_duration_min']):.2f} min")
    print(f"  Final State: {artifacts.session_states[-1]['state']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
