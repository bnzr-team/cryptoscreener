#!/usr/bin/env python3
"""
Build labels for ML training from market data.

Generates ground truth labels per LABELS_SPEC.md:
- I_tradeable(H) for horizons 30s, 2m, 5m and profiles A, B
- Toxicity labels (y_toxic)

Usage:
    # From JSONL market events file
    python -m scripts.build_labels --input data/market_events.jsonl --output data/labels.parquet

    # With custom config
    python -m scripts.build_labels --input data/events.jsonl --output labels.parquet \\
        --spread-max-bps 15 --x-bps-30s-a 5

Output:
    Parquet file with columns:
    - ts, symbol, mid_price, spread_bps
    - y_toxic, severity_toxic_bps
    - i_tradeable_30s_a, i_tradeable_30s_b, ... (for each horizon/profile)
    - mfe_bps_30s_a, cost_bps_30s_a, net_edge_bps_30s_a, ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from collections.abc import Iterator

from cryptoscreener.contracts import MarketEvent, MarketEventType
from cryptoscreener.cost_model import CostModelConfig
from cryptoscreener.label_builder import (
    LabelBuilder,
    LabelBuilderConfig,
    LabelRow,
    ToxicityConfig,
)
from cryptoscreener.label_builder.builder import PricePoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default lookahead window (must cover max horizon + toxicity tau)
DEFAULT_LOOKAHEAD_MS = 360_000  # 6 minutes


def load_market_events(filepath: Path) -> list[MarketEvent]:
    """Load market events from JSONL file.

    Args:
        filepath: Path to JSONL file.

    Returns:
        List of MarketEvent objects sorted by timestamp.
    """
    events: list[MarketEvent] = []

    with filepath.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = MarketEvent.from_json(line)
                events.append(event)
            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")

    # Sort by timestamp
    events.sort(key=lambda e: e.ts)
    logger.info(f"Loaded {len(events)} market events")
    return events


def extract_book_prices(events: list[MarketEvent], symbol: str) -> dict[int, tuple[float, float]]:
    """Extract bid/ask prices from book events for a symbol.

    Args:
        events: List of market events.
        symbol: Symbol to filter.

    Returns:
        Dict mapping timestamp to (bid, ask) tuple.
    """
    prices: dict[int, tuple[float, float]] = {}

    for event in events:
        if event.symbol != symbol or event.type != MarketEventType.BOOK:
            continue

        payload = event.payload
        bid = float(payload.get("bid", 0))
        ask = float(payload.get("ask", 0))

        if bid > 0 and ask > 0:
            prices[event.ts] = (bid, ask)

    return prices


def extract_trade_prices(events: list[MarketEvent], symbol: str) -> list[PricePoint]:
    """Extract mid prices from trade events for MFE/MAE calculation.

    Args:
        events: List of market events.
        symbol: Symbol to filter.

    Returns:
        List of PricePoint objects sorted by timestamp.
    """
    prices: list[PricePoint] = []

    for event in events:
        if event.symbol != symbol or event.type != MarketEventType.TRADE:
            continue

        payload = event.payload
        price = float(payload.get("price", 0))

        if price > 0:
            prices.append(PricePoint(ts=event.ts, mid=price))

    # Sort by timestamp
    prices.sort(key=lambda p: p.ts)
    return prices


def get_future_prices(
    all_prices: list[PricePoint],
    start_ts: int,
    lookahead_ms: int,
) -> list[PricePoint]:
    """Get future prices within lookahead window.

    Args:
        all_prices: All price points (sorted by ts).
        start_ts: Starting timestamp.
        lookahead_ms: Lookahead window in milliseconds.

    Returns:
        List of price points within [start_ts, start_ts + lookahead_ms].
    """
    end_ts = start_ts + lookahead_ms
    result: list[PricePoint] = []

    for p in all_prices:
        if p.ts < start_ts:
            continue
        if p.ts > end_ts:
            break
        result.append(p)

    return result


def build_labels_for_symbol(
    events: list[MarketEvent],
    symbol: str,
    builder: LabelBuilder,
    sample_interval_ms: int = 1000,
    lookahead_ms: int = DEFAULT_LOOKAHEAD_MS,
) -> Iterator[LabelRow]:
    """Build labels for a single symbol.

    Args:
        events: All market events.
        symbol: Symbol to process.
        builder: LabelBuilder instance.
        sample_interval_ms: Interval between label samples.
        lookahead_ms: Lookahead window for MFE/toxicity.

    Yields:
        LabelRow objects.
    """
    # Extract prices
    book_prices = extract_book_prices(events, symbol)
    trade_prices = extract_trade_prices(events, symbol)

    if not book_prices or not trade_prices:
        logger.warning(f"No data for {symbol}")
        return

    # Get timestamp range
    min_ts = min(book_prices.keys())
    max_ts = max(book_prices.keys())

    # We can only label up to max_ts - lookahead_ms (need future data)
    label_end_ts = max_ts - lookahead_ms

    if label_end_ts <= min_ts:
        logger.warning(f"Insufficient data for {symbol}: need {lookahead_ms}ms lookahead")
        return

    logger.info(f"Building labels for {symbol}: {min_ts} to {label_end_ts}")

    # Sample at regular intervals
    current_bid = 0.0
    current_ask = 0.0
    last_sample_ts = min_ts - sample_interval_ms

    for ts in sorted(book_prices.keys()):
        if ts > label_end_ts:
            break

        # Update current bid/ask
        current_bid, current_ask = book_prices[ts]

        # Skip if not at sample interval
        if ts - last_sample_ts < sample_interval_ms:
            continue

        last_sample_ts = ts

        # Get future prices for MFE/toxicity
        future_prices = get_future_prices(trade_prices, ts, lookahead_ms)

        if not future_prices:
            continue

        # Build label row
        row = builder.build_label_row(
            ts=ts,
            symbol=symbol,
            bid=current_bid,
            ask=current_ask,
            future_prices=future_prices,
            orderbook=None,  # No depth data in simple mode
            usd_volume_60s=0.0,  # Would need volume tracking
            style="scalping",
        )

        yield row


def build_all_labels(
    events: list[MarketEvent],
    builder: LabelBuilder,
    sample_interval_ms: int = 1000,
    lookahead_ms: int = DEFAULT_LOOKAHEAD_MS,
) -> list[dict]:
    """Build labels for all symbols in the dataset.

    Args:
        events: All market events.
        builder: LabelBuilder instance.
        sample_interval_ms: Interval between label samples.
        lookahead_ms: Lookahead window for MFE/toxicity.

    Returns:
        List of flat dicts (ready for DataFrame).
    """
    # Group events by symbol
    symbols: set[str] = set()
    for event in events:
        symbols.add(event.symbol)

    logger.info(f"Found {len(symbols)} symbols: {sorted(symbols)}")

    all_rows: list[dict] = []

    for symbol in sorted(symbols):
        count = 0
        for row in build_labels_for_symbol(
            events, symbol, builder, sample_interval_ms, lookahead_ms
        ):
            flat = builder.label_row_to_flat_dict(row)
            all_rows.append(flat)
            count += 1

        logger.info(f"Generated {count} labels for {symbol}")

    return all_rows


def save_to_parquet(rows: list[dict], output_path: Path) -> None:
    """Save label rows to parquet file.

    Args:
        rows: List of flat dicts.
        output_path: Output file path.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow not installed. Install with: pip install pyarrow")
        raise

    if not rows:
        logger.warning("No rows to save")
        return

    # Create table from rows
    table = pa.Table.from_pylist(rows)

    # Write to parquet
    pq.write_table(table, output_path, compression="snappy")
    logger.info(f"Saved {len(rows)} rows to {output_path}")


def save_to_jsonl(rows: list[dict], output_path: Path) -> None:
    """Save label rows to JSONL file.

    Args:
        rows: List of flat dicts.
        output_path: Output file path.
    """
    with output_path.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row))
            f.write(b"\n")

    logger.info(f"Saved {len(rows)} rows to {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build ML training labels from market data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with market events",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file (parquet or jsonl based on extension)",
    )

    # Sampling
    parser.add_argument(
        "--sample-interval-ms",
        type=int,
        default=1000,
        help="Interval between label samples in ms (default: 1000)",
    )
    parser.add_argument(
        "--lookahead-ms",
        type=int,
        default=DEFAULT_LOOKAHEAD_MS,
        help=f"Lookahead window for MFE/toxicity in ms (default: {DEFAULT_LOOKAHEAD_MS})",
    )

    # Gate thresholds
    parser.add_argument(
        "--spread-max-bps",
        type=float,
        default=10.0,
        help="Maximum spread gate threshold in bps (default: 10.0)",
    )
    parser.add_argument(
        "--impact-max-bps",
        type=float,
        default=20.0,
        help="Maximum impact gate threshold in bps (default: 20.0)",
    )

    # Net edge thresholds
    parser.add_argument(
        "--x-bps-30s-a",
        type=float,
        default=5.0,
        help="Min net edge for 30s Profile A (default: 5.0)",
    )
    parser.add_argument(
        "--x-bps-30s-b",
        type=float,
        default=8.0,
        help="Min net edge for 30s Profile B (default: 8.0)",
    )
    parser.add_argument(
        "--x-bps-2m-a",
        type=float,
        default=10.0,
        help="Min net edge for 2m Profile A (default: 10.0)",
    )
    parser.add_argument(
        "--x-bps-2m-b",
        type=float,
        default=15.0,
        help="Min net edge for 2m Profile B (default: 15.0)",
    )
    parser.add_argument(
        "--x-bps-5m-a",
        type=float,
        default=15.0,
        help="Min net edge for 5m Profile A (default: 15.0)",
    )
    parser.add_argument(
        "--x-bps-5m-b",
        type=float,
        default=20.0,
        help="Min net edge for 5m Profile B (default: 20.0)",
    )

    # Toxicity
    parser.add_argument(
        "--toxicity-tau-ms",
        type=int,
        default=30000,
        help="Toxicity check window in ms (default: 30000)",
    )
    parser.add_argument(
        "--toxicity-threshold-bps",
        type=float,
        default=10.0,
        help="Toxicity adverse movement threshold in bps (default: 10.0)",
    )

    # Fees
    parser.add_argument(
        "--fees-bps-a",
        type=float,
        default=2.0,
        help="Fees in bps for Profile A (default: 2.0)",
    )
    parser.add_argument(
        "--fees-bps-b",
        type=float,
        default=4.0,
        help="Fees in bps for Profile B (default: 4.0)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Build config
    cost_config = CostModelConfig(
        fees_bps_a=args.fees_bps_a,
        fees_bps_b=args.fees_bps_b,
    )

    toxicity_config = ToxicityConfig(
        tau_ms=args.toxicity_tau_ms,
        threshold_bps=args.toxicity_threshold_bps,
    )

    label_config = LabelBuilderConfig(
        x_bps_30s_a=args.x_bps_30s_a,
        x_bps_30s_b=args.x_bps_30s_b,
        x_bps_2m_a=args.x_bps_2m_a,
        x_bps_2m_b=args.x_bps_2m_b,
        x_bps_5m_a=args.x_bps_5m_a,
        x_bps_5m_b=args.x_bps_5m_b,
        spread_max_bps=args.spread_max_bps,
        impact_max_bps=args.impact_max_bps,
        toxicity=toxicity_config,
        cost_model=cost_config,
    )

    builder = LabelBuilder(label_config)

    # Load events
    logger.info(f"Loading events from {args.input}")
    events = load_market_events(args.input)

    if not events:
        logger.error("No events loaded")
        return 1

    # Build labels
    logger.info("Building labels...")
    rows = build_all_labels(
        events,
        builder,
        sample_interval_ms=args.sample_interval_ms,
        lookahead_ms=args.lookahead_ms,
    )

    if not rows:
        logger.error("No labels generated")
        return 1

    # Save output
    output_ext = args.output.suffix.lower()
    if output_ext == ".parquet":
        save_to_parquet(rows, args.output)
    elif output_ext in (".jsonl", ".json"):
        save_to_jsonl(rows, args.output)
    else:
        logger.error(f"Unknown output format: {output_ext}. Use .parquet or .jsonl")
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("LABEL BUILD SUMMARY")
    print("=" * 60)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Total rows:   {len(rows)}")
    print(f"Symbols:      {len({r['symbol'] for r in rows})}")

    # Count tradeability
    tradeable_counts: dict[str, int] = {}
    for row in rows:
        for horizon in ["30s", "2m", "5m"]:
            for profile in ["a", "b"]:
                key = f"i_tradeable_{horizon}_{profile}"
                if row.get(key, 0) == 1:
                    tradeable_counts[key] = tradeable_counts.get(key, 0) + 1

    print("\nTradeability counts:")
    for key, count in sorted(tradeable_counts.items()):
        pct = count / len(rows) * 100
        print(f"  {key}: {count} ({pct:.1f}%)")

    toxic_count = sum(1 for r in rows if r.get("y_toxic", 0) == 1)
    toxic_pct = toxic_count / len(rows) * 100
    print(f"\nToxicity:     {toxic_count} ({toxic_pct:.1f}%)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
