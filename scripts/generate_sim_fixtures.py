#!/usr/bin/env python3
"""Generate simulation test fixtures.

Creates 4 deterministic fixtures for trading simulator testing:
1. monotonic_up.jsonl - Steady price increase
2. mean_reverting_range.jsonl - Price oscillates in range
3. flash_crash.jsonl - Sudden large price drop
4. ws_gap.jsonl - Gap in market data (stale book)

Each fixture contains trade and book events in JSONL format
matching the MarketEvent schema.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


def generate_event(
    ts: int,
    event_type: str,
    symbol: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Generate a market event dict."""
    return {
        "ts": ts,
        "source": "sim_fixture",
        "symbol": symbol,
        "type": event_type,
        "payload": payload,
        "recv_ts": ts + 5,  # 5ms simulated latency
    }


def generate_trade(
    ts: int, symbol: str, price: Decimal, qty: Decimal, is_buyer_maker: bool
) -> dict[str, Any]:
    """Generate a trade event."""
    return generate_event(
        ts=ts,
        event_type="trade",
        symbol=symbol,
        payload={
            "price": str(price),
            "qty": str(qty),
            "m": is_buyer_maker,
        },
    )


def generate_book(ts: int, symbol: str, bid: Decimal, ask: Decimal) -> dict[str, Any]:
    """Generate a book update event."""
    return generate_event(
        ts=ts,
        event_type="book",
        symbol=symbol,
        payload={
            "bid": str(bid),
            "ask": str(ask),
            "bid_qty": "100.0",
            "ask_qty": "100.0",
        },
    )


def generate_monotonic_up(symbol: str = "BTCUSDT") -> list[dict[str, Any]]:
    """Generate monotonic upward price movement.

    Scenario: Price steadily increases from 42000 to 42500 over 60 seconds.
    Expected: Mean reversion strategy should have bounded loss (always buying
    into rising prices, selling at lower prices due to unfilled bids).
    """
    events = []
    base_ts = 1706140800000  # 2024-01-25 00:00:00 UTC
    base_price = Decimal("42000")

    # Generate events for 60 seconds (1 trade + 1 book per 500ms)
    for i in range(120):
        ts = base_ts + i * 500

        # Price increases by ~4.17 per event (500 total over 120 events)
        price_delta = Decimal(str(i * 4.17))
        price = base_price + price_delta

        # Spread of 2 USDT
        bid = price - Decimal("1")
        ask = price + Decimal("1")

        if i % 2 == 0:
            # Trade event - price going up means buyers aggressive
            events.append(generate_trade(ts, symbol, price, Decimal("0.1"), is_buyer_maker=False))
        else:
            # Book update
            events.append(generate_book(ts, symbol, bid, ask))

    return events


def generate_mean_reverting_range(symbol: str = "BTCUSDT") -> list[dict[str, Any]]:
    """Generate mean-reverting price in a range.

    Scenario: Price oscillates between 41800-42200 over 120 seconds.
    5 full oscillations = 10 crossings of midpoint.
    Expected: Market maker should profit with net_pnl > 0 and round_trips > 5.
    """
    import math

    events = []
    base_ts = 1706140800000
    mid_price = Decimal("42000")
    amplitude = Decimal("200")  # +/- 200 from mid

    # Generate events for 120 seconds
    for i in range(240):
        ts = base_ts + i * 500

        # Sine wave oscillation - 5 full cycles over 240 events
        phase = (i / 240) * 5 * 2 * math.pi
        price_offset = Decimal(str(amplitude * Decimal(str(math.sin(phase)))))
        price = mid_price + price_offset

        # Tighter spread for better fills
        spread = Decimal("0.5")
        bid = price - spread
        ask = price + spread

        if i % 2 == 0:
            # Trade event
            # Alternate buyer/seller maker based on price direction
            is_buyer_maker = math.sin(phase + 0.1) < math.sin(phase)
            events.append(generate_trade(ts, symbol, price, Decimal("0.05"), is_buyer_maker))
        else:
            events.append(generate_book(ts, symbol, bid, ask))

    return events


def generate_flash_crash(symbol: str = "BTCUSDT") -> list[dict[str, Any]]:
    """Generate flash crash scenario.

    Scenario: Price starts at 42000, then drops 5% instantly (to 39900).
    Expected: Kill switch should fire OR loss should be bounded.
    """
    events = []
    base_ts = 1706140800000
    pre_crash_price = Decimal("42000")
    post_crash_price = Decimal("39900")  # 5% drop

    # Normal trading for 30 seconds
    for i in range(60):
        ts = base_ts + i * 500
        # Small random walk before crash
        price_jitter = Decimal(str((i % 5) - 2))  # -2 to +2
        price = pre_crash_price + price_jitter
        bid = price - Decimal("1")
        ask = price + Decimal("1")

        if i % 2 == 0:
            events.append(generate_trade(ts, symbol, price, Decimal("0.05"), is_buyer_maker=True))
        else:
            events.append(generate_book(ts, symbol, bid, ask))

    # Flash crash at t+30s
    crash_ts = base_ts + 30000

    # Rapid price drop - 10 trades in 1 second going down
    for j in range(10):
        ts = crash_ts + j * 100
        # Linear interpolation from pre to post crash
        progress = Decimal(str(j / 10))
        price = pre_crash_price - (pre_crash_price - post_crash_price) * progress

        events.append(generate_trade(ts, symbol, price, Decimal("1.0"), is_buyer_maker=True))

    # Book update at bottom
    events.append(
        generate_book(crash_ts + 1000, symbol, post_crash_price - Decimal("50"), post_crash_price)
    )

    # Recovery trading for 30 seconds at crashed price
    for i in range(60):
        ts = crash_ts + 2000 + i * 500
        price = post_crash_price + Decimal(str(i * 0.5))  # Slow recovery
        bid = price - Decimal("10")
        ask = price + Decimal("10")

        if i % 2 == 0:
            events.append(generate_trade(ts, symbol, price, Decimal("0.1"), is_buyer_maker=False))
        else:
            events.append(generate_book(ts, symbol, bid, ask))

    return events


def generate_ws_gap(symbol: str = "BTCUSDT") -> list[dict[str, Any]]:
    """Generate websocket gap scenario (stale data).

    Scenario: Normal trading, then 10-second gap with no book updates,
    then recovery. During gap, quotes should be canceled and no fills
    should occur.
    Expected: Cancel quotes during gap, no fills in stale window.
    """
    events = []
    base_ts = 1706140800000
    price = Decimal("42000")

    # Normal trading for 20 seconds with regular book updates
    for i in range(40):
        ts = base_ts + i * 500
        bid = price - Decimal("1")
        ask = price + Decimal("1")

        if i % 2 == 0:
            events.append(generate_trade(ts, symbol, price, Decimal("0.05"), is_buyer_maker=True))
        else:
            events.append(generate_book(ts, symbol, bid, ask))

    # Gap starts at t+20s - only trades, no book updates for 10 seconds
    gap_start_ts = base_ts + 20000
    gap_price = price - Decimal("50")  # Price drops during gap

    # Trades during gap (but no book updates)
    for i in range(20):
        ts = gap_start_ts + i * 500
        trade_price = gap_price + Decimal(str(i * 2))  # Prices vary
        events.append(generate_trade(ts, symbol, trade_price, Decimal("0.1"), is_buyer_maker=True))

    # Gap ends at t+30s - book update resumes
    gap_end_ts = gap_start_ts + 10000

    # Recovery with book updates
    recovery_price = price - Decimal("30")
    for i in range(40):
        ts = gap_end_ts + i * 500
        bid = recovery_price - Decimal("1")
        ask = recovery_price + Decimal("1")

        if i % 2 == 0:
            events.append(
                generate_trade(ts, symbol, recovery_price, Decimal("0.05"), is_buyer_maker=False)
            )
        else:
            events.append(generate_book(ts, symbol, bid, ask))
            recovery_price += Decimal("0.5")  # Slow price increase

    return events


def write_jsonl(events: list[dict[str, Any]], filepath: Path) -> str:
    """Write events to JSONL file and return SHA256."""
    with open(filepath, "w") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    # Compute SHA256
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    """Generate all fixtures."""
    fixture_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "sim"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "monotonic_up.jsonl": generate_monotonic_up(),
        "mean_reverting_range.jsonl": generate_mean_reverting_range(),
        "flash_crash.jsonl": generate_flash_crash(),
        "ws_gap.jsonl": generate_ws_gap(),
    }

    checksums = {}
    for name, events in fixtures.items():
        filepath = fixture_dir / name
        sha256 = write_jsonl(events, filepath)
        checksums[name] = sha256
        print(f"Generated {name}: {len(events)} events, sha256={sha256[:16]}...")

    # Write manifest
    manifest = {
        "schema_version": "1.0.0",
        "created_at": "2026-01-30T12:00:00Z",
        "description": "Simulation test fixtures for trading simulator",
        "checksums": checksums,
        "fixtures": {
            "monotonic_up.jsonl": {
                "description": "Monotonic upward price movement - test bounded loss",
                "events": len(fixtures["monotonic_up.jsonl"]),
                "assertion": "bounded_loss_no_infinite_session",
            },
            "mean_reverting_range.jsonl": {
                "description": "Mean-reverting price oscillation - test profitable MM",
                "events": len(fixtures["mean_reverting_range.jsonl"]),
                "assertion": "net_pnl_positive_round_trips_gt_5",
            },
            "flash_crash.jsonl": {
                "description": "Flash crash scenario - test kill switch",
                "events": len(fixtures["flash_crash.jsonl"]),
                "assertion": "kill_switch_or_bounded_loss",
            },
            "ws_gap.jsonl": {
                "description": "Websocket gap scenario - test stale quote handling",
                "events": len(fixtures["ws_gap.jsonl"]),
                "assertion": "cancel_quotes_during_gap_no_stale_fills",
            },
        },
    }

    manifest_path = fixture_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nManifest written to {manifest_path}")
    print("\nChecksums:")
    for name, sha256 in checksums.items():
        print(f"  {name}: {sha256}")


if __name__ == "__main__":
    main()
