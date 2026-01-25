"""Deterministic FeatureSnapshot fixtures for replay testing.

These fixtures use fixed timestamps and feature values to ensure
reproducible pipeline outputs across runs.
"""

from __future__ import annotations

from cryptoscreener.contracts.events import (
    DataHealth,
    Features,
    FeatureSnapshot,
    RegimeTrend,
    RegimeVol,
)


def make_feature_snapshot(
    symbol: str,
    ts: int,
    spread_bps: float = 2.0,
    mid: float = 50000.0,
    book_imbalance: float = 0.3,
    flow_imbalance: float = 0.4,
    natr: float = 0.02,
    impact_bps: float = 5.0,
    regime_vol: RegimeVol = RegimeVol.HIGH,
    regime_trend: RegimeTrend = RegimeTrend.TREND,
    stale_book_ms: int = 0,
    stale_trades_ms: int = 0,
) -> FeatureSnapshot:
    """Create a deterministic FeatureSnapshot.

    Args:
        symbol: Trading pair symbol.
        ts: Timestamp in milliseconds.
        spread_bps: Bid-ask spread in basis points.
        mid: Mid price.
        book_imbalance: Order book imbalance [-1, 1].
        flow_imbalance: Trade flow imbalance [-1, 1].
        natr: Normalized ATR.
        impact_bps: Market impact in basis points.
        regime_vol: Volatility regime.
        regime_trend: Trend regime.
        stale_book_ms: Book data staleness.
        stale_trades_ms: Trade data staleness.

    Returns:
        FeatureSnapshot with deterministic values.
    """
    return FeatureSnapshot(
        ts=ts,
        symbol=symbol,
        features=Features(
            spread_bps=spread_bps,
            mid=mid,
            book_imbalance=book_imbalance,
            flow_imbalance=flow_imbalance,
            natr_14_5m=natr,
            impact_bps_q=impact_bps,
            regime_vol=regime_vol,
            regime_trend=regime_trend,
        ),
        data_health=DataHealth(
            stale_book_ms=stale_book_ms,
            stale_trades_ms=stale_trades_ms,
        ),
    )


# Base timestamp: 2026-01-01 00:00:00 UTC (deterministic)
BASE_TS = 1767225600000  # ms

# Fixture: 10 FeatureSnapshots across 3 symbols at 3 time points
# This creates a scenario where symbols enter/exit the ranker
REPLAY_FIXTURE: list[FeatureSnapshot] = [
    # t=0: Initial state - BTC strong, ETH moderate, SOL weak
    make_feature_snapshot(
        symbol="BTCUSDT",
        ts=BASE_TS,
        spread_bps=1.5,
        book_imbalance=0.5,
        flow_imbalance=0.6,
        natr=0.025,
        impact_bps=3.0,
    ),
    make_feature_snapshot(
        symbol="ETHUSDT",
        ts=BASE_TS,
        spread_bps=2.5,
        book_imbalance=0.3,
        flow_imbalance=0.4,
        natr=0.02,
        impact_bps=5.0,
    ),
    make_feature_snapshot(
        symbol="SOLUSDT",
        ts=BASE_TS,
        spread_bps=4.0,
        book_imbalance=0.1,
        flow_imbalance=0.2,
        natr=0.015,
        impact_bps=8.0,
    ),
    # t=2000ms: ETH strengthens, SOL improves
    make_feature_snapshot(
        symbol="BTCUSDT",
        ts=BASE_TS + 2000,
        spread_bps=1.8,
        book_imbalance=0.45,
        flow_imbalance=0.55,
        natr=0.024,
        impact_bps=3.5,
    ),
    make_feature_snapshot(
        symbol="ETHUSDT",
        ts=BASE_TS + 2000,
        spread_bps=2.0,
        book_imbalance=0.55,
        flow_imbalance=0.65,
        natr=0.028,
        impact_bps=4.0,
    ),
    make_feature_snapshot(
        symbol="SOLUSDT",
        ts=BASE_TS + 2000,
        spread_bps=3.0,
        book_imbalance=0.35,
        flow_imbalance=0.45,
        natr=0.022,
        impact_bps=6.0,
    ),
    # t=4000ms: ETH now leads, BTC drops slightly
    make_feature_snapshot(
        symbol="BTCUSDT",
        ts=BASE_TS + 4000,
        spread_bps=2.2,
        book_imbalance=0.35,
        flow_imbalance=0.4,
        natr=0.02,
        impact_bps=5.0,
    ),
    make_feature_snapshot(
        symbol="ETHUSDT",
        ts=BASE_TS + 4000,
        spread_bps=1.5,
        book_imbalance=0.6,
        flow_imbalance=0.7,
        natr=0.03,
        impact_bps=3.0,
    ),
    make_feature_snapshot(
        symbol="SOLUSDT",
        ts=BASE_TS + 4000,
        spread_bps=2.5,
        book_imbalance=0.45,
        flow_imbalance=0.55,
        natr=0.025,
        impact_bps=4.5,
    ),
    # t=6000ms: Final state - ETH dominant
    make_feature_snapshot(
        symbol="ETHUSDT",
        ts=BASE_TS + 6000,
        spread_bps=1.2,
        book_imbalance=0.65,
        flow_imbalance=0.75,
        natr=0.032,
        impact_bps=2.5,
    ),
]


def get_snapshots_by_timestamp() -> dict[int, list[FeatureSnapshot]]:
    """Group snapshots by timestamp for batch processing.

    Returns:
        Dict mapping timestamp to list of FeatureSnapshots at that time.
    """
    by_ts: dict[int, list[FeatureSnapshot]] = {}
    for snapshot in REPLAY_FIXTURE:
        if snapshot.ts not in by_ts:
            by_ts[snapshot.ts] = []
        by_ts[snapshot.ts].append(snapshot)
    return by_ts


def compute_fixture_digest() -> str:
    """Compute SHA256 digest of the fixture for verification.

    Returns:
        Hex digest of fixture JSON.
    """
    import hashlib

    data = b"".join(s.to_json() for s in REPLAY_FIXTURE)
    return hashlib.sha256(data).hexdigest()
