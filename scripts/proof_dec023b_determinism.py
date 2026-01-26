#!/usr/bin/env python3
"""
DEC-023b Replay Determinism Proof

This script demonstrates that the operational safety limiters produce
identical behavior across multiple runs when using fake time and seeded RNG.

Usage:
    python scripts/proof_dec023b_determinism.py
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from cryptoscreener.connectors.backoff import (
    BackoffConfig,
    BackoffState,
    MessageThrottler,
    MessageThrottlerConfig,
    ReconnectLimiter,
    ReconnectLimiterConfig,
    compute_backoff_delay,
)


def run_reconnect_limiter_scenario(seed: int) -> dict[str, Any]:
    """Run a deterministic ReconnectLimiter scenario."""
    fake_time = 0

    def time_fn() -> int:
        return fake_time

    config = ReconnectLimiterConfig(
        max_reconnects_per_window=3,
        window_ms=10000,
        cooldown_after_burst_ms=5000,
        per_shard_min_interval_ms=2000,
    )
    limiter = ReconnectLimiter(config=config, _time_fn=time_fn)

    results: list[dict[str, Any]] = []
    for step in range(20):
        shard_id = step % 3
        can_reconnect = limiter.can_reconnect(shard_id)
        wait_time = limiter.get_wait_time_ms(shard_id)

        if can_reconnect:
            limiter.record_reconnect(shard_id)

        results.append(
            {
                "step": step,
                "fake_time": fake_time,
                "shard_id": shard_id,
                "can_reconnect": can_reconnect,
                "wait_time": wait_time,
                "status": limiter.get_status(),
            }
        )

        fake_time += 1000  # Advance 1 second

    return {"seed": seed, "results": results}


def run_message_throttler_scenario(seed: int) -> dict[str, Any]:
    """Run a deterministic MessageThrottler scenario."""
    fake_time = 0

    def time_fn() -> int:
        return fake_time

    config = MessageThrottlerConfig(
        max_messages_per_second=10,
        safety_margin=1.0,
        burst_allowance=5,
    )
    throttler = MessageThrottler(config=config, _time_fn=time_fn)

    results: list[dict[str, Any]] = []
    for step in range(30):
        can_send = throttler.can_send(2)
        wait_time = throttler.get_wait_time_ms(2)

        if can_send:
            throttler.consume(2)

        results.append(
            {
                "step": step,
                "fake_time": fake_time,
                "can_send": can_send,
                "wait_time": wait_time,
                "status": throttler.get_status(),
            }
        )

        fake_time += 50  # Advance 50ms

    return {"seed": seed, "results": results}


def run_backoff_scenario(seed: int) -> dict[str, Any]:
    """Run a deterministic backoff with seeded jitter."""
    rng = random.Random(seed)
    config = BackoffConfig(
        base_delay_ms=1000,
        max_delay_ms=60000,
        multiplier=2.0,
        jitter_factor=0.5,
    )
    state = BackoffState()

    delays: list[int] = []
    for _ in range(10):
        state.record_error()
        delay = compute_backoff_delay(config, state, rng=rng)
        delays.append(delay)

    return {"seed": seed, "delays": delays}


def compute_digest(data: dict[str, Any]) -> str:
    """Compute SHA256 digest of JSON-serialized data."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def main() -> None:
    """Run determinism proof."""
    print("=" * 70)
    print("DEC-023b Replay Determinism Proof")
    print("=" * 70)
    print()

    # Test ReconnectLimiter determinism
    print("1. ReconnectLimiter Determinism")
    print("-" * 40)
    run1_rl = run_reconnect_limiter_scenario(seed=42)
    run2_rl = run_reconnect_limiter_scenario(seed=42)
    digest1_rl = compute_digest(run1_rl)
    digest2_rl = compute_digest(run2_rl)
    print(f"   Run 1 digest: {digest1_rl[:16]}...")
    print(f"   Run 2 digest: {digest2_rl[:16]}...")
    print(f"   Match: {'✅ YES' if digest1_rl == digest2_rl else '❌ NO'}")
    print()

    # Test MessageThrottler determinism
    print("2. MessageThrottler Determinism")
    print("-" * 40)
    run1_mt = run_message_throttler_scenario(seed=42)
    run2_mt = run_message_throttler_scenario(seed=42)
    digest1_mt = compute_digest(run1_mt)
    digest2_mt = compute_digest(run2_mt)
    print(f"   Run 1 digest: {digest1_mt[:16]}...")
    print(f"   Run 2 digest: {digest2_mt[:16]}...")
    print(f"   Match: {'✅ YES' if digest1_mt == digest2_mt else '❌ NO'}")
    print()

    # Test seeded backoff jitter determinism
    print("3. Seeded Backoff Jitter Determinism")
    print("-" * 40)
    run1_bf = run_backoff_scenario(seed=42)
    run2_bf = run_backoff_scenario(seed=42)
    digest1_bf = compute_digest(run1_bf)
    digest2_bf = compute_digest(run2_bf)
    print(f"   Run 1 digest: {digest1_bf[:16]}...")
    print(f"   Run 2 digest: {digest2_bf[:16]}...")
    print(f"   Match: {'✅ YES' if digest1_bf == digest2_bf else '❌ NO'}")
    print(f"   Delays: {run1_bf['delays']}")
    print()

    # Summary
    all_match = digest1_rl == digest2_rl and digest1_mt == digest2_mt and digest1_bf == digest2_bf
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"ReconnectLimiter:  {digest1_rl}")
    print(f"MessageThrottler:  {digest1_mt}")
    print(f"Backoff Jitter:    {digest1_bf}")
    print()
    print(f"All digests match across runs: {'✅ PASS' if all_match else '❌ FAIL'}")
    print()

    if not all_match:
        exit(1)


if __name__ == "__main__":
    main()
