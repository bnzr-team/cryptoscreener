#!/usr/bin/env python3
"""
DEC-023c REST Governor Proofing - Replay Determinism Proof

This script demonstrates that CircuitBreaker state machine transitions produce
identical behavior across multiple runs when using fake time injection.

Usage:
    python scripts/proof_dec023c_determinism.py
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cryptoscreener.connectors.backoff import (
    BackoffConfig,
    BackoffState,
    CircuitBreaker,
    compute_backoff_delay,
    handle_error_response,
)


def run_state_machine_scenario(seed: int) -> dict[str, Any]:
    """Run a deterministic CircuitBreaker state machine scenario."""
    fake_time = 0

    def time_fn() -> int:
        return fake_time

    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout_ms=30000,
        ban_recovery_timeout_ms=300000,
        half_open_max_requests=1,
        _time_fn=time_fn,
    )

    results: list[dict[str, Any]] = []

    # Simulate sequence of events
    events = [
        ("init", 0),
        ("failure", 1000),
        ("failure", 2000),
        ("rate_limit", 3000),  # Opens circuit
        ("check", 10000),  # Still blocked
        ("check", 33001),  # Transitions to HALF_OPEN
        ("success", 33002),  # Closes circuit
        ("ban", 40000),  # IP ban - immediate OPEN
        ("check", 100000),  # Still in ban cooldown
        ("check", 340001),  # Transitions to HALF_OPEN
        ("success", 340002),  # Closes circuit
    ]

    for event_type, event_time in events:
        fake_time = event_time

        if event_type == "init":
            pass
        elif event_type == "failure":
            cb.record_failure()
        elif event_type == "rate_limit":
            cb.record_failure(is_rate_limit=True)
        elif event_type == "ban":
            cb.record_failure(is_ip_ban=True)
        elif event_type == "success":
            cb.record_success()
        elif event_type == "check":
            cb.can_execute()

        results.append(
            {
                "event": event_type,
                "fake_time": fake_time,
                "state": cb.state.value,
                "can_execute": cb.can_execute(),
                "is_banned": cb.is_banned,
                "failure_count": cb.failure_count,
                "last_failure_time_ms": cb.last_failure_time_ms,
            }
        )

    return {"seed": seed, "results": results}


def run_backoff_with_retry_after_scenario(seed: int) -> dict[str, Any]:
    """Run a deterministic backoff scenario with Retry-After parsing."""
    config = BackoffConfig(
        base_delay_ms=1000,
        max_delay_ms=60000,
        multiplier=2.0,
        jitter_factor=0.0,  # No jitter for determinism
    )
    state = BackoffState()

    results: list[dict[str, Any]] = []

    # Simulate sequence with various Retry-After values
    retry_afters = [None, 0, 500, 5000, 100000]

    for retry_after in retry_afters:
        state.record_error()
        delay = compute_backoff_delay(config, state, retry_after_ms=retry_after)

        results.append(
            {
                "attempt": state.attempt,
                "retry_after_ms": retry_after,
                "computed_delay": delay,
            }
        )

    return {"seed": seed, "results": results}


def run_error_response_scenario(seed: int) -> dict[str, Any]:
    """Run deterministic error response handling."""
    test_cases = [
        (200, None, None),  # Normal response
        (429, None, None),  # Rate limit without Retry-After
        (429, None, 30000),  # Rate limit with Retry-After
        (418, None, None),  # IP ban without Retry-After
        (418, None, 60000),  # IP ban with Retry-After
        (200, -1003, None),  # -1003 error code
        (200, -1003, 45000),  # -1003 with Retry-After
        (500, None, None),  # Server error (not rate limit)
    ]

    results: list[dict[str, Any]] = []

    for status_code, error_code, retry_after_ms in test_cases:
        error = handle_error_response(status_code, error_code, retry_after_ms)

        results.append(
            {
                "status_code": status_code,
                "error_code": error_code,
                "retry_after_input": retry_after_ms,
                "is_error": error is not None,
                "error_kind": error.kind.value if error else None,
                "error_retry_after": error.retry_after_ms if error else None,
                "is_ip_ban": error.is_ip_ban if error else None,
            }
        )

    return {"seed": seed, "results": results}


def compute_digest(data: dict[str, Any]) -> str:
    """Compute SHA256 digest of JSON-serialized data."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def main() -> None:
    """Run determinism proof."""
    print("=" * 70)
    print("DEC-023c REST Governor Proofing - Replay Determinism Proof")
    print("=" * 70)
    print()

    # Test 1: State Machine Determinism
    print("1. CircuitBreaker State Machine Determinism")
    print("-" * 40)
    run1_sm = run_state_machine_scenario(seed=42)
    run2_sm = run_state_machine_scenario(seed=42)
    digest1_sm = compute_digest(run1_sm)
    digest2_sm = compute_digest(run2_sm)
    print(f"   Run 1 digest: {digest1_sm[:16]}...")
    print(f"   Run 2 digest: {digest2_sm[:16]}...")
    print(f"   Match: {'✅ YES' if digest1_sm == digest2_sm else '❌ NO'}")
    print()

    # Test 2: Backoff with Retry-After Determinism
    print("2. Backoff with Retry-After Determinism")
    print("-" * 40)
    run1_ba = run_backoff_with_retry_after_scenario(seed=42)
    run2_ba = run_backoff_with_retry_after_scenario(seed=42)
    digest1_ba = compute_digest(run1_ba)
    digest2_ba = compute_digest(run2_ba)
    print(f"   Run 1 digest: {digest1_ba[:16]}...")
    print(f"   Run 2 digest: {digest2_ba[:16]}...")
    print(f"   Match: {'✅ YES' if digest1_ba == digest2_ba else '❌ NO'}")
    print()

    # Test 3: Error Response Handling Determinism
    print("3. Error Response Handling Determinism")
    print("-" * 40)
    run1_er = run_error_response_scenario(seed=42)
    run2_er = run_error_response_scenario(seed=42)
    digest1_er = compute_digest(run1_er)
    digest2_er = compute_digest(run2_er)
    print(f"   Run 1 digest: {digest1_er[:16]}...")
    print(f"   Run 2 digest: {digest2_er[:16]}...")
    print(f"   Match: {'✅ YES' if digest1_er == digest2_er else '❌ NO'}")
    print()

    # Summary
    all_match = digest1_sm == digest2_sm and digest1_ba == digest2_ba and digest1_er == digest2_er
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"State Machine:      {digest1_sm}")
    print(f"Backoff/Retry-After: {digest1_ba}")
    print(f"Error Response:     {digest1_er}")
    print()
    print(f"All digests match across runs: {'✅ PASS' if all_match else '❌ FAIL'}")
    print()

    if not all_match:
        exit(1)


if __name__ == "__main__":
    main()
