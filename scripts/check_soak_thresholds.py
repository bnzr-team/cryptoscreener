#!/usr/bin/env python3
"""
DEC-032: Check soak summary JSON against threshold config.

Reads --baseline-json and/or --overload-json, compares fields against
thresholds from --config YAML. Exits 0 if all pass, 1 if any fail.

No external dependencies beyond PyYAML (already a project dep via CI).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_json(path: Path) -> dict[str, Any]:
    """Load and return a JSON file as dict."""
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def load_config(path: Path) -> dict[str, Any]:
    """Load and return a YAML config file as dict."""
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
        return data


def check_max(
    summary: dict[str, Any],
    field: str,
    threshold: float | int,
    failures: list[str],
    label: str,
) -> None:
    """Check that summary[field] <= threshold."""
    value = summary.get(field)
    if value is None:
        failures.append(f"  [{label}] MISSING: {field} not found in summary")
        return
    if value > threshold:
        failures.append(f"  [{label}] FAIL: {field} = {value} > {threshold}")
    else:
        print(f"  [{label}] OK: {field} = {value} <= {threshold}")


def check_min(
    summary: dict[str, Any],
    field: str,
    threshold: float | int,
    failures: list[str],
    label: str,
) -> None:
    """Check that summary[field] >= threshold."""
    value = summary.get(field)
    if value is None:
        failures.append(f"  [{label}] MISSING: {field} not found in summary")
        return
    if value < threshold:
        failures.append(f"  [{label}] FAIL: {field} = {value} < {threshold}")
    else:
        print(f"  [{label}] OK: {field} = {value} >= {threshold}")


def check_baseline(summary: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Check baseline soak summary against thresholds. Returns list of failures."""
    failures: list[str] = []
    label = "baseline"
    print("\n=== Baseline Soak Thresholds ===")

    check_max(summary, "events_dropped", thresholds.get("events_dropped_max", 0), failures, label)
    check_max(
        summary,
        "snapshots_dropped",
        thresholds.get("snapshots_dropped_max", 0),
        failures,
        label,
    )
    check_max(
        summary,
        "max_event_queue_depth",
        thresholds.get("max_event_queue_depth", 2000),
        failures,
        label,
    )
    check_max(
        summary,
        "max_snapshot_queue_depth",
        thresholds.get("max_snapshot_queue_depth", 500),
        failures,
        label,
    )
    check_max(summary, "max_rss_mb", thresholds.get("max_rss_mb", 200), failures, label)
    check_max(
        summary, "max_tick_drift_ms", thresholds.get("max_tick_drift_ms", 200), failures, label
    )
    check_max(
        summary,
        "max_reconnect_rate_per_min",
        thresholds.get("max_reconnect_rate_per_min", 6.0),
        failures,
        label,
    )

    return failures


def check_overload(summary: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Check overload soak summary against thresholds. Returns list of failures."""
    failures: list[str] = []
    label = "overload"
    print("\n=== Overload Soak Thresholds ===")

    # Under overload, drops are expected (backpressure engaged)
    check_min(
        summary, "events_dropped", thresholds.get("events_dropped_min", 1), failures, label
    )
    check_max(
        summary,
        "max_event_queue_depth",
        thresholds.get("max_event_queue_depth", 10000),
        failures,
        label,
    )
    check_max(
        summary,
        "max_snapshot_queue_depth",
        thresholds.get("max_snapshot_queue_depth", 1000),
        failures,
        label,
    )
    check_max(summary, "max_rss_mb", thresholds.get("max_rss_mb", 300), failures, label)
    check_max(
        summary, "max_tick_drift_ms", thresholds.get("max_tick_drift_ms", 1000), failures, label
    )
    check_max(
        summary,
        "max_reconnect_rate_per_min",
        thresholds.get("max_reconnect_rate_per_min", 6.0),
        failures,
        label,
    )

    return failures


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DEC-032: Check soak summary JSON against thresholds.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Path to baseline soak summary JSON",
    )
    parser.add_argument(
        "--overload-json",
        type=Path,
        default=None,
        help="Path to overload soak summary JSON",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("monitoring/soak_thresholds.yml"),
        help="Path to threshold config YAML (default: monitoring/soak_thresholds.yml)",
    )
    args = parser.parse_args()

    if args.baseline_json is None and args.overload_json is None:
        print("ERROR: At least one of --baseline-json or --overload-json is required")
        return 1

    config = load_config(args.config)
    all_failures: list[str] = []

    if args.baseline_json is not None:
        summary = load_json(args.baseline_json)
        baseline_thresholds = config.get("baseline", {})
        all_failures.extend(check_baseline(summary, baseline_thresholds))

    if args.overload_json is not None:
        summary = load_json(args.overload_json)
        overload_thresholds = config.get("overload", {})
        all_failures.extend(check_overload(summary, overload_thresholds))

    print()
    if all_failures:
        print(f"FAILED: {len(all_failures)} threshold(s) violated:")
        for f in all_failures:
            print(f)
        return 1

    print("PASSED: All soak thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
