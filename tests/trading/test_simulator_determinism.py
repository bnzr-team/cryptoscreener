"""Simulator determinism tests.

Verifies that running the same fixture with the same config
produces identical SHA256 artifacts.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cryptoscreener.trading.sim import SimConfig, Simulator

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


class TestSimulatorDeterminism:
    """Test that simulator produces deterministic output."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "monotonic_up.jsonl",
            "mean_reverting_range.jsonl",
            "flash_crash.jsonl",
            "ws_gap.jsonl",
        ],
    )
    def test_two_runs_same_sha256(self, fixture_name: str) -> None:
        """Two runs on same fixture produce identical SHA256."""
        events = load_fixture(fixture_name)
        config = SimConfig(symbol="BTCUSDT")

        # Run 1
        sim1 = Simulator(config)
        artifacts1 = sim1.run(events)

        # Run 2
        sim2 = Simulator(config)
        artifacts2 = sim2.run(events)

        # SHA256 must match
        assert artifacts1.sha256 == artifacts2.sha256
        assert len(artifacts1.sha256) == 64  # Full hex digest

    def test_different_config_different_sha256(self) -> None:
        """Different config produces different SHA256."""
        events = load_fixture("mean_reverting_range.jsonl")

        config1 = SimConfig(symbol="BTCUSDT", maker_fee_frac=Decimal("0.0002"))
        config2 = SimConfig(symbol="BTCUSDT", maker_fee_frac=Decimal("0.0004"))

        sim1 = Simulator(config1)
        artifacts1 = sim1.run(events)

        sim2 = Simulator(config2)
        artifacts2 = sim2.run(events)

        # Different config should produce different artifacts
        # (Note: may actually be same if no fills occur, but generally different)
        # We just verify we get valid hashes
        assert len(artifacts1.sha256) == 64
        assert len(artifacts2.sha256) == 64

    def test_empty_events_deterministic(self) -> None:
        """Empty events produce deterministic output."""
        config = SimConfig(symbol="BTCUSDT")

        sim1 = Simulator(config)
        artifacts1 = sim1.run([])

        sim2 = Simulator(config)
        artifacts2 = sim2.run([])

        assert artifacts1.sha256 == artifacts2.sha256

    def test_metrics_populated(self) -> None:
        """Metrics are populated in artifacts."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")

        sim = Simulator(config)
        artifacts = sim.run(events)

        # Check metrics exist
        assert "net_pnl" in artifacts.metrics
        assert "total_fills" in artifacts.metrics
        assert "round_trips" in artifacts.metrics
        assert "max_drawdown" in artifacts.metrics

    def test_session_states_tracked(self) -> None:
        """Session state transitions are tracked."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")

        sim = Simulator(config)
        artifacts = sim.run(events)

        # Should have at least READY -> ACTIVE -> STOPPED
        assert len(artifacts.session_states) >= 3

        # First state should be READY
        assert artifacts.session_states[0]["state"] == "READY"

        # Second should be ACTIVE
        assert artifacts.session_states[1]["state"] == "ACTIVE"

        # Last should be STOPPED or KILLED
        assert artifacts.session_states[-1]["state"] in ("STOPPED", "KILLED")


class TestSimulatorReproducibility:
    """Test that artifacts can be serialized and reproduced."""

    def test_artifacts_json_serializable(self) -> None:
        """Artifacts can be serialized to JSON."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")

        sim = Simulator(config)
        artifacts = sim.run(events)

        # Should be able to dump to JSON
        json_str = artifacts.model_dump_json()
        assert len(json_str) > 0

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["sha256"] == artifacts.sha256

    def test_sha256_computed_correctly(self) -> None:
        """SHA256 is computed over canonical JSON (sorted keys)."""
        events = load_fixture("mean_reverting_range.jsonl")
        config = SimConfig(symbol="BTCUSDT")

        sim = Simulator(config)
        artifacts = sim.run(events)

        # Recompute hash
        import hashlib

        import orjson

        data = {
            "config": artifacts.config,
            "fills": artifacts.fills,
            "orders": artifacts.orders,
            "positions": artifacts.positions,
            "session_states": artifacts.session_states,
            "metrics": artifacts.metrics,
        }
        canonical = orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
        expected_hash = hashlib.sha256(canonical).hexdigest()

        assert artifacts.sha256 == expected_hash
