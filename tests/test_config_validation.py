"""
DEC-030: Config validation tests for LivePipelineConfig.

Tests __post_init__ validation: port ranges, cadence bounds, symbol format,
fault flag gating, and rollout knob constraints.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest
from scripts.run_live import LivePipelineConfig


class TestConfigValidation:
    """DEC-030: LivePipelineConfig.__post_init__ validation."""

    def test_default_config_valid(self) -> None:
        """Default config passes validation."""
        config = LivePipelineConfig()
        assert config.top_n == 50
        assert config.metrics_port == 9090

    def test_invalid_metrics_port_negative(self) -> None:
        with pytest.raises(ValueError, match="metrics_port"):
            LivePipelineConfig(metrics_port=-1)

    def test_invalid_metrics_port_too_high(self) -> None:
        with pytest.raises(ValueError, match="metrics_port"):
            LivePipelineConfig(metrics_port=70000)

    def test_metrics_port_zero_valid(self) -> None:
        """Port 0 disables metrics server."""
        config = LivePipelineConfig(metrics_port=0)
        assert config.metrics_port == 0

    def test_invalid_top_n_zero(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            LivePipelineConfig(top_n=0)

    def test_invalid_top_n_too_high(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            LivePipelineConfig(top_n=3000)

    def test_invalid_cadence_too_low(self) -> None:
        with pytest.raises(ValueError, match="snapshot_cadence_ms"):
            LivePipelineConfig(snapshot_cadence_ms=50)

    def test_invalid_cadence_too_high(self) -> None:
        with pytest.raises(ValueError, match="snapshot_cadence_ms"):
            LivePipelineConfig(snapshot_cadence_ms=100000)

    def test_invalid_duration_zero(self) -> None:
        with pytest.raises(ValueError, match="duration_s"):
            LivePipelineConfig(duration_s=0)

    def test_invalid_duration_negative(self) -> None:
        with pytest.raises(ValueError, match="duration_s"):
            LivePipelineConfig(duration_s=-5)

    def test_duration_none_valid(self) -> None:
        """None duration (run forever) is valid."""
        config = LivePipelineConfig(duration_s=None)
        assert config.duration_s is None

    def test_invalid_graceful_timeout_negative(self) -> None:
        with pytest.raises(ValueError, match="graceful_timeout_s"):
            LivePipelineConfig(graceful_timeout_s=-1)

    def test_invalid_symbol_format_lowercase(self) -> None:
        with pytest.raises(ValueError, match="Invalid symbol"):
            LivePipelineConfig(symbols=["btcusdt"])

    def test_invalid_symbol_format_special_chars(self) -> None:
        with pytest.raises(ValueError, match="Invalid symbol"):
            LivePipelineConfig(symbols=["BTC-USDT"])

    def test_valid_symbols(self) -> None:
        config = LivePipelineConfig(symbols=["BTCUSDT", "ETHUSDT"])
        assert config.symbols == ["BTCUSDT", "ETHUSDT"]

    def test_fault_flags_blocked_by_default(self) -> None:
        """Fault flags raise without ALLOW_FAULTS=1."""
        with mock.patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="ALLOW_FAULTS"):
            LivePipelineConfig(fault_drop_ws_every_s=10)

    def test_fault_flags_allowed_with_env(self) -> None:
        """Fault flags pass with ALLOW_FAULTS=1."""
        with mock.patch.dict(os.environ, {"ALLOW_FAULTS": "1"}):
            config = LivePipelineConfig(fault_drop_ws_every_s=10)
            assert config.fault_drop_ws_every_s == 10

    def test_fault_flags_allowed_with_dev_env(self) -> None:
        """Fault flags pass with ENV=dev."""
        with mock.patch.dict(os.environ, {"ENV": "dev"}):
            config = LivePipelineConfig(fault_slow_consumer_ms=100)
            assert config.fault_slow_consumer_ms == 100

    def test_dry_run_flag(self) -> None:
        config = LivePipelineConfig(dry_run=True)
        assert config.dry_run is True

    def test_readiness_staleness_default(self) -> None:
        config = LivePipelineConfig()
        assert config.readiness_staleness_s == 30
