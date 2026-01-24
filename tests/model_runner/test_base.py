"""Tests for ModelRunner base class."""

import pytest

from cryptoscreener.model_runner.base import ModelRunnerConfig, RunnerMetrics


class TestModelRunnerConfig:
    """Tests for ModelRunnerConfig."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = ModelRunnerConfig()

        assert config.model_version == "baseline-v1.0.0+0000000"
        assert config.calibration_version == "cal-v1.0.0"
        assert config.default_profile == "A"
        assert config.toxic_threshold == 0.7
        assert config.tradeable_threshold == 0.6
        assert config.watch_threshold == 0.3

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = ModelRunnerConfig(
            model_version="test-v2.0.0+abc1234",
            calibration_version="cal-v2.0.0",
            default_profile="B",
            toxic_threshold=0.8,
            tradeable_threshold=0.7,
            watch_threshold=0.4,
        )

        assert config.model_version == "test-v2.0.0+abc1234"
        assert config.calibration_version == "cal-v2.0.0"
        assert config.default_profile == "B"
        assert config.toxic_threshold == 0.8
        assert config.tradeable_threshold == 0.7
        assert config.watch_threshold == 0.4


class TestRunnerMetrics:
    """Tests for RunnerMetrics."""

    @pytest.fixture
    def metrics(self) -> RunnerMetrics:
        """Create fresh metrics."""
        return RunnerMetrics()

    def test_initial_values(self, metrics: RunnerMetrics) -> None:
        """Initial metrics are zero."""
        assert metrics.predictions_made == 0
        assert metrics.predictions_tradeable == 0
        assert metrics.predictions_watch == 0
        assert metrics.predictions_trap == 0
        assert metrics.predictions_dead == 0
        assert metrics.predictions_data_issue == 0
        assert metrics.predictions_per_symbol == {}

    def test_record_tradeable(self, metrics: RunnerMetrics) -> None:
        """Record TRADEABLE prediction."""
        metrics.record_prediction("BTCUSDT", "TRADEABLE")

        assert metrics.predictions_made == 1
        assert metrics.predictions_tradeable == 1
        assert metrics.predictions_per_symbol["BTCUSDT"] == 1

    def test_record_watch(self, metrics: RunnerMetrics) -> None:
        """Record WATCH prediction."""
        metrics.record_prediction("ETHUSDT", "WATCH")

        assert metrics.predictions_made == 1
        assert metrics.predictions_watch == 1
        assert metrics.predictions_per_symbol["ETHUSDT"] == 1

    def test_record_trap(self, metrics: RunnerMetrics) -> None:
        """Record TRAP prediction."""
        metrics.record_prediction("BTCUSDT", "TRAP")

        assert metrics.predictions_made == 1
        assert metrics.predictions_trap == 1

    def test_record_dead(self, metrics: RunnerMetrics) -> None:
        """Record DEAD prediction."""
        metrics.record_prediction("BTCUSDT", "DEAD")

        assert metrics.predictions_made == 1
        assert metrics.predictions_dead == 1

    def test_record_data_issue(self, metrics: RunnerMetrics) -> None:
        """Record DATA_ISSUE prediction."""
        metrics.record_prediction("BTCUSDT", "DATA_ISSUE")

        assert metrics.predictions_made == 1
        assert metrics.predictions_data_issue == 1

    def test_multiple_predictions(self, metrics: RunnerMetrics) -> None:
        """Record multiple predictions."""
        metrics.record_prediction("BTCUSDT", "TRADEABLE")
        metrics.record_prediction("BTCUSDT", "TRADEABLE")
        metrics.record_prediction("ETHUSDT", "WATCH")

        assert metrics.predictions_made == 3
        assert metrics.predictions_tradeable == 2
        assert metrics.predictions_watch == 1
        assert metrics.predictions_per_symbol["BTCUSDT"] == 2
        assert metrics.predictions_per_symbol["ETHUSDT"] == 1

    def test_reset(self, metrics: RunnerMetrics) -> None:
        """Reset clears all metrics."""
        metrics.record_prediction("BTCUSDT", "TRADEABLE")
        metrics.record_prediction("ETHUSDT", "WATCH")

        metrics.reset()

        assert metrics.predictions_made == 0
        assert metrics.predictions_tradeable == 0
        assert metrics.predictions_watch == 0
        assert metrics.predictions_per_symbol == {}
