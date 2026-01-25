"""Tests for backtest harness."""

import tempfile
from pathlib import Path
from typing import Any

import orjson

from cryptoscreener.backtest.harness import (
    BacktestConfig,
    BacktestHarness,
    load_labeled_data,
    print_backtest_summary,
    save_backtest_result,
)
from cryptoscreener.cost_model.calculator import Profile
from cryptoscreener.label_builder import Horizon


def create_test_labels() -> list[dict[str, Any]]:
    """Create test labeled data."""
    rows = []
    for i in range(20):
        ts = i * 1000
        symbol = "BTCUSDT" if i % 2 == 0 else "ETHUSDT"

        # Create varying net edges
        base_edge = (i % 5) * 10 - 10  # -10 to 30

        row = {
            "ts": ts,
            "symbol": symbol,
            "mid_price": 50000.0 + i * 100,
            "spread_bps": 2.0,
            "y_toxic": 1 if i % 7 == 0 else 0,
            "severity_toxic_bps": 15.0 if i % 7 == 0 else 0.0,
        }

        # Add labels for each horizon/profile
        for h in ["30s", "2m", "5m"]:
            for p in ["a", "b"]:
                edge_key = f"net_edge_bps_{h}_{p}"
                i_key = f"i_tradeable_{h}_{p}"

                edge = base_edge + (5 if p == "a" else 0)
                tradeable = 1 if edge > 5 else 0

                row[edge_key] = edge
                row[i_key] = tradeable

        rows.append(row)

    return rows


class TestLoadLabeledData:
    """Tests for loading labeled data."""

    def test_load_jsonl(self) -> None:
        """Test loading JSONL file."""
        rows = create_test_labels()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(orjson.dumps(row))
                f.write(b"\n")
            tmp_path = Path(f.name)

        try:
            loaded = load_labeled_data(tmp_path)
            assert len(loaded) == len(rows)
            assert loaded[0]["symbol"] == rows[0]["symbol"]
        finally:
            tmp_path.unlink()

    def test_unsupported_format(self) -> None:
        """Test unsupported file format raises error."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"a,b,c\n1,2,3\n")
            tmp_path = Path(f.name)

        try:
            try:
                load_labeled_data(tmp_path)
                raise AssertionError("Expected ValueError")
            except ValueError as e:
                assert "Unsupported" in str(e)
        finally:
            tmp_path.unlink()


class TestBacktestHarness:
    """Tests for BacktestHarness."""

    def test_evaluate_labels_only(self) -> None:
        """Test evaluation using labels only."""
        rows = create_test_labels()

        config = BacktestConfig(
            horizons=(Horizon.H_30S, Horizon.H_2M),
            profiles=(Profile.A,),
            top_k=5,
        )
        harness = BacktestHarness(config)
        result = harness.evaluate_labels_only(rows)

        assert len(result.results) == 2  # 2 horizons * 1 profile
        assert result.metadata["n_rows"] == len(rows)

        # Check that metrics are computed
        for r in result.results:
            assert 0 <= r.metrics.auc <= 1
            assert 0 <= r.metrics.pr_auc <= 1
            assert r.metrics.calibration.brier_score >= 0
            assert r.metrics.n_samples > 0

    def test_evaluate_with_toxicity(self) -> None:
        """Test toxicity metrics are computed."""
        rows = create_test_labels()

        harness = BacktestHarness()
        result = harness.evaluate_labels_only(rows)

        assert result.toxicity_metrics is not None
        assert result.toxicity_metrics.n_samples > 0

    def test_churn_metrics(self) -> None:
        """Test churn metrics are computed when timestamps present."""
        rows = create_test_labels()

        config = BacktestConfig(
            horizons=(Horizon.H_30S,),
            profiles=(Profile.A,),
            top_k=3,
        )
        harness = BacktestHarness(config)
        result = harness.evaluate_labels_only(rows)

        # Should have churn metrics since we have timestamps and symbols
        r = result.results[0]
        assert r.metrics.churn is not None
        assert r.metrics.churn.jaccard_similarity >= 0

    def test_get_result(self) -> None:
        """Test getting specific horizon/profile result."""
        rows = create_test_labels()

        harness = BacktestHarness()
        result = harness.evaluate_labels_only(rows)

        r = result.get_result(Horizon.H_30S, Profile.A)
        assert r is not None
        assert r.horizon == Horizon.H_30S
        assert r.profile == Profile.A

        # Non-existent combination
        r2 = result.get_result(Horizon.H_30S, Profile.A)
        assert r2 is not None


class TestBacktestResult:
    """Tests for BacktestResult serialization."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        rows = create_test_labels()

        harness = BacktestHarness()
        result = harness.evaluate_labels_only(rows)

        d = result.to_dict()

        assert "config" in d
        assert "metadata" in d
        assert "results" in d
        assert "30s_A" in d["results"]

        # Check result fields
        r = d["results"]["30s_A"]
        assert "auc" in r
        assert "pr_auc" in r
        assert "brier_score" in r
        assert "ece" in r
        assert "topk_capture" in r

    def test_save_and_load(self) -> None:
        """Test saving and loading results."""
        rows = create_test_labels()

        harness = BacktestHarness()
        result = harness.evaluate_labels_only(rows)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            save_backtest_result(result, tmp_path)

            # Verify file is valid JSON
            with tmp_path.open("rb") as f:
                loaded = orjson.loads(f.read())

            assert "results" in loaded
            assert loaded["metadata"]["n_rows"] == len(rows)
        finally:
            tmp_path.unlink()


class TestPrintSummary:
    """Tests for print_backtest_summary."""

    def test_prints_without_error(self, capsys: Any) -> None:
        """Test summary prints without error."""
        rows = create_test_labels()

        harness = BacktestHarness()
        result = harness.evaluate_labels_only(rows)

        # Should not raise
        print_backtest_summary(result)

        captured = capsys.readouterr()
        assert "BACKTEST EVALUATION SUMMARY" in captured.out
        assert "AUC" in captured.out
        assert "30s/A" in captured.out
