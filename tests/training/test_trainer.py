"""Tests for ML trainer (DEC-038)."""

from __future__ import annotations

import numpy as np
import pytest

from cryptoscreener.training.feature_schema import FEATURE_ORDER, PREDICTION_HEADS
from cryptoscreener.training.trainer import (
    HeadMetrics,
    Trainer,
    TrainingConfig,
    TrainingError,
    TrainingResult,
)


def make_sample_rows(n: int = 100, seed: int = 42) -> list[dict]:
    """Generate sample data rows for testing.

    Creates synthetic data with 8 features and 4 label columns.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n):
        row = {
            # Features
            "spread_bps": rng.uniform(1, 20),
            "mid": rng.uniform(40000, 50000),
            "book_imbalance": rng.uniform(-1, 1),
            "flow_imbalance": rng.uniform(-1, 1),
            "natr_14_5m": rng.uniform(0.5, 3.0),
            "impact_bps_q": rng.uniform(0, 15),
            "regime_vol_binary": float(rng.integers(0, 2)),
            "regime_trend_binary": float(rng.integers(0, 2)),
            # Labels for profile 'a'
            "i_tradeable_30s_a": int(rng.random() > 0.7),
            "i_tradeable_2m_a": int(rng.random() > 0.6),
            "i_tradeable_5m_a": int(rng.random() > 0.5),
            "y_toxic": int(rng.random() > 0.9),
            # Labels for profile 'b' (not used in these tests)
            "i_tradeable_30s_b": int(rng.random() > 0.75),
            "i_tradeable_2m_b": int(rng.random() > 0.65),
            "i_tradeable_5m_b": int(rng.random() > 0.55),
        }
        rows.append(row)

    return rows


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self) -> None:
        """TrainingConfig has sensible defaults."""
        config = TrainingConfig()
        assert config.seed == 42
        assert config.val_ratio == 0.2
        assert config.model_type == "random_forest"
        assert config.profile == "a"

    def test_invalid_seed_raises(self) -> None:
        """Negative seed raises ValueError."""
        with pytest.raises(ValueError, match="seed must be non-negative"):
            TrainingConfig(seed=-1)

    def test_invalid_val_ratio_raises(self) -> None:
        """val_ratio outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="val_ratio must be in"):
            TrainingConfig(val_ratio=0)
        with pytest.raises(ValueError, match="val_ratio must be in"):
            TrainingConfig(val_ratio=1.0)
        with pytest.raises(ValueError, match="val_ratio must be in"):
            TrainingConfig(val_ratio=1.5)

    def test_invalid_model_type_raises(self) -> None:
        """Unknown model_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            TrainingConfig(model_type="xgboost")  # type: ignore[arg-type]

    def test_invalid_profile_raises(self) -> None:
        """Invalid profile raises ValueError."""
        with pytest.raises(ValueError, match="profile must be"):
            TrainingConfig(profile="c")


class TestTrainerPrepareData:
    """Tests for Trainer.prepare_data()."""

    def test_prepare_data_shapes(self) -> None:
        """prepare_data returns correct shapes."""
        trainer = Trainer()
        rows = make_sample_rows(n=50)

        X, y = trainer.prepare_data(rows)

        assert X.shape == (50, 8)  # 50 samples, 8 features
        assert y.shape == (50, 4)  # 50 samples, 4 heads

    def test_prepare_data_feature_order(self) -> None:
        """prepare_data extracts features in correct order."""
        trainer = Trainer()
        rows = [
            {
                "spread_bps": 5.0,
                "mid": 45000.0,
                "book_imbalance": 0.3,
                "flow_imbalance": -0.2,
                "natr_14_5m": 1.5,
                "impact_bps_q": 3.0,
                "regime_vol_binary": 1.0,
                "regime_trend_binary": 0.0,
                "i_tradeable_30s_a": 1,
                "i_tradeable_2m_a": 0,
                "i_tradeable_5m_a": 1,
                "y_toxic": 0,
            }
        ]

        X, y = trainer.prepare_data(rows)

        # Check feature values match order
        assert X[0, 0] == 5.0  # spread_bps
        assert X[0, 1] == 45000.0  # mid
        assert X[0, 2] == 0.3  # book_imbalance
        assert X[0, 3] == -0.2  # flow_imbalance
        assert X[0, 4] == 1.5  # natr_14_5m
        assert X[0, 5] == 3.0  # impact_bps_q
        assert X[0, 6] == 1.0  # regime_vol_binary
        assert X[0, 7] == 0.0  # regime_trend_binary

        # Check label values
        assert y[0, 0] == 1  # i_tradeable_30s
        assert y[0, 1] == 0  # i_tradeable_2m
        assert y[0, 2] == 1  # i_tradeable_5m
        assert y[0, 3] == 0  # y_toxic

    def test_prepare_data_profile_b(self) -> None:
        """prepare_data uses correct label columns for profile b."""
        config = TrainingConfig(profile="b")
        trainer = Trainer(config)
        rows = [
            {
                **dict.fromkeys(FEATURE_ORDER, 1.0),
                "i_tradeable_30s_a": 0,
                "i_tradeable_2m_a": 0,
                "i_tradeable_5m_a": 0,
                "y_toxic": 0,
                "i_tradeable_30s_b": 1,
                "i_tradeable_2m_b": 1,
                "i_tradeable_5m_b": 1,
            }
        ]

        _, y = trainer.prepare_data(rows)

        # Profile 'b' labels should be extracted
        assert y[0, 0] == 1  # i_tradeable_30s_b
        assert y[0, 1] == 1  # i_tradeable_2m_b
        assert y[0, 2] == 1  # i_tradeable_5m_b
        assert y[0, 3] == 0  # y_toxic (same for both profiles)

    def test_prepare_data_missing_feature_raises(self) -> None:
        """prepare_data raises TrainingError for missing feature."""
        trainer = Trainer()
        rows = [{"spread_bps": 5.0}]  # Missing other features

        with pytest.raises(TrainingError, match="missing feature column"):
            trainer.prepare_data(rows)

    def test_prepare_data_missing_label_raises(self) -> None:
        """prepare_data raises TrainingError for missing label."""
        trainer = Trainer()
        rows = [dict.fromkeys(FEATURE_ORDER, 1.0)]  # Missing labels

        with pytest.raises(TrainingError, match="missing label column"):
            trainer.prepare_data(rows)

    def test_prepare_data_empty_raises(self) -> None:
        """prepare_data raises TrainingError for empty rows."""
        trainer = Trainer()

        with pytest.raises(TrainingError, match="No data rows"):
            trainer.prepare_data([])


class TestTrainerTrain:
    """Tests for Trainer.train()."""

    def test_train_random_forest(self) -> None:
        """train() returns RandomForest model."""
        config = TrainingConfig(model_type="random_forest", n_estimators=10)
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)

        model = trainer.train(X, y)

        # Should have fitted estimators
        assert hasattr(model, "estimators_")
        assert len(model.estimators_) == 4  # One per head

    def test_train_logistic(self) -> None:
        """train() returns Logistic model."""
        config = TrainingConfig(model_type="logistic", max_iter=100)
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)

        model = trainer.train(X, y)

        assert hasattr(model, "estimators_")
        assert len(model.estimators_) == 4

    def test_train_wrong_feature_count_raises(self) -> None:
        """train() raises for wrong feature count."""
        trainer = Trainer()
        X = np.zeros((100, 5))  # Wrong: 5 features instead of 8
        y = np.zeros((100, 4), dtype=np.int32)

        with pytest.raises(TrainingError, match="Expected 8 features"):
            trainer.train(X, y)

    def test_train_wrong_head_count_raises(self) -> None:
        """train() raises for wrong head count."""
        trainer = Trainer()
        X = np.zeros((100, 8))
        y = np.zeros((100, 2), dtype=np.int32)  # Wrong: 2 heads instead of 4

        with pytest.raises(TrainingError, match="Expected 4 heads"):
            trainer.train(X, y)


class TestTrainerEvaluate:
    """Tests for Trainer.evaluate()."""

    def test_evaluate_returns_metrics_per_head(self) -> None:
        """evaluate() returns HeadMetrics for each head."""
        config = TrainingConfig(n_estimators=10)
        trainer = Trainer(config)

        rows = make_sample_rows(n=200, seed=42)
        X, y = trainer.prepare_data(rows)
        X_train, y_train = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]

        model = trainer.train(X_train, y_train)
        metrics = trainer.evaluate(model, X_val, y_val)

        assert len(metrics) == 4
        for head in PREDICTION_HEADS:
            assert head in metrics
            m = metrics[head]
            assert isinstance(m, HeadMetrics)
            assert m.head_name == head
            assert 0 <= m.auc <= 1
            assert 0 <= m.pr_auc <= 1
            assert m.brier >= 0
            assert 0 <= m.ece <= 1

    def test_evaluate_metrics_are_reasonable(self) -> None:
        """evaluate() returns reasonable metric values."""
        config = TrainingConfig(n_estimators=50)
        trainer = Trainer(config)

        # Create data with some predictable pattern
        rows = make_sample_rows(n=500, seed=123)
        X, y = trainer.prepare_data(rows)
        X_train, y_train = X[:400], y[:400]
        X_val, y_val = X[400:], y[400:]

        model = trainer.train(X_train, y_train)
        metrics = trainer.evaluate(model, X_val, y_val)

        # With random data, AUC should be around 0.5
        # Allow some variance
        for head in PREDICTION_HEADS:
            m = metrics[head]
            # Very loose bounds - just checking not broken
            assert 0.3 <= m.auc <= 0.9
            assert m.n_samples == 100


class TestTrainerDeterminism:
    """Tests for training determinism."""

    def test_same_seed_produces_same_model(self) -> None:
        """Training with same seed produces identical results."""
        config = TrainingConfig(seed=42, n_estimators=10)
        rows = make_sample_rows(n=100, seed=99)

        # Train twice with same seed
        trainer1 = Trainer(config)
        X1, y1 = trainer1.prepare_data(rows)
        model1 = trainer1.train(X1, y1)

        trainer2 = Trainer(config)
        X2, y2 = trainer2.prepare_data(rows)
        model2 = trainer2.train(X2, y2)

        # Predictions should be identical
        probs1 = trainer1.predict_proba(model1, X1)
        probs2 = trainer2.predict_proba(model2, X2)

        for head in PREDICTION_HEADS:
            np.testing.assert_array_almost_equal(probs1[head], probs2[head])

    def test_different_seeds_produce_different_models(self) -> None:
        """Training with different seeds produces different results."""
        rows = make_sample_rows(n=100, seed=99)

        trainer1 = Trainer(TrainingConfig(seed=42, n_estimators=10))
        X1, y1 = trainer1.prepare_data(rows)
        model1 = trainer1.train(X1, y1)

        trainer2 = Trainer(TrainingConfig(seed=123, n_estimators=10))
        X2, y2 = trainer2.prepare_data(rows)
        model2 = trainer2.train(X2, y2)

        probs1 = trainer1.predict_proba(model1, X1)
        probs2 = trainer2.predict_proba(model2, X2)

        # At least one head should have different predictions
        any_different = False
        for head in PREDICTION_HEADS:
            if not np.allclose(probs1[head], probs2[head]):
                any_different = True
                break
        assert any_different


class TestTrainAndEvaluate:
    """Tests for Trainer.train_and_evaluate()."""

    def test_train_and_evaluate_returns_result(self) -> None:
        """train_and_evaluate() returns TrainingResult."""
        config = TrainingConfig(n_estimators=10)
        trainer = Trainer(config)

        rows = make_sample_rows(n=200)
        X, y = trainer.prepare_data(rows)
        X_train, y_train = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]

        result = trainer.train_and_evaluate(X_train, y_train, X_val, y_val)

        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert len(result.metrics) == 4
        assert result.config is config
        assert result.feature_order == FEATURE_ORDER
        assert result.head_order == PREDICTION_HEADS


class TestHeadMetrics:
    """Tests for HeadMetrics."""

    def test_to_dict_rounds_values(self) -> None:
        """to_dict() rounds float values to 4 decimals."""
        metrics = HeadMetrics(
            head_name="p_inplay_2m",
            auc=0.7654321,
            pr_auc=0.6543219,
            brier=0.1234567,
            ece=0.0456789,
            n_samples=100,
            n_positives=30,
        )

        d = metrics.to_dict()

        assert d["auc"] == 0.7654
        assert d["pr_auc"] == 0.6543
        assert d["brier"] == 0.1235
        assert d["ece"] == 0.0457
        assert d["n_samples"] == 100
        assert d["n_positives"] == 30
