"""Tests for Scorer."""

from collections.abc import Callable

import pytest

from cryptoscreener.contracts.events import (
    DataHealth,
    ExecutionProfile,
    PredictionSnapshot,
    PredictionStatus,
)
from cryptoscreener.scoring.scorer import Scorer, ScorerConfig


class TestScorerConfig:
    """Tests for ScorerConfig."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = ScorerConfig()

        assert config.w_30s == 0.2
        assert config.w_2m == 0.5
        assert config.w_5m == 0.3
        assert config.utility_max_bps == 50.0
        assert config.toxic_alpha == 0.5

    def test_weights_sum_to_one(self) -> None:
        """Weights should sum to 1.0."""
        config = ScorerConfig()
        assert abs(config.w_30s + config.w_2m + config.w_5m - 1.0) < 1e-6


class TestScorer:
    """Tests for Scorer."""

    @pytest.fixture
    def scorer(self) -> Scorer:
        """Create scorer with default config."""
        return Scorer()

    @pytest.fixture
    def make_prediction(self) -> Callable[..., PredictionSnapshot]:
        """Factory for creating PredictionSnapshots."""

        def _make(
            symbol: str = "BTCUSDT",
            p_inplay_30s: float = 0.5,
            p_inplay_2m: float = 0.5,
            p_inplay_5m: float = 0.5,
            expected_utility_bps_2m: float = 25.0,
            p_toxic: float = 0.1,
            status: PredictionStatus = PredictionStatus.WATCH,
        ) -> PredictionSnapshot:
            return PredictionSnapshot(
                ts=1000,
                symbol=symbol,
                profile=ExecutionProfile.A,
                p_inplay_30s=p_inplay_30s,
                p_inplay_2m=p_inplay_2m,
                p_inplay_5m=p_inplay_5m,
                expected_utility_bps_2m=expected_utility_bps_2m,
                p_toxic=p_toxic,
                status=status,
                reasons=[],
                model_version="test-v1.0.0",
                calibration_version="cal-v1.0.0",
                data_health=DataHealth(),
            )

        return _make

    def test_init_default_config(self) -> None:
        """Scorer initializes with default config."""
        scorer = Scorer()
        assert scorer.config.w_2m == 0.5

    def test_init_custom_config(self) -> None:
        """Scorer initializes with custom config."""
        config = ScorerConfig(w_30s=0.3, w_2m=0.4, w_5m=0.3)
        scorer = Scorer(config)
        assert scorer.config.w_2m == 0.4

    def test_invalid_weights_sum(self) -> None:
        """Invalid weights sum raises error."""
        config = ScorerConfig(w_30s=0.5, w_2m=0.5, w_5m=0.5)
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            Scorer(config)

    def test_invalid_utility_max(self) -> None:
        """Invalid utility_max_bps raises error."""
        config = ScorerConfig(utility_max_bps=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            Scorer(config)

    def test_invalid_toxic_alpha(self) -> None:
        """Invalid toxic_alpha raises error."""
        config = ScorerConfig(toxic_alpha=1.5)
        with pytest.raises(ValueError, match="must be in"):
            Scorer(config)

    def test_compute_p_inplay(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Compute p_inplay weighted average."""
        pred = make_prediction(p_inplay_30s=0.2, p_inplay_2m=0.6, p_inplay_5m=0.4)
        p_inplay = scorer.compute_p_inplay(pred)

        # 0.2*0.2 + 0.6*0.5 + 0.4*0.3 = 0.04 + 0.30 + 0.12 = 0.46
        assert abs(p_inplay - 0.46) < 1e-6

    def test_compute_utility_factor_positive(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Compute utility factor for positive utility."""
        pred = make_prediction(expected_utility_bps_2m=25.0)
        factor = scorer.compute_utility_factor(pred)

        # 25 / 50 = 0.5
        assert abs(factor - 0.5) < 1e-6

    def test_compute_utility_factor_clamped_high(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Utility factor clamped at max."""
        pred = make_prediction(expected_utility_bps_2m=100.0)
        factor = scorer.compute_utility_factor(pred)

        # Clamped to 50, so 50/50 = 1.0
        assert abs(factor - 1.0) < 1e-6

    def test_compute_utility_factor_clamped_low(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Utility factor clamped at 0 for negative utility."""
        pred = make_prediction(expected_utility_bps_2m=-10.0)
        factor = scorer.compute_utility_factor(pred)

        # Clamped to 0, so 0/50 = 0.0
        assert abs(factor - 0.0) < 1e-6

    def test_compute_toxicity_penalty_low(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Toxicity penalty for low p_toxic."""
        pred = make_prediction(p_toxic=0.1)
        penalty = scorer.compute_toxicity_penalty(pred)

        # 1 - 0.5 * 0.1 = 0.95
        assert abs(penalty - 0.95) < 1e-6

    def test_compute_toxicity_penalty_high(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Toxicity penalty for high p_toxic."""
        pred = make_prediction(p_toxic=0.8)
        penalty = scorer.compute_toxicity_penalty(pred)

        # 1 - 0.5 * 0.8 = 0.6
        assert abs(penalty - 0.6) < 1e-6

    def test_score_formula(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Test complete score formula."""
        pred = make_prediction(
            p_inplay_30s=0.6,
            p_inplay_2m=0.8,
            p_inplay_5m=0.7,
            expected_utility_bps_2m=40.0,
            p_toxic=0.2,
        )
        score = scorer.score(pred)

        # p_inplay = 0.6*0.2 + 0.8*0.5 + 0.7*0.3 = 0.12 + 0.40 + 0.21 = 0.73
        # utility_factor = 40/50 = 0.8
        # toxicity_penalty = 1 - 0.5*0.2 = 0.9
        # score = 0.73 * 0.8 * 0.9 = 0.5256
        expected = 0.73 * 0.8 * 0.9
        assert abs(score - expected) < 1e-6

    def test_score_zero_utility(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Score is 0 when utility is 0."""
        pred = make_prediction(expected_utility_bps_2m=0.0)
        score = scorer.score(pred)
        assert score == 0.0

    def test_score_negative_utility(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Score is 0 when utility is negative."""
        pred = make_prediction(expected_utility_bps_2m=-10.0)
        score = scorer.score(pred)
        assert score == 0.0

    def test_score_clamped_to_one(self) -> None:
        """Score is clamped to 1.0 max."""
        # Use config that could produce > 1.0
        config = ScorerConfig(toxic_alpha=0.0)
        scorer = Scorer(config)

        pred = PredictionSnapshot(
            ts=1000,
            symbol="BTCUSDT",
            profile=ExecutionProfile.A,
            p_inplay_30s=1.0,
            p_inplay_2m=1.0,
            p_inplay_5m=1.0,
            expected_utility_bps_2m=100.0,
            p_toxic=0.0,
            status=PredictionStatus.TRADEABLE,
            reasons=[],
            model_version="test-v1.0.0",
            calibration_version="cal-v1.0.0",
        )
        score = scorer.score(pred)
        assert score == 1.0

    def test_score_batch(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Score batch returns correct number of scores."""
        predictions = [
            make_prediction(symbol="BTCUSDT", expected_utility_bps_2m=30.0),
            make_prediction(symbol="ETHUSDT", expected_utility_bps_2m=20.0),
            make_prediction(symbol="BNBUSDT", expected_utility_bps_2m=10.0),
        ]
        scores = scorer.score_batch(predictions)

        assert len(scores) == 3
        assert scores[0] > scores[1] > scores[2]  # Higher utility = higher score

    def test_deterministic_scoring(
        self, scorer: Scorer, make_prediction: Callable[..., PredictionSnapshot]
    ) -> None:
        """Scoring is deterministic."""
        pred = make_prediction(
            p_inplay_30s=0.5,
            p_inplay_2m=0.7,
            p_inplay_5m=0.6,
            expected_utility_bps_2m=35.0,
            p_toxic=0.15,
        )

        score1 = scorer.score(pred)
        score2 = scorer.score(pred)

        assert score1 == score2
