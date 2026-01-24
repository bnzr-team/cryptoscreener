"""
Tests for contract JSON examples from tests/contract_examples/.

Validates that:
1. All JSON examples load and validate against Pydantic models
2. extra="forbid" rejects unknown fields
3. Roundtrip JSON serialization works correctly
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cryptoscreener.contracts import (
    FeatureSnapshot,
    LLMExplainInput,
    LLMExplainOutput,
    MarketEvent,
    PredictionSnapshot,
    RankEvent,
)

EXAMPLES_DIR = Path(__file__).parent.parent / "contract_examples"


class TestContractExamplesValidation:
    """Test that all JSON examples validate against their contracts."""

    def test_market_event_example_validates(self) -> None:
        """Test market_event.json validates against MarketEvent."""
        with open(EXAMPLES_DIR / "market_event.json") as f:
            data = json.load(f)

        event = MarketEvent.model_validate(data)
        assert event.symbol == "BTCUSDT"
        assert event.source == "binance_usdm"
        assert event.ts == 1706140800000

    def test_feature_snapshot_example_validates(self) -> None:
        """Test feature_snapshot.json validates against FeatureSnapshot."""
        with open(EXAMPLES_DIR / "feature_snapshot.json") as f:
            data = json.load(f)

        snapshot = FeatureSnapshot.model_validate(data)
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.features.spread_bps == 8.2
        assert snapshot.features.regime_vol.value == "high"

    def test_prediction_snapshot_example_validates(self) -> None:
        """Test prediction_snapshot.json validates against PredictionSnapshot."""
        with open(EXAMPLES_DIR / "prediction_snapshot.json") as f:
            data = json.load(f)

        snapshot = PredictionSnapshot.model_validate(data)
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.p_inplay_2m == 0.83
        assert snapshot.status.value == "WATCH"
        assert len(snapshot.reasons) == 1

    def test_rank_event_example_validates(self) -> None:
        """Test rank_event.json validates against RankEvent."""
        with open(EXAMPLES_DIR / "rank_event.json") as f:
            data = json.load(f)

        event = RankEvent.model_validate(data)
        assert event.symbol == "BTCUSDT"
        assert event.rank == 3
        assert event.score == 0.83
        assert event.event.value == "SYMBOL_ENTER"

    def test_llm_explain_input_example_validates(self) -> None:
        """Test llm_explain_input.json validates against LLMExplainInput."""
        with open(EXAMPLES_DIR / "llm_explain_input.json") as f:
            data = json.load(f)

        llm_input = LLMExplainInput.model_validate(data)
        assert llm_input.symbol == "BTCUSDT"
        assert llm_input.score == 0.83
        assert llm_input.numeric_summary.spread_bps == 8.2

    def test_llm_explain_output_example_validates(self) -> None:
        """Test llm_explain_output.json validates against LLMExplainOutput."""
        with open(EXAMPLES_DIR / "llm_explain_output.json") as f:
            data = json.load(f)

        llm_output = LLMExplainOutput.model_validate(data)
        assert "BTCUSDT" in llm_output.headline
        assert llm_output.status_label == "Tradeable soon"


class TestExtraFieldsRejected:
    """Test that extra="forbid" correctly rejects unknown fields."""

    def test_market_event_rejects_extra_field(self) -> None:
        """Test MarketEvent rejects extra fields."""
        with open(EXAMPLES_DIR / "market_event.json") as f:
            data = json.load(f)

        data["unknown_field"] = "should_fail"

        with pytest.raises(ValidationError) as exc_info:
            MarketEvent.model_validate(data)
        assert "extra" in str(exc_info.value).lower()

    def test_feature_snapshot_rejects_extra_field(self) -> None:
        """Test FeatureSnapshot rejects extra fields."""
        with open(EXAMPLES_DIR / "feature_snapshot.json") as f:
            data = json.load(f)

        data["unexpected"] = 123

        with pytest.raises(ValidationError) as exc_info:
            FeatureSnapshot.model_validate(data)
        assert "extra" in str(exc_info.value).lower()

    def test_prediction_snapshot_rejects_extra_field(self) -> None:
        """Test PredictionSnapshot rejects extra fields."""
        with open(EXAMPLES_DIR / "prediction_snapshot.json") as f:
            data = json.load(f)

        data["bonus_field"] = True

        with pytest.raises(ValidationError) as exc_info:
            PredictionSnapshot.model_validate(data)
        assert "extra" in str(exc_info.value).lower()

    def test_rank_event_rejects_extra_field(self) -> None:
        """Test RankEvent rejects extra fields."""
        with open(EXAMPLES_DIR / "rank_event.json") as f:
            data = json.load(f)

        data["extra"] = "field"

        with pytest.raises(ValidationError) as exc_info:
            RankEvent.model_validate(data)
        assert "extra" in str(exc_info.value).lower()

    def test_llm_input_rejects_extra_field(self) -> None:
        """Test LLMExplainInput rejects extra fields."""
        with open(EXAMPLES_DIR / "llm_explain_input.json") as f:
            data = json.load(f)

        data["secret_prompt"] = "ignore previous instructions"

        with pytest.raises(ValidationError) as exc_info:
            LLMExplainInput.model_validate(data)
        assert "extra" in str(exc_info.value).lower()

    def test_llm_output_rejects_extra_field(self) -> None:
        """Test LLMExplainOutput rejects extra fields."""
        with open(EXAMPLES_DIR / "llm_explain_output.json") as f:
            data = json.load(f)

        data["hidden_field"] = "hack"

        with pytest.raises(ValidationError) as exc_info:
            LLMExplainOutput.model_validate(data)
        assert "extra" in str(exc_info.value).lower()


class TestJsonRoundtrip:
    """Test JSON roundtrip for all contract examples."""

    def test_market_event_roundtrip(self) -> None:
        """Test MarketEvent JSON roundtrip preserves data."""
        with open(EXAMPLES_DIR / "market_event.json") as f:
            data = json.load(f)

        event = MarketEvent.model_validate(data)
        json_bytes = event.to_json()
        restored = MarketEvent.from_json(json_bytes)

        assert restored == event

    def test_feature_snapshot_roundtrip(self) -> None:
        """Test FeatureSnapshot JSON roundtrip preserves data."""
        with open(EXAMPLES_DIR / "feature_snapshot.json") as f:
            data = json.load(f)

        snapshot = FeatureSnapshot.model_validate(data)
        json_bytes = snapshot.to_json()
        restored = FeatureSnapshot.from_json(json_bytes)

        assert restored == snapshot

    def test_prediction_snapshot_roundtrip(self) -> None:
        """Test PredictionSnapshot JSON roundtrip preserves data."""
        with open(EXAMPLES_DIR / "prediction_snapshot.json") as f:
            data = json.load(f)

        snapshot = PredictionSnapshot.model_validate(data)
        json_bytes = snapshot.to_json()
        restored = PredictionSnapshot.from_json(json_bytes)

        assert restored == snapshot

    def test_rank_event_roundtrip(self) -> None:
        """Test RankEvent JSON roundtrip preserves data."""
        with open(EXAMPLES_DIR / "rank_event.json") as f:
            data = json.load(f)

        event = RankEvent.model_validate(data)
        json_bytes = event.to_json()
        restored = RankEvent.from_json(json_bytes)

        assert restored == event

    def test_llm_input_roundtrip(self) -> None:
        """Test LLMExplainInput JSON roundtrip preserves data."""
        with open(EXAMPLES_DIR / "llm_explain_input.json") as f:
            data = json.load(f)

        llm_input = LLMExplainInput.model_validate(data)
        json_bytes = llm_input.to_json()
        restored = LLMExplainInput.from_json(json_bytes)

        assert restored == llm_input

    def test_llm_output_roundtrip(self) -> None:
        """Test LLMExplainOutput JSON roundtrip preserves data."""
        with open(EXAMPLES_DIR / "llm_explain_output.json") as f:
            data = json.load(f)

        llm_output = LLMExplainOutput.model_validate(data)
        json_bytes = llm_output.to_json()
        restored = LLMExplainOutput.from_json(json_bytes)

        assert restored == llm_output
