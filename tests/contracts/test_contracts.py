"""
Tests for data contracts.

Covers:
- JSON roundtrip serialization/deserialization
- Schema validation
- LLM output safety constraints (adversarial tests)
- Contract compliance with DATA_CONTRACTS.md
"""

from __future__ import annotations

import json

import pytest

from cryptoscreener.contracts import (
    ALLOWED_STATUS_LABELS,
    DataHealth,
    ExecutionProfile,
    Features,
    FeatureSnapshot,
    LLMExplainInput,
    LLMExplainOutput,
    LLMOutputViolation,
    LLMStyle,
    MarketEvent,
    MarketEventType,
    NumericSummary,
    PredictionSnapshot,
    PredictionStatus,
    RankEvent,
    RankEventPayload,
    RankEventType,
    ReasonCode,
    RegimeTrend,
    RegimeVol,
    Windows,
    compute_rank_events_digest,
    generate_fallback_output,
    validate_contract_json,
    validate_llm_output_no_new_numbers,
    validate_llm_output_strict,
    validate_or_fallback,
)

# =============================================================================
# Test Fixtures (matching DATA_CONTRACTS.md exactly)
# =============================================================================


@pytest.fixture
def sample_market_event() -> MarketEvent:
    """Sample MarketEvent matching DATA_CONTRACTS.md."""
    return MarketEvent(
        ts=1706140800000,
        source="binance_usdm",
        symbol="BTCUSDT",
        type=MarketEventType.TRADE,
        payload={"price": "42000.50", "qty": "0.1", "side": "buy"},
        recv_ts=1706140800005,
    )


@pytest.fixture
def sample_feature_snapshot() -> FeatureSnapshot:
    """Sample FeatureSnapshot matching DATA_CONTRACTS.md."""
    return FeatureSnapshot(
        ts=1706140800000,
        symbol="BTCUSDT",
        features=Features(
            spread_bps=8.2,
            mid=42000.50,
            book_imbalance=0.35,
            flow_imbalance=0.63,
            natr_14_5m=0.025,
            impact_bps_q=6.5,
            regime_vol=RegimeVol.HIGH,
            regime_trend=RegimeTrend.TREND,
        ),
        windows=Windows(
            w1s={"vol": 150.5},
            w10s={"vol": 1420.3},
            w60s={"vol": 8500.0},
        ),
        data_health=DataHealth(
            stale_book_ms=50,
            stale_trades_ms=120,
            missing_streams=[],
        ),
    )


@pytest.fixture
def sample_prediction_snapshot() -> PredictionSnapshot:
    """Sample PredictionSnapshot matching DATA_CONTRACTS.md."""
    return PredictionSnapshot(
        ts=1706140800000,
        symbol="BTCUSDT",
        profile=ExecutionProfile.A,
        p_inplay_30s=0.72,
        p_inplay_2m=0.83,
        p_inplay_5m=0.65,
        expected_utility_bps_2m=4.5,
        p_toxic=0.21,
        status=PredictionStatus.WATCH,
        reasons=[
            ReasonCode(
                code="RC_FLOW_SURGE",
                value=2.1,
                unit="z",
                evidence="flow_imbalance=0.63",
            ),
        ],
        model_version="1.0.0+abc1234",
        calibration_version="iso_2026-01-24",
        data_health=DataHealth(),
    )


@pytest.fixture
def sample_rank_event() -> RankEvent:
    """Sample RankEvent matching DATA_CONTRACTS.md."""
    return RankEvent(
        ts=1706140800000,
        event=RankEventType.SYMBOL_ENTER,
        symbol="BTCUSDT",
        rank=3,
        score=0.83,
        payload=RankEventPayload(
            prediction={"status": "WATCH", "p_inplay_2m": 0.83},
            llm_text="Flow surge detected, watching for breakout.",
        ),
    )


@pytest.fixture
def sample_llm_input() -> LLMExplainInput:
    """Sample LLMExplainInput matching DATA_CONTRACTS.md."""
    return LLMExplainInput(
        symbol="BTCUSDT",
        timeframe="2m",
        status=PredictionStatus.WATCH,
        score=0.83,
        reasons=[
            ReasonCode(
                code="RC_FLOW_SURGE",
                value=2.1,
                unit="z",
                evidence="flow_imbalance=0.63",
            ),
        ],
        numeric_summary=NumericSummary(
            spread_bps=8.2,
            impact_bps=6.5,
            p_toxic=0.21,
            regime="high-vol trend",
        ),
        style=LLMStyle(tone="friendly", max_chars=180),
    )


@pytest.fixture
def sample_llm_output() -> LLMExplainOutput:
    """Sample LLMExplainOutput matching DATA_CONTRACTS.md."""
    return LLMExplainOutput(
        headline="BTCUSDT: flow surge + tight spread, likely tradable soon.",
        subtext="Watch for quick breakout; toxicity low-to-moderate.",
        status_label="Tradeable soon",
        tooltips={"p_inplay": "Calibrated probability of net edge after costs."},
    )


# =============================================================================
# MarketEvent Tests
# =============================================================================


class TestMarketEvent:
    """Tests for MarketEvent contract."""

    def test_roundtrip_json(self, sample_market_event: MarketEvent) -> None:
        """Test JSON serialization roundtrip."""
        json_bytes = sample_market_event.to_json()
        restored = MarketEvent.from_json(json_bytes)
        assert restored == sample_market_event

    def test_roundtrip_json_string(self, sample_market_event: MarketEvent) -> None:
        """Test JSON string roundtrip."""
        json_str = sample_market_event.to_json().decode()
        restored = MarketEvent.from_json(json_str)
        assert restored == sample_market_event

    def test_symbol_uppercase(self) -> None:
        """Test that symbol is normalized to uppercase."""
        event = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="btcusdt",  # lowercase
            type=MarketEventType.TRADE,
            payload={},
            recv_ts=1005,
        )
        assert event.symbol == "BTCUSDT"

    def test_immutable(self, sample_market_event: MarketEvent) -> None:
        """Test that MarketEvent is immutable."""
        with pytest.raises((TypeError, ValueError)):  # ValidationError for frozen model
            sample_market_event.ts = 9999

    def test_invalid_ts_negative(self) -> None:
        """Test that negative timestamp is rejected."""
        with pytest.raises(ValueError):
            MarketEvent(
                ts=-1,
                source="binance_usdm",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={},
                recv_ts=1000,
            )

    def test_json_structure_matches_spec(self, sample_market_event: MarketEvent) -> None:
        """Verify JSON structure matches DATA_CONTRACTS.md spec."""
        data = json.loads(sample_market_event.to_json())
        required_keys = {"ts", "source", "symbol", "type", "payload", "recv_ts"}
        assert set(data.keys()) == required_keys


# =============================================================================
# FeatureSnapshot Tests
# =============================================================================


class TestFeatureSnapshot:
    """Tests for FeatureSnapshot contract."""

    def test_roundtrip_json(self, sample_feature_snapshot: FeatureSnapshot) -> None:
        """Test JSON serialization roundtrip."""
        json_bytes = sample_feature_snapshot.to_json()
        restored = FeatureSnapshot.from_json(json_bytes)
        assert restored == sample_feature_snapshot

    def test_features_validation(self) -> None:
        """Test that feature values are validated."""
        with pytest.raises(ValueError):
            Features(
                spread_bps=8.2,
                mid=-100,  # Invalid: must be positive
                book_imbalance=0.5,
                flow_imbalance=0.5,
                natr_14_5m=0.02,
                impact_bps_q=5.0,
                regime_vol=RegimeVol.LOW,
                regime_trend=RegimeTrend.TREND,
            )

    def test_book_imbalance_bounds(self) -> None:
        """Test book_imbalance is bounded [-1, 1]."""
        with pytest.raises(ValueError):
            Features(
                spread_bps=8.2,
                mid=42000,
                book_imbalance=1.5,  # Invalid: must be <= 1
                flow_imbalance=0.5,
                natr_14_5m=0.02,
                impact_bps_q=5.0,
                regime_vol=RegimeVol.LOW,
                regime_trend=RegimeTrend.TREND,
            )

    def test_json_structure_matches_spec(self, sample_feature_snapshot: FeatureSnapshot) -> None:
        """Verify JSON structure matches DATA_CONTRACTS.md spec."""
        data = json.loads(sample_feature_snapshot.to_json())
        required_keys = {"ts", "symbol", "features", "windows", "data_health"}
        assert set(data.keys()) == required_keys
        assert "spread_bps" in data["features"]
        assert "regime_vol" in data["features"]


# =============================================================================
# PredictionSnapshot Tests
# =============================================================================


class TestPredictionSnapshot:
    """Tests for PredictionSnapshot contract."""

    def test_roundtrip_json(self, sample_prediction_snapshot: PredictionSnapshot) -> None:
        """Test JSON serialization roundtrip."""
        json_bytes = sample_prediction_snapshot.to_json()
        restored = PredictionSnapshot.from_json(json_bytes)
        assert restored == sample_prediction_snapshot

    def test_probability_bounds(self) -> None:
        """Test probability fields are bounded [0, 1]."""
        with pytest.raises(ValueError):
            PredictionSnapshot(
                ts=1000,
                symbol="BTCUSDT",
                profile=ExecutionProfile.A,
                p_inplay_30s=1.5,  # Invalid: must be <= 1
                p_inplay_2m=0.83,
                p_inplay_5m=0.65,
                expected_utility_bps_2m=4.5,
                p_toxic=0.21,
                status=PredictionStatus.WATCH,
                reasons=[],
                model_version="1.0.0",
                calibration_version="v1",
            )

    def test_all_statuses(self) -> None:
        """Test all PredictionStatus values are valid."""
        for status in PredictionStatus:
            pred = PredictionSnapshot(
                ts=1000,
                symbol="BTCUSDT",
                profile=ExecutionProfile.A,
                p_inplay_30s=0.5,
                p_inplay_2m=0.5,
                p_inplay_5m=0.5,
                expected_utility_bps_2m=0.0,
                p_toxic=0.1,
                status=status,
                reasons=[],
                model_version="1.0.0",
                calibration_version="v1",
            )
            assert pred.status == status

    def test_json_structure_matches_spec(
        self, sample_prediction_snapshot: PredictionSnapshot
    ) -> None:
        """Verify JSON structure matches DATA_CONTRACTS.md spec."""
        data = json.loads(sample_prediction_snapshot.to_json())
        required_keys = {
            "ts",
            "symbol",
            "profile",
            "p_inplay_30s",
            "p_inplay_2m",
            "p_inplay_5m",
            "expected_utility_bps_2m",
            "p_toxic",
            "status",
            "reasons",
            "model_version",
            "calibration_version",
            "data_health",
        }
        assert set(data.keys()) == required_keys


# =============================================================================
# RankEvent Tests
# =============================================================================


class TestRankEvent:
    """Tests for RankEvent contract."""

    def test_roundtrip_json(self, sample_rank_event: RankEvent) -> None:
        """Test JSON serialization roundtrip."""
        json_bytes = sample_rank_event.to_json()
        restored = RankEvent.from_json(json_bytes)
        assert restored == sample_rank_event

    def test_digest_deterministic(self, sample_rank_event: RankEvent) -> None:
        """Test that digest is deterministic."""
        digest1 = sample_rank_event.digest()
        digest2 = sample_rank_event.digest()
        assert digest1 == digest2
        assert len(digest1) == 64  # SHA256 hex

    def test_digest_changes_with_content(self, sample_rank_event: RankEvent) -> None:
        """Test that digest changes when content changes."""
        original_digest = sample_rank_event.digest()

        # Create modified event
        modified = RankEvent(
            ts=sample_rank_event.ts,
            event=sample_rank_event.event,
            symbol=sample_rank_event.symbol,
            rank=sample_rank_event.rank + 1,  # Changed
            score=sample_rank_event.score,
            payload=sample_rank_event.payload,
        )
        modified_digest = modified.digest()

        assert original_digest != modified_digest

    def test_all_event_types(self) -> None:
        """Test all RankEventType values are valid."""
        for event_type in RankEventType:
            event = RankEvent(
                ts=1000,
                event=event_type,
                symbol="BTCUSDT",
                rank=0,
                score=0.5,
            )
            assert event.event == event_type

    def test_json_structure_matches_spec(self, sample_rank_event: RankEvent) -> None:
        """Verify JSON structure matches DATA_CONTRACTS.md spec."""
        data = json.loads(sample_rank_event.to_json())
        required_keys = {"ts", "event", "symbol", "rank", "score", "payload"}
        assert set(data.keys()) == required_keys


class TestComputeRankEventsDigest:
    """Tests for compute_rank_events_digest function."""

    def test_deterministic(self, sample_rank_event: RankEvent) -> None:
        """Test digest is deterministic."""
        events = [sample_rank_event, sample_rank_event]
        digest1 = compute_rank_events_digest(events)
        digest2 = compute_rank_events_digest(events)
        assert digest1 == digest2

    def test_order_matters(self, sample_rank_event: RankEvent) -> None:
        """Test that event order affects digest."""
        event1 = sample_rank_event
        event2 = RankEvent(
            ts=event1.ts + 1000,
            event=RankEventType.SYMBOL_EXIT,
            symbol=event1.symbol,
            rank=0,
            score=0.5,
        )

        digest_12 = compute_rank_events_digest([event1, event2])
        digest_21 = compute_rank_events_digest([event2, event1])

        assert digest_12 != digest_21

    def test_empty_list(self) -> None:
        """Test digest of empty list."""
        digest = compute_rank_events_digest([])
        # SHA256 of empty input
        assert len(digest) == 64


# =============================================================================
# LLM Contract Tests
# =============================================================================


class TestLLMExplainInput:
    """Tests for LLMExplainInput contract."""

    def test_roundtrip_json(self, sample_llm_input: LLMExplainInput) -> None:
        """Test JSON serialization roundtrip."""
        json_bytes = sample_llm_input.to_json()
        restored = LLMExplainInput.from_json(json_bytes)
        assert restored == sample_llm_input

    def test_json_structure_matches_spec(self, sample_llm_input: LLMExplainInput) -> None:
        """Verify JSON structure matches DATA_CONTRACTS.md spec."""
        data = json.loads(sample_llm_input.to_json())
        required_keys = {
            "symbol",
            "timeframe",
            "status",
            "score",
            "reasons",
            "numeric_summary",
            "style",
        }
        assert set(data.keys()) == required_keys


class TestLLMExplainOutput:
    """Tests for LLMExplainOutput contract."""

    def test_roundtrip_json(self, sample_llm_output: LLMExplainOutput) -> None:
        """Test JSON serialization roundtrip."""
        json_bytes = sample_llm_output.to_json()
        restored = LLMExplainOutput.from_json(json_bytes)
        assert restored == sample_llm_output

    def test_json_structure_matches_spec(self, sample_llm_output: LLMExplainOutput) -> None:
        """Verify JSON structure matches DATA_CONTRACTS.md spec."""
        data = json.loads(sample_llm_output.to_json())
        required_keys = {"headline", "subtext", "status_label", "tooltips"}
        assert set(data.keys()) == required_keys


# =============================================================================
# LLM Safety / Adversarial Tests
# =============================================================================


class TestLLMOutputValidation:
    """Adversarial tests for LLM output safety constraints."""

    def test_valid_output_passes(
        self, sample_llm_input: LLMExplainInput, sample_llm_output: LLMExplainOutput
    ) -> None:
        """Test that valid output passes validation."""
        violations = validate_llm_output_no_new_numbers(sample_llm_input, sample_llm_output)
        assert violations == []

    def test_valid_output_strict_passes(
        self, sample_llm_input: LLMExplainInput, sample_llm_output: LLMExplainOutput
    ) -> None:
        """Test that valid output passes strict validation."""
        # Should not raise
        validate_llm_output_strict(sample_llm_input, sample_llm_output)

    def test_rejects_new_number_in_headline(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that new numbers in headline are rejected."""
        bad_output = LLMExplainOutput(
            headline="BTCUSDT: Expected gain 15.7% tomorrow!",  # 15.7 is NEW
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(sample_llm_input, bad_output)
        assert len(violations) > 0
        assert "15.7" in violations[0]

    def test_rejects_new_number_in_subtext(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that new numbers in subtext are rejected."""
        bad_output = LLMExplainOutput(
            headline="BTCUSDT: flow surge detected.",
            subtext="Price target: $99,999.99",  # 99999.99 is NEW
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(sample_llm_input, bad_output)
        assert len(violations) > 0

    def test_rejects_new_number_in_tooltip(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that new numbers in tooltips are rejected."""
        bad_output = LLMExplainOutput(
            headline="BTCUSDT: flow surge detected.",
            subtext="",
            status_label="Watch",
            tooltips={"fake_metric": "The value is 123.456"},  # 123.456 is NEW
        )
        violations = validate_llm_output_no_new_numbers(sample_llm_input, bad_output)
        assert len(violations) > 0

    def test_allows_numbers_from_input(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that numbers from input are allowed."""
        # Using numbers that exist in input: score=0.83, spread_bps=8.2, p_toxic=0.21, value=2.1
        valid_output = LLMExplainOutput(
            headline="BTCUSDT: score 0.83 with spread 8.2 bps.",
            subtext="Toxicity at 0.21, z-score 2.1.",
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(sample_llm_input, valid_output)
        assert violations == []

    def test_rejects_percentage_conversion(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that percentage conversions are REJECTED (stringwise exact match only).

        Per LLM_INPUT_OUTPUT_SCHEMA.md: "subset of input numbers (stringwise)"
        0.83 -> 83% is NOT allowed because "83" is not in the input strings.
        """
        bad_output = LLMExplainOutput(
            headline="BTCUSDT: 83% confidence.",  # 83 is NOT in input (0.83 is)
            subtext="21% toxicity risk.",  # 21 is NOT in input (0.21 is)
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(sample_llm_input, bad_output)
        # Must reject both 83 and 21
        assert len(violations) >= 2
        violation_text = " ".join(violations)
        assert "83" in violation_text
        assert "21" in violation_text

    def test_strict_validation_raises(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that strict validation raises on violation."""
        bad_output = LLMExplainOutput(
            headline="BTCUSDT: Expected gain 999%!",
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(sample_llm_input, bad_output)
        assert "999" in str(exc_info.value)

    def test_multiple_violations_reported(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that multiple violations are all reported."""
        bad_output = LLMExplainOutput(
            headline="BTCUSDT: 111 points up!",
            subtext="Target: 222 dollars",
            status_label="333 status",
            tooltips={"key": "value is 444"},
        )
        violations = validate_llm_output_no_new_numbers(sample_llm_input, bad_output)
        assert len(violations) >= 4


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestSchemaValidation:
    """Tests for schema validation utility."""

    def test_valid_market_event(self) -> None:
        """Test valid MarketEvent JSON passes validation."""
        data = {
            "ts": 1000,
            "source": "binance_usdm",
            "symbol": "BTCUSDT",
            "type": "trade",
            "payload": {},
            "recv_ts": 1005,
        }
        is_valid, error = validate_contract_json(MarketEvent, data)
        assert is_valid
        assert error == ""

    def test_invalid_market_event_missing_field(self) -> None:
        """Test MarketEvent with missing field fails validation."""
        data = {
            "ts": 1000,
            "source": "binance_usdm",
            # "symbol" missing
            "type": "trade",
            "payload": {},
            "recv_ts": 1005,
        }
        is_valid, err = validate_contract_json(MarketEvent, data)
        assert not is_valid
        assert "symbol" in err.lower()

    def test_invalid_market_event_wrong_type(self) -> None:
        """Test MarketEvent with wrong type fails validation."""
        data = {
            "ts": "not_a_number",  # Should be int
            "source": "binance_usdm",
            "symbol": "BTCUSDT",
            "type": "trade",
            "payload": {},
            "recv_ts": 1005,
        }
        is_valid, _err = validate_contract_json(MarketEvent, data)
        assert not is_valid

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields are rejected (strict schema)."""
        data = {
            "ts": 1000,
            "source": "binance_usdm",
            "symbol": "BTCUSDT",
            "type": "trade",
            "payload": {},
            "recv_ts": 1005,
            "extra_field": "should_fail",  # Not in schema
        }
        is_valid, err = validate_contract_json(MarketEvent, data)
        assert not is_valid
        assert "extra" in err.lower()


# =============================================================================
# LLM Fallback Tests
# =============================================================================


class TestLLMFallback:
    """Tests for LLM fallback mechanism per LLM_SAFETY_GUARDRAILS.md."""

    def test_fallback_output_is_valid(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that fallback output passes all validations."""
        fallback = generate_fallback_output(sample_llm_input)

        # Should not raise any exceptions
        validate_llm_output_strict(sample_llm_input, fallback)

        # Verify structure
        assert fallback.headline
        assert fallback.status_label in ALLOWED_STATUS_LABELS
        assert len(fallback.headline) <= sample_llm_input.style.max_chars

    def test_fallback_contains_no_invented_numbers(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that fallback contains no numbers not in input."""
        fallback = generate_fallback_output(sample_llm_input)
        violations = validate_llm_output_no_new_numbers(sample_llm_input, fallback)
        assert violations == []

    def test_validate_or_fallback_returns_valid_on_good_input(
        self, sample_llm_input: LLMExplainInput, sample_llm_output: LLMExplainOutput
    ) -> None:
        """Test validate_or_fallback returns original on valid input."""
        result, was_valid = validate_or_fallback(sample_llm_input, sample_llm_output)
        assert was_valid
        assert result == sample_llm_output

    def test_validate_or_fallback_returns_fallback_on_invalid(
        self, sample_llm_input: LLMExplainInput
    ) -> None:
        """Test validate_or_fallback returns fallback on invalid input."""
        bad_output = LLMExplainOutput(
            headline="BTCUSDT: Expected 999% gain!",  # Invalid number
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        result, was_valid = validate_or_fallback(sample_llm_input, bad_output)
        assert not was_valid
        assert result != bad_output
        # Fallback should be valid
        validate_llm_output_strict(sample_llm_input, result)

    def test_fallback_status_label_mapping(self) -> None:
        """Test that fallback maps all PredictionStatus values correctly."""
        for status in PredictionStatus:
            test_input = LLMExplainInput(
                symbol="BTCUSDT",
                timeframe="2m",
                status=status,
                score=0.5,
                reasons=[],
                numeric_summary=NumericSummary(
                    spread_bps=5.0, impact_bps=3.0, p_toxic=0.1, regime="low-vol"
                ),
                style=LLMStyle(tone="friendly", max_chars=180),
            )
            fallback = generate_fallback_output(test_input)
            assert fallback.status_label in ALLOWED_STATUS_LABELS


class TestStatusLabelValidation:
    """Tests for status_label validation."""

    def test_all_allowed_labels_pass(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that all allowed status labels pass validation."""
        for label in ALLOWED_STATUS_LABELS:
            output = LLMExplainOutput(
                headline="BTCUSDT: test.",
                subtext="",
                status_label=label,
                tooltips={},
            )
            # Should not raise
            validate_llm_output_strict(sample_llm_input, output)

    def test_invalid_status_label_rejected(self, sample_llm_input: LLMExplainInput) -> None:
        """Test that invalid status labels are rejected."""
        output = LLMExplainOutput(
            headline="BTCUSDT: test.",
            subtext="",
            status_label="INVALID_LABEL_XYZ",
            tooltips={},
        )
        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(sample_llm_input, output)
        assert "status_label" in str(exc_info.value).lower()


class TestMaxLengthValidation:
    """Tests for headline max length validation."""

    def test_headline_within_limit_passes(self, sample_llm_input: LLMExplainInput) -> None:
        """Test headline within max_chars passes."""
        output = LLMExplainOutput(
            headline="A" * 180,  # Exactly at limit
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        # Should not raise
        validate_llm_output_strict(sample_llm_input, output)

    def test_headline_over_limit_rejected(self, sample_llm_input: LLMExplainInput) -> None:
        """Test headline over max_chars is rejected."""
        output = LLMExplainOutput(
            headline="A" * 181,  # Over limit
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(sample_llm_input, output)
        assert "max length" in str(exc_info.value).lower()
