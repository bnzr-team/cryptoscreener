"""
Tests for LLM number validation edge cases.

Per LLM_INPUT_OUTPUT_SCHEMA.md: "subset of input numbers (stringwise)"

This module tests edge cases around float representation:
- 0.1 + 0.2 = 0.30000000000000004 (IEEE 754 issue)
- Different string representations of same value
- Trailing zeros
- Scientific notation
"""

from __future__ import annotations

import pytest

from cryptoscreener.contracts import (
    LLMExplainInput,
    LLMExplainOutput,
    LLMOutputViolation,
    LLMStyle,
    NumericSummary,
    PredictionStatus,
    ReasonCode,
    validate_llm_output_no_new_numbers,
    validate_llm_output_strict,
)


class TestFloatRepresentationEdgeCases:
    """Test float string representation edge cases."""

    @pytest.fixture
    def input_with_float_edge_case(self) -> LLMExplainInput:
        """Input with values that have tricky float representations."""
        return LLMExplainInput(
            symbol="BTCUSDT",
            timeframe="2m",
            status=PredictionStatus.WATCH,
            score=0.3,  # Note: 0.1 + 0.2 in IEEE 754 = 0.30000000000000004
            reasons=[
                ReasonCode(
                    code="RC_TEST",
                    value=0.1,  # IEEE 754: exactly representable
                    unit="ratio",
                    evidence="test",
                ),
            ],
            numeric_summary=NumericSummary(
                spread_bps=0.7,  # Tricky in binary
                impact_bps=1.5,
                p_toxic=0.15,
                regime="test",
            ),
            style=LLMStyle(tone="friendly", max_chars=180),
        )

    def test_exact_float_string_allowed(self, input_with_float_edge_case: LLMExplainInput) -> None:
        """Test that exact float string representation is allowed."""
        # Using exact string representation of 0.3
        output = LLMExplainOutput(
            headline=f"Score is {input_with_float_edge_case.score}.",
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(input_with_float_edge_case, output)
        assert violations == []

    def test_rounded_representation_rejected(
        self, input_with_float_edge_case: LLMExplainInput
    ) -> None:
        """Test that rounded/modified representations are rejected.

        If input has 0.15, output with "15%" or "0.150" should be rejected
        because "15" and "0.150" are not in the allowed set.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: toxicity at 15%.",  # 15 is NOT 0.15
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(input_with_float_edge_case, output)
        assert len(violations) > 0
        assert "15" in " ".join(violations)

    def test_trailing_zeros_rejected(self, input_with_float_edge_case: LLMExplainInput) -> None:
        """Test that adding trailing zeros is rejected.

        0.3 != "0.30" as strings.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: score 0.30.",  # 0.30 is NOT same string as 0.3
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(input_with_float_edge_case, output)
        # 0.30 should be flagged (if it's not in allowed set)
        # Note: Python's str(0.3) = "0.3", not "0.30"
        # The regex will extract "0.30" and it won't match "0.3"
        assert len(violations) > 0

    def test_scientific_notation_rejected(
        self, input_with_float_edge_case: LLMExplainInput
    ) -> None:
        """Test that scientific notation is rejected.

        0.0001 as "1e-4" should be rejected if input used decimal notation.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: value is 3e-1.",  # 3e-1 = 0.3 but different string
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        # Note: Our regex doesn't match scientific notation, so "3e-1" won't extract "3"
        # This is actually safe behavior - it won't find numbers it can't parse
        violations = validate_llm_output_no_new_numbers(input_with_float_edge_case, output)
        # The "3" part of "3e-1" might be extracted, check behavior
        # Actually with pattern r"-?\d+\.?\d*|-?\.\d+" it will extract "3"
        # 3 is not in input, so should be rejected
        assert len(violations) > 0


class TestIntegerFloatEquivalence:
    """Test handling of integer/float equivalence (1.0 vs 1)."""

    @pytest.fixture
    def input_with_whole_numbers(self) -> LLMExplainInput:
        """Input with whole number floats."""
        return LLMExplainInput(
            symbol="BTCUSDT",
            timeframe="2m",
            status=PredictionStatus.WATCH,
            score=1.0,  # Whole number as float
            reasons=[
                ReasonCode(
                    code="RC_TEST",
                    value=2.0,
                    unit="z",
                    evidence="test",
                ),
            ],
            numeric_summary=NumericSummary(
                spread_bps=5.0,  # Whole number
                impact_bps=3.0,
                p_toxic=0.0,  # Zero
                regime="test",
            ),
            style=LLMStyle(tone="friendly", max_chars=180),
        )

    def test_integer_form_of_whole_float_allowed(
        self, input_with_whole_numbers: LLMExplainInput
    ) -> None:
        """Test that integer form of whole number float is allowed.

        If input has 5.0, output with "5" should be allowed.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: spread is 5 bps.",  # 5 instead of 5.0
            subtext="Score 1, z-score 2",  # 1 and 2 instead of 1.0 and 2.0
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(input_with_whole_numbers, output)
        assert violations == []

    def test_float_form_allowed(self, input_with_whole_numbers: LLMExplainInput) -> None:
        """Test that float form is also allowed."""
        output = LLMExplainOutput(
            headline="BTCUSDT: spread is 5.0 bps.",
            subtext="Score: 1.0.",
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(input_with_whole_numbers, output)
        assert violations == []


class TestZeroHandling:
    """Test special handling of zero values."""

    @pytest.fixture
    def input_with_zero(self) -> LLMExplainInput:
        """Input with zero values."""
        return LLMExplainInput(
            symbol="BTCUSDT",
            timeframe="2m",
            status=PredictionStatus.WATCH,
            score=0.5,
            reasons=[],
            numeric_summary=NumericSummary(
                spread_bps=0.0,
                impact_bps=0.0,
                p_toxic=0.0,
                regime="test",
            ),
            style=LLMStyle(tone="friendly", max_chars=180),
        )

    def test_zero_integer_allowed(self, input_with_zero: LLMExplainInput) -> None:
        """Test that '0' is allowed when input has 0.0."""
        output = LLMExplainOutput(
            headline="BTCUSDT: 0 spread, 0 toxicity.",
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(input_with_zero, output)
        assert violations == []

    def test_zero_float_allowed(self, input_with_zero: LLMExplainInput) -> None:
        """Test that '0.0' is also allowed."""
        output = LLMExplainOutput(
            headline="BTCUSDT: 0.0 spread.",
            subtext="",
            status_label="Watch",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(input_with_zero, output)
        assert violations == []


class TestNegativeNumbers:
    """Test handling of negative numbers."""

    @pytest.fixture
    def input_with_negative(self) -> LLMExplainInput:
        """Input with negative values (e.g., expected utility can be negative)."""
        return LLMExplainInput(
            symbol="BTCUSDT",
            timeframe="2m",
            status=PredictionStatus.TRAP,
            score=0.2,
            reasons=[
                ReasonCode(
                    code="RC_NEGATIVE",
                    value=-2.5,  # Negative z-score
                    unit="z",
                    evidence="flow_imbalance=-0.8",
                ),
            ],
            numeric_summary=NumericSummary(
                spread_bps=10.0,
                impact_bps=15.0,
                p_toxic=0.8,
                regime="high-vol",
            ),
            style=LLMStyle(tone="friendly", max_chars=180),
        )

    def test_negative_number_allowed(self, input_with_negative: LLMExplainInput) -> None:
        """Test that negative numbers from input are allowed."""
        output = LLMExplainOutput(
            headline="BTCUSDT: z-score -2.5, avoid.",
            subtext="",
            status_label="Trap",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(input_with_negative, output)
        assert violations == []

    def test_positive_version_of_negative_rejected(
        self, input_with_negative: LLMExplainInput
    ) -> None:
        """Test that converting negative to positive is rejected.

        If input has -2.5, output with "2.5" (without minus) should be rejected.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: z-score 2.5.",  # Missing minus sign
            subtext="",
            status_label="Trap",
            tooltips={},
        )
        violations = validate_llm_output_no_new_numbers(input_with_negative, output)
        # "2.5" is not in allowed set (only "-2.5" is)
        assert len(violations) > 0


class TestStrictValidationWithEdgeCases:
    """Test that strict validation properly handles edge cases."""

    def test_strict_validation_rejects_converted_percentage(self) -> None:
        """Strict validation must reject percentage conversion."""
        llm_input = LLMExplainInput(
            symbol="BTCUSDT",
            timeframe="2m",
            status=PredictionStatus.WATCH,
            score=0.85,
            reasons=[],
            numeric_summary=NumericSummary(
                spread_bps=5.0,
                impact_bps=3.0,
                p_toxic=0.20,
                regime="test",
            ),
            style=LLMStyle(tone="friendly", max_chars=180),
        )

        output = LLMExplainOutput(
            headline="BTCUSDT: 85% confidence, 20% toxicity.",
            subtext="",
            status_label="Watch",
            tooltips={},
        )

        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(llm_input, output)

        error_msg = str(exc_info.value)
        assert "85" in error_msg or "20" in error_msg
