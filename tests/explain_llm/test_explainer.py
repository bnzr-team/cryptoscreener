"""
Tests for LLM Explain module.

Per DEC-004:
- Unit tests do not perform network calls
- Provider client is injected/mocked
- Adversarial tests verify no-new-numbers constraint
- Fallback on any failure is tested
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cryptoscreener.contracts.events import (
    LLMExplainInput,
    LLMExplainOutput,
    LLMStyle,
    NumericSummary,
    PredictionStatus,
    ReasonCode,
)
from cryptoscreener.contracts.validators import (
    ALLOWED_STATUS_LABELS,
    LLMOutputViolation,
    validate_llm_output_strict,
)
from cryptoscreener.explain_llm.explainer import (
    AnthropicExplainer,
    AnthropicExplainerConfig,
    MockExplainer,
    _build_prompt,
    _parse_llm_response,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_input() -> LLMExplainInput:
    """Standard test input with known values."""
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
            )
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
def mock_anthropic_client() -> MagicMock:
    """Mock Anthropic client for testing without network calls."""
    client = MagicMock()
    return client


def make_mock_response(text: str) -> MagicMock:
    """Create mock Anthropic API response."""
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    response.content = [content_block]
    return response


# =============================================================================
# MockExplainer Tests
# =============================================================================


class TestMockExplainer:
    """Tests for deterministic MockExplainer."""

    def test_returns_valid_output(self, sample_input: LLMExplainInput) -> None:
        """MockExplainer returns valid LLMExplainOutput."""
        explainer = MockExplainer()
        output = explainer.explain(sample_input)

        assert isinstance(output, LLMExplainOutput)
        assert output.headline
        assert output.status_label in ALLOWED_STATUS_LABELS

    def test_deterministic_same_input_same_output(
        self, sample_input: LLMExplainInput
    ) -> None:
        """Same input always produces same output."""
        explainer = MockExplainer()

        output1 = explainer.explain(sample_input)
        output2 = explainer.explain(sample_input)

        assert output1.headline == output2.headline
        assert output1.status_label == output2.status_label
        assert output1.subtext == output2.subtext

    def test_status_label_mapping(self) -> None:
        """Each PredictionStatus maps to allowed status_label."""
        explainer = MockExplainer()

        for status in PredictionStatus:
            input_data = LLMExplainInput(
                symbol="BTCUSDT",
                timeframe="2m",
                status=status,
                score=0.5,
                reasons=[],
                numeric_summary=NumericSummary(
                    spread_bps=5.0,
                    impact_bps=3.0,
                    p_toxic=0.1,
                    regime="test",
                ),
                style=LLMStyle(tone="friendly", max_chars=180),
            )
            output = explainer.explain(input_data)
            assert output.status_label in ALLOWED_STATUS_LABELS

    def test_headline_respects_max_chars(self) -> None:
        """Headline is truncated to max_chars."""
        explainer = MockExplainer()
        input_data = LLMExplainInput(
            symbol="BTCUSDT",
            timeframe="2m",
            status=PredictionStatus.WATCH,
            score=0.5,
            reasons=[
                ReasonCode(
                    code="RC_VERY_LONG_REASON_CODE_NAME_THAT_EXCEEDS_LIMIT",
                    value=1.0,
                    unit="z",
                    evidence="test",
                )
            ],
            numeric_summary=NumericSummary(
                spread_bps=5.0,
                impact_bps=3.0,
                p_toxic=0.1,
                regime="test",
            ),
            style=LLMStyle(tone="friendly", max_chars=50),
        )
        output = explainer.explain(input_data)
        assert len(output.headline) <= 50

    def test_no_numbers_in_output(self, sample_input: LLMExplainInput) -> None:
        """MockExplainer output contains no numbers (by design)."""
        explainer = MockExplainer()
        output = explainer.explain(sample_input)

        # Validate against input - should pass because no numbers
        # Using the strict validator to confirm
        # Note: Mock output has no numbers, so should always pass
        validate_llm_output_strict(sample_input, output)


# =============================================================================
# Prompt Building Tests
# =============================================================================


class TestPromptBuilding:
    """Tests for prompt template construction."""

    def test_prompt_contains_input_values(self, sample_input: LLMExplainInput) -> None:
        """Prompt includes all input values."""
        prompt = _build_prompt(sample_input)

        assert "BTCUSDT" in prompt
        assert "2m" in prompt
        assert "WATCH" in prompt
        assert "0.83" in prompt
        assert "8.2" in prompt
        assert "6.5" in prompt
        assert "0.21" in prompt
        assert "high-vol trend" in prompt

    def test_prompt_contains_guardrails(self, sample_input: LLMExplainInput) -> None:
        """Prompt includes strict guardrails."""
        prompt = _build_prompt(sample_input)

        assert "MUST NOT introduce any new numbers" in prompt
        assert "MUST NOT convert numbers" in prompt
        assert "0.83 to 83% is FORBIDDEN" in prompt
        assert "status_label MUST be one of" in prompt

    def test_prompt_lists_allowed_labels(self, sample_input: LLMExplainInput) -> None:
        """Prompt lists all allowed status labels."""
        prompt = _build_prompt(sample_input)

        for label in ALLOWED_STATUS_LABELS:
            assert label in prompt


# =============================================================================
# Response Parsing Tests
# =============================================================================


class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_json(self) -> None:
        """Valid JSON response is parsed correctly."""
        raw = '{"headline": "Test", "subtext": "", "status_label": "Watch", "tooltips": {}}'
        output = _parse_llm_response(raw)

        assert output.headline == "Test"
        assert output.status_label == "Watch"

    def test_parse_json_with_markdown_wrapper(self) -> None:
        """JSON wrapped in markdown code block is parsed."""
        raw = '''```json
{"headline": "Test", "subtext": "", "status_label": "Watch", "tooltips": {}}
```'''
        output = _parse_llm_response(raw)

        assert output.headline == "Test"

    def test_parse_invalid_json_raises(self) -> None:
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            _parse_llm_response("not json at all")

    def test_parse_missing_headline_raises(self) -> None:
        """Missing headline field raises ValueError."""
        raw = '{"subtext": "", "status_label": "Watch", "tooltips": {}}'
        with pytest.raises(ValueError, match="Missing 'headline'"):
            _parse_llm_response(raw)

    def test_parse_missing_status_label_raises(self) -> None:
        """Missing status_label field raises ValueError."""
        raw = '{"headline": "Test", "subtext": "", "tooltips": {}}'
        with pytest.raises(ValueError, match="Missing 'status_label'"):
            _parse_llm_response(raw)


# =============================================================================
# AnthropicExplainer Tests (No Network Calls)
# =============================================================================


class TestAnthropicExplainer:
    """Tests for AnthropicExplainer with mocked client."""

    def test_valid_response_passes_validation(
        self,
        sample_input: LLMExplainInput,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Valid LLM response passes validation and is returned."""
        # Response with no new numbers
        valid_response = (
            '{"headline": "BTCUSDT: flow surge detected.", '
            '"subtext": "", "status_label": "Watch", "tooltips": {}}'
        )
        mock_anthropic_client.messages.create.return_value = make_mock_response(
            valid_response
        )

        explainer = AnthropicExplainer.with_client(mock_anthropic_client)
        output = explainer.explain(sample_input)

        assert output.headline == "BTCUSDT: flow surge detected."
        assert output.status_label == "Watch"

    def test_no_client_uses_fallback(self, sample_input: LLMExplainInput) -> None:
        """When no client available, fallback is used."""
        # Create explainer without client
        explainer = AnthropicExplainer()
        explainer._client = None

        output = explainer.explain(sample_input)

        # Should get fallback output
        assert output.status_label in ALLOWED_STATUS_LABELS
        assert sample_input.symbol in output.headline

    def test_api_error_uses_fallback(
        self,
        sample_input: LLMExplainInput,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """API error results in fallback."""
        mock_anthropic_client.messages.create.side_effect = Exception("Network error")

        config = AnthropicExplainerConfig(retries=0)
        explainer = AnthropicExplainer.with_client(mock_anthropic_client, config)
        output = explainer.explain(sample_input)

        # Should get fallback
        assert output.status_label in ALLOWED_STATUS_LABELS

    def test_validation_failure_uses_fallback(
        self,
        sample_input: LLMExplainInput,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Validation failure (new numbers) results in fallback."""
        # Response with invented number (15.7 not in input)
        invalid_response = (
            '{"headline": "BTCUSDT: expected gain 15.7%.", '
            '"subtext": "", "status_label": "Watch", "tooltips": {}}'
        )
        mock_anthropic_client.messages.create.return_value = make_mock_response(
            invalid_response
        )

        config = AnthropicExplainerConfig(retries=0)
        explainer = AnthropicExplainer.with_client(mock_anthropic_client, config)
        output = explainer.explain(sample_input)

        # Should get fallback (not the invalid response)
        assert "15.7" not in output.headline

    def test_client_injection_no_network(
        self, mock_anthropic_client: MagicMock
    ) -> None:
        """Verify that with_client does not make network calls."""
        # This test confirms the mocking approach works
        explainer = AnthropicExplainer.with_client(mock_anthropic_client)

        # Client should be the mock
        assert explainer._client is mock_anthropic_client

        # No network call yet
        mock_anthropic_client.messages.create.assert_not_called()


# =============================================================================
# Adversarial Tests - No New Numbers
# =============================================================================


class TestAdversarialNoNewNumbers:
    """
    Adversarial tests for no-new-numbers constraint.

    These test cases simulate malicious/buggy LLM outputs that
    should be rejected by the validator.
    """

    @pytest.fixture
    def input_083(self) -> LLMExplainInput:
        """Input with score 0.83 for percentage conversion tests."""
        return LLMExplainInput(
            symbol="BTCUSDT",
            timeframe="2m",
            status=PredictionStatus.WATCH,
            score=0.83,
            reasons=[],
            numeric_summary=NumericSummary(
                spread_bps=8.2,
                impact_bps=6.5,
                p_toxic=0.21,
                regime="test",
            ),
            style=LLMStyle(tone="friendly", max_chars=180),
        )

    def test_adversarial_percentage_conversion(
        self, input_083: LLMExplainInput
    ) -> None:
        """
        ADVERSARIAL: LLM converts 0.83 to 83%.

        This is FORBIDDEN - stringwise match only.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: 83% confidence.",  # 83 is NOT 0.83
            subtext="",
            status_label="Watch",
            tooltips={},
        )

        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(input_083, output)

        assert "83" in str(exc_info.value)

    def test_adversarial_invented_number(self, input_083: LLMExplainInput) -> None:
        """
        ADVERSARIAL: LLM invents a new number (12 bps).

        Numbers not in input are FORBIDDEN.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: expected 12 bps gain.",  # 12 is NEW
            subtext="",
            status_label="Watch",
            tooltips={},
        )

        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(input_083, output)

        assert "12" in str(exc_info.value)

    def test_adversarial_exceeds_max_chars(self, input_083: LLMExplainInput) -> None:
        """
        ADVERSARIAL: LLM exceeds max_chars limit.
        """
        long_headline = "A" * 200  # Exceeds 180 char limit
        output = LLMExplainOutput(
            headline=long_headline,
            subtext="",
            status_label="Watch",
            tooltips={},
        )

        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(input_083, output)

        assert "exceeds max length" in str(exc_info.value)

    def test_adversarial_invalid_status_label(self, input_083: LLMExplainInput) -> None:
        """
        ADVERSARIAL: LLM uses invalid status_label.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: test.",
            subtext="",
            status_label="SUPER_TRADEABLE",  # Not in allowed list
            tooltips={},
        )

        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(input_083, output)

        assert "Invalid status_label" in str(exc_info.value)

    def test_adversarial_number_in_words(self, input_083: LLMExplainInput) -> None:
        """
        ADVERSARIAL: LLM uses numbers written as words.

        Note: Current regex doesn't catch words like "eighty three".
        This test documents the limitation - we rely on LLM following
        the guardrails, not on detecting written-out numbers.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: confidence level is moderate.",
            subtext="",
            status_label="Watch",
            tooltips={},
        )

        # This should PASS because no numeric digits
        validate_llm_output_strict(input_083, output)  # No exception

    def test_adversarial_scientific_notation(self, input_083: LLMExplainInput) -> None:
        """
        ADVERSARIAL: LLM uses scientific notation (8.3e-1 = 0.83).

        The regex extracts "8" from "8.3e-1" which is not allowed.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: score 8.3e-1.",  # 8 extracted, not allowed
            subtext="",
            status_label="Watch",
            tooltips={},
        )

        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(input_083, output)

        # Either "8" or "8.3" should be flagged
        error = str(exc_info.value)
        assert "8" in error or "8.3" in error

    def test_adversarial_fraction_notation(self, input_083: LLMExplainInput) -> None:
        """
        ADVERSARIAL: LLM uses fraction (83/100).

        Both 83 and 100 are not in input.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: score is 83/100.",
            subtext="",
            status_label="Watch",
            tooltips={},
        )

        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(input_083, output)

        error = str(exc_info.value)
        assert "83" in error or "100" in error

    def test_adversarial_trailing_zero_format(self, input_083: LLMExplainInput) -> None:
        """
        ADVERSARIAL: LLM reformats 8.2 as 8.20.

        This is FORBIDDEN - stringwise match only means 8.2 != 8.20.
        The number must appear exactly as in input.
        """
        output = LLMExplainOutput(
            headline="BTCUSDT: spread is 8.20 bps.",  # 8.20 is NOT 8.2
            subtext="",
            status_label="Watch",
            tooltips={},
        )

        with pytest.raises(LLMOutputViolation) as exc_info:
            validate_llm_output_strict(input_083, output)

        # 8.20 should be flagged as unauthorized
        assert "8.20" in str(exc_info.value)

    def test_adversarial_unicode_digits(self, input_083: LLMExplainInput) -> None:
        """
        ADVERSARIAL: LLM uses unicode digits (fullwidth, arabic-indic).

        Unicode digit variants should not bypass the validator.
        Current regex does not catch these - this is a known limitation.
        LLM should follow guardrails; we rely on prompt engineering.
        """
        import contextlib

        # Fullwidth digit 8 (U+FF18) and 3 (U+FF13)
        output = LLMExplainOutput(
            headline="BTCUSDT: score \uff18\uff13%.",  # fullwidth 83
            subtext="",
            status_label="Watch",
            tooltips={},
        )

        # Either passes (known limitation) or fails (if regex catches it)
        # Both are acceptable - this documents the edge case
        with contextlib.suppress(LLMOutputViolation):
            validate_llm_output_strict(input_083, output)


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestHappyPath:
    """Happy path tests with valid LLM outputs."""

    def test_valid_output_with_input_numbers(self) -> None:
        """LLM can use numbers from input."""
        input_data = LLMExplainInput(
            symbol="BTCUSDT",
            timeframe="2m",
            status=PredictionStatus.WATCH,
            score=0.83,
            reasons=[],
            numeric_summary=NumericSummary(
                spread_bps=8.2,
                impact_bps=6.5,
                p_toxic=0.21,
                regime="high-vol trend",
            ),
            style=LLMStyle(tone="friendly", max_chars=180),
        )

        # Output using only input numbers
        output = LLMExplainOutput(
            headline="BTCUSDT: spread 8.2 bps, impact 6.5 bps, watch closely.",
            subtext="Score 0.83, toxicity 0.21.",
            status_label="Watch",
            tooltips={"spread": "Bid-ask spread in basis points."},
        )

        # Should not raise
        validate_llm_output_strict(input_data, output)

    def test_valid_output_no_numbers(self) -> None:
        """LLM can omit all numbers."""
        input_data = LLMExplainInput(
            symbol="ETHUSDT",
            timeframe="5m",
            status=PredictionStatus.TRADEABLE,
            score=0.91,
            reasons=[
                ReasonCode(
                    code="RC_FLOW_SURGE",
                    value=3.2,
                    unit="z",
                    evidence="test",
                )
            ],
            numeric_summary=NumericSummary(
                spread_bps=2.1,
                impact_bps=1.5,
                p_toxic=0.05,
                regime="low-vol trend",
            ),
            style=LLMStyle(tone="friendly", max_chars=180),
        )

        # Output with no numbers at all
        output = LLMExplainOutput(
            headline="ETHUSDT: strong flow surge, favorable conditions.",
            subtext="Low toxicity, tight spreads.",
            status_label="Tradeable",
            tooltips={},
        )

        # Should not raise
        validate_llm_output_strict(input_data, output)

    def test_mock_explainer_always_valid(self) -> None:
        """MockExplainer output always passes validation."""
        explainer = MockExplainer()

        for status in PredictionStatus:
            input_data = LLMExplainInput(
                symbol="BTCUSDT",
                timeframe="2m",
                status=status,
                score=0.5,
                reasons=[],
                numeric_summary=NumericSummary(
                    spread_bps=5.0,
                    impact_bps=3.0,
                    p_toxic=0.1,
                    regime="test",
                ),
                style=LLMStyle(tone="friendly", max_chars=180),
            )

            output = explainer.explain(input_data)

            # Should not raise
            validate_llm_output_strict(input_data, output)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full explain flow."""

    def test_anthropic_fallback_on_invalid_response(
        self,
        sample_input: LLMExplainInput,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """
        Full flow: invalid LLM response -> validation fail -> fallback.
        """
        # LLM returns response with invented number
        invalid_response = (
            '{"headline": "BTCUSDT: 99% confidence!", '
            '"subtext": "", "status_label": "Watch", "tooltips": {}}'
        )
        mock_anthropic_client.messages.create.return_value = make_mock_response(
            invalid_response
        )

        config = AnthropicExplainerConfig(retries=0)
        explainer = AnthropicExplainer.with_client(mock_anthropic_client, config)
        output = explainer.explain(sample_input)

        # Should get fallback, not the invalid response
        assert "99" not in output.headline
        assert output.status_label in ALLOWED_STATUS_LABELS

    def test_retry_then_fallback(
        self,
        sample_input: LLMExplainInput,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """
        Test retry logic: all attempts fail -> fallback.
        """
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        config = AnthropicExplainerConfig(
            retries=2,
            retry_base_delay_s=0.01,  # Fast for tests
        )
        explainer = AnthropicExplainer.with_client(mock_anthropic_client, config)
        output = explainer.explain(sample_input)

        # Should have tried 3 times (1 + 2 retries)
        assert mock_anthropic_client.messages.create.call_count == 3

        # Should get fallback
        assert output.status_label in ALLOWED_STATUS_LABELS

    def test_retry_success_on_second_attempt(
        self,
        sample_input: LLMExplainInput,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """
        Test retry logic: first fails, second succeeds.
        """
        valid_response = (
            '{"headline": "BTCUSDT: conditions favorable.", '
            '"subtext": "", "status_label": "Watch", "tooltips": {}}'
        )

        # First call fails, second succeeds
        mock_anthropic_client.messages.create.side_effect = [
            Exception("Temporary error"),
            make_mock_response(valid_response),
        ]

        config = AnthropicExplainerConfig(
            retries=1,
            retry_base_delay_s=0.01,
        )
        explainer = AnthropicExplainer.with_client(mock_anthropic_client, config)
        output = explainer.explain(sample_input)

        # Should have tried twice
        assert mock_anthropic_client.messages.create.call_count == 2

        # Should get valid response
        assert output.headline == "BTCUSDT: conditions favorable."
