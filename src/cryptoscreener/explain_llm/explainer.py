"""
LLM Explain interface and implementations.

Per CLAUDE.md §9 and DEC-004:
- Strict ExplainLLM interface (sync)
- LLM outputs are text-only; no numeric/status changes allowed
- no-new-numbers enforced via validate_llm_output_strict
- Fallback on any failure (exception/timeout/invalid output)
- Anthropic-only provider in this PR; others may be added later
- Unit tests do not perform network calls; provider client is injected/mocked
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import orjson

from cryptoscreener.contracts.events import (
    LLMExplainInput,
    LLMExplainOutput,
    PredictionStatus,
)
from cryptoscreener.contracts.validators import (
    ALLOWED_STATUS_LABELS,
    generate_fallback_output,
    validate_llm_output_strict,
)

if TYPE_CHECKING:
    from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ExplainLLM(Protocol):
    """
    Strict LLM explain interface per CLAUDE.md §9.

    LLM generates text explanations only. It MUST NOT:
    - Change any numeric values
    - Modify prediction status or score
    - Introduce new numbers not present in input

    All implementations must validate output and fall back on failure.
    """

    def explain(self, input_data: LLMExplainInput) -> LLMExplainOutput:
        """
        Generate text explanation for trading signal.

        Args:
            input_data: Structured input with symbol, status, reasons, numeric_summary.

        Returns:
            LLMExplainOutput with headline, subtext, status_label, tooltips.
            On any error, returns deterministic fallback.
        """
        ...


# Status mapping for deterministic fallback
STATUS_TO_LABEL: dict[PredictionStatus, str] = {
    PredictionStatus.TRADEABLE: "Tradeable",
    PredictionStatus.WATCH: "Watch",
    PredictionStatus.TRAP: "Trap",
    PredictionStatus.DEAD: "Dead",
    PredictionStatus.DATA_ISSUE: "Data issue",
}


class MockExplainer:
    """
    Deterministic mock explainer for tests and 'LLM off' mode.

    Returns template-based responses without any numbers.
    Fully deterministic: same input always produces same output.
    """

    def explain(self, input_data: LLMExplainInput) -> LLMExplainOutput:
        """Generate deterministic explanation without numbers."""
        status_label = STATUS_TO_LABEL.get(input_data.status, "Watch")

        # Build headline from template (no numbers)
        if input_data.reasons:
            reason = input_data.reasons[0]
            reason_text = reason.code.replace("RC_", "").replace("_", " ").lower()
            headline = f"{input_data.symbol}: {reason_text} detected."
        else:
            headline = f"{input_data.symbol}: {status_label.lower()} conditions."

        # Ensure within max_chars
        max_chars = input_data.style.max_chars if input_data.style else 180
        if len(headline) > max_chars:
            headline = headline[: max_chars - 3] + "..."

        return LLMExplainOutput(
            headline=headline,
            subtext="",
            status_label=status_label,
            tooltips={},
        )


@dataclass
class AnthropicExplainerConfig:
    """Configuration for AnthropicExplainer."""

    model: str = "claude-3-haiku-20240307"
    max_tokens: int = 300
    timeout_s: float = 10.0
    retries: int = 1
    temperature: float = 0.0  # Deterministic for predictability
    max_chars: int = 180

    # Retry settings
    retry_base_delay_s: float = 0.5
    retry_max_delay_s: float = 2.0
    retry_jitter: float = 0.1


# Prompt template with strict guardrails
PROMPT_TEMPLATE = """You are a trading signal explanation assistant.

STRICT RULES (MUST FOLLOW):
1. You MUST NOT introduce any new numbers. You can ONLY use numbers that appear in the input below.
2. You MUST NOT convert numbers (e.g., 0.83 to 83% is FORBIDDEN).
3. You MUST NOT round, truncate, or modify numbers in any way.
4. status_label MUST be one of: {allowed_labels}
5. headline MUST be <= {max_chars} characters.
6. Output MUST be valid JSON with exactly these fields: headline, subtext, status_label, tooltips

INPUT:
Symbol: {symbol}
Timeframe: {timeframe}
Status: {status}
Score: {score}
Reasons: {reasons}
Numeric Summary:
  - spread_bps: {spread_bps}
  - impact_bps: {impact_bps}
  - p_toxic: {p_toxic}
  - regime: {regime}

OUTPUT FORMAT (JSON only, no markdown):
{{
  "headline": "Short explanation (max {max_chars} chars)",
  "subtext": "Additional context (optional, can be empty)",
  "status_label": "One of allowed labels",
  "tooltips": {{"field": "explanation"}}
}}

Generate the JSON output now:"""


def _build_prompt(input_data: LLMExplainInput) -> str:
    """Build prompt from input data with guardrails."""
    reasons_text = ", ".join(f"{r.code}={r.value}{r.unit}" for r in input_data.reasons) or "none"

    max_chars = input_data.style.max_chars if input_data.style else 180

    return PROMPT_TEMPLATE.format(
        allowed_labels=", ".join(sorted(ALLOWED_STATUS_LABELS)),
        max_chars=max_chars,
        symbol=input_data.symbol,
        timeframe=input_data.timeframe,
        status=input_data.status.value,
        score=input_data.score,
        reasons=reasons_text,
        spread_bps=input_data.numeric_summary.spread_bps,
        impact_bps=input_data.numeric_summary.impact_bps,
        p_toxic=input_data.numeric_summary.p_toxic,
        regime=input_data.numeric_summary.regime,
    )


def _parse_llm_response(raw_response: str) -> LLMExplainOutput:
    """
    Parse LLM response to LLMExplainOutput.

    Expects JSON format. Raises ValueError on parse failure.
    """
    # Try to extract JSON from response (in case of markdown wrapper)
    text = raw_response.strip()

    # Remove markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Find first and last ``` lines
        start_idx = 0
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.startswith("```") and i == 0:
                start_idx = 1
            elif line.startswith("```"):
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])

    # Parse JSON
    try:
        data = orjson.loads(text.encode())
    except (orjson.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid JSON in LLM response: {e}") from e

    # Validate required fields
    if "headline" not in data:
        raise ValueError("Missing 'headline' in LLM response")
    if "status_label" not in data:
        raise ValueError("Missing 'status_label' in LLM response")

    return LLMExplainOutput(
        headline=str(data.get("headline", "")),
        subtext=str(data.get("subtext", "")),
        status_label=str(data.get("status_label", "")),
        tooltips=data.get("tooltips", {}),
    )


@dataclass
class AnthropicExplainer:
    """
    Anthropic-based LLM explainer with strict guardrails.

    Per DEC-004:
    - Anthropic-only provider (others may be added later)
    - Sync interface
    - Validates all outputs for no-new-numbers constraint
    - Falls back on any failure
    - Client is injectable for testing (no network calls in tests)
    """

    config: AnthropicExplainerConfig = field(default_factory=AnthropicExplainerConfig)
    _client: Anthropic | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize Anthropic client if not injected."""
        if self._client is None:
            # Only import and create client if not injected
            # This allows tests to inject a mock client
            try:
                from anthropic import Anthropic

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    self._client = Anthropic(api_key=api_key)
                else:
                    logger.warning(
                        "ANTHROPIC_API_KEY not set; AnthropicExplainer will use fallback"
                    )
            except ImportError:
                logger.warning(
                    "anthropic package not installed; AnthropicExplainer will use fallback"
                )

    @classmethod
    def with_client(
        cls,
        client: Anthropic,
        config: AnthropicExplainerConfig | None = None,
    ) -> AnthropicExplainer:
        """Create explainer with injected client (for testing)."""
        explainer = cls(config=config or AnthropicExplainerConfig())
        explainer._client = client
        return explainer

    def explain(self, input_data: LLMExplainInput) -> LLMExplainOutput:
        """
        Generate explanation via Anthropic API.

        On any failure (network, timeout, validation), returns fallback.
        """
        if self._client is None:
            logger.info("No Anthropic client available, using fallback")
            return generate_fallback_output(input_data)

        prompt = _build_prompt(input_data)

        # Try with retries
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                output = self._call_and_validate(input_data, prompt)
                logger.info(
                    "LLM explain success",
                    extra={
                        "provider": "anthropic",
                        "symbol": input_data.symbol,
                        "attempt": attempt + 1,
                    },
                )
                return output
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM explain attempt failed",
                    extra={
                        "provider": "anthropic",
                        "symbol": input_data.symbol,
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                        "reason": str(e)[:100],  # Truncate for safety
                    },
                )

                if attempt < self.config.retries:
                    # Exponential backoff with jitter
                    delay = min(
                        self.config.retry_base_delay_s * (2**attempt),
                        self.config.retry_max_delay_s,
                    )
                    jitter = random.uniform(0, self.config.retry_jitter)
                    time.sleep(delay + jitter)

        # All retries exhausted, return fallback
        logger.error(
            "LLM explain failed, using fallback",
            extra={
                "provider": "anthropic",
                "symbol": input_data.symbol,
                "final_error": str(last_error)[:100] if last_error else "unknown",
            },
        )
        return generate_fallback_output(input_data)

    def _call_and_validate(
        self,
        input_data: LLMExplainInput,
        prompt: str,
    ) -> LLMExplainOutput:
        """
        Call Anthropic API and validate response.

        Raises on any failure (network, parse, validation).
        """
        assert self._client is not None

        # Call Anthropic API with timeout enforcement
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout_s,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text content
        if not response.content:
            raise ValueError("Empty response from Anthropic API")

        raw_text = response.content[0].text

        # Parse response
        output = _parse_llm_response(raw_text)

        # Strict validation (no-new-numbers, status_label, max_length)
        validate_llm_output_strict(input_data, output)

        return output
