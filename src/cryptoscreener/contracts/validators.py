"""
Schema validators for data contracts.

Provides validation utilities and adversarial tests for LLM output compliance.

Per LLM_INPUT_OUTPUT_SCHEMA.md:
- No numbers may appear unless they were present in input numeric_summary (exact match).
- Extract all numbers (regex) from output and ensure subset of input numbers (stringwise).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError

from cryptoscreener.contracts.events import (
    LLMExplainInput,
    LLMExplainOutput,
    LLMStyle,
    NumericSummary,
    PredictionStatus,
)


class LLMOutputViolation(Exception):
    """Raised when LLM output violates safety constraints."""

    pass


# Allowed status labels per spec
ALLOWED_STATUS_LABELS = frozenset(
    {
        "Tradeable",
        "Tradeable soon",
        "Watch",
        "Watching",
        "Trap",
        "Avoid",
        "Dead",
        "Data issue",
    }
)


def extract_number_strings(text: str) -> list[str]:
    """
    Extract all numeric string representations from text.

    Returns raw string matches for stringwise comparison.
    Patterns matched: integers, decimals, negative numbers.

    Args:
        text: Input text to scan.

    Returns:
        List of number strings as they appear in text.
    """
    # Match integers and floats, including negative numbers
    # Patterns: 123, 12.34, -5.6, 0.83, .5
    pattern = r"-?\d+\.?\d*|-?\.\d+"
    matches = re.findall(pattern, text)
    # Filter out empty/invalid matches
    return [m for m in matches if m not in ("", "-", ".")]


def get_allowed_number_strings(input_data: LLMExplainInput) -> set[str]:
    """
    Extract all number strings from LLM input that are allowed in output.

    Per spec: "subset of input numbers (stringwise)" - exact string match only.
    No conversions (0.83 -> 83%) are allowed.

    Args:
        input_data: The LLMExplainInput provided to LLM.

    Returns:
        Set of allowed number strings (exact representations).
    """
    allowed: set[str] = set()

    # Helper to add number and common string representations
    def add_number(value: float) -> None:
        # Add the exact representation
        allowed.add(str(value))
        # Also add integer form if it's a whole number
        if value == int(value):
            allowed.add(str(int(value)))

    # Score
    add_number(input_data.score)

    # Numeric summary
    ns = input_data.numeric_summary
    add_number(ns.spread_bps)
    add_number(ns.impact_bps)
    add_number(ns.p_toxic)

    # Reason codes
    for reason in input_data.reasons:
        add_number(reason.value)

    return allowed


def validate_llm_output_no_new_numbers(
    input_data: LLMExplainInput,
    output_data: LLMExplainOutput,
) -> list[str]:
    """
    Validate that LLM output contains no numbers that weren't in the input.

    Per LLM_INPUT_OUTPUT_SCHEMA.md:
    - "No numbers may appear unless they were present in input numeric_summary (exact match)"
    - "Extract all numbers (regex) from output and ensure subset of input numbers (stringwise)"

    This is STRICT stringwise matching. No conversions allowed (e.g., 0.83 -> 83% is INVALID).

    Args:
        input_data: The LLMExplainInput provided to LLM.
        output_data: The LLMExplainOutput from LLM.

    Returns:
        List of violation messages (empty if valid).
    """
    violations: list[str] = []
    allowed = get_allowed_number_strings(input_data)

    # Check all text fields in output
    text_fields = [
        ("headline", output_data.headline),
        ("subtext", output_data.subtext),
        ("status_label", output_data.status_label),
    ]

    for field_name, text in text_fields:
        if not text:
            continue

        found_numbers = extract_number_strings(text)
        for num_str in found_numbers:
            # Stringwise exact match
            if num_str not in allowed:
                violations.append(
                    f"Field '{field_name}' contains unauthorized number: '{num_str}'. "
                    f"Allowed (stringwise): {sorted(allowed)}"
                )

    # Check tooltips
    for tooltip_key, tooltip_text in output_data.tooltips.items():
        found_numbers = extract_number_strings(tooltip_text)
        for num_str in found_numbers:
            if num_str not in allowed:
                violations.append(
                    f"Tooltip '{tooltip_key}' contains unauthorized number: '{num_str}'. "
                    f"Allowed (stringwise): {sorted(allowed)}"
                )

    return violations


def validate_status_label(status_label: str) -> list[str]:
    """
    Validate that status_label is one of allowed values.

    Args:
        status_label: The status label from LLM output.

    Returns:
        List of violation messages (empty if valid).
    """
    if status_label not in ALLOWED_STATUS_LABELS:
        return [f"Invalid status_label: '{status_label}'. Allowed: {sorted(ALLOWED_STATUS_LABELS)}"]
    return []


def validate_max_length(output_data: LLMExplainOutput, max_chars: int = 180) -> list[str]:
    """
    Validate headline length constraint.

    Args:
        output_data: The LLMExplainOutput from LLM.
        max_chars: Maximum allowed characters for headline.

    Returns:
        List of violation messages (empty if valid).
    """
    if len(output_data.headline) > max_chars:
        return [f"Headline exceeds max length: {len(output_data.headline)} > {max_chars} chars"]
    return []


def validate_llm_output_strict(
    input_data: LLMExplainInput,
    output_data: LLMExplainOutput,
) -> None:
    """
    Strictly validate LLM output per LLM_INPUT_OUTPUT_SCHEMA.md, raising exception on violation.

    Validation steps per spec:
    1. Parse JSON (already done via Pydantic)
    2. Validate required keys (headline, status_label - enforced by Pydantic)
    3. Check status_label in enum
    4. Extract all numbers (regex) from output and ensure subset of input numbers (stringwise)
    5. Enforce max length

    Args:
        input_data: The LLMExplainInput provided to LLM.
        output_data: The LLMExplainOutput from LLM.

    Raises:
        LLMOutputViolation: If any constraint is violated.
    """
    violations: list[str] = []

    # Step 3: Check status_label
    violations.extend(validate_status_label(output_data.status_label))

    # Step 4: Check numbers (stringwise exact match)
    violations.extend(validate_llm_output_no_new_numbers(input_data, output_data))

    # Step 5: Check max length
    max_chars = input_data.style.max_chars if input_data.style else 180
    violations.extend(validate_max_length(output_data, max_chars))

    if violations:
        raise LLMOutputViolation(
            "LLM output violates safety constraints:\n" + "\n".join(f"  - {v}" for v in violations)
        )


def generate_fallback_output(input_data: LLMExplainInput) -> LLMExplainOutput:
    """
    Generate deterministic fallback output when LLM output is invalid.

    Per LLM_SAFETY_GUARDRAILS.md: "If invalid → fallback deterministic template"

    This provides a safe, template-based response that:
    - Contains only numbers from input (stringwise)
    - Uses allowed status labels
    - Respects max length constraints

    Args:
        input_data: The LLMExplainInput that was provided to LLM.

    Returns:
        Safe LLMExplainOutput with deterministic template.
    """
    # Map prediction status to allowed status label
    status_to_label = {
        PredictionStatus.TRADEABLE: "Tradeable",
        PredictionStatus.WATCH: "Watch",
        PredictionStatus.TRAP: "Trap",
        PredictionStatus.DEAD: "Dead",
        PredictionStatus.DATA_ISSUE: "Data issue",
    }

    status_label = status_to_label.get(input_data.status, "Watch")

    # Build headline from template (no invented numbers)
    headline = f"{input_data.symbol}: {status_label.lower()}."

    # Add primary reason if available (uses numbers from input)
    if input_data.reasons:
        reason = input_data.reasons[0]
        headline = (
            f"{input_data.symbol}: {reason.code.replace('RC_', '').replace('_', ' ').lower()}."
        )

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


def validate_or_fallback(
    input_data: LLMExplainInput,
    output_data: LLMExplainOutput,
) -> tuple[LLMExplainOutput, bool]:
    """
    Validate LLM output and return fallback if invalid.

    Args:
        input_data: The LLMExplainInput provided to LLM.
        output_data: The LLMExplainOutput from LLM.

    Returns:
        Tuple of (validated_or_fallback_output, was_valid).
    """
    try:
        validate_llm_output_strict(input_data, output_data)
        return output_data, True
    except LLMOutputViolation:
        return generate_fallback_output(input_data), False


def validate_contract_json(
    contract_class: type[BaseModel], json_data: dict[str, Any]
) -> tuple[bool, str]:
    """
    Validate JSON data against a contract schema.

    Args:
        contract_class: The Pydantic model class to validate against.
        json_data: Dictionary of JSON data.

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        contract_class.model_validate(json_data)
        return True, ""
    except ValidationError as e:
        return False, str(e)


def create_test_llm_input() -> LLMExplainInput:
    """Create a test LLMExplainInput for validation testing."""
    from cryptoscreener.contracts.events import ReasonCode

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


def create_valid_llm_output() -> LLMExplainOutput:
    """Create a valid LLMExplainOutput that passes validation."""
    return LLMExplainOutput(
        headline="BTCUSDT: flow surge + tight spread, likely tradable soon.",
        subtext="Watch for quick breakout; toxicity low-to-moderate.",
        status_label="Watch",  # Must be in ALLOWED_STATUS_LABELS
        tooltips={"p_inplay": "Calibrated probability of net edge after costs."},
    )


def create_invalid_llm_output_with_new_numbers() -> LLMExplainOutput:
    """Create an INVALID LLMExplainOutput with unauthorized numbers."""
    return LLMExplainOutput(
        headline="BTCUSDT: flow surge detected. Expected gain: 15.7% in 2m.",  # 15.7 is NEW
        subtext="Watch for quick breakout; toxicity at 99%.",  # 99 is NEW
        status_label="Watch",
        tooltips={"p_inplay": "Probability is 0.95."},  # 0.95 is NEW
    )
