"""PolicyOutput - Output from policy evaluation.

Defines the patterns and outputs that policy rules produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class PolicyPattern(str, Enum):
    """Action patterns from DEC-043 Action Vocabulary.

    Each pattern describes a type of action the policy takes.
    """

    EMIT_ORDERS = "EMIT_ORDERS"
    SUPPRESS_ENTRY = "SUPPRESS_ENTRY"
    SUPPRESS_ALL = "SUPPRESS_ALL"
    MODIFY_PARAMS = "MODIFY_PARAMS"
    FORCE_CLOSE = "FORCE_CLOSE"


@dataclass(frozen=True)
class PolicyOutput:
    """Output from policy evaluation.

    Describes what patterns are active and any parameter overrides.
    All fields are immutable.
    """

    # Active patterns (may have multiple)
    patterns_active: frozenset[PolicyPattern] = field(default_factory=frozenset)

    # Parameter overrides (e.g., {"spread_mult": 2.0})
    param_overrides: Mapping[str, Decimal] = field(default_factory=dict)

    # Force close position (POL-006, POL-010, POL-011)
    force_close: bool = False

    # Kill session (POL-019, POL-020)
    kill: bool = False

    # Active reason codes (text-only, no digits per DEC-042)
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def should_suppress_entry(self) -> bool:
        """Check if entry orders should be suppressed."""
        return (
            PolicyPattern.SUPPRESS_ENTRY in self.patterns_active
            or PolicyPattern.SUPPRESS_ALL in self.patterns_active
            or self.force_close
            or self.kill
        )

    @property
    def should_suppress_all(self) -> bool:
        """Check if all orders should be suppressed."""
        return PolicyPattern.SUPPRESS_ALL in self.patterns_active

    @property
    def has_spread_override(self) -> bool:
        """Check if spread multiplier is overridden."""
        return "spread_mult" in self.param_overrides

    @property
    def spread_mult(self) -> Decimal:
        """Get spread multiplier (default 1.0)."""
        return self.param_overrides.get("spread_mult", Decimal("1"))

    @classmethod
    def empty(cls) -> PolicyOutput:
        """Create empty output (no policy actions)."""
        return cls()

    @classmethod
    def with_patterns(
        cls,
        patterns: set[PolicyPattern],
        reason_codes: list[str],
        *,
        param_overrides: dict[str, Decimal] | None = None,
        force_close: bool = False,
        kill: bool = False,
    ) -> PolicyOutput:
        """Create output with specified patterns.

        Args:
            patterns: Active policy patterns.
            reason_codes: Reason codes for active patterns (text-only, no digits).
            param_overrides: Optional parameter overrides.
            force_close: Whether to force close position.
            kill: Whether to kill session.

        Returns:
            Policy output with specified patterns.
        """
        return cls(
            patterns_active=frozenset(patterns),
            param_overrides=param_overrides or {},
            force_close=force_close,
            kill=kill,
            reason_codes=tuple(reason_codes),
        )
