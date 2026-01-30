"""RiskBreachEvent contract.

Notification of risk limit violation.
Producer: RiskManager
Consumer: SessionManager (triggers state change), AlertSink, Journal
"""

from __future__ import annotations

from decimal import Decimal  # noqa: TC003 - used at runtime in validators
from typing import Annotated

from pydantic import Field, field_validator

from cryptoscreener.trading.contracts.base import (
    TradingContractBase,
    parse_decimal,
)
from cryptoscreener.trading.contracts.types import (
    BreachSeverity,
    BreachType,
    RiskAction,
)


class RiskBreachEvent(TradingContractBase):
    """Notification of risk limit violation.

    Dedupe key: (session_id, breach_id).
    """

    ts: int = Field(description="Breach detection timestamp (ms)")
    breach_id: str = Field(description="Unique breach identifier (UUID)")
    breach_type: BreachType = Field(description="Type of breach")
    severity: BreachSeverity = Field(description="WARNING, CRITICAL, or FATAL")
    symbol: str | None = Field(
        default=None,
        description="Affected symbol (if symbol-specific)",
    )
    threshold: Annotated[Decimal, Field(description="Configured threshold value")] = Field()
    actual: Annotated[Decimal, Field(description="Actual value that triggered breach")] = Field()
    action_taken: RiskAction = Field(description="Automated action taken")
    details: str | None = Field(
        default=None,
        description="Additional context",
    )

    @field_validator("threshold", "actual", mode="before")
    @classmethod
    def parse_decimal_fields(cls, v: object) -> Decimal:
        """Parse Decimal fields."""
        return parse_decimal(v)

    @property
    def dedupe_key(self) -> tuple[str, str]:
        """Dedupe key: (session_id, breach_id)."""
        return (self.session_id, self.breach_id)

    @property
    def is_fatal(self) -> bool:
        """Check if breach is fatal (requires session kill)."""
        return self.severity == BreachSeverity.FATAL

    @property
    def requires_action(self) -> bool:
        """Check if breach requires automated action."""
        return self.action_taken != RiskAction.NONE
