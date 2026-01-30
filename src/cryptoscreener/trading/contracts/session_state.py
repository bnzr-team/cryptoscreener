"""SessionState contract.

Trading session state machine checkpoint.
Producer: SessionManager
Consumer: OrderGovernor, RiskManager, Dashboard
"""

from __future__ import annotations

from decimal import Decimal  # noqa: TC003 - used at runtime in validators
from typing import Annotated

from pydantic import Field, field_validator

from cryptoscreener.trading.contracts.base import (
    TradingContractBase,
    parse_decimal,
)
from cryptoscreener.trading.contracts.types import SessionStateEnum


class SessionState(TradingContractBase):
    """Trading session state machine checkpoint.

    State machine:
    INITIALIZING -> READY -> ACTIVE -> PAUSED -> ACTIVE
                      |         |        |
                  STOPPING -> STOPPED -> KILLED
    """

    ts: int = Field(description="State change timestamp (ms)")
    state: SessionStateEnum = Field(description="Current session state")
    prev_state: SessionStateEnum | None = Field(
        default=None,
        description="Previous state (for transition logging)",
    )
    reason: str | None = Field(
        default=None,
        description="Reason for state change",
    )
    symbols_active: list[str] = Field(
        default_factory=list,
        description="Currently active symbols",
    )
    open_orders_count: int = Field(description="Number of open orders")
    positions_count: int = Field(description="Number of open positions")
    daily_pnl: Annotated[Decimal, Field(description="Session PnL so far (USD)")] = Field()
    daily_trades: int = Field(description="Number of trades this session")
    risk_utilization: Annotated[Decimal, Field(description="Risk budget utilization (0.0-1.0)")] = (
        Field()
    )

    @field_validator("daily_pnl", "risk_utilization", mode="before")
    @classmethod
    def parse_decimal_fields(cls, v: object) -> Decimal:
        """Parse Decimal fields."""
        return parse_decimal(v)

    @property
    def is_active(self) -> bool:
        """Check if session is actively trading."""
        return self.state == SessionStateEnum.ACTIVE

    @property
    def is_stopped(self) -> bool:
        """Check if session is stopped or killed."""
        return self.state in (SessionStateEnum.STOPPED, SessionStateEnum.KILLED)

    @property
    def can_trade(self) -> bool:
        """Check if session can place new trades."""
        return self.state in (SessionStateEnum.READY, SessionStateEnum.ACTIVE)

    @property
    def dedupe_key(self) -> tuple[str, int]:
        """Dedupe key: (session_id, ts)."""
        return (self.session_id, self.ts)
