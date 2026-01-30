"""Simulator configuration.

SimConfig is frozen (immutable) and defines all simulation parameters.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FillModel(str, Enum):
    """Fill model type.

    CROSS: Phase 1 - optimistic fills at limit price when price crosses.
    QUEUE_LITE: Phase 2 - queue position modeling (not implemented).
    """

    CROSS = "CROSS"
    QUEUE_LITE = "QUEUE_LITE"


def _parse_decimal(v: object) -> Decimal:
    """Parse value to Decimal."""
    if isinstance(v, Decimal):
        return v
    if isinstance(v, float):
        return Decimal(str(v))
    if isinstance(v, (int, str)):
        return Decimal(v)
    raise ValueError(f"Cannot convert {type(v).__name__} to Decimal")


class SimConfig(BaseModel):
    """Simulator configuration (frozen).

    All monetary values use Decimal for precision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(description="Trading symbol (e.g., BTCUSDT)")
    latency_ms: int = Field(
        default=50,
        ge=0,
        description="Fixed latency in milliseconds (order placement to exchange)",
    )
    slippage_bps_taker: Annotated[
        Decimal,
        Field(description="Slippage for taker orders in basis points"),
    ] = Field(default=Decimal("0"))
    maker_fee_frac: Annotated[
        Decimal,
        Field(description="Maker fee as fraction (e.g., 0.0002 = 2bps)"),
    ] = Field(default=Decimal("0.0002"))
    taker_fee_frac: Annotated[
        Decimal,
        Field(description="Taker fee as fraction (e.g., 0.0004 = 4bps)"),
    ] = Field(default=Decimal("0.0004"))
    fill_model: FillModel = Field(
        default=FillModel.CROSS,
        description="Fill model type (Phase 1: CROSS only)",
    )

    # Risk limits
    max_position_qty: Annotated[
        Decimal,
        Field(description="Maximum position size in base currency"),
    ] = Field(default=Decimal("1.0"))
    max_session_loss: Annotated[
        Decimal,
        Field(description="Maximum session loss before kill switch (USD)"),
    ] = Field(default=Decimal("100.0"))
    stale_quote_ms: int = Field(
        default=5000,
        ge=0,
        description="Milliseconds before quote is considered stale",
    )

    @field_validator(
        "slippage_bps_taker",
        "maker_fee_frac",
        "taker_fee_frac",
        "max_position_qty",
        "max_session_loss",
        mode="before",
    )
    @classmethod
    def parse_decimal_fields(cls, v: object) -> Decimal:
        """Parse Decimal fields."""
        return _parse_decimal(v)
