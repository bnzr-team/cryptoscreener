"""Simulator metrics computation.

SimResult contains all metrics computed from a simulation run.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from cryptoscreener.trading.contracts import FillEvent, PositionSnapshot


def _parse_decimal(v: object) -> Decimal:
    """Parse value to Decimal."""
    if isinstance(v, Decimal):
        return v
    if isinstance(v, float):
        return Decimal(str(v))
    if isinstance(v, (int, str)):
        return Decimal(v)
    raise ValueError(f"Cannot convert {type(v).__name__} to Decimal")


class SimResult(BaseModel):
    """Simulation result metrics.

    All monetary values use Decimal for precision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # PnL metrics
    net_pnl: Annotated[Decimal, Field(description="Net PnL after commissions (USD)")] = Field()
    total_commissions: Annotated[Decimal, Field(description="Total commissions paid (USD)")] = (
        Field()
    )
    max_drawdown: Annotated[Decimal, Field(description="Maximum drawdown (USD)")] = Field()

    # Trade metrics
    total_fills: int = Field(ge=0, description="Total number of fills")
    round_trips: int = Field(ge=0, description="Number of completed round trips")
    win_rate: Annotated[Decimal, Field(description="Win rate (0.0-1.0)")] = Field()
    profit_factor: Annotated[
        Decimal,
        Field(description="Gross profit / gross loss (inf if no losses)"),
    ] = Field()

    # Position metrics
    max_position: Annotated[Decimal, Field(description="Maximum position size held")] = Field()
    avg_session_duration_min: Annotated[
        Decimal,
        Field(description="Average session duration in minutes"),
    ] = Field()

    # Fill quality metrics (Phase 1: simplified)
    fill_rate: Annotated[
        Decimal,
        Field(description="Fraction of orders that got filled (0.0-1.0)"),
    ] = Field()
    adverse_selection_rate: Annotated[
        Decimal,
        Field(description="Fraction of fills with immediate adverse move (Phase 1: 0.0)"),
    ] = Field(default=Decimal("0.0"))

    @field_validator(
        "net_pnl",
        "total_commissions",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "max_position",
        "avg_session_duration_min",
        "fill_rate",
        "adverse_selection_rate",
        mode="before",
    )
    @classmethod
    def parse_decimal_fields(cls, v: object) -> Decimal:
        """Parse Decimal fields."""
        return _parse_decimal(v)


def compute_metrics(
    fills: list[FillEvent],
    positions: list[PositionSnapshot],
    orders_placed: int,
    session_start_ts: int,
    session_end_ts: int,
) -> SimResult:
    """Compute simulation metrics from fills and positions.

    Args:
        fills: List of fill events from simulation.
        positions: List of position snapshots over time.
        orders_placed: Total number of orders placed.
        session_start_ts: Session start timestamp (ms).
        session_end_ts: Session end timestamp (ms).

    Returns:
        SimResult with computed metrics.
    """
    # Calculate PnL from fills
    total_realized_pnl = sum((f.realized_pnl for f in fills), Decimal("0"))
    total_commissions = sum((f.commission for f in fills), Decimal("0"))
    net_pnl = total_realized_pnl - total_commissions

    # Calculate drawdown from position series
    max_drawdown = Decimal("0")
    peak_pnl = Decimal("0")
    cumulative_pnl = Decimal("0")

    for pos in positions:
        cumulative_pnl = pos.realized_pnl_session + pos.unrealized_pnl
        if cumulative_pnl > peak_pnl:
            peak_pnl = cumulative_pnl
        drawdown = peak_pnl - cumulative_pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Calculate round trips (position goes flat)
    round_trips = 0
    was_flat = True
    for pos in positions:
        is_flat = pos.is_flat
        if not was_flat and is_flat:
            round_trips += 1
        was_flat = is_flat

    # Calculate win rate from round trips
    # A round trip is a "win" if it has positive realized PnL
    winning_trips = 0
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")

    # Track PnL per round trip
    trip_start_pnl = Decimal("0")
    in_position = False

    for pos in positions:
        if not in_position and not pos.is_flat:
            # Entering position
            in_position = True
            trip_start_pnl = pos.realized_pnl_session
        elif in_position and pos.is_flat:
            # Exiting position
            in_position = False
            trip_pnl = pos.realized_pnl_session - trip_start_pnl
            if trip_pnl > 0:
                winning_trips += 1
                gross_profit += trip_pnl
            else:
                gross_loss += abs(trip_pnl)

    win_rate = Decimal(str(winning_trips / round_trips)) if round_trips > 0 else Decimal("0")
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else Decimal("999999")  # Represent infinity
    )

    # Max position
    max_position = max((abs(pos.quantity) for pos in positions), default=Decimal("0"))

    # Session duration
    duration_ms = session_end_ts - session_start_ts
    duration_min = Decimal(str(duration_ms / 60000))

    # Fill rate
    fill_rate = Decimal(str(len(fills) / orders_placed)) if orders_placed > 0 else Decimal("0")

    return SimResult(
        net_pnl=net_pnl,
        total_commissions=total_commissions,
        max_drawdown=max_drawdown,
        total_fills=len(fills),
        round_trips=round_trips,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_position=max_position,
        avg_session_duration_min=duration_min,
        fill_rate=fill_rate,
        adverse_selection_rate=Decimal("0"),  # Phase 1: always 0
    )
