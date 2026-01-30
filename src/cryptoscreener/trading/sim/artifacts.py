"""Simulation artifacts and builder.

SimArtifacts contains all outputs from a simulation run.
Deterministic SHA256 computed over canonical JSON dump.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import orjson
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from cryptoscreener.trading.contracts import (
        FillEvent,
        OrderAck,
        PositionSnapshot,
        SessionState,
    )
    from cryptoscreener.trading.sim.config import SimConfig
    from cryptoscreener.trading.sim.metrics import SimResult


class SimArtifacts(BaseModel):
    """Simulation artifacts container.

    Contains all outputs from a simulation run with deterministic SHA256.
    """

    model_config = ConfigDict(extra="forbid")

    # Configuration used
    config: dict[str, Any] = Field(description="SimConfig as dict")

    # Event series (using v2 contracts)
    fills: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Fill events (FillEvent as dict)",
    )
    orders: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Order acknowledgments (OrderAck as dict)",
    )
    positions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Position snapshots (PositionSnapshot as dict)",
    )
    session_states: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Session state transitions (SessionState as dict)",
    )

    # Computed metrics
    metrics: dict[str, Any] = Field(description="SimResult as dict")

    # Determinism
    sha256: str = Field(description="SHA256 of canonical JSON dump (computed)")

    @classmethod
    def compute_sha256(cls, data: dict[str, Any]) -> str:
        """Compute SHA256 of canonical JSON dump.

        Uses orjson with sorted keys for deterministic output.

        Args:
            data: Dict to hash (without sha256 field).

        Returns:
            64-character hex SHA256 digest.
        """
        # Canonical JSON: sorted keys, no whitespace
        canonical = orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(canonical).hexdigest()


def build_artifacts(
    config: SimConfig,
    fills: list[FillEvent],
    orders: list[OrderAck],
    positions: list[PositionSnapshot],
    session_states: list[SessionState],
    metrics: SimResult,
) -> SimArtifacts:
    """Build simulation artifacts with deterministic SHA256.

    Args:
        config: Simulation configuration.
        fills: List of fill events.
        orders: List of order acknowledgments.
        positions: List of position snapshots.
        session_states: List of session state transitions.
        metrics: Computed simulation metrics.

    Returns:
        SimArtifacts with all data and computed SHA256.
    """
    # Convert all to dicts using model_dump for Pydantic models
    config_dict = config.model_dump(mode="json")
    fills_list = [f.model_dump(mode="json") for f in fills]
    orders_list = [o.model_dump(mode="json") for o in orders]
    positions_list = [p.model_dump(mode="json") for p in positions]
    states_list = [s.model_dump(mode="json") for s in session_states]
    metrics_dict = metrics.model_dump(mode="json")

    # Build data dict without sha256 first
    data = {
        "config": config_dict,
        "fills": fills_list,
        "orders": orders_list,
        "positions": positions_list,
        "session_states": states_list,
        "metrics": metrics_dict,
    }

    # Compute SHA256
    sha256 = SimArtifacts.compute_sha256(data)

    return SimArtifacts(
        config=config_dict,
        fills=fills_list,
        orders=orders_list,
        positions=positions_list,
        session_states=states_list,
        metrics=metrics_dict,
        sha256=sha256,
    )


def dump_artifacts_json(artifacts: SimArtifacts) -> bytes:
    """Dump artifacts to canonical JSON bytes.

    Args:
        artifacts: SimArtifacts to dump.

    Returns:
        JSON bytes with sorted keys (deterministic).
    """
    return orjson.dumps(
        artifacts.model_dump(mode="json"),
        option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
    )
