"""Trading v2 Simulator module.

Deterministic offline simulator for Trading/VOL v2.
Validates PnL behavior on fixed fixtures before live execution.

Phase 1: CROSS fill model only (optimistic fills at limit price).
"""

from __future__ import annotations

from cryptoscreener.trading.sim.artifacts import SimArtifacts, build_artifacts
from cryptoscreener.trading.sim.config import FillModel, SimConfig
from cryptoscreener.trading.sim.fill_model import CrossFillModel
from cryptoscreener.trading.sim.metrics import SimResult, compute_metrics
from cryptoscreener.trading.sim.runner import (
    ScenarioResult,
    ScenarioRunner,
    write_scenario_outputs,
)
from cryptoscreener.trading.sim.simulator import Simulator

__all__ = [
    "CrossFillModel",
    "FillModel",
    "ScenarioResult",
    "ScenarioRunner",
    "SimArtifacts",
    "SimConfig",
    "SimResult",
    "Simulator",
    "build_artifacts",
    "compute_metrics",
    "write_scenario_outputs",
]
