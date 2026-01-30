"""Scenario runner for trading simulation.

Runs a strategy against market events, producing:
- decisions.jsonl: Strategy decisions per tick
- sim_artifacts.json: Simulation output (fills, positions, metrics)
- Combined deterministic digest (SHA256 over both files)

See DEC-042 for design rationale.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal  # noqa: TC003 - used at runtime
from pathlib import Path  # noqa: TC003 - used at runtime in write_scenario_outputs
from typing import Any

import orjson

from cryptoscreener.trading.contracts import (
    PositionSide,
    StrategyDecision,
    StrategyDecisionOrder,
)
from cryptoscreener.trading.sim.artifacts import SimArtifacts, dump_artifacts_json
from cryptoscreener.trading.sim.config import SimConfig  # noqa: TC001 - used at runtime
from cryptoscreener.trading.sim.simulator import Simulator, SimulatorState
from cryptoscreener.trading.strategy.base import (
    Strategy,
    StrategyContext,
    StrategyOrder,
)


@dataclass(frozen=True)
class ScenarioResult:
    """Result from running a scenario.

    Contains both simulation artifacts and strategy decisions.
    """

    artifacts: SimArtifacts
    decisions: list[StrategyDecision]
    decisions_sha256: str
    artifacts_sha256: str
    combined_sha256: str


@dataclass
class ScenarioRunnerState:
    """Internal state for the scenario runner."""

    tick_seq: int = 0
    decisions: list[StrategyDecision] = field(default_factory=list)


class ScenarioRunner:
    """Runs a strategy scenario against market events.

    Produces deterministic outputs:
    - StrategyDecision journal (decisions.jsonl)
    - SimArtifacts (sim_artifacts.json)
    - Combined digest for replay verification
    """

    def __init__(
        self,
        config: SimConfig,
        strategy: Strategy,
    ) -> None:
        """Initialize scenario runner.

        Args:
            config: Simulation configuration.
            strategy: Strategy instance implementing Strategy protocol.
        """
        self.config = config
        self.strategy = strategy
        self._runner_state = ScenarioRunnerState()

    def run(self, events: list[dict[str, Any]]) -> ScenarioResult:
        """Run scenario on market events.

        Args:
            events: List of market events (dicts with ts, type, payload, symbol).

        Returns:
            ScenarioResult with artifacts, decisions, and deterministic digests.
        """
        self._runner_state = ScenarioRunnerState()

        # Create a strategy wrapper that journals decisions
        def strategy_wrapper(
            state: SimulatorState,
            bid: Decimal,
            ask: Decimal,
            ts: int,
        ) -> list[tuple[Any, Decimal, Decimal]]:
            # Build context for strategy
            ctx = StrategyContext(
                ts=ts,
                bid=bid,
                ask=ask,
                last_trade_price=state.last_trade_price,
                last_book_ts=state.last_book_ts,
                last_trade_ts=state.last_trade_ts,
                position_qty=state.position_qty,
                position_side=self._get_position_side(state.position_qty),
                entry_price=state.entry_price,
                unrealized_pnl=state.unrealized_pnl,
                realized_pnl=state.realized_pnl,
                pending_order_count=len(state.pending_orders),
                symbol=self.config.symbol,
                max_position_qty=self.config.max_position_qty,
            )

            # Call strategy
            order_intents = self.strategy.on_tick(ctx)

            # Journal the decision
            decision = self._create_decision(ctx, order_intents, state.session_id)
            self._runner_state.decisions.append(decision)
            self._runner_state.tick_seq += 1

            # Convert to simulator format
            return [
                (intent.side, intent.price, intent.quantity) for intent in order_intents
            ]

        # Run simulator with wrapper strategy
        simulator = Simulator(self.config, strategy=strategy_wrapper)
        artifacts = simulator.run(events)

        # Compute digests
        decisions_bytes = self._serialize_decisions()
        decisions_sha256 = hashlib.sha256(decisions_bytes).hexdigest()
        artifacts_sha256 = artifacts.sha256

        # Combined digest: SHA256(decisions_sha256 + artifacts_sha256)
        combined_sha256 = hashlib.sha256(
            (decisions_sha256 + artifacts_sha256).encode()
        ).hexdigest()

        return ScenarioResult(
            artifacts=artifacts,
            decisions=self._runner_state.decisions,
            decisions_sha256=decisions_sha256,
            artifacts_sha256=artifacts_sha256,
            combined_sha256=combined_sha256,
        )

    def _get_position_side(self, qty: Decimal) -> PositionSide:
        """Determine position side from quantity."""
        if qty > 0:
            return PositionSide.LONG
        elif qty < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    def _create_decision(
        self,
        ctx: StrategyContext,
        order_intents: list[StrategyOrder],
        session_id: str,
    ) -> StrategyDecision:
        """Create a StrategyDecision from context and orders."""
        orders = [
            StrategyDecisionOrder(
                session_id=session_id,
                side=intent.side,
                price=intent.price,
                quantity=intent.quantity,
                reason=intent.reason,
            )
            for intent in order_intents
        ]

        return StrategyDecision(
            session_id=session_id,
            ts=ctx.ts,
            tick_seq=self._runner_state.tick_seq,
            bid=ctx.bid,
            ask=ctx.ask,
            mid=ctx.mid,
            last_trade_price=ctx.last_trade_price,
            position_qty=ctx.position_qty,
            position_side=ctx.position_side,
            unrealized_pnl=ctx.unrealized_pnl,
            realized_pnl=ctx.realized_pnl,
            pending_order_count=ctx.pending_order_count,
            orders=orders,
            symbol=ctx.symbol,
        )

    def _serialize_decisions(self) -> bytes:
        """Serialize decisions to JSONL bytes (canonical, sorted keys)."""
        lines = []
        for decision in self._runner_state.decisions:
            # Use orjson for canonical serialization
            json_bytes = orjson.dumps(
                decision.model_dump(mode="json"),
                option=orjson.OPT_SORT_KEYS,
            )
            lines.append(json_bytes)
        return b"\n".join(lines) + b"\n" if lines else b""


def write_scenario_outputs(
    result: ScenarioResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write scenario outputs to files.

    Args:
        result: ScenarioResult from runner.run().
        output_dir: Directory to write files to.

    Returns:
        Tuple of (decisions_path, artifacts_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write decisions.jsonl
    decisions_path = output_dir / "decisions.jsonl"
    with open(decisions_path, "wb") as f:
        for decision in result.decisions:
            json_bytes = orjson.dumps(
                decision.model_dump(mode="json"),
                option=orjson.OPT_SORT_KEYS,
            )
            f.write(json_bytes + b"\n")

    # Write sim_artifacts.json
    artifacts_path = output_dir / "sim_artifacts.json"
    with open(artifacts_path, "wb") as f:
        f.write(dump_artifacts_json(result.artifacts))

    return decisions_path, artifacts_path
