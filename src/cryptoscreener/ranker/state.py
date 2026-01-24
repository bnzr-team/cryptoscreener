"""Symbol state management for hysteresis logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SymbolStateType(str, Enum):
    """State type for symbol in ranker."""

    NOT_IN_TOP_K = "NOT_IN_TOP_K"
    PENDING_ENTER = "PENDING_ENTER"  # Condition met, waiting for enter_ms
    IN_TOP_K = "IN_TOP_K"
    PENDING_EXIT = "PENDING_EXIT"  # Condition lost, waiting for exit_ms


@dataclass
class SymbolState:
    """State for a symbol in the ranker with hysteresis tracking.

    Tracks state transitions with timing for hysteresis logic:
    - Must hold enter condition for enter_ms before entering top-K
    - Must hold exit condition for exit_ms before exiting top-K
    - Minimum dwell time in any state (min_dwell_ms)

    Attributes:
        symbol: Trading pair symbol.
        state: Current state type.
        score: Current ranking score.
        rank: Current rank (0-indexed, -1 if not in top-K).
        state_entered_ts: Timestamp when current state was entered.
        last_update_ts: Timestamp of last update.
        pending_since_ts: Timestamp when pending condition started.
    """

    symbol: str
    state: SymbolStateType = SymbolStateType.NOT_IN_TOP_K
    score: float = 0.0
    rank: int = -1
    state_entered_ts: int = 0
    last_update_ts: int = 0
    pending_since_ts: int = 0

    def update_score(self, score: float, ts: int) -> None:
        """Update the score and timestamp.

        Args:
            score: New ranking score.
            ts: Current timestamp.
        """
        self.score = score
        self.last_update_ts = ts

    def transition_to(self, new_state: SymbolStateType, ts: int) -> None:
        """Transition to a new state.

        Args:
            new_state: Target state.
            ts: Timestamp of transition.
        """
        if new_state != self.state:
            self.state = new_state
            self.state_entered_ts = ts
            if new_state in (SymbolStateType.PENDING_ENTER, SymbolStateType.PENDING_EXIT):
                self.pending_since_ts = ts
            else:
                self.pending_since_ts = 0

    def dwell_time_ms(self, ts: int) -> int:
        """Calculate time spent in current state.

        Args:
            ts: Current timestamp.

        Returns:
            Milliseconds spent in current state.
        """
        return ts - self.state_entered_ts

    def pending_time_ms(self, ts: int) -> int:
        """Calculate time spent in pending condition.

        Args:
            ts: Current timestamp.

        Returns:
            Milliseconds spent in pending condition, or 0 if not pending.
        """
        if self.pending_since_ts == 0:
            return 0
        return ts - self.pending_since_ts
