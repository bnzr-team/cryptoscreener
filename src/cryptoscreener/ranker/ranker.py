"""Ranker with hysteresis for stable top-K ranking.

Implements HYSTERESIS_SPEC.md requirements:
- Enter TRADEABLE if condition holds for enter_ms (default 1500ms)
- Exit TRADEABLE only if condition fails for exit_ms (default 3000ms)
- Min dwell time in any state: min_dwell_ms (default 2000ms)
- Rank churn: top-K updates limited to 1Hz unless score delta > burst_delta
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptoscreener.contracts.events import (
    PredictionSnapshot,
    RankEvent,
    RankEventType,
)
from cryptoscreener.ranker.state import SymbolState, SymbolStateType
from cryptoscreener.scoring.scorer import Scorer, ScorerConfig


@dataclass
class RankerConfig:
    """Configuration for Ranker with hysteresis parameters.

    Attributes:
        top_k: Number of symbols to keep in top-K ranking.
        enter_ms: Time condition must hold before entering top-K.
        exit_ms: Time condition must fail before exiting top-K.
        min_dwell_ms: Minimum time to spend in any state.
        update_interval_ms: Minimum interval between rank updates (1Hz).
        burst_delta: Score delta threshold to bypass update interval.
        score_threshold: Minimum score to be considered for top-K.
    """

    top_k: int = 20
    enter_ms: int = 1500
    exit_ms: int = 3000
    min_dwell_ms: int = 2000
    update_interval_ms: int = 1000  # 1Hz
    burst_delta: float = 0.1
    score_threshold: float = 0.1


@dataclass
class RankerMetrics:
    """Metrics for ranker operations."""

    updates_processed: int = 0
    enter_events: int = 0
    exit_events: int = 0
    rank_changes: int = 0
    updates_throttled: int = 0

    def reset(self) -> None:
        """Reset all metrics."""
        self.updates_processed = 0
        self.enter_events = 0
        self.exit_events = 0
        self.rank_changes = 0
        self.updates_throttled = 0


class Ranker:
    """Ranker with hysteresis for stable top-K ranking.

    Maintains a stable top-K ranking by applying hysteresis rules:
    - Symbols must maintain score above threshold for enter_ms to enter
    - Symbols must maintain score below threshold for exit_ms to exit
    - Minimum dwell time prevents rapid state changes
    - Update throttling limits rank churn to 1Hz unless large score deltas

    Emits RankEvents for state transitions:
    - SYMBOL_ENTER: Symbol entered top-K
    - SYMBOL_EXIT: Symbol exited top-K
    """

    def __init__(
        self,
        config: RankerConfig | None = None,
        scorer: Scorer | ScorerConfig | None = None,
    ) -> None:
        """Initialize ranker.

        Args:
            config: Ranker configuration.
            scorer: Scorer instance or ScorerConfig (creates new Scorer).
        """
        self._config = config or RankerConfig()
        if isinstance(scorer, Scorer):
            self._scorer = scorer
        else:
            self._scorer = Scorer(scorer)
        self._metrics = RankerMetrics()
        self._states: dict[str, SymbolState] = {}
        self._top_k: list[str] = []
        self._last_update_ts: int = 0

    @property
    def config(self) -> RankerConfig:
        """Get ranker configuration."""
        return self._config

    @property
    def scorer(self) -> Scorer:
        """Get the scorer instance."""
        return self._scorer

    @property
    def metrics(self) -> RankerMetrics:
        """Get ranker metrics."""
        return self._metrics

    @property
    def top_k(self) -> list[str]:
        """Get current top-K symbols (ordered by rank)."""
        return list(self._top_k)

    def get_top_k(self) -> dict[str, SymbolState]:
        """Get current top-K symbols with their states.

        Returns:
            Dict of symbol to SymbolState for all symbols in top-K.
        """
        result: dict[str, SymbolState] = {}
        for symbol in self._top_k:
            state = self._states.get(symbol)
            if state:
                result[symbol] = state
        return result

    def get_state(self, symbol: str) -> SymbolState | None:
        """Get state for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            SymbolState if tracked, None otherwise.
        """
        return self._states.get(symbol)

    def _get_or_create_state(self, symbol: str, ts: int) -> SymbolState:
        """Get or create state for a symbol.

        Args:
            symbol: Trading pair symbol.
            ts: Current timestamp.

        Returns:
            SymbolState for the symbol.
        """
        if symbol not in self._states:
            self._states[symbol] = SymbolState(
                symbol=symbol,
                state_entered_ts=ts,
                last_update_ts=ts,
            )
        return self._states[symbol]

    def _should_throttle_update(self, ts: int, max_delta: float) -> bool:
        """Check if update should be throttled.

        Args:
            ts: Current timestamp.
            max_delta: Maximum score delta in current batch.

        Returns:
            True if update should be throttled.
        """
        if self._last_update_ts == 0:
            return False

        time_since_update = ts - self._last_update_ts
        if time_since_update >= self._config.update_interval_ms:
            return False

        # Allow burst if large score delta
        return max_delta < self._config.burst_delta

    def _compute_rankings(
        self, predictions: dict[str, PredictionSnapshot]
    ) -> list[tuple[str, float]]:
        """Compute rankings from predictions.

        Args:
            predictions: Map of symbol to prediction.

        Returns:
            List of (symbol, score) tuples sorted by score descending.
        """
        scored: list[tuple[str, float]] = []
        for symbol, pred in predictions.items():
            score = self._scorer.score(pred)
            if score >= self._config.score_threshold:
                scored.append((symbol, score))

        # Sort by score descending, then symbol for determinism
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def _process_state_transitions(
        self,
        symbol: str,
        score: float,
        in_new_top_k: bool,
        ts: int,
    ) -> list[RankEvent]:
        """Process state transitions for a symbol.

        Args:
            symbol: Trading pair symbol.
            score: Current score.
            in_new_top_k: Whether symbol is in new top-K.
            ts: Current timestamp.

        Returns:
            List of RankEvents generated.
        """
        events: list[RankEvent] = []
        state = self._get_or_create_state(symbol, ts)
        state.update_score(score, ts)

        current_state = state.state
        dwell = state.dwell_time_ms(ts)

        if current_state == SymbolStateType.NOT_IN_TOP_K:
            if in_new_top_k:
                if self._config.enter_ms == 0:
                    # Immediate enter (no hysteresis)
                    state.transition_to(SymbolStateType.IN_TOP_K, ts)
                    rank = self._top_k.index(symbol) if symbol in self._top_k else 0
                    events.append(
                        RankEvent(
                            ts=ts,
                            event=RankEventType.SYMBOL_ENTER,
                            symbol=symbol,
                            rank=rank,
                            score=score,
                        )
                    )
                    self._metrics.enter_events += 1
                else:
                    # Start pending enter
                    state.transition_to(SymbolStateType.PENDING_ENTER, ts)

        elif current_state == SymbolStateType.PENDING_ENTER:
            if not in_new_top_k:
                # Lost condition, go back to not in top-K
                state.transition_to(SymbolStateType.NOT_IN_TOP_K, ts)
            elif state.pending_time_ms(ts) >= self._config.enter_ms:
                # Condition held long enough, enter top-K
                state.transition_to(SymbolStateType.IN_TOP_K, ts)
                rank = self._top_k.index(symbol) if symbol in self._top_k else 0
                events.append(
                    RankEvent(
                        ts=ts,
                        event=RankEventType.SYMBOL_ENTER,
                        symbol=symbol,
                        rank=rank,
                        score=score,
                    )
                )
                self._metrics.enter_events += 1

        elif current_state == SymbolStateType.IN_TOP_K:
            if not in_new_top_k and dwell >= self._config.min_dwell_ms:
                if self._config.exit_ms == 0:
                    # Immediate exit (no hysteresis)
                    last_rank = state.rank if state.rank >= 0 else 0
                    state.transition_to(SymbolStateType.NOT_IN_TOP_K, ts)
                    events.append(
                        RankEvent(
                            ts=ts,
                            event=RankEventType.SYMBOL_EXIT,
                            symbol=symbol,
                            rank=last_rank,
                            score=score,
                        )
                    )
                    self._metrics.exit_events += 1
                else:
                    # Start pending exit
                    state.transition_to(SymbolStateType.PENDING_EXIT, ts)

        elif current_state == SymbolStateType.PENDING_EXIT:
            if in_new_top_k:
                # Regained condition, go back to in top-K
                state.transition_to(SymbolStateType.IN_TOP_K, ts)
            elif state.pending_time_ms(ts) >= self._config.exit_ms:
                # Condition failed long enough, exit top-K
                last_rank = state.rank if state.rank >= 0 else 0
                state.transition_to(SymbolStateType.NOT_IN_TOP_K, ts)
                events.append(
                    RankEvent(
                        ts=ts,
                        event=RankEventType.SYMBOL_EXIT,
                        symbol=symbol,
                        rank=last_rank,
                        score=score,
                    )
                )
                self._metrics.exit_events += 1

        return events

    def update(
        self,
        predictions: dict[str, PredictionSnapshot],
        ts: int,
    ) -> list[RankEvent]:
        """Update rankings with new predictions.

        Args:
            predictions: Map of symbol to PredictionSnapshot.
            ts: Current timestamp.

        Returns:
            List of RankEvents generated.
        """
        self._metrics.updates_processed += 1

        # Compute new rankings
        rankings = self._compute_rankings(predictions)
        new_top_k_symbols = {s for s, _ in rankings[: self._config.top_k]}

        # Check for throttling
        max_delta = 0.0
        for symbol, new_score in rankings:
            if symbol in self._states:
                delta = abs(new_score - self._states[symbol].score)
                max_delta = max(max_delta, delta)

        if self._should_throttle_update(ts, max_delta):
            self._metrics.updates_throttled += 1
            return []

        self._last_update_ts = ts

        # Process all symbols
        events: list[RankEvent] = []
        all_symbols = set(predictions.keys()) | set(self._states.keys())

        for symbol in all_symbols:
            score = 0.0
            if symbol in predictions:
                score = self._scorer.score(predictions[symbol])

            in_new_top_k = symbol in new_top_k_symbols
            symbol_events = self._process_state_transitions(symbol, score, in_new_top_k, ts)
            events.extend(symbol_events)

        # Update top-K list with confirmed members only
        confirmed_top_k: list[str] = []
        for symbol, _score in rankings:
            if len(confirmed_top_k) >= self._config.top_k:
                break
            state = self._states.get(symbol)
            if state and state.state in (
                SymbolStateType.IN_TOP_K,
                SymbolStateType.PENDING_EXIT,
            ):
                confirmed_top_k.append(symbol)
            elif state and state.state == SymbolStateType.PENDING_ENTER:
                # Include pending enters in top-K for ranking purposes
                confirmed_top_k.append(symbol)

        # Track rank changes
        if confirmed_top_k != self._top_k:
            self._metrics.rank_changes += 1

        self._top_k = confirmed_top_k

        # Update ranks in states
        for i, symbol in enumerate(self._top_k):
            if symbol in self._states:
                self._states[symbol].rank = i

        return events

    def reset(self) -> None:
        """Reset ranker state."""
        self._states.clear()
        self._top_k.clear()
        self._last_update_ts = 0
        self._metrics.reset()
