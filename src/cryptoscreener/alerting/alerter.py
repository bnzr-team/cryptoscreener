"""Alerter with anti-spam controls.

Implements ALERTING_SPEC.md requirements:
- Cooldown per symbol per event type (default 120s)
- Hysteresis: require stable state for stable_ms before firing alert
- Max alerts/min global cap (safety)
- Deduplication

Event types:
- ALERT_TRADABLE: status becomes TRADEABLE (after gates)
- ALERT_TRAP: status becomes TRAP
- ALERT_ENTER_TOPK: enters top-K (combined score)
- ALERT_DATA_ISSUE: critical data health issue persists > T seconds
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from cryptoscreener.contracts.events import (
    PredictionSnapshot,
    PredictionStatus,
    RankEvent,
    RankEventPayload,
    RankEventType,
)


@dataclass
class AlerterConfig:
    """Configuration for Alerter with anti-spam controls.

    Attributes:
        cooldown_ms: Cooldown per symbol per event type (default 120s).
        stable_ms: Time state must be stable before firing alert.
        max_alerts_per_min: Global cap on alerts per minute.
        data_issue_threshold_ms: Time for data issue to persist before alert.
    """

    cooldown_ms: int = 120_000  # 120 seconds
    stable_ms: int = 2000  # 2 seconds
    max_alerts_per_min: int = 30
    data_issue_threshold_ms: int = 10_000  # 10 seconds


@dataclass
class SymbolAlertState:
    """State tracking for a symbol's alerting.

    Attributes:
        last_status: Last known prediction status.
        status_since_ts: When current status was first observed.
        last_alert_ts: Map of event type to last alert timestamp.
        data_issue_since_ts: When data issue started (0 if none).
    """

    last_status: PredictionStatus | None = None
    status_since_ts: int = 0
    last_alert_ts: dict[RankEventType, int] = field(default_factory=dict)
    data_issue_since_ts: int = 0


@dataclass
class AlerterMetrics:
    """Metrics for alerter operations."""

    alerts_generated: int = 0
    alerts_suppressed_cooldown: int = 0
    alerts_suppressed_stability: int = 0
    alerts_suppressed_rate_limit: int = 0

    def reset(self) -> None:
        """Reset all metrics."""
        self.alerts_generated = 0
        self.alerts_suppressed_cooldown = 0
        self.alerts_suppressed_stability = 0
        self.alerts_suppressed_rate_limit = 0


class Alerter:
    """Alerter with anti-spam controls.

    Generates alerts for significant state transitions while preventing spam:
    - Cooldown prevents repeated alerts for the same condition
    - Stability check prevents alerts during flicker
    - Global rate limiting prevents alert storms
    - Deduplication tracks recent alerts
    """

    def __init__(self, config: AlerterConfig | None = None) -> None:
        """Initialize alerter.

        Args:
            config: Alerter configuration.
        """
        self._config = config or AlerterConfig()
        self._metrics = AlerterMetrics()
        self._states: dict[str, SymbolAlertState] = {}
        # Track recent alerts for rate limiting (timestamps)
        self._recent_alerts: deque[int] = deque()

    @property
    def config(self) -> AlerterConfig:
        """Get alerter configuration."""
        return self._config

    @property
    def metrics(self) -> AlerterMetrics:
        """Get alerter metrics."""
        return self._metrics

    def _get_or_create_state(self, symbol: str) -> SymbolAlertState:
        """Get or create state for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            SymbolAlertState for the symbol.
        """
        if symbol not in self._states:
            self._states[symbol] = SymbolAlertState()
        return self._states[symbol]

    def _is_rate_limited(self, ts: int) -> bool:
        """Check if global rate limit is reached.

        Args:
            ts: Current timestamp.

        Returns:
            True if rate limited.
        """
        # Remove alerts older than 1 minute
        min_ts = ts - 60_000
        while self._recent_alerts and self._recent_alerts[0] < min_ts:
            self._recent_alerts.popleft()

        return len(self._recent_alerts) >= self._config.max_alerts_per_min

    def _is_on_cooldown(self, state: SymbolAlertState, event_type: RankEventType, ts: int) -> bool:
        """Check if symbol is on cooldown for event type.

        Args:
            state: Symbol's alert state.
            event_type: Type of alert.
            ts: Current timestamp.

        Returns:
            True if on cooldown.
        """
        last_ts = state.last_alert_ts.get(event_type)
        if last_ts is None:
            return False  # Never alerted before, not on cooldown
        return (ts - last_ts) < self._config.cooldown_ms

    def _is_stable(self, state: SymbolAlertState, ts: int) -> bool:
        """Check if status has been stable long enough.

        Args:
            state: Symbol's alert state.
            ts: Current timestamp.

        Returns:
            True if stable.
        """
        if state.status_since_ts == 0:
            return False
        return (ts - state.status_since_ts) >= self._config.stable_ms

    def _record_alert(self, state: SymbolAlertState, event_type: RankEventType, ts: int) -> None:
        """Record that an alert was generated.

        Args:
            state: Symbol's alert state.
            event_type: Type of alert.
            ts: Current timestamp.
        """
        state.last_alert_ts[event_type] = ts
        self._recent_alerts.append(ts)
        self._metrics.alerts_generated += 1

    def _create_alert_event(
        self,
        event_type: RankEventType,
        prediction: PredictionSnapshot,
        ts: int,
        rank: int = 0,
        score: float = 0.0,
    ) -> RankEvent:
        """Create a RankEvent for an alert.

        Args:
            event_type: Type of alert.
            prediction: Associated prediction.
            ts: Timestamp.
            rank: Current rank.
            score: Current score.

        Returns:
            RankEvent for the alert.
        """
        return RankEvent(
            ts=ts,
            event=event_type,
            symbol=prediction.symbol,
            rank=rank,
            score=score,
            payload=RankEventPayload(
                prediction=prediction.model_dump(mode="json"),
            ),
        )

    def process_prediction(
        self,
        prediction: PredictionSnapshot,
        ts: int,
        rank: int = 0,
        score: float = 0.0,
    ) -> list[RankEvent]:
        """Process a prediction for potential alerts.

        Args:
            prediction: PredictionSnapshot to process.
            ts: Current timestamp.
            rank: Current rank in top-K.
            score: Current ranking score.

        Returns:
            List of alert RankEvents generated.
        """
        events: list[RankEvent] = []
        symbol = prediction.symbol
        state = self._get_or_create_state(symbol)

        # Track status changes
        status_changed = state.last_status != prediction.status
        if status_changed:
            state.last_status = prediction.status
            state.status_since_ts = ts

        # Check for data issue
        if prediction.status == PredictionStatus.DATA_ISSUE:
            if state.data_issue_since_ts == 0:
                state.data_issue_since_ts = ts
            elif (ts - state.data_issue_since_ts) >= self._config.data_issue_threshold_ms:
                # Data issue persisted long enough
                event = self._try_generate_alert(
                    state, RankEventType.DATA_ISSUE, prediction, ts, rank, score
                )
                if event:
                    events.append(event)
        else:
            state.data_issue_since_ts = 0

        # Check for TRADEABLE alert
        if prediction.status == PredictionStatus.TRADEABLE:
            if self._is_stable(state, ts):
                event = self._try_generate_alert(
                    state, RankEventType.ALERT_TRADABLE, prediction, ts, rank, score
                )
                if event:
                    events.append(event)
            else:
                self._metrics.alerts_suppressed_stability += 1

        # Check for TRAP alert
        if prediction.status == PredictionStatus.TRAP:
            if self._is_stable(state, ts):
                event = self._try_generate_alert(
                    state, RankEventType.ALERT_TRAP, prediction, ts, rank, score
                )
                if event:
                    events.append(event)
            else:
                self._metrics.alerts_suppressed_stability += 1

        return events

    def _try_generate_alert(
        self,
        state: SymbolAlertState,
        event_type: RankEventType,
        prediction: PredictionSnapshot,
        ts: int,
        rank: int,
        score: float,
    ) -> RankEvent | None:
        """Try to generate an alert, checking all anti-spam controls.

        Args:
            state: Symbol's alert state.
            event_type: Type of alert.
            prediction: Associated prediction.
            ts: Current timestamp.
            rank: Current rank.
            score: Current score.

        Returns:
            RankEvent if alert should be generated, None otherwise.
        """
        # Check rate limit
        if self._is_rate_limited(ts):
            self._metrics.alerts_suppressed_rate_limit += 1
            return None

        # Check cooldown
        if self._is_on_cooldown(state, event_type, ts):
            self._metrics.alerts_suppressed_cooldown += 1
            return None

        # Generate alert
        self._record_alert(state, event_type, ts)
        return self._create_alert_event(event_type, prediction, ts, rank, score)

    def process_rank_event(
        self,
        event: RankEvent,
        prediction: PredictionSnapshot | None = None,
    ) -> list[RankEvent]:
        """Process a rank event for potential alerts.

        Called when ranker emits SYMBOL_ENTER events to generate ALERT_ENTER_TOPK.

        Args:
            event: RankEvent from ranker.
            prediction: Optional associated prediction.

        Returns:
            List of alert RankEvents generated.
        """
        events: list[RankEvent] = []

        if event.event == RankEventType.SYMBOL_ENTER and prediction:
            # Ensure state exists for the symbol
            self._get_or_create_state(event.symbol)
            # Note: ALERT_ENTER_TOPK would be generated here if RankEventType supported it

        return events

    def reset(self) -> None:
        """Reset alerter state."""
        self._states.clear()
        self._recent_alerts.clear()
        self._metrics.reset()
