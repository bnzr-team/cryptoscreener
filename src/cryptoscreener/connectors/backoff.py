"""
Backoff and circuit breaker implementation for Binance API.

Per BINANCE_LIMITS.md:
- On any 429: immediate backoff
- On repeated 429: open circuit breaker
- Never "fight" the limiter
- Exponential backoff with jitter for reconnects

DEC-023 additions:
- ReconnectLimiter: Global reconnect rate control to prevent storms
- MessageThrottler: Rate limit WS subscribe/unsubscribe operations
- Seeded jitter: Deterministic backoff for replay testing
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


class RateLimitKind(str, Enum):
    """Type of rate limit error."""

    RATE_LIMIT = "RATE_LIMIT"  # 429 - normal rate limit
    IP_BAN = "IP_BAN"  # 418 - IP banned, extended cooldown required
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"  # -1003 error code


class RateLimitError(Exception):
    """Raised when rate limit is hit (429/418/-1003)."""

    def __init__(
        self,
        message: str,
        error_code: int | None = None,
        retry_after_ms: int | None = None,
        kind: RateLimitKind = RateLimitKind.RATE_LIMIT,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_after_ms = retry_after_ms
        self.kind = kind

    @property
    def is_ip_ban(self) -> bool:
        """Check if this is an IP ban (418)."""
        return self.kind == RateLimitKind.IP_BAN


class GovernorTimeoutError(Exception):
    """Raised when request times out waiting in governor queue.

    DEC-023d: Indicates timeout_ms expired while waiting for budget/concurrency.
    """

    def __init__(self, message: str, waited_ms: int = 0) -> None:
        super().__init__(message)
        self.waited_ms = waited_ms


class GovernorDroppedError(Exception):
    """Raised when request is dropped due to queue being full.

    DEC-023d: Indicates drop-new policy rejected the request because queue is at capacity.
    """

    def __init__(self, message: str, queue_depth: int = 0) -> None:
        super().__init__(message)
        self.queue_depth = queue_depth


class CircuitState(str, Enum):
    """Circuit breaker state."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Blocking requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


@dataclass
class BackoffConfig:
    """Configuration for exponential backoff.

    Per BINANCE_LIMITS.md:
    - Randomized jitter in bootstraps
    - Exponential backoff for reconnects
    - Max reconnect rate to avoid storms
    """

    base_delay_ms: int = 1000
    max_delay_ms: int = 60000
    multiplier: float = 2.0
    jitter_factor: float = 0.5  # 0.5 = ±50% jitter
    max_retries: int = 10


@dataclass
class BackoffState:
    """Mutable state for backoff tracking."""

    attempt: int = 0
    last_error_time_ms: int = 0
    consecutive_errors: int = 0

    def reset(self) -> None:
        """Reset backoff state after successful operation."""
        self.attempt = 0
        self.consecutive_errors = 0

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.attempt += 1
        self.consecutive_errors += 1
        self.last_error_time_ms = int(time.time() * 1000)


def compute_backoff_delay(
    config: BackoffConfig,
    state: BackoffState,
    retry_after_ms: int | None = None,
    *,
    rng: random.Random | None = None,
) -> int:
    """
    Compute backoff delay with exponential increase and jitter.

    Per BINANCE_LIMITS.md: exponential backoff + randomized jitter.
    Respects Retry-After header when provided.

    DEC-023: Added optional seeded RNG for deterministic replay testing.

    Args:
        config: Backoff configuration.
        state: Current backoff state.
        retry_after_ms: Server-provided retry delay (Retry-After header).
        rng: Optional seeded Random instance for deterministic jitter.

    Returns:
        Delay in milliseconds before next retry.
    """
    if state.attempt == 0:
        return 0

    # Exponential backoff
    delay = config.base_delay_ms * (config.multiplier ** (state.attempt - 1))

    # Apply jitter: delay * (1 - jitter_factor) to delay * (1 + jitter_factor)
    jitter_min = 1.0 - config.jitter_factor
    jitter_max = 1.0 + config.jitter_factor
    # DEC-023: Use seeded RNG if provided for deterministic replay
    if rng is not None:
        jitter_multiplier = rng.uniform(jitter_min, jitter_max)
    else:
        jitter_multiplier = random.uniform(jitter_min, jitter_max)
    delay = delay * jitter_multiplier

    # Cap at max delay
    delay = min(delay, config.max_delay_ms)

    # Respect Retry-After if provided (server knows best)
    if retry_after_ms is not None and retry_after_ms > 0:
        delay = max(delay, retry_after_ms)

    return int(delay)


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for API protection.

    Per BINANCE_LIMITS.md:
    - On any 429: immediate circuit OPEN
    - On 418 (IP ban): immediate circuit OPEN with extended cooldown
    - Never "fight" the limiter

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Blocking all requests, waiting for cooldown
    - HALF_OPEN: Allowing test requests to check if service recovered

    DEC-023c: Added _time_fn injection for deterministic testing.
    """

    failure_threshold: int = 5  # Consecutive failures to open circuit
    recovery_timeout_ms: int = 30000  # Time before trying half-open
    half_open_max_requests: int = 1  # Requests allowed in half-open state
    ban_recovery_timeout_ms: int = 300000  # 5 min for IP ban recovery

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time_ms: int = field(default=0)
    half_open_requests: int = field(default=0)
    _is_banned: bool = field(default=False)
    _open_until_ms: int = field(default=0)

    # DEC-023c: Optional time provider for deterministic testing
    _time_fn: Callable[[], int] | None = field(default=None)

    def _now_ms(self) -> int:
        """Get current time in milliseconds."""
        if self._time_fn is not None:
            return self._time_fn()
        return int(time.time() * 1000)

    def can_execute(self) -> bool:
        """
        Check if a request can be executed.

        Returns:
            True if request should proceed, False if blocked by circuit.
        """
        now_ms = self._now_ms()

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if forced open_until has passed
            if self._open_until_ms > 0 and now_ms < self._open_until_ms:
                return False

            # Check if recovery timeout has passed
            timeout = self.ban_recovery_timeout_ms if self._is_banned else self.recovery_timeout_ms
            if now_ms - self.last_failure_time_ms >= timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 1  # Count this request
                self._is_banned = False
                self._open_until_ms = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            if self.half_open_requests < self.half_open_max_requests:
                self.half_open_requests += 1
                return True
            return False

        return False

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            # Recovery confirmed, close circuit
            self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._is_banned = False
        self._open_until_ms = 0

    def record_failure(self, is_rate_limit: bool = False, is_ip_ban: bool = False) -> None:
        """
        Record a failed request.

        Args:
            is_rate_limit: True if failure was due to rate limiting (429).
            is_ip_ban: True if failure was IP ban (418).
        """
        now_ms = self._now_ms()
        self.last_failure_time_ms = now_ms

        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test, reopen circuit
            self.state = CircuitState.OPEN
            if is_ip_ban:
                self._is_banned = True
            return

        self.failure_count += 1

        # IP ban (418) or rate limit (429): immediate OPEN
        if is_ip_ban:
            self.state = CircuitState.OPEN
            self._is_banned = True
            return

        if is_rate_limit:
            # 429: immediate circuit OPEN per BINANCE_LIMITS.md
            self.state = CircuitState.OPEN
            return

        # Regular failures: use threshold
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def force_open(self, duration_ms: int, reason: str = "") -> None:
        """
        Force circuit open for specified duration.

        Args:
            duration_ms: How long to keep circuit open.
            reason: Reason for forcing open (for logging).
        """
        now_ms = self._now_ms()
        self.state = CircuitState.OPEN
        self.last_failure_time_ms = now_ms
        self._open_until_ms = now_ms + duration_ms

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time_ms = 0
        self.half_open_requests = 0
        self._is_banned = False
        self._open_until_ms = 0

    @property
    def is_banned(self) -> bool:
        """Check if circuit is open due to IP ban."""
        return self._is_banned

    def get_status(self) -> dict[str, str | int | bool]:
        """Get current circuit breaker status for observability."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time_ms": self.last_failure_time_ms,
            "is_banned": self._is_banned,
        }


def handle_error_response(
    status_code: int,
    error_code: int | None = None,
    retry_after_ms: int | None = None,
) -> RateLimitError | None:
    """
    Handle Binance API error response.

    Per BINANCE_LIMITS.md:
    - 429: Rate limit hit, immediate circuit OPEN
    - 418: IP auto-ban, extended cooldown required
    - -1003: TOO_MANY_REQUESTS

    Args:
        status_code: HTTP status code.
        error_code: Binance error code (e.g., -1003).
        retry_after_ms: Suggested retry delay if provided (Retry-After header).

    Returns:
        RateLimitError if rate limiting detected, None otherwise.
    """
    if status_code == 429:
        return RateLimitError(
            "Rate limit exceeded (429)",
            error_code=error_code,
            retry_after_ms=retry_after_ms,
            kind=RateLimitKind.RATE_LIMIT,
        )

    if status_code == 418:
        return RateLimitError(
            "IP banned (418) - stop all requests immediately",
            error_code=error_code,
            retry_after_ms=retry_after_ms or 300000,  # 5 min default for ban
            kind=RateLimitKind.IP_BAN,
        )

    if error_code == -1003:
        return RateLimitError(
            "TOO_MANY_REQUESTS (-1003)",
            error_code=error_code,
            retry_after_ms=retry_after_ms,
            kind=RateLimitKind.TOO_MANY_REQUESTS,
        )

    return None


# =============================================================================
# DEC-023: Reconnect Storm Protection
# =============================================================================


@dataclass
class ReconnectLimiterConfig:
    """Configuration for reconnect rate limiting.

    Per BINANCE_LIMITS.md:
    - Avoid reconnect storms during volatility
    - Exponential backoff with jitter
    - Max reconnect rate (global + per-conn)

    DEC-023: Added to prevent all shards reconnecting simultaneously.
    """

    max_reconnects_per_window: int = 5  # Max reconnects across all shards
    window_ms: int = 60000  # 1 minute sliding window
    cooldown_after_burst_ms: int = 30000  # Cooldown after hitting limit
    per_shard_min_interval_ms: int = 5000  # Min interval between reconnects per shard


@dataclass
class ReconnectLimiter:
    """
    Global reconnect rate limiter to prevent reconnect storms.

    Per BINANCE_LIMITS.md §4: "Avoid reconnect storms"
    DEC-023: Tracks reconnects across all shards and enforces rate limits.

    Features:
    - Sliding window rate limiting
    - Per-shard cooldown tracking
    - Burst protection with global cooldown
    """

    config: ReconnectLimiterConfig = field(default_factory=ReconnectLimiterConfig)

    # Sliding window of reconnect timestamps (global)
    _reconnect_timestamps: deque[int] = field(default_factory=deque)

    # Per-shard last reconnect time
    _shard_last_reconnect: dict[int, int] = field(default_factory=dict)

    # Global cooldown until timestamp (0 = no cooldown)
    _cooldown_until_ms: int = field(default=0)

    # Optional time provider for testing (DEC-023: determinism)
    _time_fn: Callable[[], int] | None = field(default=None)

    def _now_ms(self) -> int:
        """Get current time in milliseconds."""
        if self._time_fn is not None:
            return self._time_fn()
        return int(time.time() * 1000)

    def can_reconnect(self, shard_id: int) -> bool:
        """
        Check if a shard can attempt reconnection.

        Args:
            shard_id: ID of the shard wanting to reconnect.

        Returns:
            True if reconnect is allowed, False if rate limited.
        """
        now_ms = self._now_ms()

        # Check global cooldown
        if self._cooldown_until_ms > 0 and now_ms < self._cooldown_until_ms:
            return False

        # Check per-shard minimum interval (only if shard has reconnected before)
        if shard_id in self._shard_last_reconnect:
            last_reconnect = self._shard_last_reconnect[shard_id]
            if now_ms - last_reconnect < self.config.per_shard_min_interval_ms:
                return False

        # Check sliding window rate limit
        self._prune_old_timestamps(now_ms)
        if len(self._reconnect_timestamps) >= self.config.max_reconnects_per_window:
            # Hit rate limit, enter cooldown
            self._cooldown_until_ms = now_ms + self.config.cooldown_after_burst_ms
            return False

        return True

    def record_reconnect(self, shard_id: int) -> None:
        """
        Record a reconnect attempt.

        Args:
            shard_id: ID of the shard that reconnected.
        """
        now_ms = self._now_ms()
        self._reconnect_timestamps.append(now_ms)
        self._shard_last_reconnect[shard_id] = now_ms

    def get_wait_time_ms(self, shard_id: int) -> int:
        """
        Get time to wait before reconnect is allowed.

        Args:
            shard_id: ID of the shard.

        Returns:
            Milliseconds to wait (0 if can reconnect now).
        """
        now_ms = self._now_ms()

        # Check global cooldown
        if self._cooldown_until_ms > 0 and now_ms < self._cooldown_until_ms:
            return self._cooldown_until_ms - now_ms

        # Check per-shard minimum interval (only if shard has reconnected before)
        if shard_id in self._shard_last_reconnect:
            last_reconnect = self._shard_last_reconnect[shard_id]
            shard_wait = self.config.per_shard_min_interval_ms - (now_ms - last_reconnect)
            if shard_wait > 0:
                return shard_wait

        # Check sliding window - if full, compute when oldest expires
        self._prune_old_timestamps(now_ms)
        if len(self._reconnect_timestamps) >= self.config.max_reconnects_per_window:
            oldest = self._reconnect_timestamps[0]
            return (oldest + self.config.window_ms) - now_ms

        return 0

    def _prune_old_timestamps(self, now_ms: int) -> None:
        """Remove timestamps outside the sliding window."""
        cutoff = now_ms - self.config.window_ms
        while self._reconnect_timestamps and self._reconnect_timestamps[0] < cutoff:
            self._reconnect_timestamps.popleft()

    def reset(self) -> None:
        """Reset limiter state."""
        self._reconnect_timestamps.clear()
        self._shard_last_reconnect.clear()
        self._cooldown_until_ms = 0

    def get_status(self) -> dict[str, int | bool]:
        """Get current limiter status for observability."""
        now_ms = self._now_ms()
        self._prune_old_timestamps(now_ms)
        return {
            "reconnects_in_window": len(self._reconnect_timestamps),
            "max_reconnects": self.config.max_reconnects_per_window,
            "in_cooldown": self._cooldown_until_ms > now_ms,
            "cooldown_remaining_ms": max(0, self._cooldown_until_ms - now_ms),
        }


# =============================================================================
# DEC-023: Message Throttler for WS Operations
# =============================================================================


@dataclass
class MessageThrottlerConfig:
    """Configuration for WS message rate limiting.

    Per BINANCE_LIMITS.md §2:
    - 10 incoming messages per second per connection
    - Batch subscribe/unsubscribe operations

    DEC-023: Enforces rate limit on subscribe/unsubscribe operations.
    """

    max_messages_per_second: int = 10  # Binance limit
    safety_margin: float = 0.8  # Use 80% of limit for safety
    burst_allowance: int = 5  # Allow small burst


@dataclass
class MessageThrottler:
    """
    Rate limiter for WebSocket subscribe/unsubscribe messages.

    Per BINANCE_LIMITS.md §2: "10 incoming messages per second per connection"
    DEC-023: Prevents exceeding message rate limit.

    Uses token bucket algorithm with configurable rate.
    """

    config: MessageThrottlerConfig = field(default_factory=MessageThrottlerConfig)

    # Token bucket state
    _tokens: float = field(default=0.0)
    _last_update_ms: int = field(default=0)

    # Optional time provider for testing (DEC-023: determinism)
    _time_fn: Callable[[], int] | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize tokens to burst allowance."""
        self._tokens = float(self.config.burst_allowance)
        self._last_update_ms = self._now_ms()

    def _now_ms(self) -> int:
        """Get current time in milliseconds."""
        if self._time_fn is not None:
            return self._time_fn()
        return int(time.time() * 1000)

    @property
    def effective_rate(self) -> float:
        """Get effective messages per second (with safety margin)."""
        return self.config.max_messages_per_second * self.config.safety_margin

    def _refill_tokens(self, now_ms: int) -> None:
        """Refill tokens based on elapsed time."""
        elapsed_ms = now_ms - self._last_update_ms
        if elapsed_ms <= 0:
            return

        # Refill rate: effective_rate tokens per second
        tokens_to_add = (elapsed_ms / 1000.0) * self.effective_rate
        max_tokens = float(self.config.burst_allowance + self.effective_rate)
        self._tokens = min(self._tokens + tokens_to_add, max_tokens)
        self._last_update_ms = now_ms

    def can_send(self, message_count: int = 1) -> bool:
        """
        Check if messages can be sent without exceeding rate limit.

        Args:
            message_count: Number of messages to send.

        Returns:
            True if sending is allowed.
        """
        now_ms = self._now_ms()
        self._refill_tokens(now_ms)
        return self._tokens >= message_count

    def consume(self, message_count: int = 1) -> bool:
        """
        Consume tokens for sending messages.

        Args:
            message_count: Number of messages being sent.

        Returns:
            True if tokens were consumed, False if insufficient tokens.
        """
        now_ms = self._now_ms()
        self._refill_tokens(now_ms)

        if self._tokens >= message_count:
            self._tokens -= message_count
            return True
        return False

    def get_wait_time_ms(self, message_count: int = 1) -> int:
        """
        Get time to wait before sending is allowed.

        Args:
            message_count: Number of messages to send.

        Returns:
            Milliseconds to wait (0 if can send now).
        """
        now_ms = self._now_ms()
        self._refill_tokens(now_ms)

        if self._tokens >= message_count:
            return 0

        # Calculate time needed to accumulate required tokens
        tokens_needed = message_count - self._tokens
        time_needed_ms = (tokens_needed / self.effective_rate) * 1000
        return int(time_needed_ms) + 1  # +1 to ensure we have enough

    def reset(self) -> None:
        """Reset throttler state."""
        self._tokens = float(self.config.burst_allowance)
        self._last_update_ms = self._now_ms()

    def get_status(self) -> dict[str, float | int]:
        """Get current throttler status for observability."""
        now_ms = self._now_ms()
        self._refill_tokens(now_ms)
        return {
            "available_tokens": round(self._tokens, 2),
            "effective_rate_per_sec": self.effective_rate,
            "burst_allowance": self.config.burst_allowance,
        }


# =============================================================================
# DEC-023d: REST Governor for Budget/Queue/Concurrency Control
# =============================================================================


# Default endpoint weights per Binance documentation
DEFAULT_ENDPOINT_WEIGHTS: dict[str, int] = {
    "/fapi/v1/exchangeInfo": 40,
    "/fapi/v1/ticker/24hr": 40,
    "/fapi/v1/time": 1,
}
DEFAULT_WEIGHT = 10


@dataclass
class RestGovernorConfig:
    """Configuration for REST API governor.

    DEC-023d: Controls budget, queue, and concurrency limits for REST requests.

    Per BINANCE_LIMITS.md:
    - 2,400 requests per minute per IP (weight-based)
    - Use WS for live updates; avoid REST polling loops
    """

    # Budget settings (token bucket)
    budget_weight_per_minute: int = 2000  # Total weight budget per minute
    budget_refill_interval_ms: int = 1000  # Refill every second (continuous)

    # Queue settings
    max_queue_depth: int = 50  # Maximum pending requests
    default_timeout_ms: int = 30000  # Default timeout for waiting in queue

    # Concurrency settings
    max_concurrent_requests: int = 10  # Semaphore cap

    # Endpoint weights (custom weights override defaults)
    endpoint_weights: dict[str, int] = field(default_factory=dict)
    default_endpoint_weight: int = DEFAULT_WEIGHT

    def get_endpoint_weight(self, endpoint: str) -> int:
        """Get weight for an endpoint."""
        # Check custom weights first
        if endpoint in self.endpoint_weights:
            return self.endpoint_weights[endpoint]
        # Fall back to default weights
        if endpoint in DEFAULT_ENDPOINT_WEIGHTS:
            return DEFAULT_ENDPOINT_WEIGHTS[endpoint]
        return self.default_endpoint_weight


@dataclass
class RestGovernorMetrics:
    """Metrics for REST Governor observability.

    DEC-023d: Tracks budget, queue, and decision statistics.
    """

    # Counters
    requests_allowed: int = 0
    requests_deferred: int = 0  # Waited in queue, then allowed
    requests_dropped: int = 0  # Rejected due to queue full
    requests_failed_breaker: int = 0  # Rejected due to circuit breaker

    # Gauges (current state)
    current_budget_weight: float = 0.0
    current_queue_depth: int = 0
    current_concurrent: int = 0

    # Histograms (accumulators for wait time)
    total_wait_ms: int = 0
    max_wait_ms: int = 0

    # Drop reasons
    drop_reason_queue_full: int = 0
    drop_reason_timeout: int = 0
    drop_reason_breaker_open: int = 0


@dataclass
class _QueuedRequest:
    """Internal representation of a queued request."""

    endpoint: str
    weight: int
    enqueue_time_ms: int
    event: asyncio.Event
    result: str = ""  # "allowed", "timeout", "dropped"


@dataclass
class RestGovernor:
    """
    REST API governor with budget, queue, and concurrency control.

    DEC-023d: Central gatekeeper for all REST requests to prevent rate limiting.

    Features:
    - Token bucket budget (2000 weight/min default)
    - Bounded FIFO queue with drop-new policy
    - Concurrency semaphore (max 10 concurrent)
    - CircuitBreaker integration (fail-fast when OPEN)
    - Deterministic with _time_fn injection

    Usage:
        governor = RestGovernor(circuit_breaker=cb)
        await governor.acquire("/fapi/v1/exchangeInfo")  # Blocks until allowed
        # ... make HTTP request ...
        governor.release()  # Release concurrency slot
    """

    config: RestGovernorConfig = field(default_factory=RestGovernorConfig)
    circuit_breaker: CircuitBreaker | None = field(default=None)

    # Token bucket state
    _budget_tokens: float = field(default=0.0, init=False)
    _last_refill_ms: int = field(default=0, init=False)

    # Queue state
    _queue: deque[_QueuedRequest] = field(default_factory=deque, init=False)

    # Concurrency state (track count, not actual semaphore for sync dataclass)
    _concurrent_count: int = field(default=0, init=False)
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False)

    # Metrics
    metrics: RestGovernorMetrics = field(default_factory=RestGovernorMetrics, init=False)

    # Time provider for determinism (DEC-023 pattern)
    _time_fn: Callable[[], int] | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize budget tokens to full capacity."""
        self._budget_tokens = float(self.config.budget_weight_per_minute)
        self._last_refill_ms = self._now_ms()

    def _now_ms(self) -> int:
        """Get current time in milliseconds."""
        if self._time_fn is not None:
            return self._time_fn()
        return int(time.time() * 1000)

    def _refill_budget(self, now_ms: int) -> None:
        """Refill budget tokens based on elapsed time."""
        elapsed_ms = now_ms - self._last_refill_ms
        if elapsed_ms <= 0:
            return

        # Refill rate: budget_weight_per_minute / 60000 tokens per ms
        tokens_per_ms = self.config.budget_weight_per_minute / 60000.0
        tokens_to_add = elapsed_ms * tokens_per_ms
        max_tokens = float(self.config.budget_weight_per_minute)
        self._budget_tokens = min(self._budget_tokens + tokens_to_add, max_tokens)
        self._last_refill_ms = now_ms

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create asyncio semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        return self._semaphore

    async def acquire(
        self,
        endpoint: str,
        weight: int | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        """
        Acquire permission for a REST request. Blocks until allowed or raises.

        DEC-023d: Central entry point for REST rate limiting.

        Args:
            endpoint: API endpoint path (e.g., "/fapi/v1/exchangeInfo").
            weight: Request weight override (None = use endpoint default).
            timeout_ms: Maximum time to wait in queue (None = use config default).

        Raises:
            RateLimitError: If circuit breaker is OPEN (fail-fast, no queue).
            GovernorTimeoutError: If timeout expires while waiting in queue.
            GovernorDroppedError: If queue is full and request is rejected.
        """
        now_ms = self._now_ms()

        # Resolve parameters
        actual_weight = weight if weight is not None else self.config.get_endpoint_weight(endpoint)
        actual_timeout_ms = timeout_ms if timeout_ms is not None else self.config.default_timeout_ms

        # 1. Check circuit breaker (fail-fast)
        if self.circuit_breaker is not None and not self.circuit_breaker.can_execute():
            self.metrics.requests_failed_breaker += 1
            self.metrics.drop_reason_breaker_open += 1
            raise RateLimitError(
                "Circuit breaker OPEN, request rejected",
                retry_after_ms=self.circuit_breaker.recovery_timeout_ms,
            )

        # 2. Refill budget
        self._refill_budget(now_ms)

        # 3. Check if can proceed immediately (budget available + concurrency available)
        if (
            self._budget_tokens >= actual_weight
            and self._concurrent_count < self.config.max_concurrent_requests
        ):
            # Immediate allow
            self._budget_tokens -= actual_weight
            self._concurrent_count += 1
            self.metrics.requests_allowed += 1
            self.metrics.current_budget_weight = self._budget_tokens
            self.metrics.current_concurrent = self._concurrent_count
            return

        # 4. Check queue capacity (drop-new policy)
        if len(self._queue) >= self.config.max_queue_depth:
            self.metrics.requests_dropped += 1
            self.metrics.drop_reason_queue_full += 1
            raise GovernorDroppedError(
                f"Queue full ({self.config.max_queue_depth}), request dropped",
                queue_depth=len(self._queue),
            )

        # 5. Enqueue and wait
        request = _QueuedRequest(
            endpoint=endpoint,
            weight=actual_weight,
            enqueue_time_ms=now_ms,
            event=asyncio.Event(),
        )
        self._queue.append(request)
        self.metrics.current_queue_depth = len(self._queue)

        try:
            # Wait with timeout
            wait_start_ms = now_ms
            deadline_ms = now_ms + actual_timeout_ms

            while True:
                now_ms = self._now_ms()
                remaining_ms = deadline_ms - now_ms

                if remaining_ms <= 0:
                    # Timeout expired
                    request.result = "timeout"
                    self.metrics.requests_dropped += 1
                    self.metrics.drop_reason_timeout += 1
                    waited_ms = now_ms - wait_start_ms
                    raise GovernorTimeoutError(
                        f"Timeout after {waited_ms}ms waiting in queue",
                        waited_ms=waited_ms,
                    )

                # Check if we're at the front of queue and can proceed
                if self._queue and self._queue[0] is request:
                    self._refill_budget(now_ms)
                    if (
                        self._budget_tokens >= actual_weight
                        and self._concurrent_count < self.config.max_concurrent_requests
                    ):
                        # Can proceed now
                        self._budget_tokens -= actual_weight
                        self._concurrent_count += 1
                        request.result = "allowed"
                        waited_ms = now_ms - wait_start_ms
                        self.metrics.requests_deferred += 1
                        self.metrics.total_wait_ms += waited_ms
                        self.metrics.max_wait_ms = max(self.metrics.max_wait_ms, waited_ms)
                        self.metrics.current_budget_weight = self._budget_tokens
                        self.metrics.current_concurrent = self._concurrent_count
                        return

                # Wait a bit before checking again
                # Use a short sleep to avoid busy-waiting
                await asyncio.sleep(min(remaining_ms, 100) / 1000.0)

        finally:
            # Remove from queue if still present
            if request in self._queue:
                self._queue.remove(request)
            self.metrics.current_queue_depth = len(self._queue)

    def release(self) -> None:
        """
        Release a concurrency slot after request completes.

        Must be called after acquire() completes and the HTTP request is done.
        Prefer using `permit()` context manager for automatic release.
        """
        if self._concurrent_count > 0:
            self._concurrent_count -= 1
            self.metrics.current_concurrent = self._concurrent_count

    @contextlib.asynccontextmanager
    async def permit(
        self,
        endpoint: str,
        weight: int | None = None,
        timeout_ms: int | None = None,
    ) -> AsyncIterator[None]:
        """
        Async context manager for safe acquire/release.

        DEC-023d: Guarantees release even on exception.

        Usage:
            async with governor.permit("/fapi/v1/exchangeInfo"):
                response = await session.get(url)
                # ... process response ...
            # Slot automatically released here

        Args:
            endpoint: API endpoint path.
            weight: Request weight override (None = use endpoint default).
            timeout_ms: Maximum time to wait in queue.

        Raises:
            RateLimitError: If circuit breaker is OPEN.
            GovernorTimeoutError: If timeout expires while waiting.
            GovernorDroppedError: If queue is full.
        """
        await self.acquire(endpoint, weight, timeout_ms)
        try:
            yield
        finally:
            self.release()

    def get_status(self) -> dict[str, int | float | bool]:
        """Get current governor status for observability."""
        now_ms = self._now_ms()
        self._refill_budget(now_ms)
        return {
            "budget_tokens": round(self._budget_tokens, 2),
            "budget_max": self.config.budget_weight_per_minute,
            "queue_depth": len(self._queue),
            "queue_max": self.config.max_queue_depth,
            "concurrent": self._concurrent_count,
            "concurrent_max": self.config.max_concurrent_requests,
            "breaker_open": self.circuit_breaker.state == CircuitState.OPEN
            if self.circuit_breaker
            else False,
        }

    def reset(self) -> None:
        """Reset governor to initial state."""
        self._budget_tokens = float(self.config.budget_weight_per_minute)
        self._last_refill_ms = self._now_ms()
        self._queue.clear()
        self._concurrent_count = 0
        self._semaphore = None
        self.metrics = RestGovernorMetrics()
