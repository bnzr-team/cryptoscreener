"""
Backoff and circuit breaker implementation for Binance API.

Per BINANCE_LIMITS.md:
- On any 429: immediate backoff
- On repeated 429: open circuit breaker
- Never "fight" the limiter
- Exponential backoff with jitter for reconnects
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum


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
) -> int:
    """
    Compute backoff delay with exponential increase and jitter.

    Per BINANCE_LIMITS.md: exponential backoff + randomized jitter.
    Respects Retry-After header when provided.

    Args:
        config: Backoff configuration.
        state: Current backoff state.
        retry_after_ms: Server-provided retry delay (Retry-After header).

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

    def can_execute(self) -> bool:
        """
        Check if a request can be executed.

        Returns:
            True if request should proceed, False if blocked by circuit.
        """
        now_ms = int(time.time() * 1000)

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
        now_ms = int(time.time() * 1000)
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
        now_ms = int(time.time() * 1000)
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
