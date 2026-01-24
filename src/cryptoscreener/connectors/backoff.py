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


class RateLimitError(Exception):
    """Raised when rate limit is hit (429/418/-1003)."""

    def __init__(
        self,
        message: str,
        error_code: int | None = None,
        retry_after_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_after_ms = retry_after_ms


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
) -> int:
    """
    Compute backoff delay with exponential increase and jitter.

    Per BINANCE_LIMITS.md: exponential backoff + randomized jitter.

    Args:
        config: Backoff configuration.
        state: Current backoff state.

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

    return int(delay)


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for API protection.

    Per BINANCE_LIMITS.md:
    - On repeated 429: open circuit breaker
    - Never "fight" the limiter

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Blocking all requests, waiting for cooldown
    - HALF_OPEN: Allowing test requests to check if service recovered
    """

    failure_threshold: int = 5  # Consecutive failures to open circuit
    recovery_timeout_ms: int = 30000  # Time before trying half-open
    half_open_max_requests: int = 1  # Requests allowed in half-open state

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time_ms: int = field(default=0)
    half_open_requests: int = field(default=0)

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
            # Check if recovery timeout has passed
            if now_ms - self.last_failure_time_ms >= self.recovery_timeout_ms:
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
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

    def record_failure(self, is_rate_limit: bool = False) -> None:
        """
        Record a failed request.

        Args:
            is_rate_limit: True if failure was due to rate limiting (429/418).
        """
        now_ms = int(time.time() * 1000)
        self.last_failure_time_ms = now_ms

        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test, reopen circuit
            self.state = CircuitState.OPEN
            return

        self.failure_count += 1

        # Rate limit errors are more serious - lower threshold
        threshold = self.failure_threshold // 2 if is_rate_limit else self.failure_threshold

        if self.failure_count >= threshold:
            self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time_ms = 0
        self.half_open_requests = 0

    def get_status(self) -> dict[str, str | int]:
        """Get current circuit breaker status for observability."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time_ms": self.last_failure_time_ms,
        }


def handle_error_response(
    status_code: int,
    error_code: int | None = None,
    retry_after_ms: int | None = None,
) -> RateLimitError | None:
    """
    Handle Binance API error response.

    Per BINANCE_LIMITS.md:
    - 429: Rate limit hit, need backoff
    - 418: IP auto-ban after continuing post-429
    - -1003: TOO_MANY_REQUESTS

    Args:
        status_code: HTTP status code.
        error_code: Binance error code (e.g., -1003).
        retry_after_ms: Suggested retry delay if provided.

    Returns:
        RateLimitError if rate limiting detected, None otherwise.
    """
    if status_code == 429:
        return RateLimitError(
            "Rate limit exceeded (429)",
            error_code=error_code,
            retry_after_ms=retry_after_ms,
        )

    if status_code == 418:
        return RateLimitError(
            "IP banned (418) - stop all requests immediately",
            error_code=error_code,
            retry_after_ms=retry_after_ms or 300000,  # 5 min default for ban
        )

    if error_code == -1003:
        return RateLimitError(
            "TOO_MANY_REQUESTS (-1003)",
            error_code=error_code,
            retry_after_ms=retry_after_ms,
        )

    return None
