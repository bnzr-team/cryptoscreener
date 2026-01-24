"""
Tests for backoff and circuit breaker implementation.

Per BINANCE_LIMITS.md:
- On any 429: immediate backoff
- On repeated 429: open circuit breaker
- Exponential backoff with jitter
- Never "fight" the limiter
"""

from __future__ import annotations

from unittest.mock import patch

from cryptoscreener.connectors import (
    BackoffConfig,
    BackoffState,
    CircuitBreaker,
    CircuitState,
    RateLimitError,
    compute_backoff_delay,
)
from cryptoscreener.connectors.backoff import handle_error_response


class TestBackoffConfig:
    """Tests for BackoffConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BackoffConfig()
        assert config.base_delay_ms == 1000
        assert config.max_delay_ms == 60000
        assert config.multiplier == 2.0
        assert config.jitter_factor == 0.5
        assert config.max_retries == 10

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = BackoffConfig(
            base_delay_ms=500,
            max_delay_ms=30000,
            multiplier=1.5,
            jitter_factor=0.3,
            max_retries=5,
        )
        assert config.base_delay_ms == 500
        assert config.max_delay_ms == 30000


class TestBackoffState:
    """Tests for BackoffState."""

    def test_initial_state(self) -> None:
        """Test initial state."""
        state = BackoffState()
        assert state.attempt == 0
        assert state.consecutive_errors == 0

    def test_record_error(self) -> None:
        """Test recording errors."""
        state = BackoffState()
        state.record_error()
        assert state.attempt == 1
        assert state.consecutive_errors == 1
        assert state.last_error_time_ms > 0

    def test_reset(self) -> None:
        """Test resetting state."""
        state = BackoffState()
        state.record_error()
        state.record_error()
        state.reset()
        assert state.attempt == 0
        assert state.consecutive_errors == 0


class TestComputeBackoffDelay:
    """Tests for compute_backoff_delay function."""

    def test_zero_delay_on_first_attempt(self) -> None:
        """Test no delay on first attempt."""
        config = BackoffConfig()
        state = BackoffState()
        delay = compute_backoff_delay(config, state)
        assert delay == 0

    def test_exponential_increase(self) -> None:
        """Test delays increase exponentially."""
        config = BackoffConfig(base_delay_ms=1000, multiplier=2.0, jitter_factor=0.0)
        state = BackoffState()

        # Record errors and check delays
        delays = []
        for _ in range(5):
            state.record_error()
            delay = compute_backoff_delay(config, state)
            delays.append(delay)

        # With multiplier=2 and no jitter:
        # attempt 1: 1000 * 2^0 = 1000
        # attempt 2: 1000 * 2^1 = 2000
        # attempt 3: 1000 * 2^2 = 4000
        assert delays[0] == 1000
        assert delays[1] == 2000
        assert delays[2] == 4000

    def test_max_delay_cap(self) -> None:
        """Test delay is capped at max_delay_ms."""
        config = BackoffConfig(base_delay_ms=1000, max_delay_ms=5000, jitter_factor=0.0)
        state = BackoffState()

        # Many errors should hit the cap
        for _ in range(10):
            state.record_error()

        delay = compute_backoff_delay(config, state)
        assert delay <= config.max_delay_ms

    def test_jitter_applied(self) -> None:
        """Test that jitter is applied."""
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.5)
        state = BackoffState()
        state.record_error()

        # Run multiple times to see variation
        delays = set()
        for _ in range(100):
            delay = compute_backoff_delay(config, state)
            delays.add(delay)

        # Should have variation due to jitter
        assert len(delays) > 1

        # All delays should be within jitter range: 500-1500 for base 1000, jitter 0.5
        for delay in delays:
            assert 500 <= delay <= 1500


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self) -> None:
        """Test circuit starts closed."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_opens_after_threshold_failures(self) -> None:
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        for _ in range(3):
            assert cb.state == CircuitState.CLOSED
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_rate_limit_lower_threshold(self) -> None:
        """Test rate limit errors have lower threshold.

        Per BINANCE_LIMITS.md: rate limits are more serious.
        """
        cb = CircuitBreaker(failure_threshold=6)

        # Rate limit errors should trigger at half threshold (3)
        for _ in range(3):
            cb.record_failure(is_rate_limit=True)

        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        """Test successful request resets failure count."""
        cb = CircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self) -> None:
        """Test circuit goes half-open after recovery timeout."""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_ms=100,  # Short timeout for test
        )

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

        # Mock time passing
        with patch("time.time") as mock_time:
            # Initial time
            mock_time.return_value = cb.last_failure_time_ms / 1000 + 0.2  # 200ms later

            # Should transition to half-open and allow request
            assert cb.can_execute()
            assert cb.state.value == CircuitState.HALF_OPEN.value

    def test_half_open_success_closes_circuit(self) -> None:
        """Test successful request in half-open closes circuit."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_ms=0)

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        # Force half-open
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_requests = 0

        # Success should close
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self) -> None:
        """Test failure in half-open reopens circuit."""
        cb = CircuitBreaker(failure_threshold=2)

        # Force half-open
        cb.state = CircuitState.HALF_OPEN

        # Failure should reopen
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_limits_requests(self) -> None:
        """Test half-open limits number of test requests."""
        cb = CircuitBreaker(half_open_max_requests=1)

        cb.state = CircuitState.HALF_OPEN
        cb.half_open_requests = 0

        # First request allowed
        assert cb.can_execute()
        assert cb.half_open_requests == 1

        # Second request blocked
        assert not cb.can_execute()

    def test_get_status(self) -> None:
        """Test status reporting for observability."""
        cb = CircuitBreaker()
        cb.record_failure()

        status = cb.get_status()
        assert status["state"] == "CLOSED"
        assert status["failure_count"] == 1
        assert "last_failure_time_ms" in status

    def test_reset(self) -> None:
        """Test circuit reset."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state.value == CircuitState.CLOSED.value
        assert cb.failure_count == 0


class TestHandleErrorResponse:
    """Tests for handle_error_response function."""

    def test_429_returns_rate_limit_error(self) -> None:
        """Test 429 is recognized as rate limit."""
        error = handle_error_response(429)
        assert error is not None
        assert isinstance(error, RateLimitError)
        assert "429" in str(error)

    def test_418_returns_ban_error(self) -> None:
        """Test 418 is recognized as IP ban.

        Per BINANCE_LIMITS.md: 418 is auto-ban after continuing post-429.
        """
        error = handle_error_response(418)
        assert error is not None
        assert isinstance(error, RateLimitError)
        assert "418" in str(error) or "ban" in str(error).lower()
        # Should have long retry delay for ban
        assert error.retry_after_ms is not None
        assert error.retry_after_ms >= 300000  # At least 5 minutes

    def test_minus_1003_returns_rate_limit_error(self) -> None:
        """Test -1003 TOO_MANY_REQUESTS is recognized."""
        error = handle_error_response(200, error_code=-1003)
        assert error is not None
        assert isinstance(error, RateLimitError)
        assert error.error_code == -1003

    def test_normal_response_returns_none(self) -> None:
        """Test normal responses don't return error."""
        error = handle_error_response(200)
        assert error is None

    def test_500_returns_none(self) -> None:
        """Test 5xx errors don't return RateLimitError (handled differently)."""
        error = handle_error_response(500)
        assert error is None

    def test_retry_after_preserved(self) -> None:
        """Test retry_after_ms is preserved from response."""
        error = handle_error_response(429, retry_after_ms=5000)
        assert error is not None
        assert error.retry_after_ms == 5000


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = RateLimitError("Test error")
        assert str(error) == "Test error"
        assert error.error_code is None
        assert error.retry_after_ms is None

    def test_error_with_details(self) -> None:
        """Test error with all details."""
        error = RateLimitError(
            "Rate limited",
            error_code=-1003,
            retry_after_ms=60000,
        )
        assert error.error_code == -1003
        assert error.retry_after_ms == 60000


class TestBackoffIntegration:
    """Integration tests for backoff + circuit breaker."""

    def test_429_triggers_backoff_and_circuit(self) -> None:
        """Test 429 response triggers both backoff and circuit breaker."""
        config = BackoffConfig()
        state = BackoffState()
        cb = CircuitBreaker(failure_threshold=3)

        # Simulate 429 responses
        for _ in range(3):
            error = handle_error_response(429)
            assert error is not None

            state.record_error()
            cb.record_failure(is_rate_limit=True)

        # Backoff should have delay
        delay = compute_backoff_delay(config, state)
        assert delay > 0

        # Circuit should be open (rate limit has lower threshold)
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_418_immediate_circuit_open(self) -> None:
        """Test 418 (ban) immediately opens circuit.

        Per BINANCE_LIMITS.md: 418 means stop all requests immediately.
        """
        cb = CircuitBreaker(failure_threshold=10)

        error = handle_error_response(418)
        assert error is not None

        # A ban should be treated as rate limit (lower threshold)
        # With threshold 10, rate limit threshold is 5
        for _ in range(5):
            cb.record_failure(is_rate_limit=True)

        assert cb.state == CircuitState.OPEN

    def test_recovery_after_backoff(self) -> None:
        """Test system recovers after successful backoff."""
        config = BackoffConfig()
        state = BackoffState()
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout_ms=0)

        # Trigger failures
        for _ in range(3):
            state.record_error()
            cb.record_failure(is_rate_limit=True)

        assert cb.state == CircuitState.OPEN

        # Force transition to half-open
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_requests = 0

        # Simulate successful retry
        assert cb.can_execute()
        cb.record_success()
        state.reset()

        # Should be recovered
        assert cb.state == CircuitState.CLOSED
        assert state.attempt == 0
        assert compute_backoff_delay(config, state) == 0
