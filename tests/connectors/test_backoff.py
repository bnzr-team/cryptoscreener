"""
Tests for backoff and circuit breaker implementation.

Per BINANCE_LIMITS.md:
- On any 429: immediate circuit OPEN
- On 418 (IP ban): immediate OPEN with extended cooldown
- Exponential backoff with jitter
- Respect Retry-After header
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
    RateLimitKind,
    compute_backoff_delay,
    handle_error_response,
)


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

    def test_retry_after_respected(self) -> None:
        """Test that Retry-After is respected when provided.

        Per BINANCE_LIMITS.md: server knows best, respect Retry-After.
        """
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.0)
        state = BackoffState()
        state.record_error()  # attempt 1 -> delay would be 1000

        # Without retry_after: normal delay
        delay_without = compute_backoff_delay(config, state)
        assert delay_without == 1000

        # With retry_after > computed: use retry_after
        delay_with = compute_backoff_delay(config, state, retry_after_ms=5000)
        assert delay_with >= 5000

    def test_retry_after_not_less_than_computed(self) -> None:
        """Test that delay is max(computed, retry_after)."""
        config = BackoffConfig(base_delay_ms=10000, jitter_factor=0.0)  # Large base
        state = BackoffState()
        state.record_error()

        # Computed would be 10000, retry_after is smaller
        delay = compute_backoff_delay(config, state, retry_after_ms=1000)
        assert delay == 10000  # Should use computed, not smaller retry_after


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

    def test_429_immediate_open(self) -> None:
        """Test 429 immediately opens circuit.

        Per BINANCE_LIMITS.md: On any 429: immediate circuit OPEN.
        """
        cb = CircuitBreaker(failure_threshold=10)  # High threshold

        # Single 429 should immediately open
        cb.record_failure(is_rate_limit=True)
        assert cb.state == CircuitState.OPEN

    def test_418_immediate_open_with_ban_flag(self) -> None:
        """Test 418 (IP ban) immediately opens circuit with ban flag.

        Per BINANCE_LIMITS.md: 418 is IP ban, requires extended cooldown.
        """
        cb = CircuitBreaker(failure_threshold=10)

        # Single 418 should immediately open and set ban flag
        cb.record_failure(is_ip_ban=True)
        assert cb.state == CircuitState.OPEN
        assert cb.is_banned

    def test_ban_uses_extended_recovery_timeout(self) -> None:
        """Test IP ban uses extended recovery timeout."""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_ms=100,
            ban_recovery_timeout_ms=500,
        )

        # Open circuit with ban
        cb.record_failure(is_ip_ban=True)
        assert cb.state == CircuitState.OPEN
        assert cb.is_banned

        with patch("time.time") as mock_time:
            # After normal timeout (100ms) but before ban timeout (500ms)
            mock_time.return_value = cb.last_failure_time_ms / 1000 + 0.2

            # Should still be blocked (ban needs longer)
            assert not cb.can_execute()
            assert cb.state == CircuitState.OPEN

            # After ban timeout
            mock_time.return_value = cb.last_failure_time_ms / 1000 + 0.6

            # Now should transition to half-open
            assert cb.can_execute()
            assert cb.state.value == CircuitState.HALF_OPEN.value
            assert not cb.is_banned  # Ban flag cleared on transition

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

    def test_half_open_exactly_one_request_on_transition(self) -> None:
        """Test OPEN->HALF_OPEN transition allows exactly one request.

        Regression test: ensure we don't allow double requests.
        """
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_ms=0,  # Immediate transition
            half_open_max_requests=1,
        )

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # First can_execute() triggers transition and counts as request
        assert cb.can_execute()
        assert cb.state.value == CircuitState.HALF_OPEN.value
        assert cb.half_open_requests == 1

        # Second request MUST be blocked
        assert not cb.can_execute()
        assert cb.half_open_requests == 1  # Counter unchanged

    def test_force_open(self) -> None:
        """Test force_open keeps circuit open for duration."""
        cb = CircuitBreaker(recovery_timeout_ms=100)

        cb.force_open(duration_ms=500, reason="test")
        assert cb.state == CircuitState.OPEN

        with patch("time.time") as mock_time:
            # Before duration expires
            mock_time.return_value = cb.last_failure_time_ms / 1000 + 0.3

            # Should still be blocked
            assert not cb.can_execute()

            # After duration expires
            mock_time.return_value = cb.last_failure_time_ms / 1000 + 0.6

            # Now can execute
            assert cb.can_execute()

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
        cb.record_failure(is_ip_ban=True)
        assert cb.state == CircuitState.OPEN
        assert cb.is_banned

        cb.reset()
        assert cb.state.value == CircuitState.CLOSED.value
        assert cb.failure_count == 0
        assert not cb.is_banned

    def test_get_status_includes_ban(self) -> None:
        """Test status includes ban flag."""
        cb = CircuitBreaker()
        cb.record_failure(is_ip_ban=True)

        status = cb.get_status()
        assert status["is_banned"] is True


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
        # Should have IP_BAN kind
        assert error.kind == RateLimitKind.IP_BAN
        assert error.is_ip_ban

    def test_429_has_rate_limit_kind(self) -> None:
        """Test 429 has RATE_LIMIT kind."""
        error = handle_error_response(429)
        assert error is not None
        assert error.kind == RateLimitKind.RATE_LIMIT
        assert not error.is_ip_ban

    def test_minus_1003_has_too_many_requests_kind(self) -> None:
        """Test -1003 has TOO_MANY_REQUESTS kind."""
        error = handle_error_response(200, error_code=-1003)
        assert error is not None
        assert error.kind == RateLimitKind.TOO_MANY_REQUESTS

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
        assert error.kind == RateLimitKind.RATE_LIMIT  # Default kind

    def test_error_with_details(self) -> None:
        """Test error with all details."""
        error = RateLimitError(
            "Rate limited",
            error_code=-1003,
            retry_after_ms=60000,
            kind=RateLimitKind.TOO_MANY_REQUESTS,
        )
        assert error.error_code == -1003
        assert error.retry_after_ms == 60000
        assert error.kind == RateLimitKind.TOO_MANY_REQUESTS

    def test_is_ip_ban_property(self) -> None:
        """Test is_ip_ban convenience property."""
        ban_error = RateLimitError("Banned", kind=RateLimitKind.IP_BAN)
        assert ban_error.is_ip_ban

        normal_error = RateLimitError("Rate limited", kind=RateLimitKind.RATE_LIMIT)
        assert not normal_error.is_ip_ban


class TestBackoffIntegration:
    """Integration tests for backoff + circuit breaker."""

    def test_429_triggers_immediate_open(self) -> None:
        """Test single 429 immediately opens circuit.

        Per BINANCE_LIMITS.md: On any 429: immediate circuit OPEN.
        """
        config = BackoffConfig()
        state = BackoffState()
        cb = CircuitBreaker(failure_threshold=10)  # High threshold

        # Single 429 should immediately open
        error = handle_error_response(429)
        assert error is not None

        state.record_error()
        cb.record_failure(is_rate_limit=True)

        # Circuit should be open after single 429
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

        # Backoff should have delay
        delay = compute_backoff_delay(config, state)
        assert delay > 0

    def test_418_immediate_open_with_extended_cooldown(self) -> None:
        """Test 418 (ban) immediately opens circuit with extended cooldown.

        Per BINANCE_LIMITS.md: 418 means stop all requests immediately.
        """
        cb = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout_ms=100,
            ban_recovery_timeout_ms=500,
        )

        error = handle_error_response(418)
        assert error is not None
        assert error.is_ip_ban

        # Single 418 should immediately open and set ban
        cb.record_failure(is_ip_ban=error.is_ip_ban)

        assert cb.state == CircuitState.OPEN
        assert cb.is_banned

    def test_429_with_retry_after_respected(self) -> None:
        """Test 429 with Retry-After header is respected in backoff.

        Per BINANCE_LIMITS.md: respect server's Retry-After.
        """
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.0)
        state = BackoffState()

        # 429 with Retry-After
        error = handle_error_response(429, retry_after_ms=10000)
        assert error is not None
        assert error.retry_after_ms == 10000

        state.record_error()

        # Delay should respect Retry-After
        delay = compute_backoff_delay(config, state, retry_after_ms=error.retry_after_ms)
        assert delay >= 10000

    def test_recovery_after_backoff(self) -> None:
        """Test system recovers after successful backoff."""
        config = BackoffConfig()
        state = BackoffState()
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout_ms=0)

        # Trigger 429
        state.record_error()
        cb.record_failure(is_rate_limit=True)

        assert cb.state == CircuitState.OPEN

        # Transition to half-open (recovery_timeout_ms=0)
        assert cb.can_execute()
        assert cb.state.value == CircuitState.HALF_OPEN.value

        # Simulate successful retry
        cb.record_success()
        state.reset()

        # Should be recovered
        assert cb.state.value == CircuitState.CLOSED.value
        assert state.attempt == 0
        assert compute_backoff_delay(config, state) == 0

    def test_full_418_ban_recovery_flow(self) -> None:
        """Test complete 418 ban and recovery flow."""
        cb = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout_ms=100,
            ban_recovery_timeout_ms=300,
        )

        # Receive 418 ban
        error = handle_error_response(418)
        assert error is not None
        cb.record_failure(is_ip_ban=error.is_ip_ban)

        assert cb.state == CircuitState.OPEN
        assert cb.is_banned

        with patch("time.time") as mock_time:
            # After normal timeout but before ban timeout
            mock_time.return_value = cb.last_failure_time_ms / 1000 + 0.2
            assert not cb.can_execute()  # Still blocked

            # After ban timeout
            mock_time.return_value = cb.last_failure_time_ms / 1000 + 0.4
            assert cb.can_execute()  # Now allowed
            assert cb.state.value == CircuitState.HALF_OPEN.value
            assert not cb.is_banned  # Ban cleared

            # Successful recovery
            cb.record_success()
            assert cb.state == CircuitState.CLOSED
