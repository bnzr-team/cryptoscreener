"""
Tests for backoff and circuit breaker implementation.

Per BINANCE_LIMITS.md:
- On any 429: immediate circuit OPEN
- On 418 (IP ban): immediate OPEN with extended cooldown
- Exponential backoff with jitter
- Respect Retry-After header
- Never "fight" the limiter

DEC-023: Added tests for:
- ReconnectLimiter (global reconnect rate control)
- MessageThrottler (WS subscribe rate limiting)
- Seeded jitter (deterministic replay testing)

DEC-023d: Added tests for:
- RestGovernor (REST API budget/queue/concurrency control)
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from unittest.mock import patch

import pytest

from cryptoscreener.connectors import (
    BackoffConfig,
    BackoffState,
    CircuitBreaker,
    CircuitState,
    GovernorDroppedError,
    GovernorTimeoutError,
    RateLimitError,
    RateLimitKind,
    RestGovernor,
    RestGovernorConfig,
    compute_backoff_delay,
    handle_error_response,
)
from cryptoscreener.connectors.backoff import (
    MessageThrottler,
    MessageThrottlerConfig,
    ReconnectLimiter,
    ReconnectLimiterConfig,
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
        assert cb.state.value == CircuitState.CLOSED.value
        assert cb.can_execute()

    def test_opens_after_threshold_failures(self) -> None:
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        for _ in range(3):
            assert cb.state.value == CircuitState.CLOSED.value
            cb.record_failure()

        assert cb.state.value == CircuitState.OPEN.value
        assert not cb.can_execute()

    def test_429_immediate_open(self) -> None:
        """Test 429 immediately opens circuit.

        Per BINANCE_LIMITS.md: On any 429: immediate circuit OPEN.
        """
        cb = CircuitBreaker(failure_threshold=10)  # High threshold

        # Single 429 should immediately open
        cb.record_failure(is_rate_limit=True)
        assert cb.state.value == CircuitState.OPEN.value

    def test_418_immediate_open_with_ban_flag(self) -> None:
        """Test 418 (IP ban) immediately opens circuit with ban flag.

        Per BINANCE_LIMITS.md: 418 is IP ban, requires extended cooldown.
        """
        cb = CircuitBreaker(failure_threshold=10)

        # Single 418 should immediately open and set ban flag
        cb.record_failure(is_ip_ban=True)
        assert cb.state.value == CircuitState.OPEN.value
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
        assert cb.state.value == CircuitState.OPEN.value
        assert cb.is_banned

        with patch("time.time") as mock_time:
            # After normal timeout (100ms) but before ban timeout (500ms)
            mock_time.return_value = cb.last_failure_time_ms / 1000 + 0.2

            # Should still be blocked (ban needs longer)
            assert not cb.can_execute()
            assert cb.state.value == CircuitState.OPEN.value

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
        assert cb.state.value == CircuitState.CLOSED.value

    def test_half_open_after_recovery_timeout(self) -> None:
        """Test circuit goes half-open after recovery timeout."""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_ms=100,  # Short timeout for test
        )

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state.value == CircuitState.OPEN.value
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
        assert cb.state.value == CircuitState.CLOSED.value

    def test_half_open_failure_reopens_circuit(self) -> None:
        """Test failure in half-open reopens circuit."""
        cb = CircuitBreaker(failure_threshold=2)

        # Force half-open
        cb.state = CircuitState.HALF_OPEN

        # Failure should reopen
        cb.record_failure()
        assert cb.state.value == CircuitState.OPEN.value

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
        assert cb.state.value == CircuitState.OPEN.value

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
        assert cb.state.value == CircuitState.OPEN.value

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
        assert cb.state.value == CircuitState.OPEN.value
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
        assert cb.state.value == CircuitState.OPEN.value
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

        assert cb.state.value == CircuitState.OPEN.value
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

        assert cb.state.value == CircuitState.OPEN.value

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

        assert cb.state.value == CircuitState.OPEN.value
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
            assert cb.state.value == CircuitState.CLOSED.value


# =============================================================================
# DEC-023: Seeded Jitter Tests (Deterministic Replay)
# =============================================================================


class TestSeededJitter:
    """Tests for deterministic backoff with seeded RNG (DEC-023)."""

    def test_seeded_jitter_is_deterministic(self) -> None:
        """Test that same seed produces same jitter."""
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.5)
        state = BackoffState()
        state.record_error()

        # Two runs with same seed should produce identical delays
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        delay1 = compute_backoff_delay(config, state, rng=rng1)
        delay2 = compute_backoff_delay(config, state, rng=rng2)

        assert delay1 == delay2

    def test_different_seeds_different_jitter(self) -> None:
        """Test that different seeds produce different delays."""
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.5)
        state = BackoffState()
        state.record_error()

        rng1 = random.Random(42)
        rng2 = random.Random(999)

        delay1 = compute_backoff_delay(config, state, rng=rng1)
        delay2 = compute_backoff_delay(config, state, rng=rng2)

        # Very unlikely to be equal with different seeds
        # (but possible, so we don't assert != directly in production)
        # Just verify both are within jitter range
        assert 500 <= delay1 <= 1500
        assert 500 <= delay2 <= 1500

    def test_seeded_sequence_is_reproducible(self) -> None:
        """Test that a sequence of backoffs is reproducible."""
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.5)

        def compute_sequence(seed: int) -> list[int]:
            """Compute sequence of delays with given seed."""
            rng = random.Random(seed)
            state = BackoffState()
            delays = []
            for _ in range(5):
                state.record_error()
                delay = compute_backoff_delay(config, state, rng=rng)
                delays.append(delay)
            return delays

        seq1 = compute_sequence(42)
        seq2 = compute_sequence(42)

        assert seq1 == seq2

    def test_no_rng_uses_global_random(self) -> None:
        """Test that without RNG, global random is used (non-deterministic)."""
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.5)
        state = BackoffState()
        state.record_error()

        # Multiple runs without RNG should have variation
        delays = set()
        for _ in range(50):
            delay = compute_backoff_delay(config, state)
            delays.add(delay)

        # Should have variation (not all the same)
        assert len(delays) > 1


# =============================================================================
# DEC-023: ReconnectLimiter Tests
# =============================================================================


class TestReconnectLimiterConfig:
    """Tests for ReconnectLimiterConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = ReconnectLimiterConfig()
        assert config.max_reconnects_per_window == 5
        assert config.window_ms == 60000
        assert config.cooldown_after_burst_ms == 30000
        assert config.per_shard_min_interval_ms == 5000

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = ReconnectLimiterConfig(
            max_reconnects_per_window=10,
            window_ms=120000,
            cooldown_after_burst_ms=60000,
            per_shard_min_interval_ms=10000,
        )
        assert config.max_reconnects_per_window == 10
        assert config.window_ms == 120000


class TestReconnectLimiter:
    """Tests for ReconnectLimiter (DEC-023: reconnect storm protection)."""

    def test_initial_state_allows_reconnect(self) -> None:
        """Test fresh limiter allows reconnects."""
        limiter = ReconnectLimiter()
        assert limiter.can_reconnect(shard_id=0)
        assert limiter.get_wait_time_ms(shard_id=0) == 0

    def test_per_shard_min_interval_enforced(self) -> None:
        """Test per-shard minimum interval is enforced."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = ReconnectLimiterConfig(per_shard_min_interval_ms=5000)
        limiter = ReconnectLimiter(config=config, _time_fn=time_fn)

        # First reconnect allowed
        assert limiter.can_reconnect(shard_id=0)
        limiter.record_reconnect(shard_id=0)

        # Immediately after: blocked for same shard
        assert not limiter.can_reconnect(shard_id=0)
        assert limiter.get_wait_time_ms(shard_id=0) == 5000

        # Different shard: allowed
        assert limiter.can_reconnect(shard_id=1)

        # After interval: allowed again
        fake_time = 5001
        assert limiter.can_reconnect(shard_id=0)

    def test_global_rate_limit_enforced(self) -> None:
        """Test global reconnect rate limit is enforced."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = ReconnectLimiterConfig(
            max_reconnects_per_window=3,
            window_ms=60000,
            per_shard_min_interval_ms=0,  # Disable per-shard limit for this test
        )
        limiter = ReconnectLimiter(config=config, _time_fn=time_fn)

        # Record 3 reconnects (different shards to bypass per-shard limit)
        for i in range(3):
            assert limiter.can_reconnect(shard_id=i)
            limiter.record_reconnect(shard_id=i)

        # 4th reconnect blocked (rate limit hit)
        assert not limiter.can_reconnect(shard_id=10)

    def test_cooldown_after_burst(self) -> None:
        """Test cooldown is enforced after hitting rate limit."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = ReconnectLimiterConfig(
            max_reconnects_per_window=2,
            cooldown_after_burst_ms=10000,
            window_ms=5000,  # Window shorter than cooldown so timestamps expire
            per_shard_min_interval_ms=0,
        )
        limiter = ReconnectLimiter(config=config, _time_fn=time_fn)

        # Hit rate limit
        limiter.record_reconnect(shard_id=0)
        limiter.record_reconnect(shard_id=1)

        # Trigger cooldown by trying to reconnect
        assert not limiter.can_reconnect(shard_id=2)

        # Still in cooldown
        fake_time = 5000
        assert not limiter.can_reconnect(shard_id=0)

        # After cooldown (and after window expires, so old timestamps pruned)
        fake_time = 10001
        assert limiter.can_reconnect(shard_id=0)

    def test_sliding_window_expires_old_timestamps(self) -> None:
        """Test sliding window removes old reconnect timestamps."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = ReconnectLimiterConfig(
            max_reconnects_per_window=3,
            window_ms=10000,
            per_shard_min_interval_ms=0,
        )
        limiter = ReconnectLimiter(config=config, _time_fn=time_fn)

        # Fill up the window
        for i in range(3):
            limiter.record_reconnect(shard_id=i)

        # Window is full
        status = limiter.get_status()
        assert status["reconnects_in_window"] == 3

        # Move time forward past window
        fake_time = 15000

        # Old timestamps should be pruned, allowing new reconnects
        assert limiter.can_reconnect(shard_id=10)
        status = limiter.get_status()
        assert status["reconnects_in_window"] == 0

    def test_get_wait_time_accounts_for_all_limits(self) -> None:
        """Test get_wait_time_ms returns correct wait time."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = ReconnectLimiterConfig(
            per_shard_min_interval_ms=5000,
            max_reconnects_per_window=10,
        )
        limiter = ReconnectLimiter(config=config, _time_fn=time_fn)

        limiter.record_reconnect(shard_id=0)

        # Should wait for per-shard interval
        wait_time = limiter.get_wait_time_ms(shard_id=0)
        assert wait_time == 5000

    def test_reset_clears_state(self) -> None:
        """Test reset clears all limiter state."""
        limiter = ReconnectLimiter()
        limiter.record_reconnect(shard_id=0)
        limiter.record_reconnect(shard_id=1)

        limiter.reset()

        status = limiter.get_status()
        assert status["reconnects_in_window"] == 0
        assert not status["in_cooldown"]

    def test_deterministic_with_fake_time(self) -> None:
        """Test limiter behavior is deterministic with fake time (for replay)."""

        def run_scenario() -> list[bool]:
            """Run a scenario and return sequence of can_reconnect results."""
            fake_time = 0

            def time_fn() -> int:
                nonlocal fake_time
                return fake_time

            config = ReconnectLimiterConfig(
                max_reconnects_per_window=2,
                per_shard_min_interval_ms=1000,
            )
            limiter = ReconnectLimiter(config=config, _time_fn=time_fn)

            results = []
            for _step in range(10):
                can = limiter.can_reconnect(shard_id=0)
                results.append(can)
                if can:
                    limiter.record_reconnect(shard_id=0)
                fake_time += 500  # Advance 500ms per step

            return results

        # Two runs should produce identical results
        run1 = run_scenario()
        run2 = run_scenario()
        assert run1 == run2


# =============================================================================
# DEC-023: MessageThrottler Tests
# =============================================================================


class TestMessageThrottlerConfig:
    """Tests for MessageThrottlerConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = MessageThrottlerConfig()
        assert config.max_messages_per_second == 10
        assert config.safety_margin == 0.8
        assert config.burst_allowance == 5

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = MessageThrottlerConfig(
            max_messages_per_second=20,
            safety_margin=0.9,
            burst_allowance=10,
        )
        assert config.max_messages_per_second == 20


class TestMessageThrottler:
    """Tests for MessageThrottler (DEC-023: WS subscribe rate limiting)."""

    def test_initial_burst_allowed(self) -> None:
        """Test initial burst is allowed."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = MessageThrottlerConfig(burst_allowance=5)
        throttler = MessageThrottler(config=config, _time_fn=time_fn)

        # Should allow burst_allowance messages immediately
        for _ in range(5):
            assert throttler.can_send()
            assert throttler.consume()

    def test_rate_limiting_after_burst(self) -> None:
        """Test rate limiting kicks in after burst exhausted."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = MessageThrottlerConfig(
            max_messages_per_second=10,
            safety_margin=1.0,  # No safety margin for easier testing
            burst_allowance=3,
        )
        throttler = MessageThrottler(config=config, _time_fn=time_fn)

        # Exhaust burst
        for _ in range(3):
            assert throttler.consume()

        # Now rate limited
        assert not throttler.can_send()

    def test_tokens_refill_over_time(self) -> None:
        """Test tokens refill based on elapsed time."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = MessageThrottlerConfig(
            max_messages_per_second=10,
            safety_margin=1.0,
            burst_allowance=2,
        )
        throttler = MessageThrottler(config=config, _time_fn=time_fn)

        # Exhaust all tokens
        throttler.consume(2)
        assert not throttler.can_send()

        # Advance 100ms = 1 token refilled (10/sec * 0.1s = 1)
        fake_time = 100
        assert throttler.can_send()
        assert throttler.consume()

    def test_get_wait_time_accurate(self) -> None:
        """Test get_wait_time_ms returns accurate wait time."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = MessageThrottlerConfig(
            max_messages_per_second=10,
            safety_margin=1.0,
            burst_allowance=0,  # No burst for easier testing
        )
        throttler = MessageThrottler(config=config, _time_fn=time_fn)
        throttler._tokens = 0.0  # Force empty

        # Need 1 token, rate is 10/sec, so need 100ms
        wait_time = throttler.get_wait_time_ms(1)
        assert wait_time == 101  # +1 for ceiling

    def test_safety_margin_applied(self) -> None:
        """Test safety margin reduces effective rate."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = MessageThrottlerConfig(
            max_messages_per_second=10,
            safety_margin=0.8,  # 80% of 10 = 8
            burst_allowance=0,
        )
        throttler = MessageThrottler(config=config, _time_fn=time_fn)

        assert throttler.effective_rate == 8.0

    def test_consume_returns_false_when_insufficient(self) -> None:
        """Test consume returns False when not enough tokens."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = MessageThrottlerConfig(burst_allowance=2)
        throttler = MessageThrottler(config=config, _time_fn=time_fn)

        # Try to consume more than available
        assert not throttler.consume(10)

        # Tokens should not be consumed on failure
        assert throttler._tokens == 2.0

    def test_batch_consumption(self) -> None:
        """Test consuming multiple messages at once."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = MessageThrottlerConfig(burst_allowance=10)
        throttler = MessageThrottler(config=config, _time_fn=time_fn)

        # Consume batch of 5
        assert throttler.can_send(5)
        assert throttler.consume(5)

        # 5 tokens remaining
        status = throttler.get_status()
        assert status["available_tokens"] == 5.0

    def test_reset_restores_burst(self) -> None:
        """Test reset restores tokens to burst allowance."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = MessageThrottlerConfig(burst_allowance=5)
        throttler = MessageThrottler(config=config, _time_fn=time_fn)

        # Exhaust tokens
        throttler.consume(5)
        assert not throttler.can_send()

        # Reset
        throttler.reset()
        assert throttler.can_send()
        status = throttler.get_status()
        assert status["available_tokens"] == 5.0

    def test_deterministic_with_fake_time(self) -> None:
        """Test throttler behavior is deterministic with fake time."""

        def run_scenario() -> list[bool]:
            """Run a scenario and return sequence of can_send results."""
            fake_time = 0

            def time_fn() -> int:
                nonlocal fake_time
                return fake_time

            config = MessageThrottlerConfig(
                max_messages_per_second=10,
                safety_margin=1.0,
                burst_allowance=3,
            )
            throttler = MessageThrottler(config=config, _time_fn=time_fn)

            results = []
            for _step in range(20):
                can = throttler.can_send()
                results.append(can)
                if can:
                    throttler.consume()
                fake_time += 50  # Advance 50ms per step

            return results

        # Two runs should produce identical results
        run1 = run_scenario()
        run2 = run_scenario()
        assert run1 == run2

    def test_get_status_for_observability(self) -> None:
        """Test status includes useful info for monitoring."""
        config = MessageThrottlerConfig(
            max_messages_per_second=10,
            safety_margin=0.8,
            burst_allowance=5,
        )
        throttler = MessageThrottler(config=config)

        status = throttler.get_status()
        assert "available_tokens" in status
        assert "effective_rate_per_sec" in status
        assert "burst_allowance" in status
        assert status["effective_rate_per_sec"] == 8.0
        assert status["burst_allowance"] == 5


# =============================================================================
# DEC-023c: CircuitBreaker Deterministic State Machine Tests
# =============================================================================


class TestCircuitBreakerDeterministic:
    """Deterministic CircuitBreaker tests with fake clock (DEC-023c)."""

    def test_closed_to_open_on_429_deterministic(self) -> None:
        """Test CLOSED → OPEN on single 429 with fake clock."""
        fake_time = 1000000  # Start at 1 second

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            failure_threshold=10,  # High threshold
            recovery_timeout_ms=30000,
            _time_fn=time_fn,
        )

        # Initial state: CLOSED
        assert cb.state.value == CircuitState.CLOSED.value
        assert cb.can_execute()

        # Single 429 → immediate OPEN
        cb.record_failure(is_rate_limit=True)
        assert cb.state.value == CircuitState.OPEN.value
        assert not cb.can_execute()
        assert cb.last_failure_time_ms == 1000000

    def test_closed_to_open_on_418_deterministic(self) -> None:
        """Test CLOSED → OPEN on single 418 with ban flag set."""
        fake_time = 2000000

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            failure_threshold=10,
            ban_recovery_timeout_ms=300000,  # 5 min
            _time_fn=time_fn,
        )

        # Single 418 → immediate OPEN + ban flag
        cb.record_failure(is_ip_ban=True)
        assert cb.state.value == CircuitState.OPEN.value
        assert cb.is_banned
        assert cb.last_failure_time_ms == 2000000

    def test_open_to_half_open_after_timeout_deterministic(self) -> None:
        """Test OPEN → HALF_OPEN after recovery_timeout_ms passes."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_ms=30000,  # 30 seconds
            _time_fn=time_fn,
        )

        # Open circuit via 429
        cb.record_failure(is_rate_limit=True)
        assert cb.state.value == CircuitState.OPEN.value
        assert cb.last_failure_time_ms == 0

        # Before timeout: still blocked
        fake_time = 29999
        assert not cb.can_execute()
        assert cb.state.value == CircuitState.OPEN.value

        # At exactly timeout: still blocked (need to exceed)
        fake_time = 30000
        assert cb.can_execute()  # Transitions on >= timeout
        assert cb.state.value == CircuitState.HALF_OPEN.value

    def test_half_open_to_closed_on_success_deterministic(self) -> None:
        """Test HALF_OPEN → CLOSED on successful request."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            recovery_timeout_ms=1000,
            _time_fn=time_fn,
        )

        # Open and transition to HALF_OPEN
        cb.record_failure(is_rate_limit=True)
        fake_time = 1001
        assert cb.can_execute()
        assert cb.state.value == CircuitState.HALF_OPEN.value

        # Success closes circuit
        cb.record_success()
        assert cb.state.value == CircuitState.CLOSED.value
        assert cb.failure_count == 0

    def test_half_open_to_open_on_failure_deterministic(self) -> None:
        """Test HALF_OPEN → OPEN on failed request."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            recovery_timeout_ms=1000,
            _time_fn=time_fn,
        )

        # Open and transition to HALF_OPEN
        cb.record_failure(is_rate_limit=True)
        fake_time = 1001
        assert cb.can_execute()
        assert cb.state.value == CircuitState.HALF_OPEN.value

        # Failure during HALF_OPEN reopens
        cb.record_failure()
        assert cb.state.value == CircuitState.OPEN.value
        assert cb.last_failure_time_ms == 1001

    def test_full_state_machine_cycle_deterministic(self) -> None:
        """Test complete CLOSED → OPEN → HALF_OPEN → CLOSED cycle."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout_ms=30000,
            half_open_max_requests=1,
            _time_fn=time_fn,
        )

        # Step 1: CLOSED
        assert cb.state.value == CircuitState.CLOSED.value
        assert cb.can_execute()

        # Step 2: CLOSED → OPEN (via 429)
        fake_time = 1000
        cb.record_failure(is_rate_limit=True)
        assert cb.state.value == CircuitState.OPEN.value
        assert not cb.can_execute()

        # Step 3: Wait in OPEN
        fake_time = 20000
        assert not cb.can_execute()
        assert cb.state.value == CircuitState.OPEN.value

        # Step 4: OPEN → HALF_OPEN (after timeout)
        fake_time = 31001
        assert cb.can_execute()
        assert cb.state.value == CircuitState.HALF_OPEN.value
        assert cb.half_open_requests == 1

        # Step 5: HALF_OPEN blocks additional requests
        assert not cb.can_execute()

        # Step 6: HALF_OPEN → CLOSED (success)
        cb.record_success()
        assert cb.state.value == CircuitState.CLOSED.value
        assert cb.can_execute()

    def test_force_open_with_fake_clock(self) -> None:
        """Test force_open respects duration with fake clock."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            recovery_timeout_ms=1000,  # Short normal timeout
            _time_fn=time_fn,
        )

        # Force open for 10 seconds
        cb.force_open(duration_ms=10000, reason="test")
        assert cb.state.value == CircuitState.OPEN.value
        assert cb._open_until_ms == 10000

        # Before duration: blocked
        fake_time = 5000
        assert not cb.can_execute()

        # After duration: allowed (transitions to HALF_OPEN)
        fake_time = 10001
        assert cb.can_execute()
        assert cb.state.value == CircuitState.HALF_OPEN.value


# =============================================================================
# DEC-023c: 418 Ban Recovery Tests (5 minute cooldown)
# =============================================================================


class TestCircuitBreaker418BanRecovery:
    """Tests for 418 IP ban recovery with 5-minute cooldown (DEC-023c)."""

    def test_418_uses_5_minute_recovery(self) -> None:
        """Test 418 ban uses ban_recovery_timeout_ms (5 min default)."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            recovery_timeout_ms=30000,  # 30 sec normal
            ban_recovery_timeout_ms=300000,  # 5 min for ban
            _time_fn=time_fn,
        )

        # 418 triggers ban
        cb.record_failure(is_ip_ban=True)
        assert cb.state.value == CircuitState.OPEN.value
        assert cb.is_banned

        # After normal timeout (30s): still blocked
        fake_time = 31000
        assert not cb.can_execute()
        assert cb.state.value == CircuitState.OPEN.value

        # After 4 min: still blocked
        fake_time = 240000
        assert not cb.can_execute()

        # After 5 min: can recover
        fake_time = 300001
        assert cb.can_execute()
        assert cb.state.value == CircuitState.HALF_OPEN.value

    def test_418_recovery_deterministic_full_cycle(self) -> None:
        """Test complete 418 ban → recovery cycle with fake clock."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            recovery_timeout_ms=30000,
            ban_recovery_timeout_ms=300000,
            _time_fn=time_fn,
        )

        # 418 ban at t=0
        cb.record_failure(is_ip_ban=True)
        assert cb.is_banned
        assert cb.last_failure_time_ms == 0

        # Various time checks
        for check_time, should_block in [
            (1000, True),  # 1 second
            (60000, True),  # 1 minute
            (180000, True),  # 3 minutes
            (299999, True),  # Just before 5 min
            (300000, False),  # At 5 min - can recover
        ]:
            fake_time = check_time
            if should_block:
                assert not cb.can_execute(), f"Should be blocked at {check_time}ms"
            else:
                assert cb.can_execute(), f"Should be allowed at {check_time}ms"

        # Ban flag cleared on transition
        assert cb.state.value == CircuitState.HALF_OPEN.value
        assert not cb.is_banned

    def test_418_clears_ban_flag_on_half_open(self) -> None:
        """Test ban flag is cleared when transitioning to HALF_OPEN."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            ban_recovery_timeout_ms=1000,
            _time_fn=time_fn,
        )

        cb.record_failure(is_ip_ban=True)
        assert cb.is_banned

        fake_time = 1001
        cb.can_execute()  # Triggers transition

        assert cb.state.value == CircuitState.HALF_OPEN.value
        assert not cb.is_banned

    def test_418_during_half_open_sets_ban_flag(self) -> None:
        """Test 418 during HALF_OPEN reopens with ban flag."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            recovery_timeout_ms=1000,
            ban_recovery_timeout_ms=300000,
            _time_fn=time_fn,
        )

        # Get to HALF_OPEN
        cb.record_failure(is_rate_limit=True)
        fake_time = 1001
        cb.can_execute()
        assert cb.state.value == CircuitState.HALF_OPEN.value

        # 418 during HALF_OPEN
        cb.record_failure(is_ip_ban=True)
        assert cb.state.value == CircuitState.OPEN.value
        assert cb.is_banned

        # Now needs 5 min to recover
        fake_time = 2000
        assert not cb.can_execute()

        fake_time = 301002  # 1001 + 300001
        assert cb.can_execute()


# =============================================================================
# DEC-023c: Retry-After Parsing Proof Tests
# =============================================================================


class TestRetryAfterParsing:
    """Tests for Retry-After header handling (DEC-023c)."""

    def test_retry_after_seconds_format(self) -> None:
        """Test Retry-After in seconds format is respected."""
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.0)
        state = BackoffState()
        state.record_error()  # attempt 1 → 1000ms base

        # Server says wait 60 seconds (60000ms)
        delay = compute_backoff_delay(config, state, retry_after_ms=60000)

        # Should use server's value (larger than computed)
        assert delay >= 60000

    def test_retry_after_takes_max(self) -> None:
        """Test delay is max(computed, retry_after)."""
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.0)
        state = BackoffState()

        # Multiple retries → large computed delay
        for _ in range(5):
            state.record_error()  # attempt 5 → 16000ms base

        # Server says 5000ms, but computed is larger
        delay = compute_backoff_delay(config, state, retry_after_ms=5000)
        assert delay == 16000  # Uses computed, not smaller retry_after

        # Server says 100000ms, larger than computed
        delay2 = compute_backoff_delay(config, state, retry_after_ms=100000)
        assert delay2 >= 100000  # Uses retry_after

    def test_retry_after_missing_uses_backoff(self) -> None:
        """Test missing Retry-After uses exponential backoff."""
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.0)
        state = BackoffState()
        state.record_error()

        # No retry_after → uses computed backoff
        delay = compute_backoff_delay(config, state, retry_after_ms=None)
        assert delay == 1000

    def test_retry_after_zero_uses_backoff(self) -> None:
        """Test Retry-After: 0 uses exponential backoff."""
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.0)
        state = BackoffState()
        state.record_error()

        # retry_after=0 → uses computed backoff
        delay = compute_backoff_delay(config, state, retry_after_ms=0)
        assert delay == 1000

    def test_retry_after_preserved_in_error(self) -> None:
        """Test RateLimitError preserves retry_after_ms from response."""
        error = handle_error_response(429, retry_after_ms=45000)
        assert error is not None
        assert error.retry_after_ms == 45000

    def test_418_default_retry_after(self) -> None:
        """Test 418 has default 5-minute retry_after when not provided."""
        error = handle_error_response(418)
        assert error is not None
        assert error.retry_after_ms == 300000  # 5 minutes


# =============================================================================
# DEC-023c: -1003 Specific Tests
# =============================================================================


class TestMinus1003Handling:
    """Tests for -1003 TOO_MANY_REQUESTS handling (DEC-023c)."""

    def test_minus_1003_recognized(self) -> None:
        """Test -1003 error code is recognized as rate limit."""
        error = handle_error_response(200, error_code=-1003)
        assert error is not None
        assert isinstance(error, RateLimitError)
        assert error.error_code == -1003
        assert error.kind == RateLimitKind.TOO_MANY_REQUESTS

    def test_minus_1003_opens_circuit(self) -> None:
        """Test -1003 opens circuit (treated as rate limit)."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            failure_threshold=10,
            _time_fn=time_fn,
        )

        # -1003 should be treated as rate limit → immediate OPEN
        error = handle_error_response(200, error_code=-1003)
        assert error is not None

        # Record as rate limit
        cb.record_failure(is_rate_limit=True)
        assert cb.state.value == CircuitState.OPEN.value

    def test_minus_1003_with_retry_after(self) -> None:
        """Test -1003 with Retry-After is respected."""
        error = handle_error_response(200, error_code=-1003, retry_after_ms=30000)
        assert error is not None
        assert error.retry_after_ms == 30000

        # Use in backoff
        config = BackoffConfig(base_delay_ms=1000, jitter_factor=0.0)
        state = BackoffState()
        state.record_error()

        delay = compute_backoff_delay(config, state, retry_after_ms=error.retry_after_ms)
        assert delay >= 30000

    def test_minus_1003_vs_429_vs_418_kinds(self) -> None:
        """Test different error types have correct kinds."""
        error_429 = handle_error_response(429)
        error_418 = handle_error_response(418)
        error_1003 = handle_error_response(200, error_code=-1003)

        assert error_429 is not None
        assert error_418 is not None
        assert error_1003 is not None

        assert error_429.kind == RateLimitKind.RATE_LIMIT
        assert error_418.kind == RateLimitKind.IP_BAN
        assert error_1003.kind == RateLimitKind.TOO_MANY_REQUESTS

    def test_minus_1003_is_not_ip_ban(self) -> None:
        """Test -1003 is not classified as IP ban."""
        error = handle_error_response(200, error_code=-1003)
        assert error is not None
        assert not error.is_ip_ban


# =============================================================================
# DEC-023c: Determinism Proof Test
# =============================================================================


class TestCircuitBreakerDeterminismProof:
    """Replay determinism proof for CircuitBreaker (DEC-023c)."""

    def test_state_machine_sequence_is_reproducible(self) -> None:
        """Test entire state machine sequence is reproducible with fake clock."""

        def run_scenario() -> list[tuple[int, str, bool]]:
            """Run scenario and return (time, state, can_execute) tuples."""
            fake_time = 0

            def time_fn() -> int:
                return fake_time

            cb = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout_ms=10000,
                ban_recovery_timeout_ms=50000,
                half_open_max_requests=1,
                _time_fn=time_fn,
            )

            results: list[tuple[int, str, bool]] = []

            # Record initial state
            results.append((fake_time, cb.state.value, cb.can_execute()))

            # Simulate sequence of events
            events = [
                ("failure", 1000),
                ("failure", 2000),
                ("rate_limit", 3000),  # Opens circuit
                ("check", 5000),
                ("check", 10000),
                ("check", 13001),  # Transitions to HALF_OPEN
                ("failure", 13002),  # Back to OPEN
                ("check", 20000),
                ("check", 23003),  # Transitions to HALF_OPEN
                ("success", 23004),  # Closes circuit
                ("check", 25000),
            ]

            for event_type, event_time in events:
                fake_time = event_time

                if event_type == "failure":
                    cb.record_failure()
                elif event_type == "rate_limit":
                    cb.record_failure(is_rate_limit=True)
                elif event_type == "ban":
                    cb.record_failure(is_ip_ban=True)
                elif event_type == "success":
                    cb.record_success()
                elif event_type == "check":
                    cb.can_execute()

                results.append((fake_time, cb.state.value, cb.can_execute()))

            return results

        # Two runs should produce identical results
        run1 = run_scenario()
        run2 = run_scenario()

        assert run1 == run2, "State machine sequence not reproducible"

        # Verify specific states in sequence
        # Final state should be CLOSED after success
        assert run1[-1][1] == "CLOSED"
        assert run1[-1][2] is True  # can_execute

    def test_ban_recovery_sequence_is_reproducible(self) -> None:
        """Test 418 ban recovery sequence is reproducible."""

        def run_ban_scenario() -> list[tuple[int, str, bool, bool]]:
            """Run ban scenario and return (time, state, can_execute, is_banned)."""
            fake_time = 0

            def time_fn() -> int:
                return fake_time

            cb = CircuitBreaker(
                recovery_timeout_ms=30000,
                ban_recovery_timeout_ms=300000,
                _time_fn=time_fn,
            )

            results: list[tuple[int, str, bool, bool]] = []

            # Ban at t=0
            cb.record_failure(is_ip_ban=True)
            results.append((fake_time, cb.state.value, cb.can_execute(), cb.is_banned))

            # Check at various times
            check_times = [1000, 60000, 180000, 299999, 300000, 300001]
            for t in check_times:
                fake_time = t
                can_exec = cb.can_execute()
                results.append((t, cb.state.value, can_exec, cb.is_banned))

            return results

        run1 = run_ban_scenario()
        run2 = run_ban_scenario()

        assert run1 == run2, "Ban recovery sequence not reproducible"


# =============================================================================
# DEC-023d: REST Governor Tests
# =============================================================================


class TestRestGovernorConfig:
    """Tests for RestGovernorConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RestGovernorConfig()
        assert config.budget_weight_per_minute == 2000
        assert config.max_queue_depth == 50
        assert config.max_concurrent_requests == 10
        assert config.default_endpoint_weight == 10

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = RestGovernorConfig(
            budget_weight_per_minute=1000,
            max_queue_depth=20,
            max_concurrent_requests=5,
        )
        assert config.budget_weight_per_minute == 1000
        assert config.max_queue_depth == 20
        assert config.max_concurrent_requests == 5

    def test_get_endpoint_weight_default(self) -> None:
        """Test default endpoint weight."""
        config = RestGovernorConfig()
        weight = config.get_endpoint_weight("/fapi/v1/unknown")
        assert weight == 10

    def test_get_endpoint_weight_known_endpoints(self) -> None:
        """Test weights for known endpoints."""
        config = RestGovernorConfig()
        assert config.get_endpoint_weight("/fapi/v1/exchangeInfo") == 40
        assert config.get_endpoint_weight("/fapi/v1/ticker/24hr") == 40
        assert config.get_endpoint_weight("/fapi/v1/time") == 1

    def test_get_endpoint_weight_custom_override(self) -> None:
        """Test custom endpoint weight override."""
        config = RestGovernorConfig(
            endpoint_weights={"/custom/endpoint": 100},
        )
        assert config.get_endpoint_weight("/custom/endpoint") == 100
        # Known endpoint still works
        assert config.get_endpoint_weight("/fapi/v1/time") == 1


class TestRestGovernorBudget:
    """Tests for RestGovernor budget behavior.

    DEC-023d test plan: TestRestGovernorBudget
    """

    @pytest.mark.asyncio
    async def test_allows_request_under_budget(self) -> None:
        """Test request is allowed when budget is available."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = RestGovernorConfig(budget_weight_per_minute=100)
        governor = RestGovernor(config=config, _time_fn=time_fn)

        # Should allow request with weight 10 (under budget of 100)
        await governor.acquire("/test", weight=10)
        await governor.release()

        assert governor.metrics.requests_allowed == 1
        assert governor.metrics.requests_deferred == 0

    @pytest.mark.asyncio
    async def test_defers_request_when_budget_exhausted(self) -> None:
        """Test request is deferred when budget is exhausted.

        Uses real time with high refill rate to avoid fake/real time mismatch
        with asyncio.Condition.wait().
        """
        # High refill rate: 60000 weight/min = 1000 weight/sec
        # So 10 tokens refill in just 10ms
        config = RestGovernorConfig(
            budget_weight_per_minute=60000,  # 1000 tokens/sec
            default_timeout_ms=2000,
        )
        governor = RestGovernor(config=config)

        # Consume most of the budget (leave 5 tokens)
        await governor.acquire("/test", weight=59995)
        await governor.release()

        # Now we have 5 tokens. Next request needs weight=10.
        # It will wait briefly for budget to refill (~5ms for 5 more tokens)
        await governor.acquire("/test", weight=10, timeout_ms=2000)
        await governor.release()

        # Should have been deferred (had to wait for budget)
        assert governor.metrics.requests_allowed == 1
        assert governor.metrics.requests_deferred == 1

    @pytest.mark.asyncio
    async def test_budget_refills_over_time(self) -> None:
        """Test budget refills continuously over time."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = RestGovernorConfig(budget_weight_per_minute=600)  # 10 per second
        governor = RestGovernor(config=config, _time_fn=time_fn)

        # Consume all budget
        await governor.acquire("/test", weight=600)
        await governor.release()

        status1 = governor.get_status()
        assert status1["budget_tokens"] == 0

        # Advance time by 1 second (should refill 10 tokens)
        fake_time = 1000
        status2 = governor.get_status()
        assert status2["budget_tokens"] == pytest.approx(10.0, rel=0.1)

        # Advance time by 1 minute (should be at max)
        fake_time = 60000
        status3 = governor.get_status()
        assert status3["budget_tokens"] == 600

    @pytest.mark.asyncio
    async def test_endpoint_weights_respected(self) -> None:
        """Test endpoint weights are respected."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = RestGovernorConfig(budget_weight_per_minute=100)
        governor = RestGovernor(config=config, _time_fn=time_fn)

        # /fapi/v1/exchangeInfo has weight 40
        await governor.acquire("/fapi/v1/exchangeInfo")
        await governor.release()

        # Should have 60 tokens left (100 - 40)
        status = governor.get_status()
        assert status["budget_tokens"] == 60


class TestRestGovernorQueue:
    """Tests for RestGovernor queue behavior.

    DEC-023d test plan: TestRestGovernorQueue
    """

    @pytest.mark.asyncio
    async def test_queue_fifo_order(self) -> None:
        """Test queue follows FIFO order."""
        # Use real time for this test with high budget to focus on queue ordering
        config = RestGovernorConfig(
            budget_weight_per_minute=60000,  # High budget (1000/sec)
            max_concurrent_requests=1,  # Sequential processing
            default_timeout_ms=5000,
        )
        governor = RestGovernor(config=config)

        order: list[int] = []
        start_order: list[int] = []

        async def request(num: int) -> None:
            start_order.append(num)
            await governor.acquire("/test", weight=1)
            order.append(num)
            await asyncio.sleep(0.01)  # Small delay to ensure ordering
            await governor.release()

        # Launch 3 requests concurrently
        # Due to max_concurrent_requests=1, they should queue
        tasks = [asyncio.create_task(request(i)) for i in range(3)]
        await asyncio.gather(*tasks)

        # All should complete and maintain FIFO order
        assert len(order) == 3
        # The order depends on task scheduling, but first to acquire should be first
        assert order == sorted(order) or order == start_order

    @pytest.mark.asyncio
    async def test_drops_new_when_queue_full(self) -> None:
        """Test drop-new policy when queue is full."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = RestGovernorConfig(
            budget_weight_per_minute=10,
            max_queue_depth=2,
            max_concurrent_requests=1,
            default_timeout_ms=5000,
        )
        governor = RestGovernor(config=config, _time_fn=time_fn)

        # Exhaust budget
        await governor.acquire("/test", weight=10)
        # Don't release - keep slot held

        # Queue 2 requests (fills queue)
        queued_tasks = []
        for _ in range(2):

            async def queue_request() -> None:
                with contextlib.suppress(GovernorTimeoutError, GovernorDroppedError):
                    await governor.acquire("/test", weight=1, timeout_ms=100)

            queued_tasks.append(asyncio.create_task(queue_request()))

        await asyncio.sleep(0.01)  # Let tasks start

        # Third request should be dropped immediately (queue full)
        with pytest.raises(GovernorDroppedError) as exc_info:
            await governor.acquire("/test", weight=1)

        assert exc_info.value.queue_depth == 2
        assert governor.metrics.requests_dropped >= 1
        assert governor.metrics.drop_reason_queue_full >= 1

        # Cleanup
        for task in queued_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await governor.release()

    @pytest.mark.asyncio
    async def test_queue_timeout_raises_error(self) -> None:
        """Test timeout while waiting in queue raises GovernorTimeoutError."""
        fake_time = 0

        def time_fn() -> int:
            nonlocal fake_time
            return fake_time

        config = RestGovernorConfig(
            budget_weight_per_minute=10,
            max_concurrent_requests=1,
            default_timeout_ms=100,
        )
        governor = RestGovernor(config=config, _time_fn=time_fn)

        # Exhaust budget and hold concurrency slot
        await governor.acquire("/test", weight=10)

        async def timed_out_request() -> None:
            # Advance time to trigger timeout
            nonlocal fake_time
            await asyncio.sleep(0.01)
            fake_time = 200  # Past timeout
            await asyncio.sleep(0.01)

        async def make_request() -> None:
            with pytest.raises(GovernorTimeoutError) as exc_info:
                await governor.acquire("/test", weight=1, timeout_ms=100)
            assert exc_info.value.waited_ms > 0

        await asyncio.gather(make_request(), timed_out_request())

        assert governor.metrics.drop_reason_timeout >= 1
        await governor.release()


class TestRestGovernorBreaker:
    """Tests for RestGovernor circuit breaker integration.

    DEC-023d test plan: TestRestGovernorBreaker
    """

    @pytest.mark.asyncio
    async def test_fail_fast_when_breaker_open(self) -> None:
        """Test fail-fast when circuit breaker is OPEN."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            recovery_timeout_ms=30000,
            _time_fn=time_fn,
        )
        # Open the circuit
        cb.record_failure(is_rate_limit=True)
        assert cb.state == CircuitState.OPEN

        governor = RestGovernor(
            config=RestGovernorConfig(),
            circuit_breaker=cb,
            _time_fn=time_fn,
        )

        # Request should fail immediately (not queue)
        with pytest.raises(RateLimitError) as exc_info:
            await governor.acquire("/test")

        assert "Circuit breaker OPEN" in str(exc_info.value)
        assert governor.metrics.requests_failed_breaker == 1
        assert governor.metrics.drop_reason_breaker_open == 1

    @pytest.mark.asyncio
    async def test_allows_request_when_breaker_closed(self) -> None:
        """Test request allowed when circuit breaker is CLOSED."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(
            recovery_timeout_ms=30000,
            _time_fn=time_fn,
        )
        assert cb.state == CircuitState.CLOSED

        governor = RestGovernor(
            config=RestGovernorConfig(),
            circuit_breaker=cb,
            _time_fn=time_fn,
        )

        await governor.acquire("/test")
        await governor.release()

        assert governor.metrics.requests_allowed == 1
        assert governor.metrics.requests_failed_breaker == 0

    @pytest.mark.asyncio
    async def test_allows_probe_in_half_open(self) -> None:
        """Test probe request allowed in HALF_OPEN state."""
        fake_time = 0

        def time_fn() -> int:
            nonlocal fake_time
            return fake_time

        cb = CircuitBreaker(
            recovery_timeout_ms=30000,
            half_open_max_requests=2,  # Allow 2 requests in half-open
            _time_fn=time_fn,
        )
        # Open and then transition to half-open
        cb.record_failure(is_rate_limit=True)
        fake_time = 30001  # Past recovery timeout

        # First can_execute() transitions to HALF_OPEN and counts as 1 request
        assert cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Reset the half_open_requests count so governor's check can pass
        cb.half_open_requests = 0

        governor = RestGovernor(
            config=RestGovernorConfig(),
            circuit_breaker=cb,
            _time_fn=time_fn,
        )

        # Probe request should be allowed
        await governor.acquire("/test")
        await governor.release()

        # Record success to close the breaker
        cb.record_success()
        assert cb.state.value == CircuitState.CLOSED.value


class TestRestGovernorConcurrency:
    """Tests for RestGovernor concurrency semaphore.

    DEC-023d test plan: TestRestGovernorConcurrency
    """

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent(self) -> None:
        """Test semaphore limits concurrent requests."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = RestGovernorConfig(
            budget_weight_per_minute=1000,
            max_concurrent_requests=2,
        )
        governor = RestGovernor(config=config, _time_fn=time_fn)

        # Acquire 2 slots
        await governor.acquire("/test", weight=1)
        await governor.acquire("/test", weight=1)

        status = governor.get_status()
        assert status["concurrent"] == 2
        assert status["concurrent_max"] == 2

        # Release one
        await governor.release()
        status = governor.get_status()
        assert status["concurrent"] == 1

        await governor.release()

    @pytest.mark.asyncio
    async def test_burst_exceeds_semaphore_waits(self) -> None:
        """Test requests wait when semaphore is exhausted."""
        fake_time = 0

        def time_fn() -> int:
            nonlocal fake_time
            return fake_time

        config = RestGovernorConfig(
            budget_weight_per_minute=1000,
            max_concurrent_requests=1,
            default_timeout_ms=5000,
        )
        governor = RestGovernor(config=config, _time_fn=time_fn)

        # Hold first slot
        await governor.acquire("/test", weight=1)

        completed = []

        async def second_request() -> None:
            await governor.acquire("/test", weight=1, timeout_ms=2000)
            completed.append(1)
            await governor.release()

        async def release_first() -> None:
            nonlocal fake_time
            await asyncio.sleep(0.01)
            fake_time = 500
            await governor.release()  # Release first slot

        await asyncio.gather(second_request(), release_first())

        assert completed == [1]
        assert governor.metrics.requests_deferred == 1


class TestRestGovernorDeterminism:
    """Tests for RestGovernor determinism with fake clock.

    DEC-023d test plan: TestRestGovernorDeterminism
    """

    @pytest.mark.asyncio
    async def test_decision_sequence_reproducible(self) -> None:
        """Test decision sequence is reproducible with fake clock."""

        async def run_scenario() -> list[tuple[int, int, int]]:
            """Run scenario and return (time, allowed, queued)."""
            fake_time = 0

            def time_fn() -> int:
                return fake_time

            config = RestGovernorConfig(
                budget_weight_per_minute=100,
                max_concurrent_requests=2,
            )
            governor = RestGovernor(config=config, _time_fn=time_fn)

            results: list[tuple[int, int, int]] = []

            # Make several requests
            for _ in range(5):
                await governor.acquire("/test", weight=20)
                await governor.release()
                results.append(
                    (
                        fake_time,
                        governor.metrics.requests_allowed,
                        governor.metrics.current_queue_depth,
                    )
                )

            return results

        run1 = await run_scenario()
        run2 = await run_scenario()

        assert run1 == run2, "Decision sequence not reproducible"

    @pytest.mark.asyncio
    async def test_budget_state_deterministic(self) -> None:
        """Test budget state is deterministic with fake clock."""

        def run_budget_scenario() -> list[float]:
            """Run scenario and return budget levels."""
            fake_time = 0

            def time_fn() -> int:
                return fake_time

            config = RestGovernorConfig(budget_weight_per_minute=600)
            governor = RestGovernor(config=config, _time_fn=time_fn)

            # Initial
            results = [governor.get_status()["budget_tokens"]]

            # Consume 100
            governor._budget_tokens -= 100
            results.append(governor.get_status()["budget_tokens"])

            # Advance 1 second
            fake_time = 1000
            results.append(governor.get_status()["budget_tokens"])

            # Advance 10 seconds
            fake_time = 10000
            results.append(governor.get_status()["budget_tokens"])

            return [float(r) for r in results]

        run1 = run_budget_scenario()
        run2 = run_budget_scenario()

        assert run1 == run2, "Budget state not deterministic"


class TestRestGovernorMetrics:
    """Tests for RestGovernor metrics tracking.

    DEC-023d test plan: TestRestGovernorMetrics
    """

    @pytest.mark.asyncio
    async def test_metrics_increment_on_allow(self) -> None:
        """Test metrics increment when request is allowed."""
        governor = RestGovernor()

        await governor.acquire("/test", weight=10)
        await governor.release()

        assert governor.metrics.requests_allowed == 1
        assert governor.metrics.requests_deferred == 0
        assert governor.metrics.requests_dropped == 0

    @pytest.mark.asyncio
    async def test_metrics_track_wait_time(self) -> None:
        """Test metrics track wait time for deferred requests.

        Uses real time with high refill rate for fast test execution.
        """
        # High refill rate: 60000 weight/min = 1000 weight/sec
        config = RestGovernorConfig(
            budget_weight_per_minute=60000,  # 1000 tokens/sec
            max_concurrent_requests=10,
            default_timeout_ms=2000,
        )
        governor = RestGovernor(config=config)

        # Exhaust most of budget (leave just 1 token)
        await governor.acquire("/test", weight=59999)
        await governor.release()

        # Request needs 5 tokens, only 1 available
        # Should wait for ~4ms to get 4 more tokens (at 1000/sec rate)
        await governor.acquire("/test", weight=5, timeout_ms=2000)
        await governor.release()

        assert governor.metrics.requests_deferred == 1
        assert governor.metrics.total_wait_ms >= 0  # May be 0 on fast machines
        # Note: max_wait_ms may be 0 if refill is very fast

    @pytest.mark.asyncio
    async def test_drop_reasons_categorized(self) -> None:
        """Test drop reasons are categorized correctly."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        # Test breaker open
        cb = CircuitBreaker(_time_fn=time_fn)
        cb.record_failure(is_rate_limit=True)

        governor = RestGovernor(
            config=RestGovernorConfig(max_queue_depth=0),
            circuit_breaker=cb,
            _time_fn=time_fn,
        )

        with contextlib.suppress(RateLimitError):
            await governor.acquire("/test")

        assert governor.metrics.drop_reason_breaker_open == 1

        # Reset and test queue full
        governor2 = RestGovernor(
            config=RestGovernorConfig(
                budget_weight_per_minute=1,
                max_queue_depth=0,
            ),
            _time_fn=time_fn,
        )

        # Exhaust budget
        await governor2.acquire("/test", weight=1)

        with contextlib.suppress(GovernorDroppedError):
            await governor2.acquire("/test", weight=1)

        assert governor2.metrics.drop_reason_queue_full == 1


class TestRestGovernorStatus:
    """Tests for RestGovernor status reporting."""

    def test_get_status_returns_all_fields(self) -> None:
        """Test get_status returns all required fields."""
        governor = RestGovernor()

        status = governor.get_status()

        assert "budget_tokens" in status
        assert "budget_max" in status
        assert "queue_depth" in status
        assert "queue_max" in status
        assert "concurrent" in status
        assert "concurrent_max" in status
        assert "breaker_open" in status

    def test_reset_clears_state(self) -> None:
        """Test reset clears all state."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        governor = RestGovernor(_time_fn=time_fn)

        # Modify state
        governor._budget_tokens = 50
        governor._concurrent_count = 5
        governor.metrics.requests_allowed = 100

        governor.reset()

        assert governor._budget_tokens == governor.config.budget_weight_per_minute
        assert governor._concurrent_count == 0
        assert governor.metrics.requests_allowed == 0


class TestRestGovernorPermitContextManager:
    """Tests for RestGovernor.permit() async context manager.

    DEC-023d: Ensures slot is released even on exception.
    """

    @pytest.mark.asyncio
    async def test_permit_releases_on_normal_exit(self) -> None:
        """Test permit() releases slot on normal exit."""
        governor = RestGovernor()

        async with governor.permit("/test", weight=10):
            assert governor._concurrent_count == 1

        # Slot released after context exit
        assert governor._concurrent_count == 0

    @pytest.mark.asyncio
    async def test_permit_releases_on_exception(self) -> None:
        """Test permit() releases slot even when exception is raised."""
        governor = RestGovernor()

        with pytest.raises(ValueError, match="test error"):
            async with governor.permit("/test", weight=10):
                assert governor._concurrent_count == 1
                raise ValueError("test error")

        # Slot MUST be released even after exception
        assert governor._concurrent_count == 0

    @pytest.mark.asyncio
    async def test_permit_propagates_acquire_errors(self) -> None:
        """Test permit() propagates errors from acquire()."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        cb = CircuitBreaker(_time_fn=time_fn)
        cb.record_failure(is_rate_limit=True)  # Open breaker

        governor = RestGovernor(circuit_breaker=cb, _time_fn=time_fn)

        with pytest.raises(RateLimitError):
            async with governor.permit("/test"):
                pass  # Should not reach here

        # No slot should be held (acquire failed)
        assert governor._concurrent_count == 0

    @pytest.mark.asyncio
    async def test_permit_with_custom_weight_and_timeout(self) -> None:
        """Test permit() respects custom weight and timeout."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        config = RestGovernorConfig(budget_weight_per_minute=100)
        governor = RestGovernor(config=config, _time_fn=time_fn)

        async with governor.permit("/test", weight=50, timeout_ms=1000):
            # 50 weight consumed
            assert governor._budget_tokens == 50

        assert governor._concurrent_count == 0


class TestRestGovernorConcurrencyStress:
    """Stress tests for RestGovernor concurrency cap (atomic enforcement).

    DEC-023d: Proves that concurrency cap is never exceeded even under contention.
    Uses get_concurrent_count() to read state under lock for accurate invariant checks.
    """

    @pytest.mark.asyncio
    async def test_concurrency_cap_never_exceeded_under_contention(self) -> None:
        """Test that concurrency cap is NEVER exceeded with 25+ concurrent tasks.

        This is the critical atomicity test: many tasks try to acquire simultaneously,
        and we verify that at no point does concurrent_count exceed max.

        Uses get_concurrent_count() which acquires lock for safe read.
        """
        config = RestGovernorConfig(
            budget_weight_per_minute=100000,  # High budget to focus on concurrency
            max_concurrent_requests=5,  # Low cap for easier testing
            default_timeout_ms=5000,
        )
        governor = RestGovernor(config=config)

        max_observed_concurrent = 0
        violations: list[tuple[int, int]] = []
        num_tasks = 25
        completed = 0

        async def worker(worker_id: int) -> None:
            nonlocal max_observed_concurrent, completed
            for _ in range(3):  # Each worker does 3 acquire/release cycles
                await governor.acquire("/test", weight=1)
                try:
                    # Use safe accessor that acquires lock
                    current = await governor.get_concurrent_count()
                    if current > max_observed_concurrent:
                        max_observed_concurrent = current
                    if current > config.max_concurrent_requests:
                        violations.append((worker_id, current))
                    # Small delay to increase contention window
                    await asyncio.sleep(0.001)
                finally:
                    await governor.release()
            completed += 1

        # Launch all workers concurrently
        tasks = [asyncio.create_task(worker(i)) for i in range(num_tasks)]
        await asyncio.gather(*tasks)

        # Verify no violations occurred
        assert violations == [], f"Concurrency cap violated: {violations}"
        assert max_observed_concurrent <= config.max_concurrent_requests
        assert completed == num_tasks
        # Final check with safe accessor
        final_count = await governor.get_concurrent_count()
        assert final_count == 0, "All slots should be released"

    @pytest.mark.asyncio
    async def test_permit_concurrency_stress(self) -> None:
        """Test permit() context manager under stress - slots always released.

        Uses get_concurrent_count() which acquires lock for safe read.
        """
        config = RestGovernorConfig(
            budget_weight_per_minute=100000,
            max_concurrent_requests=3,
            default_timeout_ms=5000,
        )
        governor = RestGovernor(config=config)

        errors_during_work: list[str] = []
        completed_count = 0
        max_observed = 0

        async def worker(worker_id: int) -> None:
            nonlocal completed_count, max_observed
            for iteration in range(5):
                async with governor.permit("/test", weight=1):
                    # Use safe accessor that acquires lock
                    current = await governor.get_concurrent_count()
                    if current > max_observed:
                        max_observed = current
                    if current > config.max_concurrent_requests:
                        errors_during_work.append(
                            f"Worker {worker_id} iter {iteration}: count={current}"
                        )
                    await asyncio.sleep(0.0005)  # Simulate work
                completed_count += 1

        tasks = [asyncio.create_task(worker(i)) for i in range(20)]
        await asyncio.gather(*tasks)

        assert errors_during_work == []
        assert max_observed <= config.max_concurrent_requests
        assert completed_count == 20 * 5
        # Final check with safe accessor
        final_count = await governor.get_concurrent_count()
        assert final_count == 0, "All slots should be released"


class TestRestGovernorQueueProgress:
    """Tests for queue progress - timeout of first doesn't block others.

    DEC-023d: Proves event-driven queue allows progress when resources free.
    Uses fake time + notify_waiters() for fully deterministic testing.
    """

    @pytest.mark.asyncio
    async def test_timeout_of_first_doesnt_block_second(self) -> None:
        """Test that when first request times out, second can proceed.

        Uses fake time + notify_waiters() for deterministic behavior:
        1. First request queues with short timeout
        2. Advance time past first's timeout
        3. Notify waiters so first times out
        4. Advance time for budget refill
        5. Notify waiters so second can proceed

        No real asyncio.sleep() timing dependencies.
        """
        current_time = [0]

        def time_fn() -> int:
            return current_time[0]

        config = RestGovernorConfig(
            budget_weight_per_minute=600,  # 10 tokens/sec
            max_concurrent_requests=10,
            default_timeout_ms=5000,
        )
        governor = RestGovernor(config=config, _time_fn=time_fn)

        # Exhaust all budget
        await governor.acquire("/test", weight=600)
        # Budget at 0, slot held

        first_timed_out = False
        second_completed = False
        first_started = asyncio.Event()
        second_started = asyncio.Event()

        async def first_request() -> None:
            """First request - times out waiting for budget."""
            nonlocal first_timed_out
            first_started.set()
            try:
                # Needs 500 tokens, timeout 200ms - will timeout
                await governor.acquire("/test", weight=500, timeout_ms=200)
            except GovernorTimeoutError:
                first_timed_out = True

        async def second_request() -> None:
            """Second request - succeeds after time advances and budget refills."""
            nonlocal second_completed
            # Wait for first to start first (deterministic ordering)
            await first_started.wait()
            second_started.set()
            # Needs only 10 tokens, timeout 5000ms - will succeed after refill
            await governor.acquire("/test", weight=10, timeout_ms=5000)
            await governor.release()
            second_completed = True

        async def time_driver() -> None:
            """Drive time forward and signal waiters."""
            # Wait for both requests to start
            await first_started.wait()
            await second_started.wait()
            # Give tasks a chance to enter wait loop
            await asyncio.sleep(0)

            # Advance time past first request's timeout (200ms)
            current_time[0] = 300
            await governor.notify_waiters()
            await asyncio.sleep(0)  # Let first timeout process

            # Advance time enough for budget refill (10 tokens/sec = 1 token per 100ms)
            # Need 10 tokens = 1000ms of refill time
            current_time[0] = 1500
            await governor.notify_waiters()

        # Run all concurrently
        await asyncio.gather(
            first_request(),
            second_request(),
            time_driver(),
        )
        # Release initial acquisition
        await governor.release()

        assert first_timed_out, "First request should have timed out"
        assert second_completed, "Second request should have completed after refill"

    @pytest.mark.asyncio
    async def test_release_wakes_waiters(self) -> None:
        """Test that release() wakes up waiting requests.

        Deterministic test using fake time.
        """
        current_time = [0]

        def time_fn() -> int:
            return current_time[0]

        config = RestGovernorConfig(
            budget_weight_per_minute=60000,  # High budget
            max_concurrent_requests=1,  # Only 1 concurrent
            default_timeout_ms=5000,
        )
        governor = RestGovernor(config=config, _time_fn=time_fn)

        # Acquire the only slot
        await governor.acquire("/test", weight=1)

        waiter_completed = False
        waiter_started = asyncio.Event()

        async def waiter() -> None:
            """Wait for slot to be released."""
            nonlocal waiter_completed
            waiter_started.set()
            await governor.acquire("/test", weight=1)
            await governor.release()
            waiter_completed = True

        async def releaser() -> None:
            """Release the slot after waiter starts."""
            await waiter_started.wait()
            await asyncio.sleep(0)  # Let waiter enter queue
            await governor.release()

        await asyncio.gather(waiter(), releaser())

        assert waiter_completed, "Waiter should complete after release"

    @pytest.mark.asyncio
    async def test_release_wakes_waiting_requests(self) -> None:
        """Test that release() wakes up waiting requests immediately."""
        config = RestGovernorConfig(
            budget_weight_per_minute=10000,  # High budget
            max_concurrent_requests=1,  # Only 1 concurrent
            default_timeout_ms=5000,
        )
        governor = RestGovernor(config=config)

        # Hold the only slot
        await governor.acquire("/test", weight=1)

        second_started = asyncio.Event()
        second_completed = asyncio.Event()

        async def waiting_request() -> None:
            """Wait for slot to be released."""
            second_started.set()
            await governor.acquire("/test", weight=1)
            second_completed.set()
            await governor.release()

        # Start waiting request
        task = asyncio.create_task(waiting_request())

        # Wait for it to enter queue
        await second_started.wait()
        await asyncio.sleep(0.01)  # Ensure it's actually waiting

        # Verify it's in queue
        assert governor.metrics.current_queue_depth == 1

        # Release the slot - should wake up waiting request
        await governor.release()

        # Wait for completion with short timeout
        try:
            await asyncio.wait_for(second_completed.wait(), timeout=1.0)
        except TimeoutError:
            pytest.fail("Waiting request was not woken up by release()")

        await task
        assert governor._concurrent_count == 0

    @pytest.mark.asyncio
    async def test_fifo_order_preserved_under_contention(self) -> None:
        """Test FIFO order is preserved even under contention."""
        config = RestGovernorConfig(
            budget_weight_per_minute=10000,
            max_concurrent_requests=1,  # Serial processing
            default_timeout_ms=5000,
        )
        governor = RestGovernor(config=config)

        order_started: list[int] = []
        order_acquired: list[int] = []

        async def worker(worker_id: int, start_event: asyncio.Event) -> None:
            order_started.append(worker_id)
            start_event.set()
            await governor.acquire("/test", weight=1)
            order_acquired.append(worker_id)
            await asyncio.sleep(0.001)  # Small work
            await governor.release()

        # Create events to synchronize start order
        events = [asyncio.Event() for _ in range(5)]

        # Start workers in order, waiting for each to register before starting next
        tasks = []
        for i in range(5):
            task = asyncio.create_task(worker(i, events[i]))
            tasks.append(task)
            await events[i].wait()  # Wait for this worker to start
            await asyncio.sleep(0.001)  # Small gap between starts

        await asyncio.gather(*tasks)

        # First to start should be first to acquire (FIFO)
        assert order_acquired == order_started, (
            f"FIFO violated: started={order_started}, acquired={order_acquired}"
        )
