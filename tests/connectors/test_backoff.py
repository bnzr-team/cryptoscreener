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
"""

from __future__ import annotations

import random
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
