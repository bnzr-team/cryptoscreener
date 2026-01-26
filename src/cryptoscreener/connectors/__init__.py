"""Connectors for external data sources.

DEC-023: Added ReconnectLimiter and MessageThrottler for operational safety.
"""

from cryptoscreener.connectors.backoff import (
    BackoffConfig,
    BackoffState,
    CircuitBreaker,
    CircuitState,
    MessageThrottler,
    MessageThrottlerConfig,
    RateLimitError,
    RateLimitKind,
    ReconnectLimiter,
    ReconnectLimiterConfig,
    compute_backoff_delay,
    handle_error_response,
)

__all__ = [
    "BackoffConfig",
    "BackoffState",
    "CircuitBreaker",
    "CircuitState",
    "MessageThrottler",
    "MessageThrottlerConfig",
    "RateLimitError",
    "RateLimitKind",
    "ReconnectLimiter",
    "ReconnectLimiterConfig",
    "compute_backoff_delay",
    "handle_error_response",
]
