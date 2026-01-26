"""Connectors for external data sources.

DEC-023: Added ReconnectLimiter and MessageThrottler for operational safety.
DEC-023d: Added RestGovernor for REST API budget/queue/concurrency control.
"""

from cryptoscreener.connectors.backoff import (
    BackoffConfig,
    BackoffState,
    CircuitBreaker,
    CircuitState,
    GovernorDroppedError,
    GovernorTimeoutError,
    MessageThrottler,
    MessageThrottlerConfig,
    RateLimitError,
    RateLimitKind,
    ReconnectLimiter,
    ReconnectLimiterConfig,
    RestGovernor,
    RestGovernorConfig,
    RestGovernorMetrics,
    compute_backoff_delay,
    handle_error_response,
)

__all__ = [
    "BackoffConfig",
    "BackoffState",
    "CircuitBreaker",
    "CircuitState",
    "GovernorDroppedError",
    "GovernorTimeoutError",
    "MessageThrottler",
    "MessageThrottlerConfig",
    "RateLimitError",
    "RateLimitKind",
    "ReconnectLimiter",
    "ReconnectLimiterConfig",
    "RestGovernor",
    "RestGovernorConfig",
    "RestGovernorMetrics",
    "compute_backoff_delay",
    "handle_error_response",
]
