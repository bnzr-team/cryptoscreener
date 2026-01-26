"""
Structured logging configuration for CryptoScreener.

DEC-024: Provides JSON-formatted structured logging with:
- Security filtering (no secrets/PII)
- Low-cardinality labels (normalized paths, no raw payloads)
- Determinism-safe (no runtime state that affects replay)

Usage:
    from cryptoscreener.logging_config import setup_logging, get_logger

    setup_logging()  # Call once at startup
    logger = get_logger(__name__)
    logger.info("message", extra={"key": "value"})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# DEC-024: Fields that should NEVER appear in logs (security)
BLOCKED_FIELDS: frozenset[str] = frozenset(
    {
        # API credentials
        "api_key",
        "secret",
        "token",
        "password",
        "auth",
        "authorization",
        "bearer",
        "credential",
        # PII
        "ip",
        "ip_address",
        "user_agent",
        "email",
        "phone",
        # Sensitive headers
        "x-mbx-apikey",
        "x-mbx-signature",
    }
)

# DEC-024: Fields that are high-cardinality and should be normalized
# Map of field_name -> normalizer function
HIGH_CARDINALITY_FIELDS: dict[str, str] = {
    "url": "endpoint",  # Extract path only
    "body": "[BODY]",  # Redact entirely
    "payload": "[PAYLOAD]",  # Redact entirely
    "symbols": "[SYMBOLS_LIST]",  # Don't dump raw lists
    "params": "[PARAMS]",  # Query params may contain sensitive data
}


def _normalize_url(url: str) -> str:
    """Extract normalized endpoint path from URL.

    DEC-024: Prevents high-cardinality labels from query strings.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return parts.path or "/"


def _filter_log_record(record: dict[str, Any]) -> dict[str, Any]:
    """Filter sensitive and high-cardinality fields from log record.

    DEC-024: Ensures logs are safe for production observability.
    """
    filtered: dict[str, Any] = {}

    for key, value in record.items():
        key_lower = key.lower()

        # Skip blocked fields
        if key_lower in BLOCKED_FIELDS:
            continue

        # Check if field name contains any blocked word
        if any(blocked in key_lower for blocked in BLOCKED_FIELDS):
            continue

        # Handle high-cardinality fields
        if key_lower in HIGH_CARDINALITY_FIELDS:
            if key_lower == "url" and isinstance(value, str):
                filtered["endpoint"] = _normalize_url(value)
            else:
                filtered[key] = HIGH_CARDINALITY_FIELDS[key_lower]
            continue

        # Pass through safe values
        if isinstance(value, (str, int, float, bool, type(None))):
            filtered[key] = value
        elif isinstance(value, (list, tuple)):
            # Cap list size to prevent huge log lines
            if len(value) <= 10:
                filtered[key] = list(value)
            else:
                filtered[key] = f"[list:{len(value)} items]"
        elif isinstance(value, dict):
            # Recursively filter nested dicts (shallow)
            filtered[key] = {k: v for k, v in value.items() if k.lower() not in BLOCKED_FIELDS}
        else:
            # Convert other types to string representation
            filtered[key] = str(value)

    return filtered


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    DEC-024: Produces one JSON object per line for log aggregation.

    Output format:
    {"ts":"2024-01-01T00:00:00.000Z","level":"INFO","logger":"module","msg":"...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base fields (always present)
        log_dict: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Add location for errors
        if record.levelno >= logging.WARNING:
            log_dict["file"] = record.filename
            log_dict["line"] = record.lineno

        # Add exception info if present
        if record.exc_info:
            log_dict["exc"] = self.formatException(record.exc_info)

        # Add filtered extra fields
        extra = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
                "taskName",
            }:
                extra[key] = value

        if extra:
            filtered_extra = _filter_log_record(extra)
            log_dict.update(filtered_extra)

        return json.dumps(log_dict, default=str, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """Simple formatter for development/testing.

    Human-readable output with filtered fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable string."""
        # Get base message
        base = f"{record.levelname:8s} {record.name}: {record.getMessage()}"

        # Collect extra fields
        extra = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
                "taskName",
            }:
                extra[key] = value

        if extra:
            filtered = _filter_log_record(extra)
            if filtered:
                extra_str = " ".join(f"{k}={v}" for k, v in filtered.items())
                base = f"{base} | {extra_str}"

        return base


def setup_logging(
    *,
    level: int | str = logging.INFO,
    json_format: bool = True,
    stream: Any = None,
) -> None:
    """Configure structured logging for the application.

    DEC-024: Call once at application startup.

    Args:
        level: Log level (default INFO).
        json_format: Use JSON formatter (default True for production).
        stream: Output stream (default stderr).
    """
    if stream is None:
        stream = sys.stderr

    # Create handler
    handler = logging.StreamHandler(stream)

    # Select formatter
    formatter = JsonFormatter() if json_format else SimpleFormatter()
    handler.setFormatter(formatter)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Convenience wrapper that ensures consistent logger naming.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
