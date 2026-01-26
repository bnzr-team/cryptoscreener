"""Tests for logging configuration module.

DEC-024: Verifies that logging configuration:
1. Filters out secrets and PII (BLOCKED_FIELDS)
2. Normalizes high-cardinality fields (URLs to endpoints)
3. Produces valid JSON output
4. Does not affect replay determinism
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from cryptoscreener.logging_config import (
    BLOCKED_FIELDS,
    JsonFormatter,
    SimpleFormatter,
    _filter_log_record,
    _normalize_url,
    _sanitize_text,
    get_logger,
    setup_logging,
)


class TestBlockedFields:
    """Test that sensitive fields are properly blocked."""

    def test_blocked_fields_not_empty(self) -> None:
        """BLOCKED_FIELDS should contain security-relevant fields."""
        assert len(BLOCKED_FIELDS) > 0
        assert "api_key" in BLOCKED_FIELDS
        assert "secret" in BLOCKED_FIELDS
        assert "token" in BLOCKED_FIELDS
        assert "password" in BLOCKED_FIELDS

    def test_filter_removes_api_key(self) -> None:
        """api_key field should be removed from logs."""
        record = {"api_key": "supersecret123", "msg": "test"}
        filtered = _filter_log_record(record)
        assert "api_key" not in filtered
        assert "msg" in filtered

    def test_filter_removes_secret(self) -> None:
        """secret field should be removed from logs."""
        record = {"secret": "abc123", "level": "INFO"}
        filtered = _filter_log_record(record)
        assert "secret" not in filtered
        assert "level" in filtered

    def test_filter_removes_partial_matches(self) -> None:
        """Fields containing blocked words should be removed."""
        record = {
            "x_api_key_header": "value",
            "user_token": "value",
            "my_password_field": "value",
            "safe_field": "keep",
        }
        filtered = _filter_log_record(record)
        assert "x_api_key_header" not in filtered
        assert "user_token" not in filtered
        assert "my_password_field" not in filtered
        assert "safe_field" in filtered

    def test_filter_case_insensitive(self) -> None:
        """Blocked field check should be case-insensitive."""
        record = {"API_KEY": "secret", "Token": "secret2"}
        filtered = _filter_log_record(record)
        assert "API_KEY" not in filtered
        assert "Token" not in filtered

    def test_pii_fields_blocked(self) -> None:
        """PII fields should be blocked."""
        record = {
            "ip": "192.168.1.1",
            "ip_address": "10.0.0.1",
            "email": "user@example.com",
            "phone": "+1234567890",
            "msg": "test",
        }
        filtered = _filter_log_record(record)
        assert "ip" not in filtered
        assert "ip_address" not in filtered
        assert "email" not in filtered
        assert "phone" not in filtered
        assert "msg" in filtered


class TestSanitizeText:
    """Tests for _sanitize_text function."""

    def test_url_query_string_removed(self) -> None:
        """URL query strings should be removed, path kept."""
        result = _sanitize_text("Request to https://api.example.com/v1/data?key=secret&token=abc")
        assert "key=secret" not in result
        assert "token=abc" not in result
        assert "/v1/data" in result

    def test_ip_address_redacted(self) -> None:
        """IP addresses should be replaced with [IP]."""
        result = _sanitize_text("Connection from 192.168.1.100 to 10.0.0.1")
        assert "192.168.1.100" not in result
        assert "10.0.0.1" not in result
        assert "[IP]" in result

    def test_email_redacted(self) -> None:
        """Email addresses should be replaced with [EMAIL]."""
        result = _sanitize_text("User user@example.com logged in")
        assert "user@example.com" not in result
        assert "[EMAIL]" in result

    def test_bearer_token_redacted(self) -> None:
        """Bearer tokens should be redacted."""
        result = _sanitize_text("Auth: bearer abc123xyz")
        assert "abc123xyz" not in result
        assert "[TOKEN]" in result

    def test_api_key_redacted(self) -> None:
        """API keys should be redacted."""
        result = _sanitize_text("Using api_key=sk-secret-12345")
        assert "sk-secret-12345" not in result
        assert "[API_KEY]" in result

    def test_empty_string_unchanged(self) -> None:
        """Empty strings should be returned unchanged."""
        assert _sanitize_text("") == ""

    def test_safe_text_unchanged(self) -> None:
        """Text without sensitive data should be unchanged."""
        text = "Normal log message without sensitive data"
        assert _sanitize_text(text) == text


class TestHighCardinalityFields:
    """Test normalization of high-cardinality fields."""

    def test_url_normalized_to_endpoint(self) -> None:
        """URL field should be normalized to endpoint (path only)."""
        record = {"url": "https://api.binance.com/fapi/v1/exchangeInfo?symbol=BTCUSDT"}
        filtered = _filter_log_record(record)
        assert "url" not in filtered
        assert "endpoint" in filtered
        assert filtered["endpoint"] == "/fapi/v1/exchangeInfo"

    def test_normalize_url_strips_query(self) -> None:
        """_normalize_url should strip query parameters."""
        result = _normalize_url("https://example.com/api/v1/data?key=value&foo=bar")
        assert result == "/api/v1/data"

    def test_normalize_url_preserves_path(self) -> None:
        """_normalize_url should preserve the path."""
        result = _normalize_url("https://example.com/deep/nested/path")
        assert result == "/deep/nested/path"

    def test_body_redacted(self) -> None:
        """body field should be redacted."""
        record = {"body": '{"sensitive": "data"}'}
        filtered = _filter_log_record(record)
        assert filtered["body"] == "[BODY]"

    def test_payload_redacted(self) -> None:
        """payload field should be redacted."""
        record = {"payload": {"key": "value"}}
        filtered = _filter_log_record(record)
        assert filtered["payload"] == "[PAYLOAD]"

    def test_symbols_redacted(self) -> None:
        """symbols field should be redacted (could be large list)."""
        record = {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}
        filtered = _filter_log_record(record)
        assert filtered["symbols"] == "[SYMBOLS_LIST]"


class TestFilterLogRecord:
    """Test the _filter_log_record function."""

    def test_safe_fields_preserved(self) -> None:
        """Safe fields should be preserved."""
        record = {
            "level": "INFO",
            "msg": "test message",
            "count": 42,
            "ratio": 0.5,
            "enabled": True,
        }
        filtered = _filter_log_record(record)
        assert filtered["level"] == "INFO"
        assert filtered["msg"] == "test message"
        assert filtered["count"] == 42
        assert filtered["ratio"] == 0.5
        assert filtered["enabled"] is True

    def test_list_capped_at_10(self) -> None:
        """Lists larger than 10 items should be summarized."""
        record = {"items": list(range(15))}
        filtered = _filter_log_record(record)
        assert filtered["items"] == "[list:15 items]"

    def test_small_list_preserved(self) -> None:
        """Lists with 10 or fewer items should be preserved."""
        record = {"items": [1, 2, 3]}
        filtered = _filter_log_record(record)
        assert filtered["items"] == [1, 2, 3]

    def test_nested_dict_filtered(self) -> None:
        """Nested dicts should have blocked fields removed."""
        record = {
            "config": {
                "endpoint": "/api",
                "api_key": "secret",  # Should be removed
                "timeout": 30,
            }
        }
        filtered = _filter_log_record(record)
        assert "endpoint" in filtered["config"]
        assert "timeout" in filtered["config"]
        assert "api_key" not in filtered["config"]


class TestJsonFormatter:
    """Test the JSON log formatter."""

    def test_produces_valid_json(self) -> None:
        """Output should be valid JSON."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "ts" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "msg" in parsed

    def test_contains_required_fields(self) -> None:
        """JSON output should contain required fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="mylogger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "mylogger"
        assert parsed["msg"] == "hello world"

    def test_warning_includes_location(self) -> None:
        """WARNING+ logs should include file/line info."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=42,
            msg="warning message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["file"] == "test.py"
        assert parsed["line"] == 42

    def test_extra_fields_included(self) -> None:
        """Extra fields from record should be included."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.shard_id = 5
        record.endpoint = "/api/v1"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["shard_id"] == 5
        assert parsed["endpoint"] == "/api/v1"

    def test_extra_fields_filtered(self) -> None:
        """Extra fields should be filtered for security."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.api_key = "secret123"  # Should be filtered
        record.safe_field = "keep"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "api_key" not in parsed
        assert parsed["safe_field"] == "keep"


class TestSimpleFormatter:
    """Test the simple human-readable formatter."""

    def test_basic_format(self) -> None:
        """Basic log format should be readable."""
        formatter = SimpleFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "test" in output
        assert "hello" in output

    def test_extra_fields_appended(self) -> None:
        """Extra fields should be appended to message."""
        formatter = SimpleFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="message",
            args=(),
            exc_info=None,
        )
        record.shard_id = 3
        output = formatter.format(record)
        assert "shard_id=3" in output


class TestSetupLogging:
    """Test the setup_logging function."""

    def test_setup_logging_json(self) -> None:
        """setup_logging with json_format=True should use JsonFormatter."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("test_json")
        logger.info("test message", extra={"key": "value"})

        output = stream.getvalue()
        # Should be valid JSON
        parsed = json.loads(output.strip())
        assert parsed["msg"] == "test message"
        assert parsed["key"] == "value"

    def test_setup_logging_simple(self) -> None:
        """setup_logging with json_format=False should use SimpleFormatter."""
        stream = io.StringIO()
        setup_logging(json_format=False, stream=stream)

        logger = get_logger("test_simple")
        logger.info("simple test")

        output = stream.getvalue()
        assert "simple test" in output
        # Should NOT be JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(output.strip())

    def test_setup_logging_level(self) -> None:
        """setup_logging should respect log level."""
        stream = io.StringIO()
        setup_logging(level=logging.WARNING, json_format=False, stream=stream)

        logger = get_logger("test_level")
        logger.info("info message")  # Should be filtered
        logger.warning("warning message")  # Should appear

        output = stream.getvalue()
        assert "info message" not in output
        assert "warning message" in output


class TestSecurityCompliance:
    """Integration tests for security compliance (DEC-024 condition 1)."""

    def test_no_secrets_in_json_output(self) -> None:
        """Secrets should never appear in JSON output."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("security_test")
        logger.info(
            "API call",
            extra={
                "api_key": "sk-secret-key-12345",
                "token": "bearer-token-xyz",
                "authorization": "Basic abc123",
                "endpoint": "/api/v1/data",
            },
        )

        output = stream.getvalue()
        assert "sk-secret-key-12345" not in output
        assert "bearer-token-xyz" not in output
        assert "Basic abc123" not in output
        # Safe field should be present
        assert "/api/v1/data" in output

    def test_no_pii_in_json_output(self) -> None:
        """PII should never appear in JSON output."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("pii_test")
        logger.info(
            "Request received",
            extra={
                "ip": "192.168.1.100",
                "user_agent": "Mozilla/5.0...",
                "email": "user@example.com",
                "method": "GET",
            },
        )

        output = stream.getvalue()
        assert "192.168.1.100" not in output
        assert "Mozilla/5.0" not in output
        assert "user@example.com" not in output
        # Safe field should be present
        assert "GET" in output


class TestCardinalityCompliance:
    """Integration tests for cardinality compliance (DEC-024 condition 2)."""

    def test_url_normalized_in_output(self) -> None:
        """URLs should be normalized to endpoints in output."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("cardinality_test")
        logger.info(
            "HTTP request",
            extra={
                "url": "https://api.binance.com/fapi/v1/ticker/24hr?symbols=[BTCUSDT,ETHUSDT]",
            },
        )

        output = stream.getvalue()
        parsed = json.loads(output.strip())
        assert "url" not in parsed
        assert parsed["endpoint"] == "/fapi/v1/ticker/24hr"

    def test_large_payload_redacted(self) -> None:
        """Large payloads should be redacted to prevent log bloat."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("payload_test")
        large_payload = {"data": list(range(1000))}
        logger.info("Response received", extra={"payload": large_payload})

        output = stream.getvalue()
        parsed = json.loads(output.strip())
        assert parsed["payload"] == "[PAYLOAD]"


class TestMsgSanitization:
    """Tests for msg sanitization (DEC-024 critical fix)."""

    def test_url_in_msg_normalized(self) -> None:
        """URLs in msg should have query strings removed."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("msg_sanitize_test")
        logger.info("GET %s", "https://api.example.com/path?secret=abc123&token=xyz")

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        # Query string should be removed
        assert "secret=abc123" not in parsed["msg"]
        assert "token=xyz" not in parsed["msg"]
        # Path should remain
        assert "/path" in parsed["msg"]

    def test_ip_in_msg_redacted(self) -> None:
        """IP addresses in msg should be redacted."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("ip_test")
        logger.info("Connection from 192.168.1.100 failed")

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        assert "192.168.1.100" not in parsed["msg"]
        assert "[IP]" in parsed["msg"]

    def test_token_in_msg_redacted(self) -> None:
        """Tokens in msg should be redacted."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("token_test")
        logger.info("Auth failed: bearer abc123xyz")

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        assert "abc123xyz" not in parsed["msg"]
        assert "[TOKEN]" in parsed["msg"]


class TestExcSanitization:
    """Tests for exception sanitization (DEC-024 critical fix)."""

    def test_url_in_exception_sanitized(self) -> None:
        """URLs in exception messages should be sanitized."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("exc_test")
        try:
            raise ValueError("Failed to fetch https://api.example.com/data?api_key=secret123")
        except ValueError:
            logger.exception("Request failed")

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        # Query string should be removed from exception
        assert "api_key=secret123" not in parsed.get("exc", "")
        # Path should remain
        assert "/data" in parsed.get("exc", "")

    def test_ip_in_exception_sanitized(self) -> None:
        """IP addresses in exceptions should be redacted."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("exc_ip_test")
        try:
            raise ConnectionError("Cannot connect to 10.0.0.1:8080")
        except ConnectionError:
            logger.exception("Connection error")

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        assert "10.0.0.1" not in parsed.get("exc", "")
        assert "[IP]" in parsed.get("exc", "")


class TestNestedDictFiltering:
    """Tests for recursive dict filtering (DEC-024 fix)."""

    def test_nested_dict_blocked_fields_removed(self) -> None:
        """Blocked fields in nested dicts should be removed."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("nested_test")
        logger.info(
            "Request",
            extra={
                "request": {
                    "Authorization": "Bearer secret123",
                    "method": "GET",
                    "url": "https://api.example.com/data?key=value",
                }
            },
        )

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        # Authorization should be filtered (blocked field)
        assert "Authorization" not in parsed.get("request", {})
        assert "secret123" not in output
        # Safe fields should remain
        assert parsed["request"]["method"] == "GET"
        # URL should be normalized (query removed)
        assert "key=value" not in parsed["request"].get("url", "")

    def test_partial_match_in_nested_keys(self) -> None:
        """Partial matches of blocked words in nested keys should be filtered."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("partial_nested_test")
        logger.info(
            "Config",
            extra={
                "headers": {
                    "x_api_key_header": "secret_value",
                    "content_type": "application/json",
                }
            },
        )

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        # x_api_key_header contains "api_key" - should be filtered
        assert "x_api_key_header" not in parsed.get("headers", {})
        assert "secret_value" not in output
        # Safe field should remain
        assert parsed["headers"]["content_type"] == "application/json"

    def test_deeply_nested_filtering(self) -> None:
        """Deeply nested structures should be filtered up to depth limit."""
        stream = io.StringIO()
        setup_logging(json_format=True, stream=stream)

        logger = get_logger("deep_nested_test")
        logger.info(
            "Deep",
            extra={
                "level1": {
                    "level2": {
                        "token": "should_be_filtered",
                        "safe": "keep",
                    }
                }
            },
        )

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        # Token at level2 should be filtered
        assert "should_be_filtered" not in output
        # Safe field should remain
        assert parsed["level1"]["level2"]["safe"] == "keep"
