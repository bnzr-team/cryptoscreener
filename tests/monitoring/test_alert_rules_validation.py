"""
Static validation of alert_rules.yml for DEC-025 cardinality and security constraints.

DEC-025-validation: Ensures alert rules never introduce forbidden labels or selectors
that would cause cardinality explosion or leak sensitive data.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

# Forbidden label keys — must never appear in alert rule `labels:` blocks.
# Aligned with FORBIDDEN_LABELS in exporter.py.
FORBIDDEN_LABEL_KEYS: frozenset[str] = frozenset(
    {
        "symbol",
        "endpoint",
        "path",
        "query",
        "ip",
        "request_id",
        "token",
        "user_id",
        "header",
    }
)

# Forbidden selector patterns — must never appear in `expr:` blocks.
# String search for `key=` or `key="` is sufficient.
FORBIDDEN_SELECTOR_PATTERNS: tuple[str, ...] = (
    "symbol=",
    "endpoint=",
    "path=",
    "query=",
    "ip=",
    "request_id=",
    "token=",
    "user_id=",
    "header=",
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
ALERT_RULES_PATH = PROJECT_ROOT / "monitoring" / "alert_rules.yml"


def _load_alert_rules() -> dict[str, Any]:
    """Load and parse alert_rules.yml."""
    content = ALERT_RULES_PATH.read_text()
    return yaml.safe_load(content)  # type: ignore[no-any-return]


def _extract_rules(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all rule dicts from alert_rules.yml structure."""
    rules = []
    for group in data.get("groups", []):
        for rule in group.get("rules", []):
            rules.append(rule)
    return rules


class TestForbiddenLabels:
    """DEC-025-validation: No forbidden label keys in alert rules."""

    def test_no_forbidden_labels_in_rules(self) -> None:
        """Alert rules must not use forbidden high-cardinality label keys."""
        if not ALERT_RULES_PATH.exists():
            pytest.skip(f"alert_rules.yml not found at {ALERT_RULES_PATH}")

        data = _load_alert_rules()
        rules = _extract_rules(data)
        assert rules, "No rules found in alert_rules.yml"

        violations: list[str] = []
        for rule in rules:
            alert_name = rule.get("alert", "<unnamed>")
            labels = rule.get("labels", {})
            for key in labels:
                if key in FORBIDDEN_LABEL_KEYS:
                    violations.append(f"{alert_name}: labels contains '{key}'")

        assert not violations, "Forbidden labels found in alert rules:\n" + "\n".join(
            f"  - {v}" for v in violations
        )

    def test_no_forbidden_selectors_in_expr(self) -> None:
        """Alert rule expressions must not contain forbidden selectors."""
        if not ALERT_RULES_PATH.exists():
            pytest.skip(f"alert_rules.yml not found at {ALERT_RULES_PATH}")

        data = _load_alert_rules()
        rules = _extract_rules(data)
        assert rules, "No rules found in alert_rules.yml"

        violations: list[str] = []
        for rule in rules:
            alert_name = rule.get("alert", "<unnamed>")
            expr = rule.get("expr", "")
            for pattern in FORBIDDEN_SELECTOR_PATTERNS:
                if pattern in expr:
                    violations.append(f"{alert_name}: expr contains '{pattern}'")

        assert not violations, "Forbidden selectors found in alert rule expressions:\n" + "\n".join(
            f"  - {v}" for v in violations
        )


class TestForbiddenLabelsNegative:
    """Negative tests: verify detection of forbidden labels/selectors."""

    def test_detects_forbidden_label_endpoint(self) -> None:
        """Rule with 'endpoint' label must be caught."""
        bad_yaml = dedent("""\
            groups:
              - name: test
                rules:
                  - alert: BAD_RULE
                    expr: up == 0
                    labels:
                      severity: warning
                      endpoint: /api/v1
        """)
        data = yaml.safe_load(bad_yaml)
        rules = _extract_rules(data)

        found_forbidden = False
        for rule in rules:
            for key in rule.get("labels", {}):
                if key in FORBIDDEN_LABEL_KEYS:
                    found_forbidden = True
                    break

        assert found_forbidden, "Should detect 'endpoint' in labels"

    def test_detects_forbidden_selector_symbol(self) -> None:
        """Rule with 'symbol=' in expr must be caught."""
        bad_yaml = dedent("""\
            groups:
              - name: test
                rules:
                  - alert: BAD_RULE
                    expr: 'metric{symbol="BTCUSDT"} > 100'
                    labels:
                      severity: critical
        """)
        data = yaml.safe_load(bad_yaml)
        rules = _extract_rules(data)

        found_forbidden = False
        for rule in rules:
            expr = rule.get("expr", "")
            for pattern in FORBIDDEN_SELECTOR_PATTERNS:
                if pattern in expr:
                    found_forbidden = True
                    break

        assert found_forbidden, "Should detect 'symbol=' in expr"

    def test_detects_forbidden_selector_endpoint_in_expr(self) -> None:
        """Rule with 'endpoint="/api"' in expr must be caught."""
        bad_yaml = dedent("""\
            groups:
              - name: test
                rules:
                  - alert: BAD_RULE
                    expr: 'rate(requests{endpoint="/api"}[5m]) > 10'
                    labels:
                      severity: warning
        """)
        data = yaml.safe_load(bad_yaml)
        rules = _extract_rules(data)

        found_forbidden = False
        for rule in rules:
            expr = rule.get("expr", "")
            for pattern in FORBIDDEN_SELECTOR_PATTERNS:
                if pattern in expr:
                    found_forbidden = True
                    break

        assert found_forbidden, "Should detect 'endpoint=' in expr"
