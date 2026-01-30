# Trading v2 — Simulator Expectations (SSOT)

**Status:** Draft
**Date:** 2026-01-30
**DEC:** DEC-043
**Purpose:** Define acceptance criteria and KPI expectations for simulator runs.

## Overview

This document specifies what the trading system MUST achieve on each simulator fixture (DEC-041). These expectations form the acceptance criteria for policy implementations (DEC-044+).

---

## Section 1: Required Metrics

Every simulator run MUST produce these metrics in `SimArtifacts.metrics`:

### PnL Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `net_pnl` | Decimal | Net PnL after fees |
| `gross_pnl` | Decimal | PnL before fees |
| `realized_pnl` | Decimal | PnL from closed trades |
| `unrealized_pnl` | Decimal | PnL from open position at end |
| `total_fees` | Decimal | Cumulative fees paid |

### Trade Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `total_fills` | int | Number of fills |
| `buy_fills` | int | Number of buy fills |
| `sell_fills` | int | Number of sell fills |
| `round_trips` | int | Complete entry+exit cycles |
| `fill_rate` | Decimal | Fills / orders placed |

### Risk Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `max_drawdown` | Decimal | Largest peak-to-trough |
| `max_position` | Decimal | Largest absolute position |
| `time_in_position_pct` | Decimal | % of time with position |
| `kill_switch_triggered` | bool | Whether kill switch fired |

### Order Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `orders_placed` | int | Total orders submitted |
| `orders_cancelled` | int | Orders cancelled |
| `orders_filled` | int | Orders that filled |
| `order_update_rate` | Decimal | Updates per minute |

---

## Section 2: Fixture-Specific Expectations

### Fixture: `monotonic_up.jsonl`

**Market Behavior:** Steady upward price movement (trending market)

**Expected Policy Activation:**
- POL-002 (low in-play) — may trigger pause
- POL-007 (trend skew) — should apply UP bias
- POL-008 (inventory skew) — should balance
- POL-009 (counter-trend close) — should close shorts quickly
- POL-010 (soft limit unwind) — should prevent inventory buildup

**KPI Expectations:**

| Metric | Expectation | Rationale |
|--------|-------------|-----------|
| `net_pnl` | > -max_session_loss | Must not hit kill switch |
| `max_position` | <= max_position_qty | Hard limit respected |
| `round_trips` | >= 0 | May have few fills in trend |
| `kill_switch_triggered` | false | Should manage risk |

**Acceptance Criteria:**
- [ ] Position never exceeds `max_position_qty`
- [ ] Loss bounded (no runaway inventory against trend)
- [ ] Trend skew applied when trend detected
- [ ] Short positions closed quickly when trend is UP

---

### Fixture: `mean_reverting_range.jsonl`

**Market Behavior:** Price oscillates in a range (ideal for market making)

**Expected Policy Activation:**
- POL-001 (enable trading) — should be active
- POL-008 (inventory skew) — continuous balancing
- POL-010 (soft limit) — may trigger on extremes

**KPI Expectations:**

| Metric | Expectation | Rationale |
|--------|-------------|-----------|
| `net_pnl` | > 0 | Should be profitable |
| `round_trips` | >= 5 | Multiple complete cycles |
| `fill_rate` | > 0.3 | Reasonable fill rate |
| `max_drawdown` | < max_session_loss | Bounded risk |
| `kill_switch_triggered` | false | Should profit |

**Acceptance Criteria:**
- [ ] Positive net PnL
- [ ] At least 5 round trips completed
- [ ] Position stays balanced (oscillates around flat)
- [ ] Spread capture demonstrated

---

### Fixture: `flash_crash.jsonl`

**Market Behavior:** Sudden sharp price drop, then recovery

**Expected Policy Activation:**
- POL-003 (high vol) — widen spreads
- POL-004, POL-005 (toxicity) — may trigger
- POL-006 (toxic unwind) — reduce position
- POL-011 (PnL unwind) — may trigger on drawdown
- POL-019 (kill switch) — may trigger if severe

**KPI Expectations:**

| Metric | Expectation | Rationale |
|--------|-------------|-----------|
| `net_pnl` | > -max_session_loss | Bounded loss |
| `max_drawdown` | Tracked | May be significant |
| `kill_switch_triggered` | Acceptable | May fire for protection |

**Acceptance Criteria:**
- [ ] Loss bounded by `max_session_loss`
- [ ] Kill switch fires if loss threshold breached
- [ ] Spreads widen during crash
- [ ] No fills during extreme moves (if toxic detected)
- [ ] Position reduced during high volatility

---

### Fixture: `ws_gap.jsonl`

**Market Behavior:** Simulates WebSocket disconnection / data gaps

**Expected Policy Activation:**
- POL-013 (stale book pause) — MUST trigger
- POL-014 (stale trade caution) — should trigger
- POL-015 (WS reconnect grace) — should trigger

**KPI Expectations:**

| Metric | Expectation | Rationale |
|--------|-------------|-----------|
| `orders_during_gap` | 0 | No orders during stale |
| `fills_during_gap` | 0 | No fills during stale |
| `pause_triggered` | true | Staleness detected |

**Acceptance Criteria:**
- [ ] No orders placed during data gaps
- [ ] Outstanding orders cancelled on gap detection
- [ ] Trading resumes only after fresh data
- [ ] No fills occur during stale periods
- [ ] Stale detection logged in decisions

---

## Section 3: Policy Regression Matrix

Which policies should be exercised by which fixtures:

| Policy | monotonic_up | mean_reverting | flash_crash | ws_gap |
|--------|:------------:|:--------------:|:-----------:|:------:|
| POL-001 | - | ✓ | - | - |
| POL-002 | ✓ | - | ✓ | - |
| POL-003 | - | - | ✓ | - |
| POL-004 | - | - | ✓ | - |
| POL-005 | - | - | ✓ | - |
| POL-006 | - | - | ✓ | - |
| POL-007 | ✓ | - | - | - |
| POL-008 | ✓ | ✓ | ✓ | ✓ |
| POL-009 | ✓ | - | - | - |
| POL-010 | ✓ | ✓ | ✓ | ✓ |
| POL-011 | - | - | ✓ | - |
| POL-012 | ✓ | ✓ | ✓ | ✓ |
| POL-013 | - | - | - | ✓ |
| POL-014 | - | - | - | ✓ |
| POL-015 | - | - | - | ✓ |
| POL-016 | - | ✓ | - | - |
| POL-017 | - | ✓ | - | - |
| POL-018 | - | ✓ | - | - |
| POL-019 | - | - | ✓ | - |
| POL-020 | - | - | ✓ | - |

**Legend:**
- ✓ = Policy should be exercised / verified on this fixture
- `-` = Policy not expected to trigger

---

## Section 4: Acceptance Test Structure

### Test Categories

| Category | Purpose | Example |
|----------|---------|---------|
| Determinism | Same input → same output | Two runs, compare SHA256 |
| KPI | Metrics meet expectations | net_pnl > 0 for mean_reverting |
| Safety | Risk limits respected | max_position <= limit |
| Policy | Correct policies activated | POL-013 triggers on ws_gap |

### Test Naming Convention

```
test_{fixture}_{category}_{specific}

Examples:
- test_mean_reverting_kpi_positive_pnl
- test_flash_crash_safety_kill_switch
- test_ws_gap_policy_stale_pause
- test_monotonic_up_determinism_sha256
```

### Expected Test Count per Fixture

| Fixture | Determinism | KPI | Safety | Policy | Total |
|---------|:-----------:|:---:|:------:|:------:|:-----:|
| monotonic_up | 1 | 2 | 2 | 4 | 9 |
| mean_reverting | 1 | 4 | 2 | 6 | 13 |
| flash_crash | 1 | 2 | 3 | 5 | 11 |
| ws_gap | 1 | 1 | 2 | 3 | 7 |
| **Total** | 4 | 9 | 9 | 18 | **40** |

---

## Section 5: Future Fixture Proposals

These fixtures are NOT part of DEC-043 but are recommended for future implementation:

### Proposed: `high_frequency_fills.jsonl`

**Purpose:** Test rate limit compliance under heavy fill activity

**Expected Policies:** POL-016, POL-017, POL-018

### Proposed: `thin_liquidity.jsonl`

**Purpose:** Test behavior when book is thin / wide spreads

**Expected Policies:** Spread adjustments, fill rate expectations

### Proposed: `end_of_session.jsonl`

**Purpose:** Test session wind-down / unwind behavior

**Expected Policies:** Time-based unwind

### Proposed: `mixed_regime.jsonl`

**Purpose:** Test regime transitions within single session

**Expected Policies:** All regime-dependent policies

---

## Section 6: Determinism Requirements

### SHA256 Stability

For any fixture, two runs with identical:
- Fixture file
- Configuration
- Strategy parameters
- Random seed (if any)

MUST produce identical:
- `decisions_sha256`
- `artifacts_sha256`
- `combined_sha256`

### Determinism Test Pattern

```python
def test_determinism_{fixture}():
    events = load_fixture("{fixture}.jsonl")
    config = SimConfig(symbol="BTCUSDT", seed=42)
    strategy = PolicyStrategy(policy_config)

    runner1 = ScenarioRunner(config, strategy)
    result1 = runner1.run(events)

    runner2 = ScenarioRunner(config, strategy)
    result2 = runner2.run(events)

    assert result1.combined_sha256 == result2.combined_sha256
```

---

## Section 7: Reporting Requirements

### Scenario Report Structure

Each scenario run SHOULD produce a report containing:

```
=== Scenario Report ===
Fixture: {fixture_name}
Config: {config_summary}
Strategy: {strategy_name}

--- Metrics ---
Net PnL: {net_pnl}
Round Trips: {round_trips}
Max Position: {max_position}
Max Drawdown: {max_drawdown}
Kill Switch: {triggered/not_triggered}

--- Policy Activations ---
POL-001: {count} times
POL-002: {count} times
...

--- Decisions Summary ---
Total Decisions: {count}
Decisions with Orders: {count}
Unique Reasons: {list}

--- Determinism ---
Combined SHA256: {hash}
```

---

## Version History

| Date | Change | DEC |
|------|--------|-----|
| 2026-01-30 | Initial expectations | DEC-043 |
