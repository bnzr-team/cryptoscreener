# Trading v2 — Strategy Catalog (SSOT)

**Status:** Draft
**Date:** 2026-01-30
**DEC:** DEC-043
**Boundary:** Strategies operate via `StrategyDecision` / `StrategyOrder` outputs only.

## Purpose

This document defines the canonical set of **strategy modes** (composable building blocks) that the v2 trading system supports. Each mode describes:

- **Objective:** What it tries to achieve
- **When Used:** Market conditions / regime triggers
- **Decisions Emitted:** High-level behavior mapping to `StrategyDecision`
- **Must Not Do:** Hard constraints to prevent runaway behavior

Strategies are **composable**: a policy engine may combine multiple modes based on ML signals and regime detection.

---

## Mode Index

| Mode ID | Name | Primary Objective |
|---------|------|-------------------|
| `MODE_GRID` | Vol Harvesting Grid | Passive maker capture via spread |
| `MODE_SKEW` | Trend-Aware Skew | Bias quotes toward predicted direction |
| `MODE_UNWIND` | Inventory Unwind | Reduce position toward flat |
| `MODE_TOXIC_AVOID` | Toxic Flow Avoidance | Widen/disable quotes on toxic flow |
| `MODE_PAUSE` | Cooldown / Pause | Halt quoting temporarily |
| `MODE_KILL` | Kill Switch | Emergency position close + stop |

---

## MODE_GRID — Vol Harvesting Grid

### Objective
Capture spread by placing passive maker orders on both sides of the book, earning the bid-ask spread when both legs fill (round trip).

### When Used
- Market is **in-play** (sufficient volatility for fills)
- No toxic flow detected
- Position within inventory limits

### Decisions Emitted

| Condition | Action | Reason Code |
|-----------|--------|-------------|
| Flat position | Place BUY at `mid - spread_bps` | `enter_long` |
| Flat position | Place SELL at `mid + spread_bps` | `enter_short` |
| Long position | Place SELL to close at `mid + spread_bps` | `close_long` |
| Short position | Place BUY to close at `mid - spread_bps` | `close_short` |

### Configuration Parameters
- `spread_bps`: Spread in basis points from mid
- `order_qty`: Default order quantity
- `max_position_qty`: Maximum inventory cap

### Must Not Do
- Place market orders (maker only)
- Exceed `max_position_qty` in either direction
- Quote during stale market data (see `MODE_PAUSE`)
- Ignore pending orders (must respect `pending_order_count`)

---

## MODE_SKEW — Trend-Aware Skew

### Objective
Bias quote placement toward the predicted trend direction to:
- Increase fill probability on favorable side
- Reduce adverse selection from trending moves

### When Used
- `regime_trend` signal indicates directional bias
- Confidence above `trend_confidence_min`

### Decisions Emitted

| Condition | Action | Reason Code |
|-----------|--------|-------------|
| Trend UP + flat | Prioritize BUY side (tighter bid) | `skew_long_bias` |
| Trend DOWN + flat | Prioritize SELL side (tighter ask) | `skew_short_bias` |
| Trend UP + short | Aggressive close (tighter bid) | `trend_close_short` |
| Trend DOWN + long | Aggressive close (tighter ask) | `trend_close_long` |

### Configuration Parameters
- `trend_confidence_min`: Minimum trend confidence to enable skew
- `skew_bps_max`: Maximum additional bias in basis points
- `skew_aggression`: Multiplier for price improvement

### Must Not Do
- Cross the spread (no immediate fills)
- Apply skew when `regime_trend` is neutral/unknown
- Skew beyond `skew_bps_max` limit

---

## MODE_UNWIND — Inventory Unwind

### Objective
Reduce position size toward flat when:
- Inventory exceeds soft limits
- Risk signals indicate reduction needed
- Session approaching end or max loss

### When Used
- `abs(position_qty) > unwind_threshold_qty`
- `unrealized_pnl < unwind_pnl_trigger` (drawdown)
- Session time > `session_unwind_start_ms`

### Decisions Emitted

| Condition | Action | Reason Code |
|-----------|--------|-------------|
| Long > threshold | Place SELL at mid (aggressive) | `unwind_long` |
| Short > threshold | Place BUY at mid (aggressive) | `unwind_short` |
| PnL drawdown | Reduce position size | `risk_unwind` |

### Configuration Parameters
- `unwind_threshold_qty`: Position size triggering unwind
- `unwind_pnl_trigger`: PnL level triggering risk unwind
- `unwind_aggression`: Price improvement for faster fills
- `session_unwind_start_ms`: Time to start session wind-down

### Must Not Do
- Increase position while unwinding
- Place new entry orders
- Unwind faster than rate limits allow

---

## MODE_TOXIC_AVOID — Toxic Flow Avoidance

### Objective
Protect against informed flow by:
- Widening spreads when toxicity detected
- Disabling quotes entirely on high toxicity
- Reducing position exposure

### When Used
- `p_toxic >= toxicity_widen_threshold`
- `flow_imbalance` exceeds safe range
- Sudden price moves indicate informed trading

### Decisions Emitted

| Condition | Action | Reason Code |
|-----------|--------|-------------|
| `p_toxic` >= widen threshold | Widen spread by `toxic_spread_mult` | `toxic_widen` |
| `p_toxic` >= disable threshold | Cancel all quotes (NOOP) | `toxic_disable` |
| Toxicity + position | Aggressive unwind | `toxic_unwind` |

### Configuration Parameters
- `toxicity_widen_threshold`: p_toxic level to widen spreads
- `toxicity_disable_threshold`: p_toxic level to disable quoting
- `toxic_spread_mult`: Spread multiplier when toxic
- `toxic_cooldown_ms`: Minimum pause after toxic detection

### Must Not Do
- Continue normal quoting when toxicity detected
- Ignore toxicity signals from ML model
- Re-enable quotes before cooldown expires

---

## MODE_PAUSE — Cooldown / Pause

### Objective
Temporarily halt all quoting activity for safety:
- Stale market data (WS gap)
- Rate limit approaching
- Post-fill cooldown
- Manual intervention

### When Used
- `last_book_ts` older than `stale_quote_ms`
- Order count approaching rate limit
- After significant fills (anti-churn)
- External pause signal

### Decisions Emitted

| Condition | Action | Reason Code |
|-----------|--------|-------------|
| Stale data | Cancel outstanding, emit NOOP | `stale_pause` |
| Rate limit | Emit NOOP, reduce order rate | `rate_limit_pause` |
| Post-fill | Brief pause | `fill_cooldown` |

### Configuration Parameters
- `stale_quote_ms`: Max age of book data before pause
- `rate_limit_buffer`: Orders remaining before throttle
- `fill_cooldown_ms`: Pause duration after fill

### Must Not Do
- Place new orders during pause
- Ignore stale data warnings
- Resume without checking data freshness

---

## MODE_KILL — Kill Switch

### Objective
Emergency shutdown: close all positions and halt trading permanently for the session.

### When Used
- `realized_pnl + unrealized_pnl < -max_session_loss`
- Fatal error detected
- Manual kill signal

### Decisions Emitted

| Condition | Action | Reason Code |
|-----------|--------|-------------|
| Max loss breached | Cancel all, close position at market | `kill_max_loss` |
| Fatal error | Cancel all, close position | `kill_error` |
| Manual signal | Cancel all, close position | `kill_manual` |

### Configuration Parameters
- `max_session_loss`: Loss limit triggering kill switch
- `kill_close_aggression`: How aggressively to close position

### Must Not Do
- Resume trading after kill switch
- Place new orders after kill
- Delay position close for better price

---

## Mode Composition Matrix

Modes can be combined. Priority order (highest first):

| Priority | Mode | Overrides |
|----------|------|-----------|
| 1 | `MODE_KILL` | All others |
| 2 | `MODE_PAUSE` | Grid, Skew, Unwind |
| 3 | `MODE_TOXIC_AVOID` | Grid, Skew |
| 4 | `MODE_UNWIND` | Grid (entry only) |
| 5 | `MODE_SKEW` | Grid (quote placement) |
| 6 | `MODE_GRID` | Base behavior |

### Composition Rules

1. **Kill always wins:** If kill switch triggered, all other modes are ignored
2. **Pause blocks quoting:** During pause, no new orders placed
3. **Toxic modifies grid:** Toxic avoidance widens grid spreads or disables
4. **Unwind overrides entry:** When unwinding, no new entry orders
5. **Skew modifies grid:** Trend skew adjusts grid quote prices

---

## Mapping to StrategyDecision

All mode decisions MUST be expressible via `StrategyDecision` and `StrategyOrder`:

```
StrategyOrder:
  - side: OrderSide (BUY/SELL)
  - price: Decimal (computed from mode logic)
  - quantity: Decimal (from config or computed)
  - reason: str (mode's reason_code, no digits)
```

The policy engine evaluates modes in priority order and emits a single `StrategyDecision` per tick containing 0-N `StrategyOrder` entries.

---

## Version History

| Date | Change | DEC |
|------|--------|-----|
| 2026-01-30 | Initial catalog | DEC-043 |
