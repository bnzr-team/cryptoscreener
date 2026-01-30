# Trading v2 — ML Policy Library (SSOT)

**Status:** Draft
**Date:** 2026-01-30
**DEC:** DEC-043
**Purpose:** Canonical library of policy rules mapping ML metrics → trading actions.

## Overview

This document defines the **single source of truth** for how ML model outputs and market regime signals translate into concrete trading actions expressed via `StrategyDecision`.

### Design Principles

1. **Config-First:** All numeric thresholds are named parameters (no magic numbers in prose)
2. **Deterministic:** Same inputs → same outputs (reproducible via `ScenarioRunner`)
3. **Auditable:** Every action has a `reason_code` (no digits per DEC-042)
4. **Composable:** Rules can be evaluated independently and combined by priority

### Rule Format Standard

Every rule follows this template:

```
Rule ID:        POL-XXX (stable identifier)
Intent:         One-line description
Inputs:         List of metrics/signals consumed
Preconditions:  Boolean conditions using parameter names
Action:         Mapping to StrategyDecision behavior
Hysteresis:     Enter/exit thresholds (named parameters)
Cooldown:       Time-based constraints (named parameters)
Risk Gates:     Ties to 06_RISK_COST_MODEL.md params
Sim Impact:     Which fixtures should demonstrate this rule
```

---

## Configuration Parameters (SSOT)

All rules reference these named parameters. Values are defined in config, not in rule text.

### In-Play / Volatility Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `inplay_enter_prob` | Decimal | p_inplay threshold to enable trading |
| `inplay_exit_prob` | Decimal | p_inplay threshold to disable trading |
| `inplay_horizon` | str | Which horizon: `2m`, `5m`, `30s` |
| `vol_regime_high_threshold` | Decimal | regime_vol value for "high vol" |
| `natr_min_bps` | Decimal | Minimum NATR for in-play |

### Toxicity Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `toxicity_widen_threshold` | Decimal | p_toxic to widen spreads |
| `toxicity_disable_threshold` | Decimal | p_toxic to disable quoting |
| `toxicity_exit_threshold` | Decimal | p_toxic to re-enable (hysteresis) |
| `toxic_spread_mult` | Decimal | Spread multiplier when toxic |
| `toxic_cooldown_ms` | int | Cooldown after toxic detection |

### Trend / Skew Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `trend_confidence_min` | Decimal | Minimum confidence for skew |
| `trend_skew_bps_max` | Decimal | Max skew adjustment |
| `trend_regime_threshold` | Decimal | regime_trend for directional bias |

### Inventory / Unwind Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `inventory_soft_limit` | Decimal | Position triggering unwind mode |
| `inventory_hard_limit` | Decimal | Maximum allowed position |
| `unwind_aggression` | Decimal | Price improvement for unwind |
| `unwind_pnl_trigger` | Decimal | PnL drawdown triggering unwind |

### Staleness / Safety Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `stale_quote_ms` | int | Max book age before pause |
| `stale_trade_ms` | int | Max trade age before concern |
| `rate_limit_buffer` | int | Orders remaining before throttle |
| `fill_cooldown_ms` | int | Pause after fill |

### Risk / Kill Switch Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_session_loss` | Decimal | Loss limit for kill switch |
| `max_drawdown` | Decimal | Drawdown limit |
| `kill_close_aggression` | Decimal | How fast to close on kill |

---

## Section 1: In-Play Detection & Volatility Regime

### POL-001: Enable Trading on In-Play

**Rule ID:** POL-001
**Intent:** Enable grid quoting when symbol becomes tradeable

**Inputs:**
- `p_inplay_{horizon}`: ML probability of in-play state
- `regime_vol`: Volatility regime classification
- `natr_14_5m`: Normalized ATR

**Preconditions:**
```
p_inplay_{inplay_horizon} >= inplay_enter_prob
AND regime_vol != "LOW"
AND natr_14_5m >= natr_min_bps
```

**Action:**
- Enable `MODE_GRID` quoting
- Emit orders per grid configuration

**Hysteresis:**
- Enter: `p_inplay >= inplay_enter_prob`
- Exit: `p_inplay < inplay_exit_prob`

**Cooldown:** None

**Risk Gates:** Must pass `max_session_loss` check

**Sim Impact:** `mean_reverting_range.jsonl` — should show trading activity

---

### POL-002: Disable Trading on Low In-Play

**Rule ID:** POL-002
**Intent:** Stop quoting when in-play probability drops

**Inputs:**
- `p_inplay_{horizon}`: ML probability

**Preconditions:**
```
p_inplay_{inplay_horizon} < inplay_exit_prob
```

**Action:**
- Disable new entry orders
- Keep exit orders for existing position
- Reason: `low_inplay_pause`

**Hysteresis:**
- Uses `inplay_exit_prob` (lower than enter for hysteresis)

**Cooldown:** `inplay_cooldown_ms` before re-evaluation

**Risk Gates:** None

**Sim Impact:** `monotonic_up.jsonl` — should limit exposure during trend

---

### POL-003: High Volatility Regime Adjustment

**Rule ID:** POL-003
**Intent:** Widen spreads in high volatility regime

**Inputs:**
- `regime_vol`: Volatility regime (LOW/NORMAL/HIGH)
- `natr_14_5m`: Normalized ATR

**Preconditions:**
```
regime_vol == "HIGH"
OR natr_14_5m > vol_regime_high_threshold
```

**Action:**
- Multiply `spread_bps` by `high_vol_spread_mult`
- Reason: `vol_spread_adjust`

**Hysteresis:**
- Enter: `regime_vol == HIGH`
- Exit: `regime_vol != HIGH` for `vol_exit_ticks` consecutive ticks

**Cooldown:** None

**Risk Gates:** None

**Sim Impact:** `flash_crash.jsonl` — should show wider spreads

---

## Section 2: Toxicity Avoidance & Spread Widening

### POL-004: Widen Spread on Toxicity

**Rule ID:** POL-004
**Intent:** Protect against informed flow by widening spreads

**Inputs:**
- `p_toxic`: ML toxicity probability
- `flow_imbalance`: Order flow imbalance metric

**Preconditions:**
```
p_toxic >= toxicity_widen_threshold
```

**Action:**
- Multiply `spread_bps` by `toxic_spread_mult`
- Reason: `toxic_widen`

**Hysteresis:**
- Enter: `p_toxic >= toxicity_widen_threshold`
- Exit: `p_toxic < toxicity_exit_threshold`

**Cooldown:** `toxic_cooldown_ms`

**Risk Gates:** None

**Sim Impact:** All fixtures — toxicity should trigger defensive behavior

---

### POL-005: Disable Quoting on High Toxicity

**Rule ID:** POL-005
**Intent:** Stop quoting entirely when toxicity is severe

**Inputs:**
- `p_toxic`: ML toxicity probability

**Preconditions:**
```
p_toxic >= toxicity_disable_threshold
```

**Action:**
- Cancel all outstanding orders
- Emit NOOP (no new orders)
- Reason: `toxic_disable`

**Hysteresis:**
- Enter: `p_toxic >= toxicity_disable_threshold`
- Exit: `p_toxic < toxicity_exit_threshold` AND cooldown elapsed

**Cooldown:** `toxic_cooldown_ms` (mandatory)

**Risk Gates:** None

**Sim Impact:** Should prevent fills during toxic periods

---

### POL-006: Toxic Flow Unwind

**Rule ID:** POL-006
**Intent:** Aggressively reduce position when toxicity + position

**Inputs:**
- `p_toxic`: ML toxicity probability
- `position_qty`: Current position
- `position_side`: LONG/SHORT/FLAT

**Preconditions:**
```
p_toxic >= toxicity_widen_threshold
AND position_side != FLAT
```

**Action:**
- Place aggressive close order at mid price
- Reason: `toxic_unwind`

**Hysteresis:** None (immediate action)

**Cooldown:** None (urgency overrides)

**Risk Gates:** Must respect `inventory_hard_limit`

**Sim Impact:** Should limit losses during toxic periods

---

## Section 3: Trend Regime Skew & Inventory Bias

### POL-007: Apply Trend Skew to Quotes

**Rule ID:** POL-007
**Intent:** Bias quotes toward predicted trend direction

**Inputs:**
- `regime_trend`: Trend direction (UP/DOWN/NEUTRAL)
- `trend_confidence`: Confidence in trend signal
- `microprice_proxy`: Microprice estimate (if available)

**Preconditions:**
```
regime_trend != "NEUTRAL"
AND trend_confidence >= trend_confidence_min
```

**Action:**
- If `regime_trend == UP`: Tighten bid by `trend_skew_bps_max`, widen ask
- If `regime_trend == DOWN`: Tighten ask by `trend_skew_bps_max`, widen bid
- Reason: `trend_skew_{direction}`

**Hysteresis:**
- Enter: `trend_confidence >= trend_confidence_min`
- Exit: `trend_confidence < trend_confidence_exit`

**Cooldown:** None

**Risk Gates:** Skew capped at `trend_skew_bps_max`

**Sim Impact:** `monotonic_up.jsonl` — should show directional bias

---

### POL-008: Inventory Skew

**Rule ID:** POL-008
**Intent:** Bias quotes to reduce inventory toward flat

**Inputs:**
- `position_qty`: Current position
- `position_side`: LONG/SHORT/FLAT

**Preconditions:**
```
abs(position_qty) > inventory_skew_start
```

**Action:**
- If LONG: Tighten ask (want to sell)
- If SHORT: Tighten bid (want to buy)
- Skew magnitude proportional to position size
- Reason: `inventory_skew`

**Hysteresis:** Linear scaling based on position size

**Cooldown:** None

**Risk Gates:** Position must stay within `inventory_hard_limit`

**Sim Impact:** All fixtures — should prevent inventory buildup

---

### POL-009: Counter-Trend Position Close

**Rule ID:** POL-009
**Intent:** Aggressively close position when trend is against us

**Inputs:**
- `regime_trend`: Trend direction
- `position_side`: Current position side
- `unrealized_pnl`: Current unrealized PnL

**Preconditions:**
```
(regime_trend == UP AND position_side == SHORT)
OR (regime_trend == DOWN AND position_side == LONG)
```

**Action:**
- Place aggressive close order
- Price: mid with `unwind_aggression` improvement
- Reason: `counter_trend_close`

**Hysteresis:** None (immediate)

**Cooldown:** None

**Risk Gates:** None

**Sim Impact:** `monotonic_up.jsonl` — should close shorts quickly

---

## Section 4: Unwind / Risk-Off Behaviors

### POL-010: Soft Limit Unwind

**Rule ID:** POL-010
**Intent:** Begin reducing position when soft limit exceeded

**Inputs:**
- `position_qty`: Current position
- `inventory_soft_limit`: Configuration parameter

**Preconditions:**
```
abs(position_qty) > inventory_soft_limit
```

**Action:**
- Disable new entry orders
- Place close orders only
- Reason: `soft_limit_unwind`

**Hysteresis:**
- Enter: `abs(position_qty) > inventory_soft_limit`
- Exit: `abs(position_qty) <= inventory_soft_limit * 0.8`

**Cooldown:** None

**Risk Gates:** Must respect `inventory_hard_limit`

**Sim Impact:** All fixtures — should prevent position growth

---

### POL-011: PnL Drawdown Unwind

**Rule ID:** POL-011
**Intent:** Reduce position on unrealized loss

**Inputs:**
- `unrealized_pnl`: Current unrealized PnL
- `unwind_pnl_trigger`: Configuration parameter

**Preconditions:**
```
unrealized_pnl < unwind_pnl_trigger
```

**Action:**
- Place aggressive close order
- Reason: `pnl_drawdown_unwind`

**Hysteresis:** None (immediate action on trigger)

**Cooldown:** `unwind_cooldown_ms`

**Risk Gates:** Ties to `max_drawdown`

**Sim Impact:** `flash_crash.jsonl` — should limit losses

---

### POL-012: Hard Limit Block

**Rule ID:** POL-012
**Intent:** Absolutely prevent position exceeding hard limit

**Inputs:**
- `position_qty`: Current position
- `inventory_hard_limit`: Configuration parameter

**Preconditions:**
```
abs(position_qty) >= inventory_hard_limit
```

**Action:**
- Cancel all entry orders immediately
- Only allow close orders
- Reason: `hard_limit_block`

**Hysteresis:** None (hard constraint)

**Cooldown:** None

**Risk Gates:** This IS the risk gate

**Sim Impact:** All fixtures — position never exceeds limit

---

## Section 5: Staleness / WS Gap Safety

### POL-013: Stale Book Pause

**Rule ID:** POL-013
**Intent:** Pause quoting when book data is stale

**Inputs:**
- `last_book_ts`: Timestamp of last book update
- `ts`: Current tick timestamp

**Preconditions:**
```
(ts - last_book_ts) > stale_quote_ms
```

**Action:**
- Cancel all outstanding orders
- Emit NOOP
- Reason: `stale_book_pause`

**Hysteresis:** None (immediate)

**Cooldown:** Resume only after fresh data

**Risk Gates:** None

**Sim Impact:** `ws_gap.jsonl` — should pause during gaps

---

### POL-014: Stale Trade Data Warning

**Rule ID:** POL-014
**Intent:** Reduce confidence when trade data is stale

**Inputs:**
- `last_trade_ts`: Timestamp of last trade
- `ts`: Current tick timestamp

**Preconditions:**
```
(ts - last_trade_ts) > stale_trade_ms
```

**Action:**
- Widen spreads by `stale_spread_mult`
- Reason: `stale_trade_caution`

**Hysteresis:**
- Enter: stale for `stale_trade_ms`
- Exit: fresh trade received

**Cooldown:** None

**Risk Gates:** None

**Sim Impact:** `ws_gap.jsonl` — should show wider spreads

---

### POL-015: WS Reconnect Grace

**Rule ID:** POL-015
**Intent:** Brief pause after WS reconnection

**Inputs:**
- `ws_reconnect_flag`: Boolean indicating recent reconnect

**Preconditions:**
```
ws_reconnect_flag == true
```

**Action:**
- Cancel existing orders
- Wait `ws_reconnect_grace_ms` before resuming
- Reason: `ws_reconnect_grace`

**Hysteresis:** None

**Cooldown:** `ws_reconnect_grace_ms`

**Risk Gates:** None

**Sim Impact:** Should prevent stale order fills

---

## Section 6: Order Budget / Rate-Limit Safety

### POL-016: Rate Limit Throttle

**Rule ID:** POL-016
**Intent:** Reduce order frequency approaching rate limit

**Inputs:**
- `orders_remaining`: Orders left in rate limit window
- `rate_limit_buffer`: Configuration parameter

**Preconditions:**
```
orders_remaining <= rate_limit_buffer
```

**Action:**
- Reduce order update frequency
- Skip non-essential order updates
- Reason: `rate_limit_throttle`

**Hysteresis:**
- Enter: `orders_remaining <= rate_limit_buffer`
- Exit: `orders_remaining > rate_limit_buffer * 2`

**Cooldown:** Window-based

**Risk Gates:** Per BINANCE_LIMITS.md §6

**Sim Impact:** Should prevent rate limit violations

---

### POL-017: Post-Fill Cooldown

**Rule ID:** POL-017
**Intent:** Brief pause after fill to prevent churn

**Inputs:**
- `last_fill_ts`: Timestamp of last fill
- `ts`: Current tick timestamp

**Preconditions:**
```
(ts - last_fill_ts) < fill_cooldown_ms
```

**Action:**
- Delay new order placement
- Reason: `fill_cooldown`

**Hysteresis:** None

**Cooldown:** `fill_cooldown_ms`

**Risk Gates:** None

**Sim Impact:** Should reduce order churn

---

### POL-018: Order Churn Detection

**Rule ID:** POL-018
**Intent:** Detect and prevent excessive order updates

**Inputs:**
- `orders_placed_window`: Orders placed in recent window
- `max_orders_per_window`: Configuration parameter

**Preconditions:**
```
orders_placed_window > max_orders_per_window
```

**Action:**
- Pause order updates
- Reason: `churn_prevention`

**Hysteresis:** Window-based

**Cooldown:** Until window resets

**Risk Gates:** Per BINANCE_LIMITS.md §6

**Sim Impact:** Should limit order frequency

---

## Section 7: Kill Switch / Emergency

### POL-019: Max Session Loss Kill

**Rule ID:** POL-019
**Intent:** Emergency shutdown on max loss breach

**Inputs:**
- `realized_pnl`: Session realized PnL
- `unrealized_pnl`: Current unrealized PnL
- `max_session_loss`: Configuration parameter

**Preconditions:**
```
(realized_pnl + unrealized_pnl) < -max_session_loss
```

**Action:**
- KILL_SWITCH activated
- Cancel all orders
- Close position at market
- Halt all trading for session
- Reason: `kill_max_loss`

**Hysteresis:** None (irreversible for session)

**Cooldown:** Permanent for session

**Risk Gates:** This IS the ultimate risk gate

**Sim Impact:** `flash_crash.jsonl` — should trigger and limit losses

---

### POL-020: Max Drawdown Kill

**Rule ID:** POL-020
**Intent:** Kill switch on max drawdown from peak

**Inputs:**
- `session_peak_pnl`: Highest PnL reached in session
- `realized_pnl + unrealized_pnl`: Current total PnL
- `max_drawdown`: Configuration parameter

**Preconditions:**
```
(session_peak_pnl - (realized_pnl + unrealized_pnl)) > max_drawdown
```

**Action:**
- KILL_SWITCH activated
- Same as POL-019
- Reason: `kill_max_drawdown`

**Hysteresis:** None (irreversible)

**Cooldown:** Permanent for session

**Risk Gates:** Ties to `max_drawdown`

**Sim Impact:** Should protect against runaway losses

---

## Rule Priority Matrix

Rules are evaluated in this order. Higher priority rules can override lower.

| Priority | Rules | Description |
|----------|-------|-------------|
| 1 | POL-019, POL-020 | Kill switch (absolute) |
| 2 | POL-012 | Hard inventory limit |
| 3 | POL-013, POL-015 | Staleness pause |
| 4 | POL-005 | Toxic disable |
| 5 | POL-016, POL-018 | Rate limit safety |
| 6 | POL-004, POL-006 | Toxic widen/unwind |
| 7 | POL-010, POL-011 | Soft limit/PnL unwind |
| 8 | POL-002 | Low in-play pause |
| 9 | POL-007, POL-008, POL-009 | Trend/inventory skew |
| 10 | POL-003, POL-014, POL-017 | Adjustments |
| 11 | POL-001 | Enable trading |

---

## Policy-to-Fixture Regression Matrix

| Policy | monotonic_up | mean_reverting | flash_crash | ws_gap |
|--------|--------------|----------------|-------------|--------|
| POL-001 | - | ✓ Active | - | - |
| POL-002 | ✓ Limit | - | ✓ Pause | - |
| POL-003 | - | - | ✓ Widen | - |
| POL-004 | - | - | ✓ Widen | - |
| POL-005 | - | - | ✓ Disable | - |
| POL-006 | - | - | ✓ Unwind | - |
| POL-007 | ✓ Skew | - | - | - |
| POL-008 | ✓ Balance | ✓ Balance | ✓ Balance | ✓ Balance |
| POL-009 | ✓ Close | - | - | - |
| POL-010 | ✓ Limit | ✓ Limit | ✓ Limit | ✓ Limit |
| POL-011 | - | - | ✓ Unwind | - |
| POL-012 | ✓ Block | ✓ Block | ✓ Block | ✓ Block |
| POL-013 | - | - | - | ✓ Pause |
| POL-014 | - | - | - | ✓ Caution |
| POL-015 | - | - | - | ✓ Grace |
| POL-019 | - | - | ✓ Kill | - |

---

## Version History

| Date | Change | DEC |
|------|--------|-----|
| 2026-01-30 | Initial library with 20 policies | DEC-043 |
