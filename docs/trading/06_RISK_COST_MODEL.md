# Trading v2 — Risk & Cost Model (SSOT)

**Status:** Draft
**Date:** 2026-01-30
**DEC:** DEC-043
**Purpose:** Define SSOT assumptions for fees, slippage, latency, and risk constraints.

## Overview

This document specifies the cost and risk parameters that all trading strategies and policies MUST respect. These values form the basis for simulator configuration (`SimConfig`) and policy risk gates.

---

## Section 1: Fee Model

### Binance Futures Fee Structure

| Fee Type | Rate | Notes |
|----------|------|-------|
| Maker Fee | 0.0200% | Post-only limit orders |
| Taker Fee | 0.0500% | Market orders / crossing |
| VIP Tiers | Variable | Assume base tier for safety |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maker_fee_frac` | Decimal | 0.0002 | Maker fee as fraction |
| `taker_fee_frac` | Decimal | 0.0005 | Taker fee as fraction |
| `fee_rebate_frac` | Decimal | 0.0 | Any rebate (usually 0) |

### Fee Computation

```
fill_fee = fill_notional * fee_frac
where:
  fill_notional = fill_price * fill_qty
  fee_frac = maker_fee_frac (if maker) or taker_fee_frac (if taker)
```

### Constraints

- Strategies MUST assume maker fills for profitability calculations
- Simulator MUST track cumulative fees in `SimArtifacts.metrics`
- Kill switch loss includes realized fees

---

## Section 2: Slippage Model

### Slippage Assumptions

| Scenario | Expected Slippage | Notes |
|----------|-------------------|-------|
| Normal liquidity | 0-1 bps | Maker orders at BBO |
| High volatility | 1-5 bps | Wider spreads |
| Flash crash | 5-20+ bps | May not fill at limit |
| Unwind (aggressive) | 2-5 bps | Crossing spread |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slippage_base_bps` | Decimal | 1.0 | Base slippage estimate |
| `slippage_vol_mult` | Decimal | 2.0 | Multiplier for high vol |
| `slippage_unwind_bps` | Decimal | 3.0 | Slippage for aggressive close |

### Slippage in Simulator

The current simulator (DEC-041) uses **CROSS fill model**:
- BUY fills if trade_price <= bid_price (optimistic)
- SELL fills if trade_price >= ask_price (optimistic)

This is **optimistic** (no slippage on fills). Future fill models may add:
- Partial fills
- Price improvement / degradation
- Queue position modeling

### Constraints

- Profitability estimates MUST account for expected slippage
- Unwind orders assume `slippage_unwind_bps` degradation
- Kill switch close assumes worst-case slippage

---

## Section 3: Latency Model

### Latency Assumptions

| Component | Expected Latency | P99 Latency |
|-----------|------------------|-------------|
| WS book update | 10-50ms | 200ms |
| WS trade update | 10-50ms | 200ms |
| Order submission | 50-100ms | 500ms |
| Order fill report | 20-50ms | 200ms |
| Round trip (decision → fill) | 100-200ms | 1000ms |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_latency_ms` | int | 100 | Expected decision-to-fill |
| `max_latency_ms` | int | 500 | Max tolerable latency |
| `stale_quote_ms` | int | 1000 | Book data considered stale |
| `stale_trade_ms` | int | 5000 | Trade data considered stale |

### Latency in Simulator

Current simulator is **synchronous** (instant fills within same tick). Latency effects are modeled via:
- Stale data detection (POL-013, POL-014)
- WS gap fixtures (`ws_gap.jsonl`)
- Timestamp comparisons in `StrategyContext`

### Constraints

- Strategies MUST NOT assume instant fills
- Stale data detection is mandatory (see `05_ML_POLICY_LIBRARY.md` POL-013)
- Order updates rate-limited to prevent churn

---

## Section 4: Inventory Constraints

### Position Limits

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_position_qty` | Decimal | 0.01 | Hard position limit (BTC) |
| `inventory_soft_limit` | Decimal | 0.005 | Soft limit triggering unwind |
| `inventory_skew_start` | Decimal | 0.002 | Position triggering skew |

### Position Notional Limits

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_notional_usd` | Decimal | 5000 | Max position notional |
| `max_order_notional_usd` | Decimal | 1000 | Max single order notional |

### Constraints

- Position MUST NOT exceed `max_position_qty` (hard block, POL-012)
- Exceeding `inventory_soft_limit` triggers unwind mode (POL-010)
- All constraints apply to absolute position (long or short)

---

## Section 5: Loss Limits / Kill Switch

### Session Loss Limits

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_session_loss` | Decimal | 50.0 | Max loss before kill (USD) |
| `max_drawdown` | Decimal | 30.0 | Max drawdown from peak (USD) |
| `max_consecutive_losses` | int | 5 | Consecutive losing trades |

### Kill Switch Semantics

When any loss limit is breached:

1. **Immediate action:**
   - Cancel all outstanding orders
   - Close position at market (aggressive)
   - Log kill switch event with reason

2. **Session state:**
   - Set `SessionState.status = KILLED`
   - Record `kill_reason` in session state
   - No further trading allowed this session

3. **Recovery:**
   - Requires manual intervention or new session
   - Session cannot be "un-killed"

### Constraints

- Loss limits are **absolute** (no exceptions)
- Kill switch MUST fire within one tick of breach
- Position close attempts even if rate-limited (best effort)

---

## Section 6: Order Budget / Rate Limits

Per BINANCE_LIMITS.md §6:

### Rate Limit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_orders_per_second` | int | 10 | Orders per second |
| `max_orders_per_minute` | int | 100 | Orders per minute |
| `rate_limit_buffer` | int | 20 | Orders remaining to trigger throttle |
| `max_orders_per_window` | int | 50 | Anti-churn window limit |

### Order Budget Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order_window_ms` | int | 60000 | Window for order counting |
| `fill_cooldown_ms` | int | 500 | Pause after fill |
| `order_update_min_ms` | int | 100 | Min time between updates |

### Constraints

- Orders MUST NOT exceed Binance rate limits
- Approaching limits triggers throttling (POL-016)
- Excessive updates trigger churn prevention (POL-018)

---

## Section 7: Spread / Quote Constraints

### Spread Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spread_bps_min` | Decimal | 5.0 | Minimum spread |
| `spread_bps_default` | Decimal | 10.0 | Default spread |
| `spread_bps_max` | Decimal | 100.0 | Maximum spread |
| `toxic_spread_mult` | Decimal | 3.0 | Spread multiplier for toxicity |
| `high_vol_spread_mult` | Decimal | 2.0 | Spread multiplier for vol |

### Quote Price Constraints

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_tick_size` | Decimal | 0.01 | Minimum price increment |
| `max_price_deviation_bps` | Decimal | 500 | Max deviation from mid |

### Constraints

- Quotes MUST be at least `spread_bps_min` from mid
- Quotes MUST NOT exceed `max_price_deviation_bps` from mid
- Spread adjustments capped at `spread_bps_max`

---

## Section 8: Config vs Computed

### Static Configuration (defined at session start)

These values are set in config and do not change during session:

| Category | Parameters |
|----------|------------|
| Fees | `maker_fee_frac`, `taker_fee_frac` |
| Limits | `max_position_qty`, `max_session_loss`, `max_drawdown` |
| Rate limits | `max_orders_per_minute`, `rate_limit_buffer` |
| Base spread | `spread_bps_default`, `spread_bps_min`, `spread_bps_max` |
| Thresholds | All `*_threshold` parameters |

### Computed Online (updated each tick)

These values are computed from market data and state:

| Computed Value | Source |
|----------------|--------|
| Current spread | From `bid`, `ask` in `StrategyContext` |
| Position PnL | Computed from entry price and current mid |
| Inventory ratio | `position_qty / max_position_qty` |
| Orders remaining | From rate limit tracking |
| Staleness | `ts - last_book_ts`, `ts - last_trade_ts` |

### ML Model Outputs (external)

These come from ML models, not computed in policy engine:

| ML Output | Used By |
|-----------|---------|
| `p_inplay_*` | POL-001, POL-002 |
| `p_toxic` | POL-004, POL-005, POL-006 |
| `regime_vol` | POL-003 |
| `regime_trend` | POL-007, POL-009 |
| `flow_imbalance` | POL-004 |

---

## Section 9: Default Configuration Profile

### Profile: Conservative (Default)

```yaml
# Fees
maker_fee_frac: 0.0002
taker_fee_frac: 0.0005

# Position limits
max_position_qty: 0.01
inventory_soft_limit: 0.005
inventory_skew_start: 0.002

# Loss limits
max_session_loss: 50.0
max_drawdown: 30.0

# Spreads
spread_bps_default: 10.0
spread_bps_min: 5.0
spread_bps_max: 100.0

# Staleness
stale_quote_ms: 1000
stale_trade_ms: 5000

# Rate limits
max_orders_per_minute: 100
rate_limit_buffer: 20
fill_cooldown_ms: 500

# Thresholds (ML)
inplay_enter_prob: 0.6
inplay_exit_prob: 0.4
toxicity_widen_threshold: 0.3
toxicity_disable_threshold: 0.7
toxicity_exit_threshold: 0.2
trend_confidence_min: 0.6
```

---

## Version History

| Date | Change | DEC |
|------|--------|-----|
| 2026-01-30 | Initial risk/cost model | DEC-043 |
