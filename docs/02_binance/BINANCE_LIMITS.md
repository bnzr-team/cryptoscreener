# Binance Limits & Blockers — USD‑M Futures (Implementation Notes)
**Date:** 2026-01-24

> This file is a practical checklist. Always re-check official Binance docs periodically.

---

## 1) Futures REST rate limits (high level)

- Binance Futures FAQ states default **2,400 requests per minute per IP** and default **1,200 orders per minute** per account/sub‑account (tiers may adjust by volume).
- Exceeding request limits triggers throttling; continuing after receiving `429` may lead to an IP ban (`418`) depending on behavior described in Binance API docs.

**Implementation rules**
- Use WS for live updates; avoid REST polling loops.
- Maintain a per-endpoint weight budget; centralize rate limit accounting.
- On any `429`: immediate backoff; on repeated `429`: open circuit breaker; never “fight” the limiter.

---

## 2) USD‑M Futures WebSocket Market Streams limits

- **10 incoming messages per second per connection**.
- **Maximum 1024 streams per connection**.
- Over the limit → disconnected; repeated disconnects may lead to bans.

**Design**
- Use **combined streams** and shard symbols across multiple WS connections.
- Keep subscription count under ~70–80% of max to preserve headroom.
- Batch subscribe messages and throttle subscribe/unsubscribe operations.

---

## 3) Error codes and bans

- Futures error code `-1003 TOO_MANY_REQUESTS`: too many requests; may include ban-until timestamp in message.
- HTTP `418` is used for IP auto-bans after continuing to send requests post-`429` (documented).
- `5XX` and `503` may occur.

---

## 4) “ML” and WAF limits (important)

Binance states there are:
- Hard-limits
- Machine Learning limits
- WAF limits

Meaning: throttling can occur based on behavior patterns, not only raw weights.

**Mitigation**
- Randomized jitter in bootstraps
- WS-first approach
- Avoid reconnect storms
- Central “API governor” process with budgets and global cooldowns

---

## 5) Practical blockers for this project

1. **Reconnect storms during volatility** → risk of bans  
   Mitigation: exponential backoff, max reconnect rate, persistence of last good state.

2. **Too many streams** if you track all symbols with depth+trades+klines  
   Mitigation: dynamic universe selection + multiplex + fewer depth levels.

3. **REST misuse** for high-frequency data  
   Mitigation: only bootstrap snapshots; rely on WS diffs thereafter.

---

## 6) Trading Endpoints (v2 — VOL Harvesting)

> **Scope:** This section applies to Trading/VOL Harvesting v2 (see DEC-040).
> **Snapshot date:** 2026-01-29. MUST be verified against live Binance docs before Phase 1 implementation.

### 6.1 Endpoint Weights (Futures USDT-M)

| Endpoint | Method | Path | Weight | Notes |
|----------|--------|------|--------|-------|
| Place order | POST | `/fapi/v1/order` | 1 | Single limit/market order |
| Cancel order | DELETE | `/fapi/v1/order` | 1 | By orderId or clientOrderId |
| Cancel all | DELETE | `/fapi/v1/allOpenOrders` | 1 | Per symbol |
| Query order | GET | `/fapi/v1/order` | 1 | |
| Open orders | GET | `/fapi/v1/openOrders` | 1 (with symbol), 5 (all) | |
| Position risk | GET | `/fapi/v2/positionRisk` | 5 | |
| Account info | GET | `/fapi/v2/account` | 5 | |
| User trades | GET | `/fapi/v1/userTrades` | 5 | |
| Listen key | POST/PUT/DELETE | `/fapi/v1/listenKey` | 1 | Create/keepalive/close |

### 6.2 Order Rate Limits

| Limit type | Value | Window | Scope |
|------------|-------|--------|-------|
| Request weight | 2400 | 1 minute | Per IP |
| Order rate | 300 | 10 seconds | Per IP |
| Order rate | 1200 | 1 minute | Per IP |
| Order count | 200,000 | 24 hours | Per account |

### 6.3 OrderGovernor Budget Allocation (v2 guidance)

For vol harvesting use case (max 3 symbols × 6 levels):

```
Worst case full requote: 3 × 6 cancel + 3 × 6 place = 36 orders
With diff-based requote: typically 6-12 orders per cycle

Budget policy:
- Reserve 20% of order rate for KILL/EMERGENCY (always passthrough)
- Allocate 80% for normal operations
- Effective budget: 240 orders / 10s for normal ops → 24/s
- At worst case 36/requote → max 1 full requote per ~1.5s

Weight budget:
- 2400/min total → ~40/sec
- Trading orders: max 36 weight per full requote
- Leaves ~1600/min for queries (reconcile, position checks)
```

### 6.4 Trading-Specific Error Codes

| Code | Meaning | OrderGovernor Action | Bot Action |
|------|---------|---------------------|------------|
| 429 | Rate limit exceeded | FREEZE state, backoff from `Retry-After` | Queue pending (except KILL), log, alert |
| 418 | IP ban (repeated 429) | FREEZE indefinite | Alert, manual intervention, do NOT retry |
| -1003 | Too many requests | Same as 429 | Same as 429 |
| -2011 | Unknown order | No action (normal race) | Check if fill occurred |
| -1015 | Too many new orders | FREEZE 1min | Reduce requote frequency |
| -4131 | Reduce only rejected | No action | Position already flat, ignore |
| -1013 | Filter failure (tick/step) | No action | Refresh SymbolSpec → re-sanitize |
| -1111 | Precision too high | No action | Refresh SymbolSpec → adjust rounding |

### 6.5 User Data Stream (Trading)

- `listenKey` via REST: POST to create, PUT every 30min keepalive, DELETE to close
- Events: `ORDER_TRADE_UPDATE`, `ACCOUNT_UPDATE`
- Single WS connection, separate from market data streams
- Auto-reconnect with exponential backoff on disconnect

> **Verification requirement (Phase 1):**
> 1. Verify endpoints/weights/limits against current Binance Futures docs
> 2. Confirm via response headers (`X-MBX-USED-WEIGHT-*`, `X-MBX-ORDER-COUNT-*`, `Retry-After`)
> 3. Log used-weight headers from every REST response (audit trail)
> 4. Update this section with verified snapshot date

---

## Sources (check for updates)
- Futures rate limits FAQ: https://www.binance.com/en/support/faq/detail/281596e222414cdd9051664ea621cdc3
- USD‑M Futures WebSocket market streams limits (10 msg/s, 1024 streams): https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams
- USD‑M Futures error codes (-1003 TOO_MANY_REQUESTS): https://developers.binance.com/docs/derivatives/usds-margined-futures/error-code
- 429/418 semantics (auto-ban behavior): https://binance-docs.github.io/apidocs/delivery_testnet/en/
- Binance API FAQ describing Hard/ML/WAF limits: https://www.binance.com/en/support/faq/detail/360004492232
- Futures trading endpoints: https://developers.binance.com/docs/derivatives/usds-margined-futures/trade
