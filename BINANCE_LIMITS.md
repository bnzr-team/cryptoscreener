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

## Sources (check for updates)
- Futures rate limits FAQ: https://www.binance.com/en/support/faq/detail/281596e222414cdd9051664ea621cdc3
- USD‑M Futures WebSocket market streams limits (10 msg/s, 1024 streams): https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams
- USD‑M Futures error codes (-1003 TOO_MANY_REQUESTS): https://developers.binance.com/docs/derivatives/usds-margined-futures/error-code
- 429/418 semantics (auto-ban behavior): https://binance-docs.github.io/apidocs/delivery_testnet/en/
- Binance API FAQ describing Hard/ML/WAF limits: https://www.binance.com/en/support/faq/detail/360004492232
