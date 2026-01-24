# Error Codes Playbook

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## When receiving REST errors
### 429 (rate limit)
1. Log structured event: endpoint, weight, headers, retry-after if any
2. ApiGovernor → OPEN circuit breaker for cooldown
3. Reduce REST to minimum (disable universe refresh)
4. Keep WS running

### 418 (IP ban)
1. Enter SAFE MODE immediately (disable REST, reduce WS load)
2. Record ban-until time if present
3. Alert operator/user

### -1003 TOO_MANY_REQUESTS
Treat as 429. Capture message; some include ban-until.

## WS disconnects
- If close code indicates policy violation: reduce streams and increase backoff.

## 5xx / 503
- Assume transient. Retry with backoff. Do not spam.
