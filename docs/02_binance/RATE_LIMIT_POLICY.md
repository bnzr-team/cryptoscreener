# Rate Limit Policy (API Governor)

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Objective
Never trigger bans. Always stay below known limits with headroom.

## Global rules
1. Centralize all REST calls through one governor.
2. Maintain budgets per endpoint group (weight-based) + global rpm cap.
3. On `429`:
   - stop new REST calls for `cooldown_s`
   - backoff with jitter
4. On `418`:
   - immediate “safe mode”: disable REST for extended cooldown
   - reduce WS subscriptions if reconnect storms occur
5. Use WS for live data; REST only for:
   - exchangeInfo bootstrap
   - low-frequency universe refresh
   - occasional snapshots if required

## Implementation steps
- Implement `ApiGovernor` with:
  - token bucket per endpoint
  - moving window counters
  - circuit breaker states: CLOSED → OPEN → HALF_OPEN
- Provide unit tests for budget math and state transitions.
