# Reconnect Strategy

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Principles
- Never reconnect aggressively; avoid storms.
- Prefer exponential backoff + full jitter.

## Backoff schedule (default)
- base=0.5s, cap=60s
- backoff = random(0, min(cap, base*2^attempt))

## Storm controls
- Max reconnect attempts per minute per process: 10
- Global “cooldown” if multiple connections drop simultaneously

## Steps on disconnect
1. Mark affected streams as stale immediately.
2. Emit DATA_ISSUE events for impacted symbols.
3. Start reconnect with backoff.
4. On reconnect: resubscribe in throttled batches.
5. If repeated: reduce universe (drop low priority) and/or reduce depth frequency.
