# Data Freshness Rules

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Thresholds (defaults)
- book stale: > 1000ms (for fast markets) → gate TRADEABLE
- trades stale: > 2000ms → downgrade to WATCH/DATA_ISSUE
- mark stale: > 5000ms → minor, but log

## Actions
- If book stale: status → DATA_ISSUE for symbol; score forced to 0
- If missing critical stream: DATA_ISSUE and remove from top list
- If partial degradation: allow WATCH but never TRADEABLE

## Tests
- Unit tests for staleness transitions
- Replay tests with simulated stalls
