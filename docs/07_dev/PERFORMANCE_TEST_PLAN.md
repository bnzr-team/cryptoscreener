# Performance Test Plan

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Tools
- pytest-benchmark (optional)
- simple time/perf counters

## Scenarios
- steady-state 200 symbols
- burst 400 symbols with depth updates
- disconnect/reconnect storms (simulated)

## Pass/fail
- p95 < 250ms
- memory growth bounded
