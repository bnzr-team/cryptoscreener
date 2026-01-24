# SLO / SLI

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## SLIs
- Latency p95/p99 end-to-end
- WS drop rate
- Stale book rate
- 429/418 incidents
- Top list churn

## SLOs (v1 targets)
- Latency p95 < 250ms
- No 418 incidents in 30 days
- Data freshness: book stale < 1% of time

## Error budget
If error budget exhausted: freeze feature additions, focus reliability.
