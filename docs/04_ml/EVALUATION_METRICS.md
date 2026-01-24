# Evaluation Metrics

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Offline
- AUC, PR-AUC per horizon
- Brier, ECE
- Top-K capture: fraction of tradeable events contained in top-K
- Mean net_edge_bps in top-K
- Churn: rank changes per minute, state changes per minute

## Online
- Alert precision/recall
- Latency p95/p99
- Drop rate
- Drift: feature distribution shifts, calibration drift
