# Risk Register

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


| ID | Risk | Likelihood | Impact | Detection | Mitigation |
|---|---|---:|---:|---|---|
| R1 | Binance ban (418) due to limiter abuse | M | H | 429/418 counters, reconnect storms | WS-first, governor, backoff, circuit breaker |
| R2 | Reconnect storm during volatility | H | H | reconnect rate, drop rate | jitter, max reconnect/min, sharding |
| R3 | Feature/label mismatch online vs offline | M | H | replay tests diff | single feature lib, golden fixtures |
| R4 | Model miscalibration under drift | H | M | ECE/Brier drift | rolling recalibration, regime segmentation |
| R5 | LLM hallucination | M | M | schema validator, “no new numbers” check | deterministic reasons, strict outputs |
| R6 | Top list flicker (unusable UX) | H | M | churn metric | hysteresis, dwell time |
