# Assumptions & Constraints

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Assumptions
- Target exchange: Binance USD‑M perps.
- Product is read-only (no order placement) in v1.
- Latency target assumes local compute near user, no heavy cloud hops.

## Constraints
- Binance WS per-connection constraints (10 msg/s, 1024 streams).
- Multiple limiter layers (Hard/ML/WAF).
- LLM may be disabled; system must function with deterministic explanations.

## What we will NOT assume
- That REST limits are stable forever.
- That probability outputs are automatically calibrated without explicit calibration pipeline.
