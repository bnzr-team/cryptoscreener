# Drift Monitoring

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Signals
- Feature drift: PSI, KS test vs baseline
- Prediction drift: mean/variance changes, calibration drift
- Outcome drift: alert precision changes (if you log outcomes)

## Actions
- WARN: increase logging, mark model as “suspect”
- CRITICAL: switch to baseline heuristic mode and alert operator
