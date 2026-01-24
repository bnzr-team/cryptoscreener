# Calibration Spec

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Why
Raw ML probabilities are not guaranteed to be calibrated; we require calibration for trust.

## Steps
1. For each head p_inplay_H, fit calibrator on validation:
   - isotonic regression (default)
   - or Platt scaling
2. Evaluate:
   - Brier score
   - ECE/MCE
   - reliability diagram buckets
3. Persist calibrator artifact with version stamp.
4. Online: apply calibrator after model prediction.

## Drift handling
- If ECE exceeds threshold for N days: trigger retrain/recalibration.
