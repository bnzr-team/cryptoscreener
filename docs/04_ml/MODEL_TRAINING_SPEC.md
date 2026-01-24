# Model Training Spec

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Training loop (v1)
1. Load dataset + split.
2. Train multi-head:
   - classification heads for p_inplay_{30s,2m,5m}
   - regression head for expected_utility_bps (or two-stage: classify then regress)
   - classification for p_toxic
3. Early stopping on validation.
4. Calibrate each probability head (CALIBRATION_SPEC).
5. Export artifacts:
   - model file(s)
   - calibrators
   - feature list + ordering
   - schema version

## Hyperparameters
- Must be configurable via yaml.
- Default baseline should be conservative (avoid overfitting).

## Tests
- Smoke training on tiny dataset fixture.
