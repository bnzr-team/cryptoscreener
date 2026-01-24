# ML Overview

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Targets
- p_inplay_{30s,2m,5m}: probability of tradeability event after costs
- expected_utility_bps_{30s,2m,5m}: expected net edge conditional on being tradeable
- p_toxic: probability of adverse selection

## Model types (v1)
- GBDT (LightGBM/CatBoost) multi-head
- Calibrators per head (isotonic/Platt)
- Optional regime model

## Constraints
- Inference must be fast (<5ms per symbol batch on typical CPU)
- Deterministic feature parity with offline.
