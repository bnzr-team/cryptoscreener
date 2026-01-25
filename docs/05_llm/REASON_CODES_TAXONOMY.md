# Reason Codes Taxonomy

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Categories
### Liquidity/Costs
- RC_SPREAD_TIGHT / RC_SPREAD_WIDE
- RC_IMPACT_LOW / RC_IMPACT_HIGH
- RC_LIQUIDITY_OK / RC_LIQUIDITY_THIN

### Flow
- RC_FLOW_SURGE
- RC_FLOW_IMBALANCE_LONG / SHORT
- RC_TRADE_INTENSITY_UP

### Volatility/Regime
- RC_VOL_EXPANDING
- RC_REGIME_HIGH_VOL / LOW_VOL
- RC_TRENDING / CHOPPY

### Risk/Toxic
- RC_TOXIC_RISK_UP
- RC_TRAP_SIGNATURE

### Gates (Trading Blockers)
- RC_GATE_SPREAD_FAIL — spread exceeds max threshold
- RC_GATE_IMPACT_FAIL — impact exceeds max threshold

### Data Quality
- RC_DATA_STALE — stale market data detected

### ML/Calibration (MLRunner-specific)
- RC_CALIBRATION_ADJ — calibration adjustment applied to raw probability

### Artifact Errors (PROD mode only)
- RC_MODEL_UNAVAILABLE — model artifact missing or failed to load
- RC_CALIBRATION_MISSING — calibration artifact missing or failed to load
- RC_ARTIFACT_INTEGRITY_FAIL — artifact hash mismatch (SHA256 verification failed)

## Rule format
Each code must have:
- trigger condition (formula, thresholds)
- evidence fields to include
- severity score (0..1)
