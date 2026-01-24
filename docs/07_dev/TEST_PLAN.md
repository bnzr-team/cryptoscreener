# Test Plan

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Unit tests
- feature computations (synthetic fixtures)
- cost model (synthetic books)
- scoring + gates
- hysteresis transitions
- LLM validator

## Integration tests
- WS message parsing → MarketEvent
- end-to-end in-process pipeline (with fake connector)

## Replay tests (critical)
- Record small fixture dataset (1–5 min)
- Replay must reproduce:
  - same FeatureSnapshot outputs
  - same RankEvent sequence (within tolerance)

## Performance tests
- 400 symbols simulated
- Ensure latency SLO and bounded memory
