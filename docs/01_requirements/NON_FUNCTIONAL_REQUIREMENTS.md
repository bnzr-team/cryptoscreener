# Non-Functional Requirements

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Latency
- p95 end-to-end < 250ms (ingest→features→inference→rank)
- p99 < 750ms under burst conditions (graceful degradation allowed)

## Availability
- 99%+ during run; degrade to DATA_ISSUE states on missing data

## Scale
- 400 symbols, multiple WS connections, 1024 streams per conn limit respected

## Determinism
- Offline/online feature parity; replay must reproduce same outputs (within tolerance)

## Safety
- No aggressive REST polling
- Circuit breaker on 429/418
