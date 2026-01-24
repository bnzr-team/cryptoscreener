# Event Bus Spec

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## In-process (v1)
Use asyncio queues between stages:
- market_events_q
- feature_snapshots_q
- prediction_snapshots_q
- rank_events_q

## Backpressure
- Each queue has maxsize; on overflow:
  - drop lowest priority symbols first
  - log metric `dropped_events_total`

## Serialization
- Contracts are dataclasses with `.to_json()` producing the exact schema.
