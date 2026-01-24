# Modules Overview

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Responsibilities (single owner per concern)
- connectors/binance: connectivity, normalize raw messages → MarketEvent
- stream_router: sharding, routing, health
- features: ring buffers, feature computation, emits FeatureSnapshot
- models: model loading + inference + calibration
- scoring: costs, gates, profile combine, reasons
- ranker: top-K, hysteresis, emits RankEvent deltas
- explain_llm: text only, schema validation
- notifiers: Telegram (and optional UI feed)
- storage: recorder + parquet + metadata DB
- observability: metrics/logging; dashboards configs

## Anti-patterns (forbidden)
- Features computed ad-hoc in multiple places
- LLM allowed to alter numbers or ranking
- REST polling loops for live data
