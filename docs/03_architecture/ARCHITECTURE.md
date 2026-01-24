# Architecture — In‑Play Predictor (CryptoScreener‑X) — ML + LLM
**Version:** 1.0  
**Date:** 2026-01-24

---

## 1. High-level system diagram

**Binance WS/REST → Stream Router → Feature Engine → ML Inference → Ranker/Selector → (UI/Telegram/Storage/Monitoring)**

Key principle: **WebSockets first** for market data; REST only for bootstrap & slow-changing metadata.

---

## 2. Runtime pipeline (online)

### 2.1 Modules (services / packages)
1. `connectors/binance/`
   - WS market streams (combined), optional user-data
   - REST bootstrap: exchangeInfo, funding/OI snapshots, symbol list
   - Rate-limit middleware + backoff + circuit breaker

2. `stream_router/`
   - normalizes messages into canonical events
   - sharding across WS connections
   - “stream health” metrics (stale, drop, reconnect)

3. `features/`
   - rolling windows per symbol (ring buffers)
   - microstructure features + regime features
   - publishes `FeatureSnapshot` objects at fixed cadence (e.g. every 250ms or 1s)

4. `models/`
   - loads model artifacts (GBDT + calibrators + regime model)
   - runs fast inference over snapshots
   - produces `PredictionSnapshot` with reasons (top SHAP/feature attributions)

5. `scoring/`
   - cost model per execution profile
   - gates & penalties (toxicity, liquidity)
   - aggregation across horizons/profiles

6. `ranker/`
   - top‑K selection and hysteresis (anti‑flicker)
   - emits events: `SYMBOL_ENTER`, `SYMBOL_EXIT`, `ALERT`

7. `explain_llm/`
   - takes `PredictionSnapshot` + reason codes
   - returns short narrative & UX status label
   - strict: cannot modify numbers, only text

8. `notifiers/`
   - Telegram charting/alerts
   - optional web UI websocket feed

9. `storage/`
   - online: Redis / in-memory cache
   - offline: parquet (S3/local), plus metadata db (SQLite/Postgres)

10. `observability/`
   - structured logs
   - metrics: Prometheus compatible
   - dashboards + alert rules

---

## 3. Offline pipeline (training/backtest)

### 3.1 Stages
1. `ingest_raw/` — store WS market data (compressed) for replay
2. `build_features_offline/` — deterministic feature builder
3. `label_builder/` — compute move/cost/net_edge labels for both profiles & horizons
4. `train/` — train multi‑task model + calibrators
5. `validate/` — metrics + reliability + ablations
6. `package/` — model registry artifacts, versioned with git commit + hash
7. `replay_backtest/` — run full pipeline on recorded data to test end‑to‑end logic

### 3.2 Model versioning contract
- `model_version = semver + git_sha + data_cutoff + training_hash`
- Every inference log includes model_version

---

## 4. Data contracts (canonical)
See `DATA_CONTRACTS.md` in repo. The key online objects:
- `MarketEvent`
- `FeatureSnapshot`
- `PredictionSnapshot`
- `RankEvent`

---

## 5. Resilience patterns
- **Backpressure**: drop low-priority streams first (e.g., lower‑cap symbols) if message rate spikes.
- **Exponential backoff + jitter** on disconnect or 429/418.
- **Circuit breaker**: stop REST calls for cooldown after repeated 429.
- **Stale detection**: if book/trades stale → mark DATA_ISSUE and gate.
