# STATE

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM
**Updated:** 2026-01-24

---

## Current status

- **Phase:** MVP (scaffold + contracts)
- **PR:** #1 — Repository scaffold, contracts, replay harness

### Done (PR#1)
- [x] Project structure: `pyproject.toml`, `src/cryptoscreener/`, `tests/`, `scripts/`
- [x] Data contracts: `MarketEvent`, `FeatureSnapshot`, `PredictionSnapshot`, `RankEvent`, `LLMExplainInput/Output`
- [x] JSON schema validation with Pydantic v2 (strict mode, frozen models)
- [x] LLM safety validators: no-new-numbers adversarial tests
- [x] Replay harness: `scripts/run_replay.py` with determinism verification
- [x] Test fixture: `tests/fixtures/sample_run/` (11 market events, 3 rank events)
- [x] 39 unit tests passing (contracts, roundtrip, LLM guardrails)
- [x] Tooling: ruff + mypy (strict) configured and passing

### Done (DEC-025: Prometheus Observability)
- [x] Alert thresholds & runbooks planning (PR#81)
- [x] WS storm metrics added to ShardMetrics/ConnectorMetrics (PR#82)
- [x] 16 PromQL alert rules in `monitoring/alert_rules.yml`, validated with `promtool` (PR#83)
- [x] `MetricsExporter` — 12 Prometheus metrics (6 Gauge + 6 Counter), low-cardinality only (PR#84)
- [x] `GET /metrics` HTTP endpoint via aiohttp.web, `--metrics-port` CLI flag (PR#85)
- [x] 1024+ tests passing (ruff ✓, mypy ✓, pytest ✓)

### In Progress
- None

### Blocked
- None

## Known issues
- Replay pipeline uses stub logic (deterministic but not ML-based)
- LLM module not yet implemented (contracts only)

## Artifact checksums (PR#1)
- `market_events.jsonl`: `ba7d6e2018426517893ac4de3052a145e72b88f20d82f8d864558fca99eea277`
- `expected_rank_events.jsonl`: `901a6cc399a2de563f55c1b3458edba8250b08a785978848ef890ca435e34335`
- RankEvent stream digest: `08f158e3d78b74e0b75318772bf0cd689859783de25c3b404ad501153efcd44d`
