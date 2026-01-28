# STATE

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM
**Updated:** 2026-01-28

---

## Current status

- **Phase:** Pre-live (all offline components built, CI hardened, observability ready)
- **Test count:** 1047+ passing (ruff ✓, mypy ✓, pytest ✓)
- **Next milestone:** DEC-033 — TBD

### Done — Infrastructure & CI (PR#43–90)

- [x] Acceptance packet automation (DEC-008/009): `acceptance_packet.sh`, `proof_guard.yml`, CI ARTIFACT mode
- [x] Stacked PR detection, merge safety, verbatim-only proof policy
- [x] CODEOWNERS + branch protection: 7 required checks, `enforce_admins: true` (PR#88)
- [x] ML/Artifacts Hardening CI gates (PR#90):
  - `checksum-guard`: SHA256 verification of fixture manifests + untracked file detection
  - `replay-determinism`: double-run digest comparison
  - `contracts-roundtrip`: roundtrip tests on contract changes

### Done — Binance Operational Safety (DEC-023a–d, PR#62–75)

- [x] Exponential backoff with jitter, CircuitBreaker (CLOSED→OPEN→HALF_OPEN)
- [x] RestGovernor: rate limiting, queue budgeting, concurrency cap
- [x] Wired into BinanceRestClient, ShardManager, StreamManager
- [x] Deterministic CircuitBreaker proofing (DEC-023c)
- [x] Async resource cleanup for aiohttp sessions (PR#75)

### Done — Observability (DEC-024/025, PR#76–89)

- [x] Structured logging with security filtering (DEC-024, PR#79)
- [x] Connector observability metrics: disconnects, reconnects, ping timeouts (PR#77)
- [x] 16 PromQL alert rules in `monitoring/alert_rules.yml` (PR#83)
- [x] `MetricsExporter`: 12 Prometheus metrics (6 Gauge + 6 Counter), low-cardinality (PR#84)
- [x] `GET /metrics` HTTP endpoint via aiohttp.web (PR#85)
- [x] E2E smoke test: exporter + endpoint runtime correctness (PR#89)
- [x] DEC-025-validation: `promtool` CI + forbidden label checks (PR#87)
- [x] Live runner metrics wiring (DEC-026): `MetricsExporter.update()` called every cadence tick
- [x] WS resilience validation & soak tests (DEC-027): fault injection, SoakSummary JSON, fake WS integration test (PR#93)

### Done — ML Pipeline (DEC-012–021, PR#54–71)

- [x] Label builder + cost model (DEC-010, PR#54)
- [x] Offline backtest harness with AUC/Brier/ECE/TopK (DEC-011, PR#55)
- [x] Training dataset split with anti-leakage (DEC-012)
- [x] Platt calibration (DEC-013)
- [x] MLRunner with calibration integration (DEC-014)
- [x] Baseline E2E determinism acceptance (DEC-015)
- [x] Artifact registry / manifest with SHA256 (DEC-016)
- [x] Production profile gates: DEV/PROD strictness (DEC-017)
- [x] Reason codes SSOT alignment (DEC-018)
- [x] MLRunner E2E acceptance as CI gate (DEC-019)
- [x] LLM timeout enforcement (DEC-020)
- [x] Model package E2E smoke (DEC-021)
- [x] Replay gate trigger expansion (DEC-022)

### Done — Foundations (PR#1–25)

- [x] Data contracts: MarketEvent, FeatureSnapshot, PredictionSnapshot, RankEvent, LLMExplainInput/Output
- [x] LLM guardrails: no-new-numbers, status-label validators, ExplainLLM + MockExplainer + AnthropicExplainer
- [x] Replay harness with determinism verification
- [x] Live pipeline scaffold (`scripts/run_live.py` — DEC-006)
- [x] Record→replay bridge (`scripts/run_record.py` — DEC-007)
- [x] Feature engine, scorer, ranker, alerter

### Done — Observability & Resilience (PR#91–94)

- [x] Backpressure, resource bounds, queue-growth acceptance (DEC-028, PR#94): bounded queues, drop-oldest policy, tick drift/RSS/queue depth instrumentation, 6 Prometheus metrics, runtime soak proof

### Done — Deployment Readiness (DEC-029)

- [x] `GET /healthz` endpoint with pipeline health JSON
- [x] Dockerfile (multi-stage, non-root, HEALTHCHECK)
- [x] docker-compose.yml (cryptoscreener + Prometheus)
- [x] CI docker smoke workflow
- [x] Ops runbook (`docs/RUNBOOK_DEPLOYMENT.md`)
- [x] 4 healthz endpoint tests

### Done — Production Readiness v1.5 (DEC-030)

- [x] `GET /readyz` endpoint (200 ready, 503 not ready)
- [x] Config validation (`__post_init__`): port/cadence/symbol/duration, fault flag gating
- [x] `--dry-run` + `--graceful-timeout-s` rollout knobs
- [x] Readiness transition + config validation tests
- [x] Runbook: readiness stuck, reconnect storm, backpressure checklists

### Done — Kubernetes Manifests MVP (DEC-031)

- [x] `k8s/deployment.yaml` with liveness/readiness probes, resource limits, security context
- [x] `k8s/service.yaml` ClusterIP on port 9090
- [x] `k8s/configmap.yaml` non-secret pipeline config
- [x] `k8s/secret.yaml` template (empty values, created out-of-band)
- [x] `k8s/kustomization.yaml` Kustomize entrypoint
- [x] `docs/RUNBOOK_K8S.md` quick start + troubleshooting

### Done — Nightly Soak Regression Gate (DEC-032)

- [x] `--ws-url` CLI flag for offline soak (overrides ConnectorConfig.base_ws_url)
- [x] `scripts/run_fake_soak.py` — ContinuousFakeWSServer + pipeline runner
- [x] `scripts/check_soak_thresholds.py` — threshold checker (baseline + overload)
- [x] `monitoring/soak_thresholds.yml` — threshold config
- [x] `.github/workflows/nightly_soak.yml` — nightly + dispatch CI workflow
- [x] `tests/test_check_soak_thresholds.py` — 15 threshold checker tests

### In Progress
- None

### Blocked
- None

## Known issues
- `run_live.py` uses stub/minimal pipeline logic — real ML inference path not yet wired
- No trained model artifact in registry (MLRunner falls back to BaselineRunner)
- No frontend/dashboard (RankEvents go to logs only)

## Artifact checksums (PR#1 fixture)
- `market_events.jsonl`: `ba7d6e2018426517893ac4de3052a145e72b88f20d82f8d864558fca99eea277`
- `expected_rank_events.jsonl`: `901a6cc399a2de563f55c1b3458edba8250b08a785978848ef890ca435e34335`
- RankEvent stream digest: `08f158e3d78b74e0b75318772bf0cd689859783de25c3b404ad501153efcd44d`
