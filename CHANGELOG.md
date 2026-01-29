# CHANGELOG

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM
**Updated:** 2026-01-29

---

## Unreleased

### Changed

#### Replay Pipeline Parity (DEC-037)
- `scripts/run_replay.py`: Replaced `MinimalReplayPipeline` stub with `ReplayPipeline` using real components (FeatureEngine → BaselineRunner → Ranker → Alerter)
- `tests/fixtures/sample_run/expected_rank_events.jsonl`: Regenerated for real pipeline output
- `tests/fixtures/sample_run/manifest.json`: Bumped to v2.0.0 with updated checksums and digest
- `tests/contracts/test_replay_determinism.py`: Updated fixture checksums

### Added

#### Trading/VOL Harvesting v2 Docs Scaffold (DEC-040)
- `docs/trading/` directory with v2 SSOT templates:
  - `DOCS_INDEX_TRADING.md`: v2 document index and SSOT rules
  - `01_SCOPE_BOUNDARY_SSOT.md`: v1/v2 boundary definition (RankEvent as only interface)
  - `TRADING_DECISIONS.md`: TRD-001 (boundary), TRD-002 (LLM limits), TRD-003 (simulator determinism)
  - `TRADING_SPEC.md`: Invariants, state machine, risk gates
  - `TRADING_STATE.md`: Milestones M0-M3, status tracking
  - `TRADING_CHANGELOG.md`: v2 changelog template
- `DECISIONS.md`: Added DEC-040 (v2 Scope & Boundary) with forbidden dependencies list
- `docs/00_product/PRD.md`: Updated non-goals to clarify v2 scope
- `docs/02_binance/BINANCE_LIMITS.md`: Added §6 Trading Endpoints (weights, limits, error codes)
- `docs/03_architecture/DATA_CONTRACTS.md`: Added §6 Trading Contracts pointer
- v2 Boundary: v2 consumes `RankEvent` as ONLY SSOT input from v1, no v1 modifications

#### RankEvent Delivery UX Pack (DEC-039)
- `src/cryptoscreener/delivery/` package: config, formatter, dedupe, router, sinks
- Three delivery sinks:
  - `sinks/telegram.py`: Telegram Bot API (sendMessage with HTML parse mode)
  - `sinks/slack.py`: Slack Incoming Webhooks (mrkdwn format)
  - `sinks/webhook.py`: Generic HTTP webhook (JSON payload with text/html/markdown variants)
- Anti-spam controls:
  - Per-symbol cooldown (default 120s, configurable via `--delivery-cooldown-s`)
  - Global rate limit (30 deliveries/minute)
  - Status transition filtering (only notify on status changes)
- Deterministic template formatting (`formatter.py`): no LLM required
- `scripts/run_live.py` integration:
  - `--delivery-telegram` / `--delivery-slack` / `--delivery-webhook` enable flags
  - `--delivery-dry-run` for testing without sending
  - `--delivery-cooldown-s N` for custom cooldown
- `docs/RUNBOOK_DELIVERY.md`: setup guide, K8s secrets, troubleshooting
- 58 new tests (`tests/delivery/`): formatter, dedupe, all sinks, router

#### Grafana Dashboards Pack (DEC-036)
- `monitoring/grafana/dashboards/cryptoscreener-overview.json`: 12-panel dashboard covering WS health (reconnects, disconnects, ping timeouts, subscribe delays), circuit breaker (transitions, OPEN duration), REST governor (queue depth, concurrency, request rates, saturation gauge)
- `monitoring/grafana/dashboards/cryptoscreener-backpressure.json`: 8-panel dashboard covering pipeline queue depths, event/snapshot drop rates, tick drift, process RSS
- `docs/RUNBOOK_GRAFANA.md`: Import guide (manual, provisioning, docker-compose), template variables, panel inventory, troubleshooting
- Template variables: `$datasource`, `$namespace`, `$pod`, `$job` with "All" defaults

#### Dual-mode Prometheus Scrape (DEC-035)
- `k8s/service.yaml`: Added `prometheus.io/scrape`, `prometheus.io/port`, `prometheus.io/path` annotations for plain Prometheus service discovery
- `monitoring/prometheus_scrape_k8s_example.yml`: Example scrape job with `kubernetes_sd_configs` (role: service), label filtering, and static target fallback
- `docs/RUNBOOK_K8S.md`: Extended with plain Prometheus setup path, troubleshooting for empty targets, port mismatch, 404, and relabeling issues

#### Secrets Strategy (DEC-034)
- `k8s/externalsecret.yaml`: ExternalSecret syncing 3 keys (`BINANCE_API_KEY`, `BINANCE_SECRET_KEY`, `ANTHROPIC_API_KEY`) from ESO backend into `cryptoscreener-secrets`
- `k8s/secretstore.yaml`: SecretStore template with AWS Secrets Manager, HashiCorp Vault, and Kubernetes dev backend examples
- `scripts/secret_guard.py`: CI scanner detecting AWS key patterns, long hex strings, and base64 env var assignments
- `.github/workflows/secret_guard.yml`: CI gate running secret guard on push and PR
- `docs/RUNBOOK_SECRETS.md`: Setup guide for ESO, manual secrets, CI guard, runtime redaction rules
- Kustomize updated with ESO resources (commented out by default for non-ESO clusters)
- 14 unit tests for secret guard (`tests/test_secret_guard.py`)

#### Prometheus Operator Integration (DEC-033)
- `k8s/servicemonitor.yaml`: ServiceMonitor for auto-discovery (15s scrape interval, `/metrics` path)
- `k8s/prometheusrule.yaml`: All 16 alert rules from `monitoring/alert_rules.yml` packaged as PrometheusRule CRD
- K8s label standardization: `app.kubernetes.io/part-of: cryptoscreener-x` added to Deployment, Service, and CRD metadata
- Kustomize updated to include ServiceMonitor + PrometheusRule (commentable for non-operator clusters)
- `docs/RUNBOOK_K8S.md`: Prometheus Operator setup, verification, and troubleshooting section

#### Nightly Soak Regression Gate (DEC-032)
- `--ws-url` CLI flag on `run_live.py` for offline soak with FakeWSServer
- `scripts/run_fake_soak.py`: ContinuousFakeWSServer + pipeline runner (no outbound network)
- `scripts/check_soak_thresholds.py`: threshold checker for baseline + overload soak summaries
- `monitoring/soak_thresholds.yml`: configurable thresholds (queue depth, RSS, tick drift, drops)
- `.github/workflows/nightly_soak.yml`: scheduled nightly + manual dispatch CI workflow
- 15 unit tests for threshold checking logic

#### Kubernetes Manifests MVP (DEC-031)
- `k8s/deployment.yaml`: Deployment with liveness (`/healthz`) and readiness (`/readyz`) probes, resource requests/limits, non-root security context, read-only root filesystem
- `k8s/service.yaml`: ClusterIP Service exposing metrics port 9090
- `k8s/configmap.yaml`: Non-secret pipeline configuration (TOP_N, CADENCE, etc.)
- `k8s/secret.yaml`: Template with empty values (real secrets created via `kubectl create secret`)
- `k8s/kustomization.yaml`: Kustomize entrypoint
- `docs/RUNBOOK_K8S.md`: Quick start, probe semantics, troubleshooting guide

#### Production Readiness v1.5 (DEC-030)
- `GET /readyz` endpoint: 200 when ready, 503 when not (WS connected + events fresh)
- Config validation (`__post_init__`): port/cadence/symbol/duration bounds, fault flag gating (`ALLOW_FAULTS=1`)
- `--dry-run` flag: validate config + start metrics server, exit without processing
- `--graceful-timeout-s` flag for shutdown timeout
- Readiness transition tests (503→200→503 on staleness)
- Config validation tests (14 cases)
- Runbook: readiness stuck 503, reconnect storm, backpressure checklists

#### Deployment Readiness MVP (DEC-029)
- `GET /healthz` endpoint on metrics server returning pipeline health JSON (`status`, `uptime_s`, `ws_connected`, `last_event_ts`)
- Multi-stage Dockerfile with non-root user and built-in HEALTHCHECK
- `docker-compose.yml` with cryptoscreener + Prometheus services
- `monitoring/prometheus.yml` scrape configuration
- CI docker smoke workflow (`.github/workflows/docker_smoke.yml`)
- Ops runbook (`docs/RUNBOOK_DEPLOYMENT.md`)
- 4 new healthz endpoint tests

#### Backpressure, Resource Bounds & Queue-Growth Acceptance (DEC-028, PR#94)
- Bounded event queue (`maxsize=10,000`) with drop-newest policy on `BinanceStreamManager`
- Bounded snapshot queue (`maxsize=1,000`) on `FeatureEngine` with drop counter
- 6 new Prometheus metrics: `pipeline_event_queue_depth`, `pipeline_snapshot_queue_depth`, `pipeline_tick_drift_ms`, `pipeline_rss_mb`, `pipeline_events_dropped`, `pipeline_snapshots_dropped`
- Tick drift, RSS, and queue depth sampling in main loop cadence block
- `SoakSummary` extended with backpressure fields (`max_event_queue_depth`, `max_rss_mb`, etc.)
- 6 new integration tests for backpressure acceptance (`TestBackpressureAcceptance`)

#### WS Resilience Validation & Soak Tests (DEC-027, PR#93)
- `SoakSummary` dataclass + `--summary-json PATH` CLI flag for post-run JSON report
- Fault injection: `--fault-drop-ws-every-s N` (periodic WS disconnect), `--fault-slow-consumer-ms M` (consumer delay)
- `BinanceStreamManager.force_disconnect()` method for clean fault injection
- Reconnect rate tracking (rolling window) with `max_reconnect_rate_per_min` in summary
- Fake WS server integration test (`test_ws_resilience.py`): reconnect discipline + backoff verification
- 4 new tests for reconnect behavior and force_disconnect

#### Live Runner Metrics Wiring (DEC-026, PR#92)
- Wired `MetricsExporter.update()` into `run_live.py` main loop (every cadence tick ~1s)
- Added read-only `circuit_breaker` and `governor` properties to `BinanceStreamManager`
- 12 Prometheus metrics now update continuously during live operation
- No new metrics, no new labels — reuses DEC-025 exporter contract
- 4 new tests for wiring correctness and cardinality safety

#### ML/Artifacts Hardening CI Gates (PR#90)
- Three path-triggered CI execution gates preventing silent drift:
  - `checksum-guard`: SHA256 verification of fixture/artifact manifests, untracked file detection, symlink/path traversal protection
  - `replay-determinism`: double-run digest comparison for replay fixtures
  - `contracts-roundtrip`: `pytest tests/contracts/` triggered on contract changes
- Fixed stale `mlrunner_model/manifest.json` (removed runtime-generated `model.pkl`)
- Updated `docs/branch_protection.md` with 7 required checks
- Branch protection applied: `strict: true`, `enforce_admins: true`

#### DEC-025 E2E Smoke Test (PR#89)
- `tests/monitoring/test_metrics_endpoint_smoke.py`: 4 async tests via `AioHTTPTestCase`
- Verifies all 12 `REQUIRED_METRIC_NAMES` present, `# TYPE` counter/gauge correct, counter monotonicity, gauge latest-value semantics

#### CODEOWNERS + Branch Protection (PR#88)
- `.github/CODEOWNERS`: `@bnzr-hub` for all paths (sole maintainer)
- `docs/branch_protection.md`: full branch protection checklist + `gh api` scriptable setup

#### DEC-025 Validation CI (PR#87)
- `promtool check rules` CI workflow on `monitoring/**` changes
- Forbidden label/selector checks in alert rules (cardinality protection)
- 5 tests: positive (no forbidden labels) + negative (detect forbidden labels/selectors)

#### DEC-025 SSOT Backfill (PR#86)
- DECISIONS.md, STATE.md, CHANGELOG.md updated with DEC-025 implementation details

#### Prometheus Observability Stack (DEC-025)
- `MetricsExporter` in `src/cryptoscreener/connectors/exporter.py` (PR#84):
  - 12 low-cardinality Prometheus metrics (6 Gauge + 6 Counter)
  - Metric names aligned 1:1 with `monitoring/alert_rules.yml`
  - `FORBIDDEN_LABELS` contract (no symbol/endpoint/path/query/ip)
  - `REQUIRED_METRIC_NAMES` frozenset for test validation
  - Counter delta tracking for monotonic increment semantics
- Minimal HTTP `/metrics` endpoint in `src/cryptoscreener/connectors/metrics_server.py` (PR#85):
  - `aiohttp.web` server serving `generate_latest(registry)`
  - Content-Type: `text/plain; version=0.0.4; charset=utf-8`
  - `--metrics-port` CLI flag in `scripts/run_live.py` (default 9090, 0 to disable)
- 16 PromQL alert rules in `monitoring/alert_rules.yml` (PR#83):
  - `promtool check rules` validated (SUCCESS 16 rules)
  - Counter references use `_total` suffix per prometheus_client convention
- `prometheus_client>=0.19.0` added to dependencies

#### Replay Gate Trigger Expansion (DEC-022)
- Expanded `require_replay()` in `acceptance_packet.sh` to include critical inference modules
- New trigger paths:
  - `src/cryptoscreener/registry/` — model package loading
  - `src/cryptoscreener/model_runner/` — MLRunner inference
  - `src/cryptoscreener/calibration/` — probability calibration
  - `src/cryptoscreener/training/` — dataset split (affects artifacts)
- PRs touching these paths now trigger full replay determinism verification
- Updated DEC-008 replay-required detection documentation

#### Model Package E2E Smoke (DEC-021)
- New `tests/registry/test_package_e2e_smoke.py` with 15 tests
- Full path validation: ModelPackage → MLRunner load → calibration → inference
- Test coverage:
  - Package integrity: manifest valid, checksums match, required files exist
  - MLRunner loads: PROD mode (no fallback), actual inference (not DATA_ISSUE)
  - Determinism: double run produces identical digests
- Fixtures generated on-the-fly (no binaries in git)
- JSON examples for PredictionSnapshot and RankEvent

#### LLM Timeout Enforcement (DEC-020)
- Fixed: `timeout_s` config now passed to Anthropic API call
- Timeout → fallback (per DEC-004 contract)
- New tests: `test_timeout_uses_fallback`, `test_timeout_parameter_passed_to_api`

#### MLRunner E2E Acceptance as CI Gate (DEC-019)
- New `tests/replay/test_mlrunner_e2e_determinism.py` with 17 tests
- MLRunner determinism contract:
  - DEV mode (no model): falls back to BaselineRunner → deterministic
  - PROD mode (no model): returns DATA_ISSUE with `RC_MODEL_UNAVAILABLE` → deterministic
- Test coverage:
  - DEV mode: digest stability, field-by-field comparison, fallback verification
  - PROD mode: digest stability, DATA_ISSUE assertion, artifact error verification
  - Replay proof generation (2 runs, digest comparison)
  - JSON roundtrip for RankEvent and PredictionSnapshot
  - Edge cases: empty fixture, single frame, single symbol
- CI gate: triggered by `acceptance_packet.sh` when PR touches `tests/replay/`
- Total tests: 745+

#### Production Profile Gates (DEC-017)
- New `InferenceStrictness` enum (`DEV`/`PROD`) for controlling fail-fast behavior
- PROD mode: artifact errors return `DATA_ISSUE` status (never raises, never fallback)
- DEV mode: lenient behavior with fallback to BaselineRunner
- Configurable data freshness thresholds aligned with SSOT:
  - `stale_book_max_ms`: 1000ms (was hardcoded 5000ms)
  - `stale_trades_max_ms`: 2000ms (was hardcoded 30000ms)
- New artifact error reason codes for precise diagnostics:
  - `RC_MODEL_UNAVAILABLE`: model artifact missing or failed to load
  - `RC_CALIBRATION_MISSING`: calibration artifact missing or failed to load
  - `RC_ARTIFACT_INTEGRITY_FAIL`: artifact hash mismatch (SHA256 verification failed)
- `MLRunnerConfig.strictness` field (default: DEV)
- 12 new tests for PROD/DEV behavior matrix

#### SSOT Reason Codes Alignment (DEC-018)
- Aligned all `RC_*` codes in `BaselineRunner` and `MLRunner` to `REASON_CODES_TAXONOMY.md`
- Code renames:
  - `RC_BOOK_PRESSURE` → `RC_FLOW_IMBALANCE_LONG` / `RC_FLOW_IMBALANCE_SHORT`
  - `RC_TIGHT_SPREAD` → `RC_SPREAD_TIGHT`
  - `RC_WIDE_SPREAD` → `RC_SPREAD_WIDE`
  - `RC_TOXIC_RISK` → `RC_TOXIC_RISK_UP`
  - `RC_HIGH_VOL` → `RC_REGIME_HIGH_VOL`
- Updated `REASON_CODES_TAXONOMY.md` with missing implementation codes:
  - Gates: `RC_GATE_SPREAD_FAIL`, `RC_GATE_IMPACT_FAIL`
  - Data Quality: `RC_DATA_STALE`
  - ML/Calibration: `RC_CALIBRATION_ADJ`
- Updated tests to assert SSOT-compliant code names

#### GitHub PR-C — MLRunner with Calibration Integration (DEC-014)
- New `src/cryptoscreener/model_runner/ml_runner.py`:
  - `MLRunner` class inheriting from `ModelRunner`
  - Loads model artifacts (pickle/joblib/ONNX)
  - Loads calibration artifacts from DEC-013
  - Applies calibration to raw model probabilities
  - Fallback to `BaselineRunner` when model unavailable
- New `MLRunnerConfig` with model_path, calibration_path, require_calibration, fallback_to_baseline
- Error types: `ModelArtifactError`, `CalibrationArtifactError`
- Calibration flow: raw probs → calibrated probs → PredictionSnapshot → Scorer/Ranker
- Tests: 25+ tests for fallback, calibration loading, determinism, gates
- Replay determinism: same input → identical PredictionSnapshot JSON
- Per PRD §11 Milestone 3: "Training pipeline skeleton"

#### GitHub PR-B — Probability Calibration (DEC-013)
- New `src/cryptoscreener/calibration/` module:
  - `PlattCalibrator` with Platt scaling (sigmoid calibration)
  - `fit_platt()` using gradient descent on cross-entropy
  - `CalibrationArtifact` with calibrators + metadata
  - `CalibrationMetadata` with schema_version, git_sha, config_hash, data_hash
  - `save_calibration_artifact()` / `load_calibration_artifact()`
  - `Calibrator` protocol for pluggable implementations
- New `scripts/fit_calibration.py` CLI:
  - Fits calibrators on validation data
  - Reports Brier/ECE before and after calibration
  - Outputs calibration.json with calibrators + metadata
  - Configurable heads, max_iter, learning rate
- Tests: 36 unit tests for Platt, artifacts, roundtrip, adversarial
- Guarantees: probabilities in [0,1], monotonicity, determinism
- Per PRD §11 Milestone 3: "Training pipeline skeleton"

#### GitHub PR-A — Training Dataset Split (DEC-012)
- New `src/cryptoscreener/training/` module:
  - `time_based_split()` with strict temporal ordering
  - `SplitConfig`, `SplitMetadata`, `SplitResult` dataclasses
  - `verify_no_leakage()` anti-leakage validation
  - `_find_boundary_shift()` to prevent duplicate ts leakage
  - Fail-fast on empty splits
  - Optional purge gap between splits
  - `load_labeled_dataset()` for parquet/JSONL
  - `validate_schema()` with strict mode
  - `get_feature_columns()`, `get_label_columns()` extractors
- New `scripts/split_dataset.py` CLI:
  - Configurable train/val/test ratios (default 0.7/0.15/0.15)
  - Outputs: train.jsonl, val.jsonl, test.jsonl, metadata.json
  - LEAKAGE CHECK status with pass/fail
- Tests: 25 anti-leakage tests verify `max(train_ts) < min(val_ts) < min(test_ts)`
- Metadata tracking: git_sha, config_hash, data_hash, schema_version
- Per PRD §11 Milestone 3: "Training pipeline skeleton"

#### GitHub PR#55 — Offline Backtest Harness (DEC-011)
- New `src/cryptoscreener/backtest/` module:
  - `compute_auc()`, `compute_pr_auc()` for classification quality
  - `compute_brier_score()`, `compute_ece()` for calibration metrics
  - `compute_topk_capture()`, `compute_topk_mean_edge()` for top-K metrics
  - `compute_churn_metrics()` for rank stability analysis
  - `BacktestHarness` class for running evaluations
  - `BacktestResult` with serialization to JSON
- New `scripts/run_backtest.py` CLI:
  - Evaluates labeled data against metrics
  - Configurable horizons, profiles, top-K
  - Outputs JSON report with all metrics
  - Exit code based on ECE acceptance criteria
- Tests: 41 unit tests for metrics and harness
- Per PRD §10: AUC, PR-AUC, Brier, ECE, Top-K capture, churn

#### GitHub PR#54 — Label Builder for ML Ground Truth (DEC-010)
- New `src/cryptoscreener/cost_model/` module:
  - `CostCalculator` for execution cost estimation
  - `compute_spread_bps()`, `compute_impact_bps()` functions
  - `ExecutionCosts` dataclass with spread, fees, impact, total
  - Configurable fees by profile (A: maker-ish, B: taker-ish)
  - Clip size calculation: `Q_usd = k * usd_volume_60s`
- New `src/cryptoscreener/label_builder/` module:
  - `LabelBuilder` for ML ground truth generation
  - `I_tradeable(H)` for horizons 30s, 2m, 5m per LABELS_SPEC.md
  - MFE/MAE calculation (Maximum Favorable/Adverse Excursion)
  - Toxicity labels (`y_toxic`) with configurable tau and threshold
  - Gate checks (spread, impact) before tradeability
  - `LabelRow` with all labels for (symbol, timestamp)
  - Flat dict conversion for DataFrame/parquet export
- New `scripts/build_labels.py` CLI:
  - Reads market events from JSONL
  - Outputs labels to parquet or JSONL
  - Configurable thresholds: `--x-bps-*`, `--spread-max-bps`, `--toxicity-*`
  - Summary report with tradeability/toxicity statistics
- Tests: 25+ unit tests for cost_model and label_builder

#### GitHub PR#52 — CI Pipeline Verified (Smoke Test)
- **Verified commit:** `31c292d38b64936ac2e2379d11cdd1f94e2ca082`
- **What was tested:**
  - FULL VERBATIM mode for small diffs (README.md single-line change)
  - Manual marker detection outside managed block (PR#51 feature)
  - acceptance-packet auto-updates PR body between managed markers
  - proof-guard validates FULL VERBATIM content correctly
- **Known behavior:** proof-guard may need re-run if it starts before acceptance-packet updates body (timing, not a bug)
- **Result:** All checks pass, CI pipeline works as designed after PR#49-#51

#### GitHub PR#51 — Harden Manual Marker Detection Outside Managed Block
- `proof_guard.yml`: detect acceptance markers (`PENDING`, `CI ARTIFACT`, `IDENTITY`) outside managed block
- Manual markers in user-editable PR description now FAIL with explicit error message
- Error message explains: CI only updates inside `<!-- ACCEPTANCE_PACKET_START -->...<!-- ACCEPTANCE_PACKET_END -->`
- `acceptance_packet.yml`: fix `re.sub` backslash escape error when packet contains regex patterns (use lambda)
- CLAUDE.md updated with manual marker warning and fix instructions

#### GitHub PR#49 — CI Acceptance Packet + Auto PR Body Proof + Proof Guard CI Artifact Mode
- New workflow `.github/workflows/acceptance_packet.yml`:
  - Runs on `pull_request` (opened, synchronize, reopened, ready_for_review) + `workflow_dispatch`
  - Installs dev dependencies (`pip install -e ".[dev]"`)
  - Runs `./scripts/acceptance_packet.sh <PR_NUMBER>`
  - Always uploads artifact `acceptance_packet_pr<N>` with SHA256/size
  - Auto-updates PR body between `<!-- ACCEPTANCE_PACKET_START -->` and `<!-- ACCEPTANCE_PACKET_END -->` markers
  - Two modes: FULL VERBATIM (packet < 45KB with `diff --git`) or CI ARTIFACT (reference block)
  - Anti-loop: skips if actor is `github-actions[bot]`, no `edited` trigger
- Updated `proof_guard.yml` to support CI ARTIFACT mode:
  - PENDING mode now FAILS (was security hole - prevented merge without proof)
  - FULL VERBATIM: all markers + `diff --git` required
  - CI ARTIFACT: validates via GitHub API that check-run `Acceptance Packet` passed for HEAD_SHA
  - CI ARTIFACT mode does NOT require `diff --git` in body
- CLAUDE.md updated: "CI is the source of truth", local runs optional

#### GitHub PR#48 — Merge Safety & Stacked PR Detection
- `acceptance_packet.sh`: added `== ACCEPTANCE PACKET: MERGE SAFETY ==` section
- PR chain resolution: walks up branch tree to find all prerequisite PRs by number/URL
- Fails if stacked base branch has no corresponding open PR (`STACKED_CHAIN_BROKEN`)
- New readiness classification: `Ready for review` vs `Ready for final merge`
- STACKED PRs show `Ready for final merge: false` (must merge prerequisites first)
- New `--require-main-base` flag to fail if base is not main/master
- CLAUDE.md updated with detailed stacked PR workflow and best practices

#### GitHub PR#47 — Reviewer Message Generator
- New `scripts/reviewer_message.sh <PR>` — generates ready-to-paste reviewer chat message
- Wraps acceptance_packet.sh output in clean "copy from here" format
- Shows status indicator (ready/not ready) at the end
- CLAUDE.md updated with usage instructions

#### GitHub PR#46 — Robust Replay Detection in acceptance_packet.sh
- `acceptance_packet.sh`: switched from `gh pr view --json files` to `gh api /repos/.../pulls/.../files`
- File list retrieval now matches `proof_guard.yml` logic (paginated API call)
- Added fail-safe: empty file list now fails packet (prevents false `replay_required: false`)
- Added repo detection via `gh repo view --json nameWithOwner`

#### GitHub PR#45 — Enforce Verbatim Acceptance Packet
- `acceptance_packet.sh`: removed SUMMARY section, single `== ACCEPTANCE PACKET: END ==` marker
- `acceptance_packet.sh`: added "Paste verbatim, do NOT summarize" banner
- `proof_guard.yml`: disabled proof_bundle fallback — acceptance_packet is now mandatory
- `proof_guard.yml`: added `diff --git` check to ensure full patch is included
- No more shortcuts: PR body must contain complete verbatim acceptance packet output

#### GitHub PR#44 — Global Proof & Reporting Policy (DEC-009)
- Verbatim-only reporting policy: summaries/tables/paraphrasing are NOT valid proof
- DEC-009: Global proof policy — acceptable evidence is only verbatim packet output
- `proof_guard.yml` dual-mode marker validation:
  - Preferred: acceptance_packet markers (`== ACCEPTANCE PACKET: *`)
  - Fallback: proof_bundle markers (backward compatible)
- Replay determinism enforcement in acceptance_packet mode requires `ALL DIGESTS MATCH`
- CLAUDE.md "Формат отчёта" replaced with verbatim-only mandatory format

#### GitHub PR#43 — Acceptance Packet Automation (DEC-008)
- New `scripts/acceptance_packet.sh <PR>` — one-command "ready for ACCEPT" generator
  - Waits for CI checks to pass (polls with timeout)
  - Runs quality gates (ruff, mypy, pytest)
  - Auto-generates proof bundle artifacts if missing
  - Shows full PR diff
  - Validates replay determinism if PR touches replay-related files
- Updated `scripts/proof_bundle.sh`: python→python3, use `gh pr diff` for full PR diff
- Updated `scripts/proof_bundle_chat.sh`: auto-generate artifacts if missing
- Expanded `.github/workflows/proof_guard.yml` replay detection to include `run_record.py` and `tests/replay/`
- Added mandatory `acceptance_packet.sh` usage rule to CLAUDE.md

#### GitHub PR#25 — Live Pipeline End-to-End (DEC-006)
- `scripts/run_live.py`: Full live pipeline connecting all components
- Data flow: BinanceStreamManager → StreamRouter → FeatureEngine → BaselineRunner → Ranker → Alerter
- Aggregated `PipelineMetrics` with event counts, latencies, LLM stats
- Graceful shutdown via SIGINT/SIGTERM (flag-based, no race conditions)
- CLI flags: `--symbols`, `--top N`, `--cadence`, `--llm`, `--output`, `--duration-s`, `--verbose`
- `--duration-s` for controlled test runs without manual SIGTERM
- One-time REST call for `--top N` mode (symbol list only, not polling)
- LLM disabled by default in live mode (per DEC-005)

#### GitHub PR#42 — Record→Replay Bridge (DEC-007)
- `scripts/run_record.py` — recording harness for fixture generation
  - CLI: `--symbols`, `--duration-s`, `--out-dir`, `--cadence-ms`, `--llm`, `--source`
  - Outputs: `market_events.jsonl`, `expected_rank_events.jsonl`, `manifest.json`
- `SyntheticMarketEventGenerator` — deterministic synthetic market event generation
- `MinimalRecordPipeline` — mirrors `MinimalReplayPipeline` for determinism
- Manifest format v1.0.0 with SHA256 checksums and replay digest
- LLM OFF by default in recording mode
- `tests/replay/test_record_replay_roundtrip.py` — 16 tests for record→replay determinism

#### GitHub PR#23 — LLM Explain Pipeline Integration (DEC-005)
- Integrated LLM explain into Alerter: `RankEvent.payload.llm_text` populated for alert events
- `ExplainLLMProtocol` for dependency injection (no hard import of explain_llm module)
- Per-symbol LLM cooldown (60s default) with caching to reduce API calls
- `AlerterConfig.llm_enabled` flag for global LLM disable
- LLM metrics: `llm_calls`, `llm_cache_hits`, `llm_failures`
- 8 integration tests for LLM functionality in Alerter

#### GitHub PR#22 — LLM Explain Module (DEC-004)
- `ExplainLLM` interface with `AnthropicExplainer` and `MockExplainer`
- Strict no-new-numbers validation (`validate_llm_output_strict`)
- Fallback on any LLM failure (timeout, validation, exception)
- 9 adversarial tests for LLM guardrails

#### PR#1 — Scaffold + Contracts
- Project structure with `pyproject.toml`, editable install, dev dependencies
- Data contract models in `src/cryptoscreener/contracts/`:
  - `MarketEvent` — canonical market data from exchange
  - `FeatureSnapshot` — feature vector with windows and data health
  - `PredictionSnapshot` — ML predictions with reasons and status
  - `RankEvent` — ranking state transitions with deterministic digest
  - `LLMExplainInput/Output` — strict LLM contract (text-only)
- Schema validation utilities with adversarial LLM tests
- Replay harness (`scripts/run_replay.py`) with determinism verification
- Test fixture (`tests/fixtures/sample_run/`) with checksums
- 39 unit tests covering roundtrip, validation, and LLM guardrails
- Tooling: ruff (lint) + mypy (strict types) configured

### Changed
- None

### Fixed
- None

---

## 2026-01-24
- Documentation pack created (PRD, architecture, contracts, ops, dev)
