# DECISIONS

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


Record decisions with:
- Date
- Decision
- Alternatives considered
- Rationale
- Impact

---

## DEC-001: Integer/Float Equivalence in LLM Number Validation

**Date:** 2026-01-24

**Decision:** When validating LLM output numbers against input numeric_summary, treat whole-number floats as equivalent to their integer representation.

**Rule:** `5.0` in input allows both `"5"` and `"5.0"` in output.

**Alternatives considered:**
1. Strict string matching only (`"5.0"` must appear as `"5.0"`)
2. Normalize all to floats before comparison
3. Allow any numeric equivalence (rejected: too permissive)

**Rationale:**
- LLMs naturally output `"5"` when the value is a whole number, even if input was `5.0`
- Strict matching causes false-positive validation failures
- Whole-number equivalence is mathematically sound and human-intuitive
- Non-whole numbers (e.g., `5.123`) still require exact string match

**Impact:**
- `LLMExplainOutputValidator` must implement whole-number equivalence check
- Tests in `test_llm_float_edge_cases.py` cover edge cases
- No security implications (still prevents LLM from inventing new numbers)

---

## DEC-002: BaselineRunner Status Logic with PRD Critical Gates

**Date:** 2026-01-24 (Updated)

**Decision:** BaselineRunner implements PRD critical gates as HARD blocks before TRADEABLE, plus simplified heuristic-based status classification.

**PRD Critical Gates (HARD - must pass for TRADEABLE):**
1. **Spread gate:** `spread_bps <= spread_max_bps` (default: 10.0 bps)
2. **Impact gate:** `impact_bps_q <= impact_max_bps` (default: 20.0 bps)
3. **Data health gate:** freshness checks (stale/missing streams)

If any gate fails → TRADEABLE is blocked → downgrade to WATCH.

**Status Classification:**
1. **DATA_ISSUE** (checked first):
   - `stale_book_ms > 5000` → DATA_ISSUE
   - `stale_trades_ms > 30000` → DATA_ISSUE
   - `missing_streams` not empty → DATA_ISSUE

2. **TRAP** (checked second, before gates):
   - `p_toxic >= 0.7` → TRAP

3. **TRADEABLE** (requires passing ALL gates):
   - `p_inplay >= 0.6` AND `spread_bps <= 10.0` AND `impact_bps_q <= 20.0`
   - If gates fail → downgrade to WATCH

4. **WATCH:** `p_inplay >= 0.3`

5. **DEAD:** `p_inplay < 0.3`

**Gate Failure Reasons:**
- `RC_GATE_SPREAD_FAIL`: Spread exceeds max threshold
- `RC_GATE_IMPACT_FAIL`: Impact exceeds max threshold

**Expected utility:** Computed but not used for gating (future MLRunner will use it).

**Alternatives considered:**
1. Soft gates (spread/impact as factors in p_inplay) — rejected: PRD requires HARD gates
2. Full utility-based state machine from STATE_MACHINE.md — deferred to MLRunner
3. Hysteresis for state transitions — deferred to MLRunner

**Rationale:**
- PRD Section 9 requires critical gates before TRADEABLE
- Baseline mode enforces gates with configurable thresholds
- Gate failures produce explicit reason codes for transparency
- Configuration allows tuning without code changes

**Impact:**
- TRADEABLE is impossible if spread or impact exceed limits
- Gate thresholds included in `compute_digest()` for replay verification
- 10 dedicated gate tests verify behavior
- Clear upgrade path to MLRunner with more sophisticated logic

---

## DEC-003: RankEvent Payload Semantics (Lightweight vs Rich)

**Date:** 2026-01-24

**Decision:** RankEvent `payload.prediction` has different semantics based on event source:
- **Ranker events** (SYMBOL_ENTER, SYMBOL_EXIT): Empty dict `{}` — lightweight events optimized for high-frequency updates.
- **Alerter events** (ALERT_TRADABLE, ALERT_TRAP, DATA_ISSUE): Full `PredictionSnapshot` dict — self-contained payloads for downstream consumers.

**Alternatives considered:**
1. Always include full PredictionSnapshot — rejected: excessive payload size for high-frequency ranker events
2. Make `payload.prediction` nullable (`None`) — rejected: breaks JSON schema compatibility, `{}` is valid JSON
3. Separate event types with different schemas — rejected: increases contract complexity

**Rationale:**
- Ranker emits events at 1Hz rate; full prediction payload would multiply bandwidth unnecessarily
- Downstream consumers (e.g., UI) can fetch prediction from prediction store using symbol+timestamp
- Alerter events are lower frequency and benefit from self-contained payloads for notification systems
- `{}` is a valid dict value that distinguishes "no prediction attached" from "prediction is null"

**Impact:**
- DATA_CONTRACTS.md updated with explicit payload semantics
- Downstream consumers must check `payload.prediction != {}` before accessing prediction fields
- RankEvent contract in `events.py` uses `RankEventPayload(prediction={})` as default
- Score normalization formula documented: `score = p_inplay * (utility/Umax) * (1 - α*p_toxic)` with score ∈ [0, 1]

---

## DEC-004: LLM Explain Module Architecture (PR#23)

**Date:** 2026-01-24

**Decision:** Implement LLM Explain module with strict guardrails per CLAUDE.md §9:

1. **Anthropic-only provider** in initial implementation; other providers may be added later.
2. **Sync interface** (`ExplainLLM.explain()` returns `LLMExplainOutput` directly, no async).
3. **No-new-numbers constraint:** LLM outputs are text-only; MUST NOT change numeric values, status, or score. Numbers in output must be exact stringwise match to input.
4. **Fallback on any failure:** Exception, timeout, or validation violation → deterministic fallback via `generate_fallback_output()`.
5. **Client injection for testing:** `AnthropicExplainer.with_client()` allows injecting mock client; unit tests do NOT perform network calls.

**Alternatives considered:**
1. Async interface with `asyncio` — rejected: adds complexity, sync is sufficient for batch/interactive use
2. Multiple providers (OpenAI, Gemini) — deferred: Anthropic-only simplifies initial implementation
3. Allow LLM to compute percentages (0.83 → 83%) — rejected: violates no-new-numbers principle
4. Strict retry without fallback — rejected: availability > precision for explanatory text

**Rationale:**
- LLM explanations are auxiliary text; they MUST NOT affect trading decisions
- Strict validation prevents LLM hallucinations from introducing spurious numbers
- Fallback ensures system always produces valid output, even if LLM fails
- Client injection enables deterministic testing without network dependencies
- Sync interface matches existing codebase patterns (no async in pipeline)

**Impact:**
- `validate_llm_output_strict()` enforces no-new-numbers via regex extraction
- 9 adversarial tests verify constraint: percentage conversion, invented numbers, max_chars, invalid status_label, numbers in words, scientific notation, fraction notation, trailing zeros, unicode digits
- MockExplainer provides deterministic output for "LLM off" mode and tests
- AnthropicExplainer includes retry logic with exponential backoff + jitter

**Known risks:**
1. Regex-based number extraction may miss edge cases (unicode digits, locale-specific thousand separators)
2. Anthropic API/SDK format may change; fallback ensures graceful degradation

---

## DEC-005: Alerter-LLM Integration (GitHub PR#23)

**Date:** 2026-01-24

**Decision:** Integrate LLM explain into Alerter with the following constraints:

1. **Per-symbol cooldown (60s default):** LLM calls are rate-limited per symbol, not globally. Cached `llm_text` is reused within cooldown window.
2. **Hardcoded numeric_summary fields:** `spread_bps=0.0`, `impact_bps=0.0` in LLM input — these values are not available in `PredictionSnapshot`. Will be populated when BaselineRunner provides them.
3. **Optional dependency:** Alerter accepts `explainer: ExplainLLMProtocol | None`. If None, `llm_text` remains empty.
4. **Failure isolation:** LLM exceptions are caught and logged; alerting continues with empty `llm_text`.

**Alternatives considered:**
1. Global cooldown across all symbols — rejected: would throttle unrelated symbols
2. Fetch spread/impact from FeatureSnapshot — deferred: requires pipeline plumbing
3. Make LLM mandatory — rejected: system must work without LLM

**Rationale:**
- Per-symbol cooldown matches alert semantics (each symbol's explanation evolves independently)
- Hardcoded 0.0 is safe: LLM uses these for context, not trading decisions
- Optional explainer allows gradual rollout and testing
- Failure isolation per DEC-004: LLM failures must not break core functionality

**Impact:**
- `AlerterConfig.llm_cooldown_ms` (default 60000) controls rate limiting
- `AlerterConfig.llm_enabled` (default True) allows global disable
- Metrics: `llm_calls`, `llm_cache_hits`, `llm_failures` for observability
- 8 integration tests verify caching, cooldown, and failure handling

---

## DEC-008: Acceptance Packet Automation (PR#43)

**Date:** 2026-01-24

**Decision:** Implement one-command "ready for ACCEPT" generator that produces complete proof bundle with all required evidence.

**Command:**
```bash
./scripts/acceptance_packet.sh <PR_NUMBER>
```

**What it does:**
1. **Waits for CI** — polls `gh pr checks` until all pass (fails on timeout/failure)
2. **Runs quality gates** — ruff, mypy, pytest (fails if any fail)
3. **Ensures artifacts exist** — auto-generates via `proof_bundle.sh` if missing
4. **Shows PR diff** — full `gh pr diff` output
5. **Validates replay** (if required) — generates synthetic fixture, runs replay twice, verifies digest match

**Replay-required detection:**
PR requires replay proof if it touches any of:
- `scripts/run_replay.py`
- `scripts/run_record.py`
- `tests/replay/*`
- `tests/fixtures/*`

**Exit codes:**
- `0` — All checks passed, ready for ACCEPT
- `1` — Some check failed
- `2` — Usage error

**Alternatives considered:**
1. Manual multi-step process — rejected: error-prone, inconsistent evidence
2. Separate scripts for each check — rejected: fragmented, easy to miss steps
3. CI-only validation — rejected: doesn't help during development/review

**Rationale:**
- Single source of truth for "what does ACCEPT require"
- Eliminates back-and-forth about missing evidence
- Auto-generates missing artifacts
- Fails fast on any issue

**Impact:**
- New `scripts/acceptance_packet.sh` with CLI interface
- Updated `scripts/proof_bundle.sh` (python→python3, PR diff instead of HEAD)
- Updated `scripts/proof_bundle_chat.sh` (auto-generate artifacts)
- Expanded `proof_guard.yml` replay detector
- CLAUDE.md updated with mandatory acceptance_packet usage

---

## DEC-009: Proof & Reporting Policy (Global)

**Date:** 2026-01-24

**Decision:** Acceptable review evidence is **only** a verbatim proof packet pasted into PR body:
- Preferred: output of `./scripts/acceptance_packet.sh <PR>`
- Fallback: full output of `./scripts/proof_bundle.sh <PR>`

Summary/tables/paraphrasing **are NOT valid proof**. "No proof = NOT DONE" applies to all PRs and all DEC scopes.

**Required packet contents (minimum):**
- PR identity (url/state/base/head/title/number)
- CI status (no pending; fail if any failed)
- Quality gates raw output: ruff/mypy/pytest
- PR diff (full, patch-level)
- Artifacts/proof bundle file path (or proof markers)

**Conditional requirements:**
- If PR touches replay-related areas (`scripts/run_replay.py`, `scripts/run_record.py`, `tests/replay/*`, `tests/fixtures/*`) → packet must include replay determinism proof (2 runs + digest match).
- If PR touches LLM guardrails/contracts → packet must include adversarial tests ("no-new-numbers", enums, max chars, fallback).

**Enforcement:** `proof_guard.yml` validates presence of packet markers in PR body (dual-mode: acceptance_packet or proof_bundle markers).

**Alternatives considered:**
1. Summary-based reporting — rejected: summaries can omit/hide failures
2. Manual marker checks — rejected: error-prone, inconsistent
3. CI-only enforcement — rejected: doesn't catch missing evidence in PR body

**Rationale:**
- Verbatim output is tamper-evident and complete
- Reviewer can verify exact command output without re-running
- Eliminates "it works on my machine" claims
- Standardizes evidence across all PRs

**Impact:**
- CLAUDE.md "Формат отчёта" replaced with verbatim-only policy
- proof_guard.yml updated with dual-mode marker validation
- All future PRs must include verbatim packet in body

---

## DEC-006: Live Pipeline Architecture (GitHub PR#25)

**Date:** 2026-01-24

**Decision:** Implement `scripts/run_live.py` as end-to-end live pipeline with the following design:

1. **Data flow order:** BinanceStreamManager → StreamRouter → FeatureEngine → BaselineRunner → Ranker → Alerter → Output
2. **Async event loop:** Uses `asyncio` with `async for event in stream_manager.events()` pattern
3. **Snapshot cadence:** Configurable via `--cadence` (default 1000ms), controls feature emission and prediction frequency
4. **LLM disabled by default:** Live mode sets `llm_enabled=False`; use `--llm` flag to enable
5. **Graceful shutdown:** SIGINT/SIGTERM sets `_running=False`; main loop exits naturally; `finally` block calls `stop()`
6. **Duration limit:** `--duration-s N` for controlled runs without requiring SIGINT/SIGTERM
7. **Symbol selection:** Either explicit `--symbols BTCUSDT,ETHUSDT` or `--top N` for top N by volume

**REST vs WebSocket:**
- **One-time REST at startup:** `--top N` mode calls `get_top_symbols_by_volume()` which makes two REST calls: exchangeInfo + 24hr tickers. Symbols are sorted by 24h quote volume (descending).
- **No REST in main loop:** All market data streaming is via WebSocket. No high-frequency REST polling occurs.
- **Explicit symbols mode:** `--symbols X,Y,Z` bypasses REST entirely — pure WebSocket from start.

**Alternatives considered:**
1. Separate process per component — rejected: adds IPC complexity, single process is sufficient for 50-500 symbols
2. Celery/Redis for task distribution — rejected: overkill for MVP, single async loop handles throughput
3. LLM enabled by default — rejected: per DEC-005, LLM adds latency and cost; disabled by default
4. Signal handler calls `stop()` directly — rejected: causes race conditions with double-stop; handler only sets flag

**Rationale:**
- Single async process simplifies deployment and debugging
- Cadence-based emission (not event-driven) ensures predictable resource usage
- LLM opt-in prevents unexpected API costs during development
- Graceful shutdown via flag-setting avoids double-stop race conditions
- `--duration-s` enables CI testing without manual SIGTERM

**Impact:**
- `scripts/run_live.py` is the primary entry point for live trading signals
- `PipelineMetrics` collects: event counts, latencies, LLM stats, router/ranker stats
- Output is JSONL (RankEvents) to stdout or `--output` file
- CLI designed for production use: `python -m scripts.run_live --top 50 --output events.jsonl`
- CI can use: `python -m scripts.run_live --symbols BTCUSDT --duration-s 10` for smoke tests

---

## DEC-007: Record→Replay Bridge (PR#42)

**Date:** 2026-01-24

**Decision:** Implement recording harness to generate replay fixtures from synthetic or live data:

1. **New script `scripts/run_record.py`**:
   - CLI: `--symbols`, `--duration-s`, `--out-dir`, `--cadence-ms`, `--llm` (default OFF), `--source synthetic|live`
   - Outputs: `market_events.jsonl`, `expected_rank_events.jsonl`, `manifest.json`

2. **Manifest format (schema_version 1.0.0)**:
   - Required fields: `schema_version`, `recorded_at`, `source`, `symbols`, `duration_s`
   - SHA256 checksums: `sha256.market_events.jsonl`, `sha256.expected_rank_events.jsonl`
   - Replay verification: `replay.rank_event_stream_digest`
   - Stats: `total_market_events`, `total_rank_events`, `time_range_ms`

3. **MinimalRecordPipeline mirrors MinimalReplayPipeline**:
   - Same deterministic logic (SYMBOL_ENTER at trade 2, ALERT_TRADABLE at trade 4)
   - Ensures record→replay digest match

4. **LLM OFF by default**: Recording does not include LLM explanations unless `--llm` flag is set

**Alternatives considered:**
1. Record only market events, recompute expected on replay — rejected: no baseline truth for verification
2. Store expected digest only (not full events) — rejected: need events for debugging mismatches
3. Live-only recording — rejected: synthetic mode enables CI testing without external dependencies

**Rationale:**
- Recording expected outputs provides ground truth for determinism verification
- SHA256 checksums detect file tampering or corruption
- Synthetic mode enables fast, deterministic CI tests
- Manifest documents recording parameters for reproducibility

**Impact:**
- New `scripts/run_record.py` with CLI interface
- New `tests/replay/test_record_replay_roundtrip.py` with 16 tests
- Fixtures can be generated for any symbol set and duration
- Replay verification uses manifest digest for comparison

---

## DEC-010: Label Builder Architecture (PR#54)

**Date:** 2026-01-25

**Decision:** Implement offline label builder for ML ground truth generation per LABELS_SPEC.md:

1. **Cost Model** (`src/cryptoscreener/cost_model/`):
   - `cost_bps = spread_bps + fees_bps + impact_bps(Q)`
   - Configurable fees per profile: A (maker-ish, 2 bps), B (taker-ish, 4 bps)
   - Clip size: `Q_usd = k * usd_volume_60s` where k=0.01 (scalping) or k=0.03 (intraday)
   - Impact estimation via orderbook depth walking

2. **Tradeability Labels**:
   - `I_tradeable(H, profile) = 1` if `net_edge_bps(H) >= X_bps(profile, H)` AND gates pass
   - `net_edge_bps(H) = MFE_bps(H) - cost_bps`
   - MFE = Maximum Favorable Excursion within horizon window
   - Gates: spread <= spread_max_bps, impact <= impact_max_bps

3. **Toxicity Labels**:
   - `y_toxic = 1` if price moves against by > threshold_bps within tau_ms
   - Default: tau=30s, threshold=10 bps

4. **CLI** (`scripts/build_labels.py`):
   - Input: JSONL market events
   - Output: parquet or JSONL with flat schema
   - Configurable thresholds via CLI args

**Alternatives considered:**
1. Online label generation during recording — rejected: labels need future data (lookahead)
2. Single profile — rejected: PRD requires both maker and taker profiles
3. Fixed thresholds — rejected: thresholds should be tunable per trading style

**Rationale:**
- Offline labeling allows using future data (MFE, toxicity)
- Separate cost model enables unit testing and reuse
- Flat output format compatible with pandas/parquet for ML training
- Configurable thresholds support experimentation

**Impact:**
- New `src/cryptoscreener/cost_model/` module
- New `src/cryptoscreener/label_builder/` module
- New `scripts/build_labels.py` CLI
- 25+ unit tests for cost model and label builder
- Labels ready for ML training pipeline (Milestone 3)

---

## DEC-011: Offline Backtest Harness (PR#55)

**Date:** 2026-01-25

**Decision:** Implement offline backtest evaluation module per PRD §10 and EVALUATION_METRICS.md.

**Components:**
1. **Metrics Module** (`src/cryptoscreener/backtest/metrics.py`):
   - AUC, PR-AUC for classification quality
   - Brier score for probabilistic accuracy
   - ECE (Expected Calibration Error), MCE (Maximum Calibration Error)
   - Top-K capture rate, mean net_edge_bps in top-K
   - Churn metrics for rank stability (Jaccard similarity, state changes)

2. **Harness Module** (`src/cryptoscreener/backtest/harness.py`):
   - `BacktestConfig` for evaluation configuration
   - `BacktestHarness` for running evaluations
   - `Predictor` protocol for model integration
   - `evaluate_labels_only()` for baseline analysis
   - `evaluate_with_predictor()` for model evaluation

3. **CLI** (`scripts/run_backtest.py`):
   - Input: labeled parquet/JSONL from build_labels.py
   - Output: JSON report with all metrics
   - Exit code: 0 if ECE < 5% (acceptance criteria), 1 otherwise

**Metrics per PRD §10:**
- Offline: AUC, PR-AUC, Brier, ECE, Top-K capture, net edge in top-K, churn
- Acceptance criteria: ECE < 5%, top-K captures >= X% of tradeable events

**Alternatives considered:**
1. Use scikit-learn metrics — rejected: adds heavy dependency for simple calculations
2. Inline evaluation in training — rejected: separate module enables reuse
3. Only model evaluation — rejected: baseline analysis (labels-only) useful for data quality

**Rationale:**
- Pure Python metrics avoid sklearn dependency
- Protocol-based predictor allows any model implementation
- Separate harness enables both offline eval and CI integration
- JSON output format for easy inspection and automation

**Impact:**
- New `src/cryptoscreener/backtest/` module
- New `scripts/run_backtest.py` CLI
- 41 unit tests for metrics and harness
- Completes PRD §11 Milestone 2: "Label builder + offline backtest harness"

---

## DEC-012: Training Dataset Split (PR-A)

**Date:** 2026-01-25

**Decision:** Implement time-based train/val/test splitting for ML training per PRD Section 11 Milestone 3.

**Components:**
1. **Dataset Module** (`src/cryptoscreener/training/dataset.py`):
   - Schema validation against LABELS_SPEC.md output format
   - `DatasetSchema` with version tracking (1.0.0)
   - `load_labeled_dataset()` for parquet/JSONL loading
   - `validate_schema()` with strict mode option
   - `get_feature_columns()`, `get_label_columns()` extractors

2. **Split Module** (`src/cryptoscreener/training/split.py`):
   - `SplitConfig` with train/val/test ratios (default 0.7/0.15/0.15)
   - `time_based_split()` with strict temporal ordering
   - Optional purge gap between splits to prevent leakage
   - `SplitResult.verify_no_leakage()` for validation
   - `SplitMetadata` with git_sha, config_hash, data_hash

3. **CLI** (`scripts/split_dataset.py`):
   - Input: labeled parquet/JSONL from build_labels.py
   - Output: train.jsonl, val.jsonl, test.jsonl, metadata.json
   - Configurable ratios, purge gap, output format
   - Prints LEAKAGE CHECK status with pass/fail

**Anti-Leakage Guarantees:**
- All train samples have timestamps strictly before all val samples
- All val samples have timestamps strictly before all test samples
- Invariant: `max(train_ts) < min(val_ts) < min(test_ts)`
- Optional purge gap removes samples near split boundaries

**Metadata Tracking:**
- `schema_version`: Split schema version (1.0.0)
- `git_sha`: Git commit at split time (12 chars)
- `config_hash`: SHA256 of split configuration (16 chars)
- `data_hash`: SHA256 of input data (16 chars)
- `split_timestamp`: ISO timestamp of split creation
- Timestamp ranges for each split (train/val/test)

**Alternatives considered:**
1. Random split - rejected: causes temporal leakage, future data in train set
2. Symbol-stratified split - deferred: global split sufficient for MVP
3. K-fold cross-validation - deferred: single split for initial training
4. Purge gap always on - rejected: should be configurable (default 0)

**Rationale:**
- Time-based splits prevent data leakage in time-series ML
- Strict ordering is verifiable via `verify_no_leakage()`
- Metadata enables reproducibility and provenance tracking
- Schema validation ensures compatibility with label_builder output
- Optional purge gap handles cases where features have lookback windows

**Impact:**
- New `src/cryptoscreener/training/` module
- New `scripts/split_dataset.py` CLI
- Anti-leakage tests verify temporal ordering invariants
- Starts PRD Section 11 Milestone 3: "Training pipeline skeleton"

---

## DEC-013: Probability Calibration (PR-B)

**Date:** 2026-01-25

**Decision:** Implement Platt scaling calibration for model probability outputs per PRD §11 Milestone 3.

**Components:**
1. **Platt Module** (`src/cryptoscreener/calibration/platt.py`):
   - `PlattCalibrator` dataclass with a, b parameters
   - `fit_platt()` using gradient descent on cross-entropy
   - Platt's target adjustment for better calibration
   - Numerically stable sigmoid/logit transforms

2. **Artifact Module** (`src/cryptoscreener/calibration/artifact.py`):
   - `CalibrationMetadata` with schema_version, git_sha, config_hash, data_hash
   - `CalibrationArtifact` containing calibrators + metadata
   - `save_calibration_artifact()` / `load_calibration_artifact()`
   - Metrics before/after tracking (Brier, ECE)

3. **Calibrator Interface** (`src/cryptoscreener/calibration/calibrator.py`):
   - `Calibrator` protocol for pluggable implementations
   - `CalibratorMethod` enum (PLATT, future: ISOTONIC)
   - `fit_calibrator()` unified API

4. **CLI** (`scripts/fit_calibration.py`):
   - Input: validation split from PR-A (val.jsonl)
   - Output: calibration.json with calibrators + metadata
   - Reports Brier/ECE before and after calibration
   - Configurable heads, max_iter, learning rate

**Calibration Heads:**
- `p_inplay_30s` → `i_tradeable_30s_a`
- `p_inplay_2m` → `i_tradeable_2m_a`
- `p_inplay_5m` → `i_tradeable_5m_a`
- `p_toxic` → `y_toxic`

**Guarantees:**
- All calibrated probabilities remain in [0, 1]
- Monotonicity preserved for positive slope
- Deterministic: same input → same output
- Fail-fast on invalid input (empty, single class, mismatched lengths)
- **Anti-ranking-inversion:** Rejects calibrators with `a <= 0` (NegativeSlopeError)

**Negative Slope Handling (CRITICAL):**
- If fitted `a <= 0`, the calibrator would invert rankings (catastrophic for ranker)
- `fit_platt()` raises `NegativeSlopeError` by default when `a <= 0`
- CLI exits with code 1 and logs actionable error message
- Possible causes: anti-correlated model, data corruption, too small validation window
- Diagnostic mode: `reject_negative_slope=False` allows fitting for investigation only

**Metadata Tracking:**
- `schema_version`: Calibration schema version (1.0.0)
- `git_sha`: Git commit at calibration time (12 chars)
- `config_hash`: SHA256 of calibration configuration (16 chars)
- `data_hash`: SHA256 of validation data (16 chars)
- `metrics_before`/`metrics_after`: Brier, ECE per head

**Alternatives considered:**
1. scikit-learn CalibratedClassifierCV — rejected: heavy dependency
2. Isotonic regression — deferred: Platt sufficient for MVP
3. Temperature scaling — considered: simpler but Platt more flexible
4. Calibrate on train — rejected: must fit on val to avoid overfitting

**Rationale:**
- Platt scaling is standard for binary classifier calibration
- Pure Python avoids sklearn dependency
- Artifact storage enables reproducibility
- Metrics before/after demonstrate improvement
- Fit on validation set prevents calibration overfitting

**Impact:**
- New `src/cryptoscreener/calibration/` module
- New `scripts/fit_calibration.py` CLI
- 42 unit tests for Platt, artifacts, roundtrip, adversarial, negative slope rejection
- Continues PRD §11 Milestone 3: "Training pipeline skeleton"

---

## DEC-015: Baseline E2E Determinism Acceptance (PR#59)

**Date:** 2026-01-25

**Decision:** Add end-to-end replay determinism tests for the **BaselineRunner** pipeline path.

**Scope Clarification:**
- This PR tests the **BaselineRunner** path (heuristic-based predictions)
- MLRunner E2E acceptance will be a separate PR after PR#58 (MLRunner) is merged
- BaselineRunner is the production fallback when ML model is unavailable

**Goal:** Demonstrate that identical input produces identical output:
`FeatureSnapshot → BaselineRunner → Scorer → Ranker → RankEvent`

**Components:**

1. **E2E Determinism Test Suite** (`tests/replay/test_e2e_determinism.py`):
   - `TestE2EDeterminism`: Core determinism assertions (12 tests)
   - `TestE2EReplayProof`: Generates SHA256 proof artifacts
   - `TestE2EEdgeCases`: Empty, single-frame, single-symbol scenarios

2. **Fixture** (`tests/fixtures/replay_baseline/`):
   - Deterministic FeatureSnapshots for BTC/ETH/SOL
   - Fixed timestamps (BASE_TS = 2026-01-01 00:00:00 UTC)
   - 5 time frames with varying feature values

3. **Pipeline Function** (`run_e2e_pipeline`):
   - Explicitly uses `BaselineRunner` (not MLRunner)
   - Configures `RankerConfig(score_threshold=0.001)` to ensure events are generated
   - Asserts `len(events) > 0` to prevent empty digest regression
   - Validates runner type via `PredictionSnapshot.model_version`

**Digest Computation (deterministic serialization):**
```python
# From cryptoscreener/contracts/events.py:
def compute_rank_events_digest(events: list[RankEvent]) -> str:
    """Compute SHA256 of concatenated RankEvent JSON bytes."""
    data = b"".join(e.to_json() for e in events)
    return hashlib.sha256(data).hexdigest()

# to_json() uses msgspec which produces deterministic output
# (stable field order, no whitespace variations)
```

**Proof Format (corrected, with non-empty events):**
```
=== Replay Proof ===
Input fixture digest:  d80958017df4ea948e910f67f2662c4f838368632da49253c8d962d60d8881fd
RankEvent digest (r1): fc846d6aea56d76c16287e553b4f3d8bd579f3803788e264df84f0f041efc62c
RankEvent digest (r2): fc846d6aea56d76c16287e553b4f3d8bd579f3803788e264df84f0f041efc62c
Prediction digest:     eed9cf4967696a5b75b3c78c7604f9410a2b39986c37f28df03e815b6ca84e52
Total RankEvents: 3
Digests match: True
```

**Test Coverage:**
- Same input → same events: `test_same_input_produces_same_output`
- Stable digests: `test_rank_events_digest_stable`, `test_predictions_digest_stable`
- JSON roundtrip: `test_rank_event_json_roundtrip`, `test_prediction_snapshot_json_roundtrip`
- Proof generation: `test_generate_replay_proof`
- Different input → different output: `test_different_input_different_output`
- **Non-empty events assertion**: `test_produces_rank_events` (prevents empty digest regression)

**Runner Type Validation:**
- Test asserts `PredictionSnapshot.model_version == "baseline-v1"`
- Ensures tests are running against expected BaselineRunner, not accidentally MLRunner

**Why BaselineRunner first:**
1. BaselineRunner is the production fallback (critical path)
2. Fully deterministic (heuristic-based, no ML model variance)
3. MLRunner requires PR#58 merge; will be tested separately
4. MLRunner with `fallback_to_baseline=True` delegates here anyway

**Future work (separate PR):**
- MLRunner E2E Determinism Acceptance (after PR#58 merge, DEC number TBD)
- Will test calibrated ML predictions path
- Note: DEC-016 is reserved for Artifact Registry/Manifest per roadmap

**Alternatives considered:**
1. Wait for MLRunner merge — rejected: baseline path is critical and testable now
2. Compare event-by-event instead of digest — rejected: digest is more robust
3. Use real MarketEvents — deferred: FeatureSnapshots are the pipeline input

**Rationale:**
- Proves replay determinism for baseline fallback path per PRD requirement
- Enables CI/CD to catch non-determinism regressions
- Provides proof artifacts for audit trail
- Tests full pipeline integration without external dependencies or ML models

**Impact:**
- New `tests/replay/test_e2e_determinism.py` (12 tests)
- New `tests/fixtures/replay_baseline/` fixture module
- Extends existing replay test infrastructure
- Partially completes Milestone 3 replay acceptance (baseline path)

---

## DEC-016: Artifact Registry / Manifest (PR#60)

**Date:** 2026-01-25

**Decision:** Implement unified artifact manifest format with fail-fast validation per MODEL_REGISTRY_VERSIONING.md.

**Scope (Option A - Minimal):**
1. Unified manifest format (checksums.txt + manifest.json)
2. Schema validation with fail-fast
3. ModelPackage loader API (single entry point)
4. Tests for good/bad package scenarios

**Components:**

1. **Version Parsing** (`src/cryptoscreener/registry/version.py`):
   - `ModelVersion` frozen dataclass with semver, git_sha, data_cutoff, train_hash
   - `parse_model_version()` supports full format and baseline-v format
   - Pattern: `{semver}+{git_sha}+{data_cutoff}+{train_hash}`

2. **Manifest Management** (`src/cryptoscreener/registry/manifest.py`):
   - `Manifest` and `ArtifactEntry` dataclasses
   - `parse_checksums_txt()` / `generate_checksums_txt()` for checksums.txt format
   - `load_manifest()` / `save_manifest()` for dual-format support
   - `validate_manifest()` for hash/size verification
   - `compute_file_sha256()` for file hashing

3. **ModelPackage Loader** (`src/cryptoscreener/registry/package.py`):
   - `ModelPackage` dataclass with path, manifest, schema, features, version, calibrators
   - `load_package()` as single entry point with optional validation
   - `validate_package()` for comprehensive validation
   - `FeatureSpec` and `SchemaVersion` helper dataclasses

**checksums.txt Format (SSOT for hashes):**
```
# checksums.txt - SHA256 hashes for 1.0.0+abc1234+20260125+12345678
# Generated: 2026-01-25T12:00:00Z

abc123...  model.bin
def456...  calibrator_p_inplay_2m.pkl
```

**manifest.json Format (metadata + checksums):**
```json
{
  "schema_version": "1.0.0",
  "model_version": "1.0.0+abc1234+20260125+12345678",
  "created_at": "2026-01-25T12:00:00Z",
  "artifacts": [
    {"name": "model.bin", "sha256": "abc123...", "size_bytes": 12345}
  ]
}
```

**Required Artifacts (per MODEL_REGISTRY_VERSIONING.md):**
- `schema_version.json` — Package schema and model version
- `features.json` — Ordered feature list
- `checksums.txt` — SHA256 hashes for all artifacts

**Optional Artifacts:**
- `model.bin` — Trained model binary
- `calibrator_{head}.pkl` — Probability calibrators
- `training_report.md` — Training metrics

**Validation Behavior:**
- Fail-fast on SHA256 mismatch
- Fail-fast on size mismatch
- Fail-fast on missing required artifacts
- Schema version compatibility check
- Feature list exact match validation (optional)

**API Usage:**
```python
from cryptoscreener.registry import load_package

# Load with full validation
package = load_package("/path/to/model")
print(package.version.semver)  # "1.0.0"
print(package.features.features)  # ["spread_bps", "book_imbalance", ...]
print(package.calibrators)  # {"p_inplay_2m": Path(...)}

# Skip validation for development
package = load_package("/path/to/model", validate=False)

# Validate with expected schema/features
package = load_package(
    "/path/to/model",
    expected_schema_version="1.0.0",
    expected_features=["spread_bps", "book_imbalance"]
)
```

**Alternatives considered:**
1. Full registry with remote storage — deferred: local-first MVP sufficient
2. Only checksums.txt format — rejected: manifest.json provides richer metadata
3. Only manifest.json format — rejected: checksums.txt is human-readable SSOT
4. No validation — rejected: fail-fast prevents corrupted model usage

**Rationale:**
- Dual format provides both machine-readable metadata and human-readable checksums
- checksums.txt is SSOT for hashes (easy to verify manually: `sha256sum -c checksums.txt`)
- Fail-fast validation prevents silent corruption
- Single entry point (`load_package`) simplifies consumer code
- Compatible with MODEL_REGISTRY_VERSIONING.md specification

**Impact:**
- New `src/cryptoscreener/registry/` module with 4 files
- New `tests/registry/` with 73 tests
- Single API for loading model packages
- Foundation for future remote registry (Option B/C)

---

## DEC-018: Reason Codes SSOT Alignment

**Date:** 2026-01-25

**Decision:** Align all `RC_*` reason codes in implementation with `REASON_CODES_TAXONOMY.md` as the Single Source of Truth (SSOT).

**Problem:**
Implementation in `BaselineRunner` and `MLRunner` used ad-hoc reason code names that diverged from the canonical taxonomy:
- `RC_BOOK_PRESSURE` → not in SSOT
- `RC_TIGHT_SPREAD` → SSOT defines `RC_SPREAD_TIGHT`
- `RC_WIDE_SPREAD` → SSOT defines `RC_SPREAD_WIDE`
- `RC_TOXIC_RISK` → SSOT defines `RC_TOXIC_RISK_UP`
- `RC_HIGH_VOL` → SSOT defines `RC_REGIME_HIGH_VOL`

**Changes:**

1. **Code aligned to SSOT** (`baseline.py`, `ml_runner.py`):
   - `RC_BOOK_PRESSURE` → `RC_FLOW_IMBALANCE_LONG` / `RC_FLOW_IMBALANCE_SHORT` (direction-aware)
   - `RC_TIGHT_SPREAD` → `RC_SPREAD_TIGHT`
   - `RC_WIDE_SPREAD` → `RC_SPREAD_WIDE`
   - `RC_TOXIC_RISK` → `RC_TOXIC_RISK_UP`
   - `RC_HIGH_VOL` → `RC_REGIME_HIGH_VOL`

2. **SSOT updated** (`REASON_CODES_TAXONOMY.md`):
   - Added `### Gates (Trading Blockers)`: `RC_GATE_SPREAD_FAIL`, `RC_GATE_IMPACT_FAIL`
   - Added `### Data Quality`: `RC_DATA_STALE`
   - Added `### ML/Calibration (MLRunner-specific)`: `RC_CALIBRATION_ADJ`

3. **Tests updated** (`test_baseline.py`):
   - All assertions updated to expect SSOT-compliant codes

**Alternatives considered:**
1. Alias mapping layer (translate old→new) — rejected: adds complexity, masks the problem
2. Update only SSOT (add old names as aliases) — rejected: dilutes SSOT authority
3. Ignore divergence — rejected: violates SSOT principle, causes confusion

**Rationale:**
- SSOT must be authoritative; implementation diverging creates confusion
- Direction-aware flow imbalance codes (`_LONG`/`_SHORT`) are more informative
- Explicit gate failure codes improve observability
- Backward compatibility is NOT a concern (no production consumers yet)

**Impact:**
- All `RC_*` codes in codebase now match `REASON_CODES_TAXONOMY.md`
- Tests verify SSOT compliance
- Future additions must go through SSOT first

---

## DEC-019: MLRunner E2E Acceptance as CI Gate

**Date:** 2026-01-25

**Decision:** Add MLRunner E2E determinism tests to validate the MLRunner inference path produces stable, reproducible output as a CI gate.

**Problem:**
1. BaselineRunner E2E determinism tests exist (DEC-015), but MLRunner path not covered
2. Need to validate both DEV mode (fallback) and PROD mode (fail-safe) determinism
3. CI should fail if MLRunner path becomes non-deterministic

**MLRunner Determinism Contract:**
- **DEV mode** (no model): Falls back to BaselineRunner → deterministic heuristic output
- **PROD mode** (no model): Returns DATA_ISSUE with `RC_MODEL_UNAVAILABLE` → deterministic error state
- Both paths produce stable SHA256 digests across multiple runs

**Test Coverage:**

1. **DEV Mode Tests** (`TestMLRunnerDevModeDeterminism`):
   - Same input → same output (field-by-field comparison)
   - RankEvent digest stable across runs
   - PredictionSnapshot digest stable across runs
   - Pipeline produces at least 1 RankEvent
   - Runner uses fallback to baseline

2. **PROD Mode Tests** (`TestMLRunnerProdModeDeterminism`):
   - Same input → same output
   - Prediction digest stable across runs
   - All predictions are DATA_ISSUE with `RC_MODEL_UNAVAILABLE`
   - Runner has artifact error flag set

3. **Replay Proof Tests** (`TestMLRunnerE2EReplayProof`):
   - Generate DEV mode replay proof (2 runs, digest comparison)
   - Generate PROD mode replay proof (2 runs, digest comparison)
   - RankEvent JSON roundtrip
   - PredictionSnapshot JSON roundtrip

4. **Edge Cases** (`TestMLRunnerE2EEdgeCases`):
   - Empty fixture → no events
   - Single frame → deterministic
   - Single symbol → deterministic
   - Fixture digest is stable

**CI Integration:**
- Test file: `tests/replay/test_mlrunner_e2e_determinism.py`
- Triggered automatically by `acceptance_packet.sh` when PR touches `tests/replay/`
- Uses same fixture pattern as BaselineRunner tests

**Fixture:**
- 15 FeatureSnapshots across 3 symbols (BTCUSDT, ETHUSDT, SOLUSDT)
- 5 time points (0ms, 2000ms, 4000ms, 6000ms, 8000ms)
- Varied feature values to exercise ranker state transitions

**Real Model Mode** (`TestMLRunnerRealModelDeterminism`):
- Uses actual model.pkl + calibration.json fixtures from `tests/fixtures/mlrunner_model/`
- DeterministicModel class produces hash-based reproducible probabilities
- Verifies non-empty RankEvents (prevents empty digest regression)
- Verifies model_version reflects fixture artifact, not baseline
- Verifies calibration is applied (calibration_version contains fixture SHA)

**Security: Pickle Usage in Tests**

The test fixture uses pickle format (`model.pkl`) for compatibility with MLRunner's
existing model loading code. Security mitigations:

1. **Test-only scope**: Pickle is loaded ONLY in test code paths, never in production
2. **Path restriction**: `tests/fixtures/mlrunner_model/` is never referenced in prod configs
3. **Integrity verification**: SHA256 hash checked before loading (`MODEL_SHA256`)
4. **Minimal payload**: DeterministicModel contains only seed (int) and version (str)
5. **Source controlled**: Model class is in `deterministic_model.py`, not arbitrary code

Future consideration: Replace pickle with ONNX or JSON-based model format for
enhanced security (no arbitrary code execution risk).

**Rationale:**
- Determinism is critical for replay verification and debugging
- MLRunner path must be tested before production use
- DEV, PROD, and Real Model modes all have deterministic contracts
- CI gate prevents regression in determinism
- Real model tests prove actual inference path is deterministic

**Impact:**
- New `tests/replay/test_mlrunner_e2e_determinism.py` (27 tests)
- Total tests: 754+
- MLRunner path validated as deterministic in all three modes
- CI gate for MLRunner changes

---

## DEC-017: Production Profile Gates

**Date:** 2026-01-25

**Decision:** Add `InferenceStrictness` enum to control fail-fast behavior in DEV vs PROD modes, and align data freshness thresholds with SSOT.

**Problem:**
1. Data freshness thresholds in implementation (5000ms/30000ms) diverged from SSOT (`DATA_FRESHNESS_RULES.md`: 1000ms/2000ms)
2. No distinction between development and production strictness
3. MLRunner exceptions could crash the pipeline in production

**Changes:**

1. **InferenceStrictness enum** (`base.py`):
   ```python
   class InferenceStrictness(str, Enum):
       DEV = "dev"   # Lenient: fallback allowed, calibration optional
       PROD = "prod" # Strict: fail-safe, never fallback, never TRADEABLE without model
   ```

2. **Data freshness thresholds** (configurable, SSOT defaults):
   - `stale_book_max_ms`: 1000 (per `DATA_FRESHNESS_RULES.md`)
   - `stale_trades_max_ms`: 2000 (per `DATA_FRESHNESS_RULES.md`)

3. **PROD mode behavior** (fail-safe, not fail-fast):
   - Missing model → `DATA_ISSUE` status with `RC_MODEL_UNAVAILABLE`
   - Missing calibration → `DATA_ISSUE` status with `RC_CALIBRATION_MISSING`
   - Hash mismatch → `DATA_ISSUE` status with `RC_ARTIFACT_INTEGRITY_FAIL`
   - Never raises exceptions (deterministic, auditable)
   - Never returns `TRADEABLE` without valid model+calibration

4. **DEV mode behavior** (lenient, default):
   - `fallback_to_baseline` honored (default True)
   - `require_calibration` honored (default True)
   - Model/calibration errors → fallback to BaselineRunner

**Behavior Matrix:**

| Condition | DEV Mode | PROD Mode (Reason Code) |
|-----------|----------|-------------------------|
| Model missing | BaselineRunner fallback | DATA_ISSUE (`RC_MODEL_UNAVAILABLE`) |
| Calibration missing | Raw probs (if allowed) | DATA_ISSUE (`RC_CALIBRATION_MISSING`) |
| Hash mismatch | Fallback (if allowed) | DATA_ISSUE (`RC_ARTIFACT_INTEGRITY_FAIL`) |
| Book stale > threshold | DATA_ISSUE | DATA_ISSUE |
| Trades stale > threshold | DATA_ISSUE | DATA_ISSUE |

**New Reason Codes (per REASON_CODES_TAXONOMY.md):**
- `RC_MODEL_UNAVAILABLE` — model artifact missing or failed to load
- `RC_CALIBRATION_MISSING` — calibration artifact missing or failed to load
- `RC_ARTIFACT_INTEGRITY_FAIL` — artifact hash mismatch (SHA256 verification failed)

**SSOT Updates:**
- `REASON_CODES_TAXONOMY.md`: Added artifact error codes section
- `CONFIG_SPEC.md`: Added `models.inference_strictness` key

**Alternatives considered:**
1. Raise exceptions in PROD → rejected: can crash pipeline, hard to debug
2. Different threshold profiles → rejected: overcomplicates, thresholds are per SSOT
3. Always require calibration → rejected: need flexibility for development

**Rationale:**
- PROD mode must never silently proceed without valid artifacts
- Returning `DATA_ISSUE` instead of exceptions ensures deterministic, auditable behavior
- SSOT-aligned thresholds prevent configuration drift
- DEV mode allows rapid iteration without strict artifact requirements

**Impact:**
- `InferenceStrictness` enum added to `base.py`
- `MLRunnerConfig.strictness` field (default: DEV)
- Data freshness thresholds configurable in `ModelRunnerConfig`
- 12 new tests for PROD/DEV behavior matrix (7 strictness + 5 freshness)
- 728 total tests pass

---

## DEC-014: MLRunner with Calibration Integration (PR-C)

**Date:** 2026-01-25

**Decision:** Implement MLRunner as production model runner with calibration integration per PRD §11 Milestone 3.

**Components:**
1. **MLRunner Class** (`src/cryptoscreener/model_runner/ml_runner.py`):
   - Inherits from `ModelRunner` base class
   - Loads model artifact (pickle/joblib/ONNX)
   - Loads calibration artifact from DEC-013
   - Applies calibration to raw model probabilities
   - Falls back to BaselineRunner when model unavailable

2. **MLRunnerConfig**:
   - `model_path`: Path to model artifact
   - `calibration_path`: Path to calibration artifact JSON
   - `require_calibration`: If True, fail if calibration unavailable (default: True)
   - `fallback_to_baseline`: If True, use baseline on model failure (default: True)

3. **Calibration Application**:
   - MLRunner applies `CalibrationArtifact.transform()` to each raw probability
   - Calibrated probabilities flow to `PredictionSnapshot`
   - Downstream consumers (Scorer, Ranker, Alerter) use calibrated values
   - No changes needed in downstream modules

**Probability Flow:**
```
FeatureSnapshot → MLRunner._run_inference() → raw probs
                → MLRunner._calibrate() → calibrated probs
                → PredictionSnapshot
                → Scorer.score() uses calibrated probs
                → Ranker ranks by calibrated scores
```

**Fallback Behavior:**
1. Model unavailable + `fallback_to_baseline=True` → uses BaselineRunner
2. Calibration unavailable + `require_calibration=False` → uses raw probabilities
3. Calibration unavailable + `require_calibration=True` → raises CalibrationArtifactError
4. Per-head calibrator missing → uses raw probability for that head

**Error Types:**
- `ModelArtifactError`: Model file not found or invalid format
- `CalibrationArtifactError`: Calibration required but unavailable

**Supported Model Formats:**
- `.pkl`: Python pickle (scikit-learn models)
- `.joblib`: Joblib serialization
- `.onnx`: ONNX runtime inference

**Replay Determinism:**
- Same input + same config → identical PredictionSnapshot JSON
- `MLRunner.compute_digest()` includes model_version + calibration_version
- Batch prediction matches sequential prediction
- Tests verify SHA256 hash stability across runs

**Ranker Integration:**
- Scorer consumes `PredictionSnapshot.p_inplay_*` and `p_toxic` directly
- These contain **calibrated** probabilities from MLRunner
- No changes to Scorer or Ranker code needed
- Calibration is transparent to downstream

**Alternatives considered:**
1. Calibrate in Scorer — rejected: calibration is model-specific, belongs in runner
2. Separate CalibrationRunner wrapper — rejected: adds complexity, single runner simpler
3. Always require calibration — rejected: need flexibility for development/testing
4. Async model loading — rejected: startup-time loading is sufficient

**Rationale:**
- MLRunner is drop-in replacement for BaselineRunner
- Calibration integrated at inference time for correctness
- Fallback ensures graceful degradation during development
- Deterministic for reproducible backtests

**Artifact Integrity (added in review):**
- `MLRunnerConfig.model_sha256` and `calibration_sha256` fields for expected hashes
- `ArtifactIntegrityError` raised on hash mismatch (or fallback if configured)
- Hash verification is case-insensitive, computed via SHA256
- Tests: `TestMLRunnerArtifactIntegrity` (7 tests)

**Evidence No-Digits Policy (added in review):**
- `ReasonCode.evidence` field must NOT contain digits (0-9)
- Numbers belong in the `value` field; evidence is for LLM consumption
- Enforces "no-new-numbers" policy for downstream LLM consumers
- Tests: `TestEvidenceNoDigitsPolicy` (4 tests)

**Known Limitations (future work):**
- **Artifact registry not implemented:** While hash verification exists, there's no central
  manifest/registry that maps model versions to expected SHA256 hashes. Currently hashes
  must be manually configured in `MLRunnerConfig`.
- **LLM not involved:** MLRunner `reasons` field contains deterministic `ReasonCode` objects
  built from feature values, NOT LLM-generated text. The `evidence` field is a template-based
  string without numbers. This is intentional separation per DEC-004/DEC-005.

**Impact:**
- New `src/cryptoscreener/model_runner/ml_runner.py`
- Extended `src/cryptoscreener/model_runner/__init__.py` exports
- 32 unit tests for fallback, calibration, determinism, gates, artifact integrity, evidence policy
- Completes MLRunner portion of PRD §11 Milestone 3
