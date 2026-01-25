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
