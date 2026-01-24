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
5. **No digits in llm_text:** LLM prompt instructs model to avoid numeric values in output text. This eliminates risk of number formatting mismatches (trailing zeros, percentages, locale separators). Numbers are visible in `payload.prediction` — no need to repeat in text.

**Alternatives considered:**
1. Global cooldown across all symbols — rejected: would throttle unrelated symbols
2. Fetch spread/impact from FeatureSnapshot — deferred: requires pipeline plumbing
3. Make LLM mandatory — rejected: system must work without LLM
4. Allow digits if exact string match — rejected: too fragile, LLM may reformat numbers unpredictably

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
