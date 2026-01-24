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

## DEC-002: BaselineRunner Simplified Status Logic

**Date:** 2026-01-24

**Decision:** BaselineRunner uses simplified heuristic-based status classification without full utility-gate state machine from architecture docs.

**Simplifications:**
1. **Status derived from two thresholds only:**
   - `p_toxic >= 0.7` → TRAP (hard gate, checked first)
   - `p_inplay >= 0.6` → TRADEABLE
   - `p_inplay >= 0.3` → WATCH
   - else → DEAD

2. **Data health as hard gate:**
   - `stale_book_ms > 5000` → DATA_ISSUE
   - `stale_trades_ms > 30000` → DATA_ISSUE
   - `missing_streams` not empty → DATA_ISSUE

3. **Expected utility computed but not used for gating:**
   - `expected_utility_bps_2m` is calculated and included in output
   - Not used for TRADEABLE/WATCH thresholds (future ML runner will use it)

**Alternatives considered:**
1. Full utility-based state machine from STATE_MACHINE.md
2. Hard spread/impact gates before TRADEABLE
3. Hysteresis for state transitions

**Rationale:**
- Baseline mode is for initial deployment without trained ML models
- Heuristic approach is transparent and debuggable
- Configuration allows threshold tuning without code changes
- Future MLRunner can implement full state machine with calibrated models

**Impact:**
- BaselineRunner is deterministic and reproducible
- `compute_digest()` captures all thresholds for replay verification
- All status transitions are testable via unit tests
- Clear upgrade path to MLRunner with more sophisticated logic
