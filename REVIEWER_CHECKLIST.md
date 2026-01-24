# REVIEWER_CHECKLIST.md
One-page checklist for reviewing Claude’s PRs in CryptoScreener‑X.

---

## 0) Gate: SSOT & Scope
- [ ] Change is within PRD/SPEC scope. If not: DECISIONS.md updated + SPEC/STATE/CHANGELOG updated.
- [ ] No “drive-by refactors” or extra features.
- [ ] Versioning: any behavior/interface change is documented.

---

## 1) Proof bundle (NO PROOF = NOT DONE)
Required in every update:
- [ ] `git diff` or commit hash
- [ ] raw outputs: `ruff check .`, `mypy .`, `pytest -q`
- [ ] logs: `run_live` and/or `run_replay` (as applicable)
- [ ] sample JSON payloads matching DATA_CONTRACTS + schema validation output
- [ ] replay determinism evidence (RankEvent digest/hash)
- [ ] sha256 for fixtures/model artifacts (if touched)

If any item missing → CHANGES REQUIRED.

---

## 2) Contracts & Interfaces (DATA_CONTRACTS.md)
- [ ] MarketEvent/FeatureSnapshot/PredictionSnapshot/RankEvent match schema exactly (fields, types, enums).
- [ ] Roundtrip tests exist (to_json/from_json).
- [ ] Backward compatibility or explicit schema_version bump (with migration note).
- [ ] No ad-hoc dicts between modules.

---

## 3) Binance safety (BINANCE_LIMITS + binance docs)
### WS
- [ ] Sharding implemented (streams/conn <= headroom target; never near 1024).
- [ ] Subscribe/unsubscribe throttled; no control-message storms.
- [ ] Reconnect: exponential backoff + jitter; max reconnect/min enforced.
- [ ] Staleness detection: book/trades stale gates trigger DATA_ISSUE.

### REST
- [ ] No high-frequency polling for market data.
- [ ] ApiGovernor: budgets + backoff + circuit breaker.
- [ ] 429/418/-1003 handling: immediate cooldown, no “fight the limiter”.

---

## 4) Determinism & Replay (REPRODUCIBILITY + TEST_PLAN)
- [ ] Same feature library online/offline (no drift in formulas).
- [ ] Replay produces same RankEvent sequence within tolerance.
- [ ] Fixtures are hashed and verified.
- [ ] Time in replay driven by event timestamps, not wall clock.

---

## 5) Feature engine correctness (FEATURES_CATALOG)
- [ ] Feature formulas + windows match catalog.
- [ ] Missing data handling specified and tested.
- [ ] Clipping/normalization deterministic.
- [ ] Performance: O(1) updates with ring buffers; bounded memory.

---

## 6) ML correctness (LABELS_SPEC / COST_MODEL_SPEC / CALIBRATION_SPEC)
- [ ] Labels implement net_edge after costs; no leakage.
- [ ] Cost model uses spread+fees+impact and respects execution profiles.
- [ ] Calibration applied per head (isotonic/Platt); metrics reported (ECE/Brier).
- [ ] Model artifacts versioned (semver+git_sha+hash) + checksums.
- [ ] Baseline mode works if artifacts missing (and has tests).

---

## 7) LLM strictness (LLM_ROLE_POLICY + LLM_*_SCHEMA)
- [ ] LLM is **text-only**: cannot alter numbers, score, or status.
- [ ] Output validated: JSON schema, enum status_label, max length.
- [ ] “No-new-numbers” validator present + adversarial tests.
- [ ] Fallback deterministic templates used on any invalid output or outage.

---

## 8) Ranker UX behavior (HYSTERESIS_SPEC + ALERTING_SPEC)
- [ ] Hysteresis: enter/exit thresholds + dwell time implemented & tested.
- [ ] Alerts: cooldown/dedupe/global cap implemented & tested.
- [ ] Event rate bounded (no alert spam, no flicker).
- [ ] RankEvent deltas, not full spam snapshots.

---

## 9) Observability (OBSERVABILITY + SLO_SLI)
- [ ] Metrics: latency hist, drop rate, stale ms, reconnect count, 429/418 counters.
- [ ] Logs structured, no secrets.
- [ ] Incident playbooks referenced (ERROR_CODES_PLAYBOOK / RUNBOOK).

---

## 10) Verdict rules
- ACCEPT only if: scope OK + proof bundle complete + all relevant checklist sections pass.
- CHANGES REQUIRED if: any missing proof, contract mismatch, Binance-safety risk, replay nondeterminism, or LLM guardrail gap.
- REJECT if: repeated noncompliance, unsafe limiter behavior, or untracked scope expansion.

---

## Quick response template (copy/paste)
Verdict: ACCEPT / CHANGES REQUIRED / REJECT  
Findings:
- …
Required proof bundle items:
- …
Concrete next steps:
1) …
2) …
3) …
