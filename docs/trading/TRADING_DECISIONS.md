# Trading/VOL Harvesting v2 — Decisions (SSOT)

**Format:** TRD-XXX  
**Rule:** Any behavior/contract change must be captured here first.

---

## TRD-001 — v2 Scope and SSOT Boundary (RankEvent only)
**Date:** 2026-01-29  
**Status:** Draft  

### Decision
Trading/VOL Harvesting v2 is a separate subproject. It consumes `RankEvent` as its only SSOT input and does not modify v1 behavior, metrics, or contracts.

### Rationale
Prevents scope creep into v1 and keeps v2 iteration fast.

### Consequences
- v2 contracts live under `docs/trading/` (or root `DATA_CONTRACTS.md` references them)
- Any need to change RankEvent schema triggers a root-level DEC update

---

## TRD-002 — LLM boundaries (advisory only, no-new-numbers)
**Date:** 2026-01-29  
**Status:** Draft  

### Decision
If an LLM is used, it may only select from a finite set of **enum presets** / qualitative labels. It must not introduce or modify numbers, probabilities, thresholds, or scores.

### Rationale
Keeps decisions deterministic and audit-friendly.

### Consequences
- Schema validation + no-new-numbers guard
- Deterministic fallback templates

---

## TRD-003 — Simulator determinism as a release gate
**Date:** 2026-01-29  
**Status:** Draft  

### Decision
Replay/simulator must be deterministic: same inputs ⇒ same journal outputs (within defined tolerances), with fixtures checksummed.

### Rationale
Enables trustworthy evaluation and prevents hidden regressions.

### Consequences
- Fixture checksum guard
- Replay determinism CI gate
