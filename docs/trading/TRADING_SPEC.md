# Trading/VOL Harvesting v2 — SPEC (Invariants & Acceptance)

**Status:** Draft  
**Date:** 2026-01-29  

## Invariants (hard)
- Units:
  - NATR is a **fraction** (e.g., 0.015), not percent
  - Fees are **fractional** (e.g., 0.0004), not percent
- Money/quantities represented with Decimal where required
- v2 consumes RankEvent only (SSOT boundary)
- No new numbers may be introduced by LLM (if used)

## Trading state machine (high level)
- FLAT → ENTERING → OPEN → EXITING → FLAT
- Transitions must be explicit, logged, and journaled

## Risk safety gates (MVP)
- Per-symbol cooldown (anti-churn)
- Global cap (max actions/alerts per minute)
- Daily loss limit / kill-switch (paper mode first)
- Exposure caps per symbol and total

## Determinism requirements
- Simulator/replay must be deterministic on fixed fixtures
- Fixtures and expected outputs must be checksummed
- Any nondeterminism must be documented + tolerated explicitly

## Acceptance criteria (v2 docs stage)
- All v2 docs exist and are indexed
- Scope boundary rules are explicit and enforceable
- Formal root SSOT updates for v2 scope are planned (or done)
