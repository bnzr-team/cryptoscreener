# Trading/VOL Harvesting v2 — Scope, Boundary, SSOT

**Status:** Draft  
**Date:** 2026-01-29  
**Boundary:** RankEvent (CryptoScreener v1) is the **only** SSOT input.

## Scope statement
Trading/VOL Harvesting v2 is a standalone subproject that **consumes RankEvents** and produces:
- Trading decisions (intents)
- Simulated/live execution reports
- Journals / PnL summaries

It is **not** an extension of CryptoScreener v1 detection pipeline.

## SSOT boundary
- **Input SSOT:** `RankEvent` (from v1)
- v2 must treat RankEvent as read-only and must not infer or invent new numeric fields that are not present in RankEvent.

## Forbidden dependencies (hard)
v2 code and docs MUST NOT import or depend directly on v1 internal modules such as:
- scoring / ranker / alerter / feature engine internals
- any "hidden" live pipeline state

Allowed:
- RankEvent contract (read-only)
- public v1 utilities (only if explicitly marked stable and contract-safe)

## Required formal actions before Phase 1 implementation
Before any trading code implementation begins, the repo must contain:
1) Root `DECISIONS.md`: new DEC record formalizing v2 scope & boundary
2) Root `PRD.md`: non-goals updated to keep trading/execution out of v1 scope
3) `BINANCE_LIMITS.md`: trading endpoints weights/limits added and verified
4) `DATA_CONTRACTS.md`: trading contracts included (or linked to this folder as SSOT)
5) Root `CHANGELOG.md`: entry noting v2 docs/spec start

## Acceptance criteria (scope)
✅ Pass if:
- v2 only consumes RankEvent and does not require changes to v1 schema/contracts
- docs clearly separate v1 vs v2 responsibilities

❌ Fail (scope violation) if:
- v2 requires modifying v1 RankEvent schema/fields
- v2 requires embedding trading logic inside v1 pipeline
- v2 pulls market data via high-frequency REST polling

## Open questions (to be decided in TRD-XXX)
- Which RankEvent fields are mandatory for v2 gating policies?
- What is the minimal RankEvent “event identity” for dedupe?
