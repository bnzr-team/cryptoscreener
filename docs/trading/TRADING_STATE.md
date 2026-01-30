# Trading/VOL Harvesting v2 — STATE

**Status:** Draft
**Date:** 2026-01-30

## Milestones
- M0: Docs hardening + SSOT gates ✅
- M1: Contracts + simulator fixtures ✅
- M2: Paper mode end-to-end (RankEvent → intents → sim fills → PnL) — in progress
- M3: Live execution safety layer (separate decision)

## Done
- **DEC-040:** Trading v2 docs pack (SSOT templates)
- **DEC-041:** Simulator fixtures + ScenarioRunner (4 fixtures, deterministic replay)
- **DEC-042:** Strategy Interface + Contracts (StrategyDecision, BaselineStrategy)
- **DEC-043:** Policy Library docs (20 POL rules, Action vocabulary, Risk/Cost model)

## In progress
- **DEC-044:** Policy Engine MVP (next)

## Blocked
- None

## Known issues
- Binance trading endpoints limits not yet verified in repo SSOT
