# Trading/VOL Harvesting v2 — Docs Index (SSOT)

**Status:** Draft  
**Owner:** TBD  
**Last updated:** 2026-01-29  

## Purpose
This folder contains the v2 (Trading/VOL Harvesting) documentation pack. v2 is a **separate subproject** that **consumes RankEvent as SSOT boundary** and does **not** extend CryptoScreener v1 scope.

## Source of Truth rules
- Any change of behavior/contracts/constraints must go through:
  1) `TRADING_DECISIONS.md` (new TRD-XXX record)
  2) Update `TRADING_SPEC.md`, `TRADING_STATE.md`, `TRADING_CHANGELOG.md` as needed
- v2 MUST NOT modify v1 RankEvent schema/contracts without a root-level DEC.

## Stop conditions
Stop and escalate (new decision required) if any of the below is needed:
- Change RankEvent schema or any v1 data contract
- Add new Prometheus metrics/alerts in v1 as a dependency
- New Binance high-frequency REST polling for market data (must be WS-first)

## Document map
| Doc | Purpose | Status |
|---|---|---|
| `01_SCOPE_BOUNDARY_SSOT.md` | v1/v2 boundary, forbidden deps, formal gates | Draft |
| `03_CONTRACTS.md` | v2 data contracts (SSOT), JSON examples, test plan | Draft |
| `TRADING_DECISIONS.md` | v2 decisions (TRD-XXX) | Draft |
| `TRADING_SPEC.md` | invariants/acceptance/spec | Draft |
| `TRADING_STATE.md` | current status, milestones, known issues | Draft |
| `TRADING_CHANGELOG.md` | v2 changelog | Draft |

## PR links
- PR: TBD
