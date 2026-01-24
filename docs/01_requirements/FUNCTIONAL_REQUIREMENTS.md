# Functional Requirements

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


This document is normative; implementation must satisfy all items.

## FR-1 Universe selection
- Support dynamic universe (top-N by volume/OI) with configurable exclusions.
- Universe recalculated at low cadence (e.g., 5–15 min) via REST.

## FR-2 Market data ingestion
- WS combined streams for: trades, depth, miniTicker/mark, klines (configurable).
- Normalize into `MarketEvent` contract.

## FR-3 Feature snapshots
- Emit `FeatureSnapshot` per symbol on cadence.
- Deterministic features: offline and online must match given same events.

## FR-4 ML inference + calibration
- Predict p_inplay for 30s/2m/5m + expected_utility_bps + p_toxic.
- Apply calibrators per head.
- If no artifacts: run baseline heuristics (documented in SPEC).

## FR-5 Scoring and gates
- Compute costs per profile A/B; gate TRADEABLE if constraints violated.
- Output `PredictionSnapshot` with reasons.

## FR-6 Ranker
- Maintain top‑K; apply hysteresis rules; emit `RankEvent`.

## FR-7 Explainability
- Deterministic reason codes always present.
- LLM narrative optional; must follow strict schema and “no new numbers”.

## FR-8 Notifications
- Telegram notifier that consumes RankEvents.

## FR-9 Storage/Replay
- Record raw events and allow deterministic replay.
