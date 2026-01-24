# UI Feed Spec

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Transport
- WebSocket server (local or remote) emitting `RankEvent` deltas (see DATA_CONTRACTS)

## Payload rules
- Only deltas; client can request snapshot on connect
- Heartbeat every 5s
- Backpressure: drop low-priority events if client slow

## Frequency
- Rank deltas no more than 1Hz unless major change
- Symbol detail updates: on demand or capped 2Hz

## Security
- If remote: token auth; no exchange keys ever transmitted
