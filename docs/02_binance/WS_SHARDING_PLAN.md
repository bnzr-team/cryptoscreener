# WS Sharding Plan

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Goal
Stay under 1024 streams/conn and under practical msg-rate per conn.

## Streams to consider (configurable)
- trades/aggTrades (per symbol)
- depth@100ms or depth@250ms (choose carefully)
- markPrice@1s (optional)
- kline@1m (optional for context)

## Sharding algorithm (v1)
1. Build list of required stream names for current universe.
2. Partition into chunks of size `max_streams_per_conn` (default 800).
3. For each chunk: open a WS connection with combined streams URL.
4. Subscribe in batches (e.g., 50 streams per SUB message) with throttling.

## Health & rebalancing
- If connection sees high lag/drops: reduce load (drop lowest priority symbols)
- Periodically rebalance when universe changes (but rate-limit SUB/UNSUB).
