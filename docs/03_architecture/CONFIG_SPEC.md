# Config Spec

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


All config is loaded from `/configs/*.yaml`.

## Core keys
- universe:
  - mode: volume|oi|manual
  - top_n: 200
  - exclude: []
  - refresh_s: 900
- ws:
  - max_streams_per_conn: 800
  - depth_interval_ms: 250
  - max_reconnect_per_min: 10
- features:
  - snapshot_interval_ms: 1000
  - windows: [1,10,60,300]
- scoring:
  - profileA: {fees_bps: ..., spread_max_bps: ..., impact_max_bps: ...}
  - profileB: {...}
  - toxicity_max: 0.6
  - weights: {w30:0.4, w2m:0.4, w5m:0.2}
- llm:
  - enabled: false
  - provider: openai|local
  - max_chars: 180
