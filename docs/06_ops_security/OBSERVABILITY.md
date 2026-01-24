# Observability

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Logging
- Structured JSON logs with fields: ts, module, symbol, event, severity, correlation_id
- Separate audit log for rate-limit incidents

## Metrics
- ws_messages_total, ws_reconnects_total
- stale_book_ms (hist)
- e2e_latency_ms (hist)
- dropped_events_total
- api_429_total, api_418_total

## Dashboards
- Latency & throughput
- Data health
- Rate-limits
- Top-K churn
