# Runbook (On-call)

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Incident: 429 spike
1. Confirm ApiGovernor OPEN
2. Reduce REST to minimum
3. Verify WS stable
4. Record incident report

## Incident: 418 ban
1. SAFE MODE
2. Stop REST, reduce WS
3. Wait ban-until
4. Postmortem: why did we continue after 429?

## Incident: reconnect storm
1. Increase backoff cap
2. Reduce streams
3. Check network and Binance status
