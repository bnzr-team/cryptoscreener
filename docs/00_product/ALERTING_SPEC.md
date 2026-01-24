# Alerting Spec

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Event types
- `ALERT_TRADABLE`: status becomes TRADEABLE (after gates)
- `ALERT_TRAP`: status becomes TRAP
- `ALERT_ENTER_TOPK`: enters top‑K (combined score)
- `ALERT_DATA_ISSUE`: critical data health issue persists > T seconds

## Anti-spam controls
- Cooldown per symbol per event type: default 120s
- Hysteresis: require stable state for `stable_ms` before firing alert
- Max alerts/min global cap (safety)

## Telegram payload
- Title: “{SYM} — {STATUS}”
- Line1: headline (LLM) or deterministic
- Line2: “p2m={p} util={u}bps spread={s}bps tox={t}”
- Line3: top reason codes (2–3)
- Attachment: chart image (mplfinance) optional

## Tests
- Unit: dedupe, cooldown, hysteresis transitions
- Replay: ensure alert count within expected bounds
