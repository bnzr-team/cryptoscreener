# Hysteresis Spec (Anti-flicker)

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Goals
- Avoid rank/status flicker that kills usability.
- Keep responsiveness to genuine regime changes.

## Rules
- Enter TRADEABLE if condition holds continuously for `enter_ms` (default 1500ms)
- Exit TRADEABLE only if condition fails for `exit_ms` (default 3000ms)
- Min dwell time in any state: `min_dwell_ms` (default 2000ms)
- Rank churn: top-K updates limited to 1Hz unless score delta > `burst_delta`

## Tests
- Property-based test with random score streams; ensure bounded event rate.
