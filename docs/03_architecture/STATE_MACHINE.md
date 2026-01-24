# Symbol State Machine

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## States
- DEAD
- WATCH
- TRADEABLE_SOON
- TRADEABLE
- HOT_BUT_DIRTY
- TRAP
- DATA_ISSUE

## Inputs
- p_inplay (per horizon)
- expected_utility_bps
- p_toxic
- gates (spread/impact/freshness)

## Transition rules (normative)
1. If DATA_ISSUE gate fails → DATA_ISSUE (sticky for min 5s)
2. Else if p_toxic >= T_trap → TRAP
3. Else if utility>=U_min and p_inplay>=P_tradeable and all gates pass → TRADEABLE
4. Else if p_inplay>=P_soon and utility>=U_soon → TRADEABLE_SOON
5. Else if p_toxic>=T_dirty and p_inplay high → HOT_BUT_DIRTY
6. Else WATCH
7. DEAD when utility <= 0 or costs dominate

All thresholds come from config.
