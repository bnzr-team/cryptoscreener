# Labels Spec

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Purpose
Create ground truth for p_inplay and utility that matches execution reality.

## Step-by-step labeling (per symbol, time t)
For each horizon H in {30s,2m,5m} and profile in {A,B}:
1. Compute spread_bps(t) and estimate fees_bps(profile).
2. Determine clip size:
   - Q_usd = k * usd_volume_60s(t)  (k differs by style)
3. Estimate impact_bps(profile, t, Q_usd) via depth/impact model.
4. cost_bps = spread_bps + fees_bps + impact_bps
5. Compute realized favorable move:
   - MFE_bps(H) = max_{t..t+H}( (price - entry_price)/entry_price * 1e4 )
   - For short, also compute MAE if needed.
6. net_edge_bps(H) = MFE_bps(H) - cost_bps
7. I_tradeable(H) = 1[ net_edge_bps(H) >= X_bps(profile,H) ] AND gates_pass

## Toxicity labels (p_toxic)
Define an “adverse selection” event:
- After hypothetical entry at t, within τ (e.g., 10–30s) price moves against by > Y_bps
- Or spread widens + reversal signature after flow spike

Store:
- y_toxic = 1/0
- severity_toxic_bps (optional)

## Dataset splitting
Time-based split; no leakage.
