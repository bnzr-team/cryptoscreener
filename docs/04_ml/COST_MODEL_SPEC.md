# Cost Model Spec

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Components
cost_bps = spread_bps + fees_bps + impact_bps(Q)

### spread_bps
Compute from best bid/ask.

### fees_bps
Configurable per profile:
- Profile A (maker-ish): lower
- Profile B (taker-ish): higher
(Exact fee schedule can be configured; do not hardcode.)

### impact_bps(Q)
Approximate using orderbook depth:
- Determine price levels needed to fill quote size Q
- Convert implied slippage to bps
- Clip to sane range to avoid outlier explosions

## Clip size
Q_usd = k * usd_volume_60s
Defaults:
- scalping k=0.01
- intraday k=0.03

## Validation
- Backtest sanity checks: impact_bps increases when depth decreases
- Unit tests with synthetic books
