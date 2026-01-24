# Features Catalog

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


Each feature must define: name, formula, window, unit, clipping, missing handling.

## Core microstructure
- spread_bps = (ask-bid)/mid*1e4
- microprice = (ask*bid_size + bid*ask_size)/(bid_size+ask_size)
- book_imbalance = (sum_bid_depth - sum_ask_depth)/(sum_bid_depth+sum_ask_depth)
- depth_slope = regression slope of depth vs price levels
- flow_imbalance_w = (buy_vol - sell_vol)/(buy_vol + sell_vol) for window w
- trade_intensity = trades_count / window_s
- realized_vol_w = std(log returns) * sqrt(annualization optional)

## Regime/context
- natr_14_5m (if using kline features)
- btc_beta_short (corr vs BTC returns over 60s)

## Liquidity/impact proxies
- impact_bps_q = estimated price impact for clip Q (see COST_MODEL_SPEC)

## Standardization
- Store raw + z-scored per symbol (rolling median/MAD)
