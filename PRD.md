# PRD — In‑Play Predictor (CryptoScreener‑X) — ML + LLM
**Version:** 1.0 (enterprise implementation spec)  
**Date:** 2026-01-24  
**Source:** User concept draft “In‑Play Predictor for Perpetual Futures (Tradeability ‘from the future’)” (attached).  

---

## 0. Executive summary

We will build a real‑time ranking system for Binance **USD‑M perpetual futures** that predicts **near‑future tradeability** per symbol (seconds–minutes) and outputs:

- `p_inplay_30s`, `p_inplay_2m`, `p_inplay_5m` — **calibrated** probabilities that an opportunity meeting execution constraints will occur
- `expected_utility_bps` — expected tradable edge **after** spread, fees, and impact
- `p_toxic` — probability of adverse selection / “toxic flow”
- `p_inplay` — single scalar score used for ranking + alerting
- `explain` — human‑readable reasons (reason codes + LLM narrative)

Core concept: **net edge after costs** is the target; the model is forbidden to “hallucinate opportunity” when execution is not feasible.

---

## 1. Goals and non‑goals

### 1.1 Goals
1. Real‑time ranking of 200–400+ perp symbols with latency target **< 250 ms** end‑to‑end.
2. Produce stable top‑N list (10–30 symbols) for scalping and intraday attention allocation.
3. Provide **explainability** and “why this is tradable now” at a glance.
4. Use ML to predict opportunity probability + expected edge; use LLM to:
   - convert model/execution diagnostics into short, UX‑friendly narratives
   - generate consistent microcopy/status labels/tooltips
   - assist debugging (summarize anomalies, drift, data issues)
5. Robustness to API limits, disconnects, volatility spikes, and data quality issues.

### 1.2 Non‑goals (v1)
- Fully automatic trading/execution. (We only score + alert; later can integrate trader/bot.)
- Cross‑exchange arbitrage. (Future optional.)
- Spot markets. (v1 focuses on USD‑M perps.)

---

## 2. Personas and user stories (v1)

**Primary user:** experienced scalper/intraday trader monitoring fast markets.

User stories:
1. “Show me the **next** symbols likely to become tradable in the next 2–5 minutes.”
2. “Warn me when a symbol is **Hot but Dirty** (likely toxicity) so I don’t get trapped.”
3. “Explain in one line *why* a symbol ranks high (spread, imbalance, flow, regime).”
4. “Let me tune my execution profile: limit‑heavy vs aggressive.”

---

## 3. Definitions and target formulation

### 3.1 Cost‑aware “tradeability”
For horizon H in {30s, 2m, 5m} define:

- `move_bps(H)` — expected achievable range/price move on H
- `cost_bps` = `spread_bps` + `fees_bps` + `impact_bps(Q)`  
- `net_edge_bps(H)` = `move_bps(H)` − `cost_bps`

Tradeability event at horizon H:
- `I_tradeable(H) = 1` if `net_edge_bps(H) >= X_bps(profile, H)` AND execution gates pass

Where `X_bps` differs by execution profile (maker-ish vs taker-ish).

### 3.2 Two execution profiles (must‑have)
- **Profile A (limit‑heavy):** lower fees, but requires fill probability / queue position
- **Profile B (aggressive):** higher fees/impact, but higher completion probability

We compute:
- `p_inplay_H_A`, `p_inplay_H_B`, `net_edge_H_A`, `net_edge_H_B`
and combine conservatively (see §8).

---

## 4. Functional requirements

### 4.1 Data ingestion (Binance USD‑M Futures)
Must support:
- Trades/aggTrades stream
- Best bid/ask + book updates (depth)
- Mark price, funding, OI (where available)
- Klines for feature aggregation windows
- Optional: index/mark price streams to detect regime & dislocations

### 4.2 Real‑time feature engine
Compute microstructure‑first features:
- spread, mid, microprice, volatility proxies (NATR/realized)
- order book imbalance, depth slope, liquidity/impact estimates
- order flow imbalance, trade intensity, delta volume
- regime signals: low/high vol, trend/chop, liquidity regime
- cross‑market signals: BTC/ETH “risk on/off” gating, correlation bursts

Feature windows: 1s–60s (tick) and 1m–30m (context).

### 4.3 ML inference service
Online inference must output for each symbol:
- `p_inplay_30s`, `p_inplay_2m`, `p_inplay_5m`
- `expected_utility_bps_30s/2m/5m`
- `p_toxic`
- Uncertainty/health flags (missing data, stale book, outliers)

### 4.4 Ranker + selector
- Produce top‑N list per profile + combined list
- Apply gates (see §9) and staleness rules
- Emit events to UI/Telegram: `SYMBOL_UP`, `SYMBOL_DOWN`, `ALERT_TRADABLE`, `ALERT_TRAP`, `DATA_ISSUE`

### 4.5 Explainability layer (reason codes + LLM)
Two levels:

1) **Deterministic reason codes** (non‑LLM, auditable):
- `RC_SPREAD_TIGHTENING`, `RC_BOOK_IMBALANCE_LONG`, `RC_FLOW_SURGE`, `RC_VOL_EXPANDING`, `RC_LIQUIDITY_OK`, `RC_TOXIC_RISK`, etc.
- Each reason code includes numeric evidence (bps, z‑score, quantile)

2) **LLM narrative**:
- Input: JSON payload with top reason codes, deltas, regime, and risk flags
- Output: 1–2 lines in friendly tone + suggested “status label”
- Hard rule: LLM **never** changes numeric scores; it only summarizes.

### 4.6 Storage + observability
- Raw stream buffers (rolling, optional)
- Feature store (online snapshot + offline parquet)
- Predictions/logs + model versioning
- Monitoring dashboards: latency, dropped msgs, drift, calibration, top symbols churn

---

## 5. Non‑functional requirements

- Latency: p95 < 250 ms (ingest→features→inference→rank)
- Availability: 99%+ during market hours; graceful degradation when data missing
- Scalability: 400 symbols, 1k+ streams (via multiplexing)
- Security: API keys stored encrypted; principle of least privilege
- Compliance: obey Binance limits; auto backoff; never spam reconnect

---

## 6. Binance constraints, limits, and blockers (implementation rules)

> **Note:** exact limits may change; implementation must read headers/`rateLimits` where possible and treat docs as minimum constraints.

### 6.1 Futures REST/API rate limits (high-level)
- Binance Futures FAQ: default **2,400 requests/min per IP** and default **1,200 orders/min** per account/sub‑account (tiered by volume).
- Exceeding limits returns `429`; continuing after `429` can lead to `418` IP ban (documented behavior).
- Futures error codes include `-1003 TOO_MANY_REQUESTS` and “IP banned until …”.

### 6.2 USD‑M Futures WebSocket market stream limits (critical)
- **10 incoming messages per second per connection**; exceeding disconnects; repeated disconnects may ban IP.
- **Max 1024 streams per connection**.

### 6.3 Additional “ML/WAF” limits
Binance documents multiple limit layers: **Hard limits**, **Machine Learning limits**, and **WAF limits**.

### 6.4 Blockers and design implications
- **Polling is a trap**: must prefer WebSockets for live market data.
- **Stream multiplexing**: use combined streams and shard across connections (e.g., 3–6 connections) to stay under 1024 streams/conn and message rate.
- **Backoff discipline**: exponential backoff + jitter; circuit breaker when `429/418` observed.
- **Reconnect discipline**: avoid thrashing during volatility spikes.

(Exact citations are maintained in BINANCE_LIMITS.md in this repo.)

**References:** See `BINANCE_LIMITS.md` for a source-linked checklist.


---

## 7. ML system design (core)

### 7.1 Labels (ground truth)
For each symbol and time t, for each horizon H:
1. Simulate both execution profiles to estimate `cost_bps(profile, t)` using:
   - spread
   - fees
   - impact using clip size `Q_usd = k * usd_volume_60s` (k default 1% scalping, 3% intraday)
2. Compute realized `move_bps(H)` (future max favorable excursion minus slippage model)
3. Set:
   - `net_edge_bps(H) = move_bps(H) - cost_bps`
   - `I_tradeable(H) = 1[ net_edge_bps(H) >= X_bps(profile,H) ]`

Also define toxicity labels for `p_toxic` (adverse selection):
- e.g., “price moves against within τ after entry” or “spread widens + reverses after flow spike”.

### 7.2 Model family (v1 recommended)
- **Base models**: gradient boosted trees (LightGBM/CatBoost) for fast inference and strong tabular performance.
- **Optional sequence layer**: lightweight temporal model (TCN/1D‑CNN) over last N seconds features, if needed.
- **Multi‑task heads**: predict p_inplay for 30s/2m/5m + expected utility + p_toxic.

### 7.3 Calibration (non‑negotiable)
- Calibrate each horizon head with **isotonic regression** or **Platt scaling** on recent validation.
- Monitor: Brier score, ECE/MCE, reliability diagrams; detect drift.

### 7.4 Regime awareness and gating
- Regime classifier (simple ML head) to segment: low‑vol, high‑vol, trend, chop, illiquid.
- Either:
  - train separate models per regime, or
  - use regime as features + thresholds.

### 7.5 Online inference contract
Per symbol snapshot (JSON):
```json
{
  "ts": 0,
  "symbol": "BTCUSDT",
  "profile": "A|B|COMBINED",
  "p_inplay_30s": 0.0,
  "p_inplay_2m": 0.0,
  "p_inplay_5m": 0.0,
  "expected_utility_bps_2m": 0.0,
  "p_toxic": 0.0,
  "status": "TRADEABLE|WATCH|TRAP|DEAD|DATA_ISSUE",
  "reasons": [{"code":"RC_FLOW_SURGE","value":2.1,"unit":"z"}],
  "model_version": "semver+hash",
  "data_health": {"stale_book_ms":0,"missing_streams":[]}
}
```

---

## 8. Final scoring and aggregation

### 8.1 Combine horizons
Compute:
- `p_inplay = w30*p30 + w2m*p2m + w5m*p5m`, with w tuned for scalping vs intraday.

### 8.2 Combine execution profiles conservatively
Let `pH_A`, `pH_B` be per horizon probabilities.
Recommended conservative union:
- `pH = min( max(pH_A, pH_B), 1 - (1-pH_A)*(1-pH_B) )`
and/or hard penalty if `p_toxic` high.

### 8.3 Utility-aware ranking
Rank by:
- `score = p_inplay * clamp(expected_utility_bps, 0, Umax) * (1 - alpha*p_toxic)`,
plus gates.

---

## 9. Critical gates (anti-hallucination)

Before a symbol can be TRADEABLE:
- Spread gate: `spread_bps <= S_max(profile)`
- Liquidity gate: depth/impact below max
- Data freshness gate: book/trade streams not stale
- Volatility sanity gate: avoid regime where labels unreliable
- Toxicity gate: `p_toxic <= T_max` OR downgrade to “Hot but Dirty”

---

## 10. Evaluation and acceptance criteria

### 10.1 Offline metrics
- AUC/PR‑AUC per horizon
- Calibration (ECE, Brier)
- Profit proxy: “top‑K capture rate” and “net edge bps in top‑K”
- Churn: stability of top‑N (avoid flicker)

### 10.2 Online metrics
- Alert precision: fraction of alerts that become tradeable
- Latency and drop rate
- Drift alarms and recalibration frequency

### 10.3 Acceptance (v1)
- In backtests, top‑20 list contains ≥ X% of all tradeable events (per horizon)
- Calibration ECE < target (e.g., 3–5%)
- p95 latency < 250ms on target hardware
- No Binance bans under sustained operation (30 days) with safety margins

---

## 11. Rollout plan (milestones)

1) Data + streaming MVP (WS ingestion, feature snapshots, no ML)
2) Label builder + offline backtest harness
3) ML v1 (GBDT) + calibration + ranker
4) Toxicity model + gates
5) LLM explainability + microcopy generation
6) Observability + hardening (limits, reconnects, drift)
7) Beta + tuning + production stabilization

---

## 12. Open questions (tracked in DECISIONS.md in repo)
- Exact thresholds X_bps per profile/horizon for your trading style
- Which symbols universe (top‑N by volume? exclude low OI?)
- Preferred alert channels and UX format
