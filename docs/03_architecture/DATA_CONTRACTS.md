# Data Contracts (v1)
**Date:** 2026-01-24

This document defines canonical schemas used between modules. All contracts are JSON-serializable and must be versioned.

---

## 1) MarketEvent

```json
{
  "ts": 0,
  "source": "binance_usdm",
  "symbol": "BTCUSDT",
  "type": "trade|book|kline|mark|oi|funding",
  "payload": {},
  "recv_ts": 0
}
```

Notes:
- `ts` is event timestamp from exchange (ms).
- `recv_ts` is local receive time (ms) for latency metrics.

---

## 2) FeatureSnapshot

```json
{
  "ts": 0,
  "symbol": "BTCUSDT",
  "features": {
    "spread_bps": 0.0,
    "mid": 0.0,
    "book_imbalance": 0.0,
    "flow_imbalance": 0.0,
    "natr_14_5m": 0.0,
    "impact_bps_q": 0.0,
    "regime_vol": "low|high",
    "regime_trend": "trend|chop"
  },
  "windows": {
    "w1s": {},
    "w10s": {},
    "w60s": {}
  },
  "data_health": {
    "stale_book_ms": 0,
    "stale_trades_ms": 0,
    "missing_streams": []
  }
}
```

---

## 3) PredictionSnapshot

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
  "reasons": [
    {"code":"RC_FLOW_SURGE","value":2.1,"unit":"z","evidence":"flow_imbalance=0.63"}
  ],
  "model_version": "1.0.0+gitsha",
  "calibration_version": "iso_2026-01-24",
  "data_health": {}
}
```

---

## 4) RankEvent

```json
{
  "ts": 0,
  "event": "SYMBOL_ENTER|SYMBOL_EXIT|ALERT_TRADABLE|ALERT_TRAP|DATA_ISSUE",
  "symbol": "BTCUSDT",
  "rank": 3,
  "score": 0.83,
  "payload": {"prediction": {}, "llm_text": ""}
}
```

Notes:
- `rank`: 0-indexed position in top-K ranking (must be >= 0).
- `score`: Normalized ranking score in [0, 1]. Formula: `p_inplay * (utility/Umax) * (1 - α*p_toxic)`.
- `payload.prediction`:
  - **Ranker events** (SYMBOL_ENTER, SYMBOL_EXIT): Empty dict `{}` — lightweight events for high-frequency updates. Downstream consumers should fetch prediction from the prediction store if needed.
  - **Alerter events** (ALERT_TRADABLE, ALERT_TRAP, DATA_ISSUE): Full `PredictionSnapshot` dict for self-contained alert payloads.
- `payload.llm_text`: LLM-generated explanation (empty string if not yet populated).

---

## 5) LLM Explain Contract (strict)

Input:
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "2m",
  "status": "WATCH",
  "score": 0.83,
  "reasons": [],
  "numeric_summary": {
    "spread_bps": 8.2,
    "impact_bps": 6.5,
    "p_toxic": 0.21,
    "regime": "high-vol trend"
  },
  "style": {"tone":"friendly","max_chars":180}
}
```

Output:
```json
{
  "headline": "BTCUSDT: flow surge + tight spread, likely tradable soon.",
  "subtext": "Watch for quick breakout; toxicity low-to-moderate.",
  "status_label": "Tradeable soon",
  "tooltips": {"p_inplay":"Calibrated probability of net edge after costs."}
}
```

LLM MUST NOT output numbers different from inputs. It can only rephrase or omit.

---

## 6) Trading Contracts (v2 — separate SSOT)

Trading/VOL Harvesting v2 defines its own contracts in **`docs/trading/`**.

**v2 contracts** (NOT part of v1):
- `OrderIntent` — order request to OMS
- `OrderAck` — exchange acknowledgement
- `FillEvent` — fill notification
- `PositionSnapshot` — current position state
- `SessionState` — trading session state machine
- `RiskBreachEvent` — risk limit violation

**Boundary contract** (v1 → v2):
- **`RankEvent`** (§4 above) is the ONLY v1 contract consumed by v2

**Key invariants** (v2):
- All monetary values use `Decimal`, not `float`
- NATR is fraction (0.025 = 2.5%), NOT percentage
- Fees are fraction (0.0002 = 0.02%), NOT bps

See `docs/trading/TRADING_SPEC.md` for full v2 contract definitions.
