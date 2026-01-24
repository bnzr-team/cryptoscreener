# UX Requirements

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## 1) Главный экран (Top list)
Колонки:
- Symbol
- Score (0..1)
- Status: `TRADEABLE | Tradeable soon | WATCH | Hot but Dirty | TRAP | DEAD | DATA_ISSUE`
- p_inplay (30s/2m/5m компактно)
- expected_utility_bps (по выбранному горизонту/профилю)
- spread_bps / impact_bps (иконки/кратко)
- p_toxic (иконка + число)
- “Why” (headline 1 строка)

## 2) Карточка символа (detail)
- 3 оси: **Tradeable / Heat / Trap**
- Временная шкала: последние 60s (мини‑графики): spread, flow, imbalance, utility
- Reason codes (топ‑5) + evidence (числа)
- Data health: stale, missing streams, reconnect count

## 3) UX правила
- LLM текст не должен “придумывать” причины: только на основе reason codes
- Показывать “DATA_ISSUE” заметно, чтобы пользователь не торговал на мусоре
- Anti-flicker: изменения статуса не чаще N секунд, если нет резкого события

## 4) Экспорт/шеринг
- Telegram alert: короткий headline + 2 причины + 1 риск + ссылка на картинку/чарт
