# Dataset Build Pipeline

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Offline pipeline steps
1. Collect raw WS data (trades, depth, mark) into time-partitioned files.
2. Normalize events → `MarketEvent` format.
3. Build features using the SAME feature library as online.
4. Build labels using LABELS_SPEC + COST_MODEL_SPEC.
5. Assemble training rows per symbol/time.
6. Split by time into train/val/test.
7. Save parquet + metadata JSON (schema, windows, hash).

## Reproducibility
- Store git SHA, config hash, data hash with each dataset build.
