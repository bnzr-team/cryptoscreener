# LLM Input/Output Schema

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


See also `docs/03_architecture/DATA_CONTRACTS.md`.

## Output constraints
- Max chars: config (default 180)
- Required keys: headline, status_label
- status_label must be one of allowed labels.
- No numbers may appear unless they were present in input numeric_summary (exact match).

## Validation algorithm (step-by-step)
1. Parse JSON.
2. Validate required keys.
3. Check status_label in enum.
4. Extract all numbers (regex) from output and ensure subset of input numbers (stringwise).
5. Enforce max length.
6. If fail → fallback.

## Tests
- Unit tests with adversarial outputs.
