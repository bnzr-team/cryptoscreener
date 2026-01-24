# Reproducibility

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Golden fixtures
- Store small raw event sequences + expected outputs
- Hash fixtures (sha256) and verify in CI

## Determinism rules
- Seed all RNGs
- Time-based operations in replay use event timestamps, not wall clock

## Proof bundle requirement
- For each milestone: attach logs + test output + artifact checksums
