# LLM Safety Guardrails

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Guardrails
- Schema validation
- No-new-numbers validator
- Enum lock for status_label
- Max length
- Rate limit LLM calls (LLM is optional)

## Failure behavior
- If LLM unavailable or invalid output: use deterministic template:
  - headline = “{SYM}: {top_reason_1} + {top_reason_2}”
  - status_label based on state

## Tests
- Golden tests for LLM validator
