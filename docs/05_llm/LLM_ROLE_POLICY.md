# LLM Role Policy

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Allowed
- Produce human-readable explanations from provided reason codes + numeric summary.
- Generate microcopy/status labels from allowed set.
- Summarize incidents/drift reports from logs.

## Forbidden
- Creating or altering numeric scores, probabilities, thresholds.
- Introducing new numbers not present in input JSON.
- Making trading promises/predictions (“will pump”).

## Enforcement
- Validate LLM output with schema + “no-new-numbers” checker.
- If invalid: fallback to deterministic template.
