# Model Registry & Versioning

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Version string
`{semver}+{git_sha}+{data_cutoff}+{train_hash}`

## Stored artifacts
- model.bin
- calibrator_{head}.pkl
- features.json (ordered list)
- schema_version.json
- training_report.md (metrics)
- checksums.txt (sha256)

## Compatibility
- Any contract/schema change bumps schema_version and requires migration notes.
