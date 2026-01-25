# Model Registry & Versioning

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM
**Date:** 2026-01-24 (Updated: 2026-01-25)

---

## Version string
`{semver}+{git_sha}+{data_cutoff}+{train_hash}`

Example: `1.0.0+abc1234+20260125+12345678`

Baseline models use: `baseline-v{semver}+{git_sha}`

## Stored artifacts

### Required artifacts
- `schema_version.json` — Package schema version and model version
- `features.json` — Ordered list of feature names
- `checksums.txt` — SHA256 hashes (SSOT for integrity verification)

### Optional artifacts
- `model.bin` — Trained model binary
- `calibrator_{head}.pkl` — Probability calibrators per prediction head
- `training_report.md` — Training metrics and notes
- `manifest.json` — Machine-readable metadata (auto-generated, derived from checksums.txt)

## checksums.txt Format (SSOT)

Human-readable, `sha256sum -c` compatible:

```
# checksums.txt - SHA256 hashes for 1.0.0+abc1234+20260125+12345678
# Generated: 2026-01-25T12:00:00Z

abc123...  model.bin
def456...  calibrator_p_inplay_2m.pkl
789abc...  features.json
```

## manifest.json Format (Optional/Derived)

Machine-readable metadata with artifact checksums:

```json
{
  "schema_version": "1.0.0",
  "model_version": "1.0.0+abc1234+20260125+12345678",
  "created_at": "2026-01-25T12:00:00Z",
  "artifacts": [
    {"name": "model.bin", "sha256": "abc123...", "size_bytes": 12345}
  ]
}
```

When `manifest.json` is absent, the loader generates metadata from `checksums.txt`.

## Artifact Name Constraints (Security)

Artifact names MUST:
- Be non-empty, non-whitespace
- Contain only the filename (no path separators `/` or `\`)
- Not be `.` or `..`
- Not be absolute paths

Symlinks are NOT allowed as artifacts.

## Compatibility
- Any contract/schema change bumps schema_version and requires migration notes.
- Loader supports both `checksums.txt`-only and `checksums.txt + manifest.json` packages.
