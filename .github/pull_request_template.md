## Summary (1–3 bullets)
-

## Scope (what changed)
-

## Proof Bundle (REQUIRED — raw output only, no summaries)

Run:
```bash
./scripts/proof_bundle.sh
```

Paste the **full output** below (do NOT edit the `== ... ==` markers):

<PASTE_PROOF_BUNDLE_OUTPUT_HERE>

Required markers (must appear in PR body):
- `== GIT SHOW --STAT ==`
- `== GIT SHOW ==`
- `== RUFF CHECK . ==`
- `== MYPY . ==`
- `== PYTEST -Q ==`

### CI confirmation (REQUIRED)

Paste raw output:
```bash
gh pr checks <PR_NUMBER>
```

### If this PR touches contracts/events/models (REQUIRED when applicable)

* Attach 3 JSON payload examples matching DATA_CONTRACTS.md
* Attach roundtrip proof (to_json/from_json equality) or schema validation output (raw)

### If this PR touches runner/ranker/alerter/replay/live pipeline (REQUIRED when applicable)

* Determinism proof:
  * input fixture sha256
  * output RankEvent stream digest sha256
  * run #1 == run #2 (raw logs)
