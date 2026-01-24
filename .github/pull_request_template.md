## Summary (1–3 bullets)
-

## Scope (what changed)
-

## Proof Bundle (MUST ATTACH — raw output, no summaries)

### Git evidence
```bash
git show --stat
git show
```

### Quality gates (repo-scope)

```bash
ruff check .
mypy .
pytest -q
```

### CI confirmation

```bash
gh pr checks <PR_NUMBER>
```

### Contracts (if touches contracts/events/models)

* 3 JSON payload examples that match DATA_CONTRACTS
* schema validation output (raw)

### Replay/determinism (if touches ranker/alerter/runner/replay)

* input fixture sha256:
* output RankEvent stream digest:
* run #1 == run #2:
