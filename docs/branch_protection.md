# Branch Protection Settings — `main`

**Repo:** `bnzr-team/cryptoscreener`
**Branch:** `main`
**Last updated:** 2026-01-27

---

## Required Settings (GitHub UI: Settings → Branches → Branch protection rules)

### Pull Request Requirements

- [x] **Require a pull request before merging**
  - [x] Require approvals: **1**
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners (enforced via `.github/CODEOWNERS`)

### Status Checks

- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - Required checks:
    - `checks` (CI workflow: ruff, mypy, pytest, replay fixture)
    - `check-rules` (Promtool workflow: `promtool check rules`, triggers on `monitoring/**`)
    - `proof-guard` (Proof Guard workflow: acceptance packet validation)
    - `acceptance-packet` (Acceptance Packet workflow: auto-generates packet in PR body)

### Conversation Resolution

- [x] **Require conversation resolution before merging**

### Push Restrictions

- [x] **Restrict who can push to matching branches**
  - Allowed: `@bnzr-hub` only
  - No force pushes
  - No deletions

### Admin Override

- [ ] **Do not allow bypassing the above settings** (recommended)
  - If bypass is needed: only `@bnzr-hub` with documented reason in PR body
  - Any `--admin` merge must include justification comment

---

## CODEOWNERS Enforcement

File: `.github/CODEOWNERS`

All PRs touching the following paths require `@bnzr-hub` approval:

| Path Pattern | Reason |
|---|---|
| `*` (default) | Sole maintainer |
| `DECISIONS.md`, `STATE.md`, `CHANGELOG.md` | SSOT documents |
| `monitoring/**` | Alert rules (cardinality/security risk) |
| `.github/workflows/**` | CI/CD pipeline integrity |
| `scripts/**` | Proof/acceptance tooling |
| `src/cryptoscreener/connectors/**` | Exporter, metrics endpoint, backoff |
| `src/cryptoscreener/contracts/**` | Data contracts (breaking change risk) |
| `tests/fixtures/**` | Replay determinism fixtures |

---

## How to Apply

### Option A: GitHub UI (manual)

1. Go to **Settings → Branches → Add branch protection rule**
2. Branch name pattern: `main`
3. Check all boxes listed above
4. Add required status checks by name
5. Save changes

### Option B: GitHub API (scriptable)

```bash
gh api -X PUT repos/bnzr-team/cryptoscreener/branches/main/protection \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["checks", "check-rules", "proof-guard", "acceptance-packet"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1
  },
  "restrictions": {
    "users": ["bnzr-hub"],
    "teams": []
  },
  "required_conversation_resolution": true
}
EOF
```

---

## Verification

After applying, verify with:

```bash
gh api repos/bnzr-team/cryptoscreener/branches/main/protection \
  --jq '{
    required_status_checks: .required_status_checks.contexts,
    enforce_admins: .enforce_admins.enabled,
    required_reviews: .required_pull_request_reviews.required_approving_review_count,
    dismiss_stale: .required_pull_request_reviews.dismiss_stale_reviews,
    code_owner_reviews: .required_pull_request_reviews.require_code_owner_reviews
  }'
```

Expected output:
```json
{
  "required_status_checks": ["checks", "check-rules", "proof-guard", "acceptance-packet"],
  "enforce_admins": true,
  "required_reviews": 1,
  "dismiss_stale": true,
  "code_owner_reviews": true
}
```
