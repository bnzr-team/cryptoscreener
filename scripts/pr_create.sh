#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/pr_create.sh "<branch>" "<title>" [fixture_dir]
BRANCH="${1:?branch required, e.g. feature/pr-0002-ws}"
TITLE="${2:?title required, e.g. PR#2: minimal WS connector scaffold}"
FIXTURE_DIR="${3:-tests/fixtures/sample_run}"

# Ensure we're in repo root
git rev-parse --show-toplevel >/dev/null

# Ensure venv tools are available
if ! command -v ruff >/dev/null 2>&1; then
  echo "ruff not found. Activate venv: source .venv/bin/activate"
  exit 1
fi

# Ensure there are changes
if [ -z "$(git status --porcelain)" ]; then
  echo "Nothing to commit. Make changes first."
  exit 1
fi

# Create or switch to branch
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH"
fi

# Commit
git add -A
git commit -m "$TITLE"

# Push branch
git push -u origin "$BRANCH"

# Generate proof bundle
mkdir -p proof
PROOF_FILE="proof/proof_${BRANCH//\//_}.md"
./scripts/gen_proof_bundle.sh "$FIXTURE_DIR" "$PROOF_FILE"

# Create PR
PR_URL="$(gh pr create \
  --base main \
  --head "$BRANCH" \
  --title "$TITLE" \
  --body-file "$PROOF_FILE")"

echo "PR: $PR_URL"
echo "Proof bundle: $PROOF_FILE"

# Enable auto-merge (squash) + delete branch after merge
gh pr merge --auto --squash --delete-branch "$PR_URL"

echo "Auto-merge enabled."
