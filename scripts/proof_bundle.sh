#!/usr/bin/env bash
set -euo pipefail

PR_NUMBER="${1:-}"
if [[ -z "${PR_NUMBER}" ]]; then
  echo "Usage: ./scripts/proof_bundle.sh <PR_NUMBER>"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="artifacts/proof_bundle_pr${PR_NUMBER}_${TS}.txt"
mkdir -p artifacts

# tee everything to a persistent file
exec > >(tee "$OUT") 2>&1

echo "== PROOF_BUNDLE_FILE =="
echo "$OUT"
echo

echo "== PR URL =="
gh pr view "${PR_NUMBER}" --json url | cat
echo

echo "== GH PR VIEW =="
gh pr view "${PR_NUMBER}" --json number,title,state,mergedAt,mergeCommit,baseRefName,headRefName,headRepositoryOwner,headRepository,url | cat
echo

echo "== GH PR CHECKS =="
gh pr checks "${PR_NUMBER}" || true
echo

echo "== CHANGED FILES =="
# Use gh pr to get all changed files in PR (not just HEAD commit)
gh pr view "${PR_NUMBER}" --json files --jq '.files[].path' || git show --name-only --pretty="" HEAD
echo

echo "== TOOLCHAIN VERSIONS =="
python3 --version
ruff --version
mypy --version
pytest --version
echo

echo "== GIT SHOW --STAT =="
git show --stat
echo

echo "== GIT SHOW =="
# Show full PR diff instead of just HEAD commit
echo "--- Full PR diff (gh pr diff ${PR_NUMBER}) ---"
gh pr diff "${PR_NUMBER}" || git show
echo

echo "== RUFF CHECK . =="
ruff check .
echo

echo "== MYPY . =="
mypy .
echo

echo "== PYTEST -Q =="
pytest -q
echo
