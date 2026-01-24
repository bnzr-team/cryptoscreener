#!/usr/bin/env bash
set -euo pipefail

PR_NUMBER="${1:-}"
if [[ -z "${PR_NUMBER}" ]]; then
  echo "Usage: ./scripts/proof_bundle.sh <PR_NUMBER>"
  exit 2
fi

echo "== PR URL =="
gh pr view "${PR_NUMBER}" --json url,state,headRefName,baseRefName,commits | cat
echo

echo "== GH PR CHECKS =="
gh pr checks "${PR_NUMBER}" || true
echo

echo "== GIT SHOW --STAT =="
git show --stat
echo

echo "== GIT SHOW =="
git show
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
