#!/usr/bin/env bash
set -euo pipefail

PR_NUMBER="${1:-}"
if [[ -z "${PR_NUMBER}" ]]; then
  echo "Usage: ./scripts/proof_bundle_chat.sh <PR_NUMBER>"
  exit 2
fi

echo "== CHAT PROOF: IDENTITY =="
PR_JSON="$(gh pr view "${PR_NUMBER}" --json url,state,mergedAt,mergeCommit,baseRefName,headRefName,title,number)"
echo "${PR_JSON}"
echo

PR_URL="$(echo "${PR_JSON}" | python -c 'import json,sys; print(json.load(sys.stdin)["url"])')"
MERGE_COMMIT="$(echo "${PR_JSON}" | python -c 'import json,sys; d=json.load(sys.stdin); mc=d.get("mergeCommit"); print(mc["oid"] if mc else "")')"

echo "== CHAT PROOF: CI =="
gh pr checks "${PR_NUMBER}" || true
echo

echo "== CHAT PROOF: GIT EVIDENCE =="
if [[ -n "${MERGE_COMMIT}" ]]; then
  echo "--- git show --stat ${MERGE_COMMIT} ---"
  git show "${MERGE_COMMIT}" --stat
  echo
  echo "--- git show --name-only ${MERGE_COMMIT} ---"
  git show --name-only --pretty="" "${MERGE_COMMIT}"
else
  echo "--- git show --stat (HEAD) ---"
  git show --stat
  echo
  echo "--- git show --name-only (HEAD) ---"
  git show --name-only --pretty="" HEAD
fi
echo
echo "Files changed: ${PR_URL}/files"
echo

echo "== CHAT PROOF: GATES =="
echo "--- ruff check . ---"
ruff check .
echo
echo "--- mypy . ---"
mypy .
echo
echo "--- pytest -q ---"
pytest -q
echo

echo "== CHAT PROOF: PROOF FILE =="
LATEST="$(ls -1t artifacts/proof_bundle_pr${PR_NUMBER}_*.txt 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST}" ]]; then
  echo "${LATEST}"
else
  echo "MISSING: run ./scripts/proof_bundle.sh ${PR_NUMBER} to generate artifacts file"
  exit 1
fi
