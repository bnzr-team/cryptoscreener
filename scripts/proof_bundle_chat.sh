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
STATE="$(echo "${PR_JSON}" | python -c 'import json,sys; print(json.load(sys.stdin)["state"])')"
MERGE_COMMIT="$(echo "${PR_JSON}" | python -c 'import json,sys; d=json.load(sys.stdin); mc=d.get("mergeCommit"); print(mc.get("oid","") if isinstance(mc,dict) else "")')"

if [[ "${STATE}" == "MERGED" && -z "${MERGE_COMMIT}" ]]; then
  echo "ERROR: PR is MERGED but mergeCommit is missing in gh pr view JSON"
  echo "Raw PR JSON:"
  echo "${PR_JSON}"
  exit 1
fi

echo "PR URL: ${PR_URL}"
echo "PR STATE: ${STATE}"
echo "MERGE COMMIT: ${MERGE_COMMIT:-<none>}"
echo

echo "== CHAT PROOF: CI =="
gh pr checks "${PR_NUMBER}" || true
echo

echo "== CHAT PROOF: GIT EVIDENCE =="
if [[ -n "${MERGE_COMMIT}" ]]; then
  # For MERGED PRs: fetch from origin to ensure we have the merge commit locally
  git fetch origin "${MERGE_COMMIT}" --quiet 2>/dev/null || git fetch origin --quiet 2>/dev/null || true
  echo "--- git show --stat ${MERGE_COMMIT} ---"
  git show "${MERGE_COMMIT}" --stat
  echo
  echo "--- git show --name-only --pretty=\"\" ${MERGE_COMMIT} ---"
  git show --name-only --pretty="" "${MERGE_COMMIT}"
else
  echo "--- git show --stat (HEAD) ---"
  git show --stat
  echo
  echo "--- git show --name-only --pretty=\"\" HEAD ---"
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

echo
echo "== CHAT PROOF: END =="
