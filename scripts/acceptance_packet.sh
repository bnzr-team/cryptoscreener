#!/usr/bin/env bash
# Acceptance Packet Generator for CryptoScreener-X
#
# One-command "ready for ACCEPT" generator.
# Waits for CI, runs quality gates, generates artifacts, validates replay if required.
#
# Usage: ./scripts/acceptance_packet.sh <PR_NUMBER>
#
# Exit codes:
#   0 - All checks passed, ready for ACCEPT
#   1 - Some check failed (CI, gates, replay)
#   2 - Usage error

set -euo pipefail

PR_NUMBER="${1:-}"
if [[ -z "${PR_NUMBER}" ]]; then
  echo "Usage: ./scripts/acceptance_packet.sh <PR_NUMBER>"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Colors for terminal (disabled if not tty)
if [[ -t 1 ]]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  NC='\033[0m' # No Color
else
  RED=''
  GREEN=''
  YELLOW=''
  NC=''
fi

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Track overall status
PACKET_STATUS="PASS"
FAILED_CHECKS=()

fail_check() {
  PACKET_STATUS="FAIL"
  FAILED_CHECKS+=("$1")
  log_error "$1"
}

echo "== ACCEPTANCE PACKET: IDENTITY =="
PR_JSON="$(gh pr view "${PR_NUMBER}" --json url,state,mergedAt,mergeCommit,baseRefName,headRefName,title,number)"
echo "${PR_JSON}" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), indent=2))'
echo

PR_URL="$(echo "${PR_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["url"])')"
STATE="$(echo "${PR_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')"
BASE_REF="$(echo "${PR_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["baseRefName"])')"
HEAD_REF="$(echo "${PR_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["headRefName"])')"

echo "PR URL: ${PR_URL}"
echo "PR STATE: ${STATE}"
echo "Base: ${BASE_REF}"
echo "Head: ${HEAD_REF}"
echo
echo "NOTE: Paste this output verbatim in full. Do NOT summarize."
echo

# Get list of changed files via gh api (robust, matches proof_guard.yml)
# This is more reliable than gh pr view --json files which can be empty on some gh versions
echo "== ACCEPTANCE PACKET: CHANGED FILES =="
REPO_NWO="$(gh repo view --json nameWithOwner -q '.nameWithOwner')"
if [[ -z "${REPO_NWO}" ]]; then
  fail_check "REPO_DETECT_FAILED: Could not determine repository (gh repo view failed)"
  CHANGED_FILES=""
else
  CHANGED_FILES="$(gh api "/repos/${REPO_NWO}/pulls/${PR_NUMBER}/files" --paginate --jq '.[].filename' 2>&1)" || {
    fail_check "FILES_API_FAILED: gh api /repos/${REPO_NWO}/pulls/${PR_NUMBER}/files failed"
    CHANGED_FILES=""
  }
fi

# Validate we got file list (empty = suspicious, fail to avoid false negative)
if [[ -z "${CHANGED_FILES}" && "${PACKET_STATUS}" == "PASS" ]]; then
  fail_check "FILES_EMPTY: PR file list is empty (unexpected for any valid PR)"
fi

echo "${CHANGED_FILES}"
echo

# Detect if replay is required
require_replay() {
  local files="$1"
  echo "${files}" | grep -qE '^(scripts/run_replay\.py|scripts/run_record\.py|tests/replay/|tests/fixtures/)' && return 0
  return 1
}

if require_replay "${CHANGED_FILES}"; then
  REPLAY_REQUIRED="true"
  log_info "Replay verification REQUIRED (PR touches replay-related files)"
else
  REPLAY_REQUIRED="false"
  log_info "Replay verification not required"
fi
echo "replay_required: ${REPLAY_REQUIRED}"
echo

echo "== ACCEPTANCE PACKET: CI =="
log_info "Waiting for CI checks to complete..."

# Poll CI status until all checks complete
MAX_WAIT=600  # 10 minutes max
POLL_INTERVAL=10
WAITED=0

while true; do
  CHECKS_OUTPUT="$(gh pr checks "${PR_NUMBER}" 2>&1 || true)"
  echo "${CHECKS_OUTPUT}"

  # Check for pending
  if echo "${CHECKS_OUTPUT}" | grep -q "pending"; then
    if [[ ${WAITED} -ge ${MAX_WAIT} ]]; then
      fail_check "CI_TIMEOUT: Checks still pending after ${MAX_WAIT}s"
      break
    fi
    log_warn "Checks still pending, waiting ${POLL_INTERVAL}s... (${WAITED}/${MAX_WAIT}s)"
    sleep "${POLL_INTERVAL}"
    WAITED=$((WAITED + POLL_INTERVAL))
    continue
  fi

  # Check for failures
  if echo "${CHECKS_OUTPUT}" | grep -qE "(fail|cancelled)"; then
    fail_check "CI_FAILED: One or more checks failed"
    break
  fi

  # All passed
  log_info "All CI checks passed"
  break
done
echo

echo "== ACCEPTANCE PACKET: GATES =="

echo "--- ruff check . ---"
if ruff check .; then
  log_info "ruff: PASS"
else
  fail_check "RUFF_FAILED"
fi
echo

echo "--- mypy . ---"
if mypy .; then
  log_info "mypy: PASS"
else
  fail_check "MYPY_FAILED"
fi
echo

echo "--- pytest -q ---"
if pytest -q; then
  log_info "pytest: PASS"
else
  fail_check "PYTEST_FAILED"
fi
echo

echo "== ACCEPTANCE PACKET: PROOF BUNDLE =="
LATEST_ARTIFACT="$(ls -1t artifacts/proof_bundle_pr${PR_NUMBER}_*.txt 2>/dev/null | head -n 1 || true)"

if [[ -z "${LATEST_ARTIFACT}" ]]; then
  log_warn "No artifacts found, generating..."
  "${SCRIPT_DIR}/proof_bundle.sh" "${PR_NUMBER}" > /dev/null 2>&1 || true
  LATEST_ARTIFACT="$(ls -1t artifacts/proof_bundle_pr${PR_NUMBER}_*.txt 2>/dev/null | head -n 1 || true)"
fi

if [[ -n "${LATEST_ARTIFACT}" ]]; then
  echo "Artifacts file: ${LATEST_ARTIFACT}"
  log_info "Proof bundle artifacts available"
else
  fail_check "ARTIFACTS_MISSING: Could not generate proof bundle"
fi
echo

echo "== ACCEPTANCE PACKET: PR DIFF =="
echo "--- gh pr diff ${PR_NUMBER} ---"
gh pr diff "${PR_NUMBER}" || fail_check "PR_DIFF_FAILED"
echo

echo "== ACCEPTANCE PACKET: REPLAY DETERMINISM =="
if [[ "${REPLAY_REQUIRED}" == "true" ]]; then
  log_info "Running replay determinism verification..."

  # Create temp directory for synthetic fixture
  TEMP_FIXTURE="$(mktemp -d)"
  trap 'rm -rf "${TEMP_FIXTURE}"' EXIT

  # Generate synthetic fixture
  echo "--- Generating synthetic fixture ---"
  if python3 scripts/run_record.py --symbols BTCUSDT --duration-s 2 --out-dir "${TEMP_FIXTURE}" --cadence-ms 100 --source synthetic 2>&1; then
    log_info "Fixture generated successfully"
  else
    fail_check "RECORD_FAILED: Could not generate synthetic fixture"
  fi
  echo

  # Run replay twice
  echo "--- Replay run #1 ---"
  REPLAY1_OUTPUT="$(python3 scripts/run_replay.py --fixture "${TEMP_FIXTURE}" -v 2>&1)"
  echo "${REPLAY1_OUTPUT}"
  DIGEST1="$(echo "${REPLAY1_OUTPUT}" | grep "RankEvent stream digest:" | awk '{print $NF}' || true)"
  echo

  echo "--- Replay run #2 ---"
  REPLAY2_OUTPUT="$(python3 scripts/run_replay.py --fixture "${TEMP_FIXTURE}" -v 2>&1)"
  echo "${REPLAY2_OUTPUT}"
  DIGEST2="$(echo "${REPLAY2_OUTPUT}" | grep "RankEvent stream digest:" | awk '{print $NF}' || true)"
  echo

  # Extract expected digest from manifest
  EXPECTED_DIGEST="$(python3 -c "
import json
with open('${TEMP_FIXTURE}/manifest.json') as f:
    m = json.load(f)
    print(m.get('replay', {}).get('rank_event_stream_digest', ''))
" 2>/dev/null || true)"

  echo "--- Digest verification ---"
  echo "Run #1 digest:   ${DIGEST1}"
  echo "Run #2 digest:   ${DIGEST2}"
  echo "Expected digest: ${EXPECTED_DIGEST}"

  if [[ -n "${DIGEST1}" && "${DIGEST1}" == "${DIGEST2}" && "${DIGEST1}" == "${EXPECTED_DIGEST}" ]]; then
    echo "✓ ALL DIGESTS MATCH"
    log_info "Replay determinism: PASS"
  else
    fail_check "REPLAY_MISMATCH: Digests do not match"
  fi
else
  echo "replay_required: false"
  echo "Skipping replay verification (PR does not touch replay-related files)"
fi
echo

echo "== ACCEPTANCE PACKET: END =="

if [[ "${PACKET_STATUS}" == "FAIL" ]]; then
  echo "RESULT: FAIL"
  echo "PR: #${PR_NUMBER}"
  echo "URL: ${PR_URL}"
  echo "State: ${STATE}"
  echo "Replay required: ${REPLAY_REQUIRED}"
  echo ""
  echo "Failed checks:"
  for check in "${FAILED_CHECKS[@]}"; do
    echo "  - ${check}"
  done
  exit 1
else
  echo "RESULT: PASS"
  echo "PR: #${PR_NUMBER}"
  echo "URL: ${PR_URL}"
  echo "State: ${STATE}"
  echo "Replay required: ${REPLAY_REQUIRED}"
  echo ""
  echo "✓ All checks passed. Ready for ACCEPT."
  exit 0
fi
