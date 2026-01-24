#!/usr/bin/env bash
# Reviewer Message Generator for CryptoScreener-X
#
# Generates a ready-to-paste message for the reviewer chat.
# Runs acceptance_packet.sh and wraps output in a clean format.
#
# Usage: ./scripts/reviewer_message.sh <PR_NUMBER>
#
# Exit codes:
#   0 - Message generated (even if acceptance packet failed - reviewer needs to see it)
#   2 - Usage error

set -euo pipefail

PR_NUMBER="${1:-}"
if [[ -z "${PR_NUMBER}" ]]; then
  echo "Usage: ./scripts/reviewer_message.sh <PR_NUMBER>"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "REVIEWER MESSAGE FOR PR #${PR_NUMBER}"
echo "================================================================================"
echo ""
echo "Copy everything below this line and paste into reviewer chat:"
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# Run acceptance_packet.sh and capture output + exit code
PACKET_OUTPUT="$("${SCRIPT_DIR}/acceptance_packet.sh" "${PR_NUMBER}" 2>&1)" || PACKET_EXIT=$?
PACKET_EXIT="${PACKET_EXIT:-0}"

# Print the packet verbatim
echo "${PACKET_OUTPUT}"

echo ""
echo "--------------------------------------------------------------------------------"
echo ""

if [[ "${PACKET_EXIT}" -eq 0 ]]; then
  echo "STATUS: ✓ Ready for ACCEPT"
else
  echo "STATUS: ✗ NOT ready (exit code ${PACKET_EXIT})"
  echo "Fix the issues above and re-run this script."
fi

echo ""
echo "================================================================================"

# Always exit 0 - the message was generated successfully
# The reviewer needs to see the output regardless of packet status
exit 0
