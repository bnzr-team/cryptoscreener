#!/usr/bin/env bash
set -euo pipefail

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

echo "== DONE =="
