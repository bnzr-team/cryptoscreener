#!/usr/bin/env bash
set -euo pipefail

FIXTURE_DIR="${1:-tests/fixtures/sample_run}"
OUT_FILE="${2:-proof_bundle.md}"

echo "# Proof bundle" > "$OUT_FILE"
echo "" >> "$OUT_FILE"

echo "## 1) Git proof" >> "$OUT_FILE"
echo '```bash' >> "$OUT_FILE"
git status >> "$OUT_FILE"
echo "" >> "$OUT_FILE"
git log --oneline --decorate -n 10 >> "$OUT_FILE"
echo '```' >> "$OUT_FILE"
echo "" >> "$OUT_FILE"

COMMIT_SHA="$(git rev-parse HEAD)"
echo "Commit: \`$COMMIT_SHA\`" >> "$OUT_FILE"
echo "" >> "$OUT_FILE"

echo '```bash' >> "$OUT_FILE"
git show --stat "$COMMIT_SHA" >> "$OUT_FILE"
echo '```' >> "$OUT_FILE"
echo "" >> "$OUT_FILE"

echo "## 2) Tool versions" >> "$OUT_FILE"
echo '```bash' >> "$OUT_FILE"
python --version >> "$OUT_FILE" 2>&1 || true
ruff --version >> "$OUT_FILE" 2>&1 || true
mypy --version >> "$OUT_FILE" 2>&1 || true
pytest --version >> "$OUT_FILE" 2>&1 || true
echo '```' >> "$OUT_FILE"
echo "" >> "$OUT_FILE"

echo "## 3) Quality gates (raw output)" >> "$OUT_FILE"
echo '```bash' >> "$OUT_FILE"
echo '$ ruff check .' >> "$OUT_FILE"
ruff check . >> "$OUT_FILE"
echo "" >> "$OUT_FILE"
echo '$ mypy .' >> "$OUT_FILE"
mypy . >> "$OUT_FILE"
echo "" >> "$OUT_FILE"
echo '$ pytest -q' >> "$OUT_FILE"
pytest -q >> "$OUT_FILE"
echo '```' >> "$OUT_FILE"
echo "" >> "$OUT_FILE"

if [ -d "$FIXTURE_DIR" ]; then
  echo "## 4) Fixtures checksums" >> "$OUT_FILE"
  echo "Fixture dir: \`$FIXTURE_DIR\`" >> "$OUT_FILE"
  echo '```bash' >> "$OUT_FILE"
  (cd "$FIXTURE_DIR" && sha256sum * ) >> "$OUT_FILE"
  echo '```' >> "$OUT_FILE"
  echo "" >> "$OUT_FILE"

  if [ -f "scripts/run_replay.py" ]; then
    echo "## 5) Replay determinism" >> "$OUT_FILE"
    echo '```bash' >> "$OUT_FILE"
    echo "\$ python -m scripts.run_replay --fixture $FIXTURE_DIR -v" >> "$OUT_FILE"
    python -m scripts.run_replay --fixture "$FIXTURE_DIR" -v >> "$OUT_FILE"
    echo '```' >> "$OUT_FILE"
    echo "" >> "$OUT_FILE"
  fi
fi

echo "## 6) Patch (git show)" >> "$OUT_FILE"
echo '```diff' >> "$OUT_FILE"
git show "$COMMIT_SHA" >> "$OUT_FILE"
echo '```' >> "$OUT_FILE"
