# CryptoScreener-X (In-Play Predictor)

ML + LLM crypto screener for identifying tradeable market conditions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development

```bash
# Lint
ruff check .

# Type check
mypy .

# Test
pytest
```

## Replay

```bash
python -m scripts.run_replay --fixture tests/fixtures/sample_run/
```
smoke 2026-01-25T00:06:43Z
