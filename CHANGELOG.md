# CHANGELOG

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM
**Updated:** 2026-01-24

---

## Unreleased

### Added

#### GitHub PR#23 — LLM Explain Pipeline Integration (DEC-005)
- Integrated LLM explain into Alerter: `RankEvent.payload.llm_text` populated for alert events
- `ExplainLLMProtocol` for dependency injection (no hard import of explain_llm module)
- Per-symbol LLM cooldown (60s default) with caching to reduce API calls
- `AlerterConfig.llm_enabled` flag for global LLM disable
- LLM metrics: `llm_calls`, `llm_cache_hits`, `llm_failures`
- 8 integration tests for LLM functionality in Alerter

#### GitHub PR#22 — LLM Explain Module (DEC-004)
- `ExplainLLM` interface with `AnthropicExplainer` and `MockExplainer`
- Strict no-new-numbers validation (`validate_llm_output_strict`)
- Fallback on any LLM failure (timeout, validation, exception)
- 9 adversarial tests for LLM guardrails

#### PR#1 — Scaffold + Contracts
- Project structure with `pyproject.toml`, editable install, dev dependencies
- Data contract models in `src/cryptoscreener/contracts/`:
  - `MarketEvent` — canonical market data from exchange
  - `FeatureSnapshot` — feature vector with windows and data health
  - `PredictionSnapshot` — ML predictions with reasons and status
  - `RankEvent` — ranking state transitions with deterministic digest
  - `LLMExplainInput/Output` — strict LLM contract (text-only)
- Schema validation utilities with adversarial LLM tests
- Replay harness (`scripts/run_replay.py`) with determinism verification
- Test fixture (`tests/fixtures/sample_run/`) with checksums
- 39 unit tests covering roundtrip, validation, and LLM guardrails
- Tooling: ruff (lint) + mypy (strict types) configured

### Changed
- None

### Fixed
- None

---

## 2026-01-24
- Documentation pack created (PRD, architecture, contracts, ops, dev)
