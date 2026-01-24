# CHANGELOG

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM
**Updated:** 2026-01-24

---

## Unreleased

### Added

#### GitHub PR#41 — Record→Replay Bridge (DEC-007)
- `scripts/run_record.py` — recording harness for fixture generation
  - CLI: `--symbols`, `--duration-s`, `--out-dir`, `--cadence-ms`, `--llm`, `--source`
  - Outputs: `market_events.jsonl`, `expected_rank_events.jsonl`, `manifest.json`
- `SyntheticMarketEventGenerator` — deterministic synthetic market event generation
- `MinimalRecordPipeline` — mirrors `MinimalReplayPipeline` for determinism
- Manifest format v1.0.0 with SHA256 checksums and replay digest
- LLM OFF by default in recording mode
- `tests/replay/test_record_replay_roundtrip.py` — 15+ tests for record→replay determinism

#### GitHub PR#32-40 — Proof Bundle v3 (DEC-006)
- 11 mandatory markers in PR body (enforced by CI)
- `scripts/proof_bundle_chat.sh` — compact chat report with strict mergeCommit handling
- File logging to `artifacts/proof_bundle_pr{N}_{timestamp}.txt`
- Verbatim paste rule for reviewer messages

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
