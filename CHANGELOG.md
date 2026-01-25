# CHANGELOG

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM
**Updated:** 2026-01-24

---

## Unreleased

### Added

#### GitHub PR#52 — CI Pipeline Verified (Smoke Test)
- **Verified commit:** `31c292d38b64936ac2e2379d11cdd1f94e2ca082`
- **What was tested:**
  - FULL VERBATIM mode for small diffs (README.md single-line change)
  - Manual marker detection outside managed block (PR#51 feature)
  - acceptance-packet auto-updates PR body between managed markers
  - proof-guard validates FULL VERBATIM content correctly
- **Known behavior:** proof-guard may need re-run if it starts before acceptance-packet updates body (timing, not a bug)
- **Result:** All checks pass, CI pipeline works as designed after PR#49-#51

#### GitHub PR#51 — Harden Manual Marker Detection Outside Managed Block
- `proof_guard.yml`: detect acceptance markers (`PENDING`, `CI ARTIFACT`, `IDENTITY`) outside managed block
- Manual markers in user-editable PR description now FAIL with explicit error message
- Error message explains: CI only updates inside `<!-- ACCEPTANCE_PACKET_START -->...<!-- ACCEPTANCE_PACKET_END -->`
- `acceptance_packet.yml`: fix `re.sub` backslash escape error when packet contains regex patterns (use lambda)
- CLAUDE.md updated with manual marker warning and fix instructions

#### GitHub PR#49 — CI Acceptance Packet + Auto PR Body Proof + Proof Guard CI Artifact Mode
- New workflow `.github/workflows/acceptance_packet.yml`:
  - Runs on `pull_request` (opened, synchronize, reopened, ready_for_review) + `workflow_dispatch`
  - Installs dev dependencies (`pip install -e ".[dev]"`)
  - Runs `./scripts/acceptance_packet.sh <PR_NUMBER>`
  - Always uploads artifact `acceptance_packet_pr<N>` with SHA256/size
  - Auto-updates PR body between `<!-- ACCEPTANCE_PACKET_START -->` and `<!-- ACCEPTANCE_PACKET_END -->` markers
  - Two modes: FULL VERBATIM (packet < 45KB with `diff --git`) or CI ARTIFACT (reference block)
  - Anti-loop: skips if actor is `github-actions[bot]`, no `edited` trigger
- Updated `proof_guard.yml` to support CI ARTIFACT mode:
  - PENDING mode now FAILS (was security hole - prevented merge without proof)
  - FULL VERBATIM: all markers + `diff --git` required
  - CI ARTIFACT: validates via GitHub API that check-run `Acceptance Packet` passed for HEAD_SHA
  - CI ARTIFACT mode does NOT require `diff --git` in body
- CLAUDE.md updated: "CI is the source of truth", local runs optional

#### GitHub PR#48 — Merge Safety & Stacked PR Detection
- `acceptance_packet.sh`: added `== ACCEPTANCE PACKET: MERGE SAFETY ==` section
- PR chain resolution: walks up branch tree to find all prerequisite PRs by number/URL
- Fails if stacked base branch has no corresponding open PR (`STACKED_CHAIN_BROKEN`)
- New readiness classification: `Ready for review` vs `Ready for final merge`
- STACKED PRs show `Ready for final merge: false` (must merge prerequisites first)
- New `--require-main-base` flag to fail if base is not main/master
- CLAUDE.md updated with detailed stacked PR workflow and best practices

#### GitHub PR#47 — Reviewer Message Generator
- New `scripts/reviewer_message.sh <PR>` — generates ready-to-paste reviewer chat message
- Wraps acceptance_packet.sh output in clean "copy from here" format
- Shows status indicator (ready/not ready) at the end
- CLAUDE.md updated with usage instructions

#### GitHub PR#46 — Robust Replay Detection in acceptance_packet.sh
- `acceptance_packet.sh`: switched from `gh pr view --json files` to `gh api /repos/.../pulls/.../files`
- File list retrieval now matches `proof_guard.yml` logic (paginated API call)
- Added fail-safe: empty file list now fails packet (prevents false `replay_required: false`)
- Added repo detection via `gh repo view --json nameWithOwner`

#### GitHub PR#45 — Enforce Verbatim Acceptance Packet
- `acceptance_packet.sh`: removed SUMMARY section, single `== ACCEPTANCE PACKET: END ==` marker
- `acceptance_packet.sh`: added "Paste verbatim, do NOT summarize" banner
- `proof_guard.yml`: disabled proof_bundle fallback — acceptance_packet is now mandatory
- `proof_guard.yml`: added `diff --git` check to ensure full patch is included
- No more shortcuts: PR body must contain complete verbatim acceptance packet output

#### GitHub PR#44 — Global Proof & Reporting Policy (DEC-009)
- Verbatim-only reporting policy: summaries/tables/paraphrasing are NOT valid proof
- DEC-009: Global proof policy — acceptable evidence is only verbatim packet output
- `proof_guard.yml` dual-mode marker validation:
  - Preferred: acceptance_packet markers (`== ACCEPTANCE PACKET: *`)
  - Fallback: proof_bundle markers (backward compatible)
- Replay determinism enforcement in acceptance_packet mode requires `ALL DIGESTS MATCH`
- CLAUDE.md "Формат отчёта" replaced with verbatim-only mandatory format

#### GitHub PR#43 — Acceptance Packet Automation (DEC-008)
- New `scripts/acceptance_packet.sh <PR>` — one-command "ready for ACCEPT" generator
  - Waits for CI checks to pass (polls with timeout)
  - Runs quality gates (ruff, mypy, pytest)
  - Auto-generates proof bundle artifacts if missing
  - Shows full PR diff
  - Validates replay determinism if PR touches replay-related files
- Updated `scripts/proof_bundle.sh`: python→python3, use `gh pr diff` for full PR diff
- Updated `scripts/proof_bundle_chat.sh`: auto-generate artifacts if missing
- Expanded `.github/workflows/proof_guard.yml` replay detection to include `run_record.py` and `tests/replay/`
- Added mandatory `acceptance_packet.sh` usage rule to CLAUDE.md

#### GitHub PR#25 — Live Pipeline End-to-End (DEC-006)
- `scripts/run_live.py`: Full live pipeline connecting all components
- Data flow: BinanceStreamManager → StreamRouter → FeatureEngine → BaselineRunner → Ranker → Alerter
- Aggregated `PipelineMetrics` with event counts, latencies, LLM stats
- Graceful shutdown via SIGINT/SIGTERM (flag-based, no race conditions)
- CLI flags: `--symbols`, `--top N`, `--cadence`, `--llm`, `--output`, `--duration-s`, `--verbose`
- `--duration-s` for controlled test runs without manual SIGTERM
- One-time REST call for `--top N` mode (symbol list only, not polling)
- LLM disabled by default in live mode (per DEC-005)

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
