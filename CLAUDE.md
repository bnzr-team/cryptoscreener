# CLAUDE.md — Implementation Instructions for In‑Play Predictor (CryptoScreener‑X) — ML + LLM

You are the primary coder for this repository. Build exactly what is specified in the PRD/ARCH/DATA_CONTRACTS documents.

## 0) Ground rules (non-negotiable)
1. Maintain **single source of truth** files and keep them updated when changes happen:
   - `DECISIONS.md` — decisions + rationale + date
   - `SPEC.md` — current spec (normative)
   - `STATE.md` — current state/progress + known issues
   - `CHANGELOG.md` — user-visible changes
2. When reporting progress, provide a **proof bundle**:
   - raw command output (e.g., pytest), logs, checksums/hashes, git diff, fixtures, JSON contracts
   - do not claim completion without reproducible evidence
3. Do not change numeric scoring logic using the LLM. LLM is *text only*.
4. **Auto-PR rule**: After completing any task, automatically:
   - Run quality gates (ruff, mypy, pytest)
   - If all pass → create PR with proof bundle body (CLAUDE.md format)
   - Enable auto-merge (squash)
   - Provide Proof Bundle Report after merge
   - No user confirmation needed — just do it

## 1) Repo layout (create if missing)
```
/connectors/binance/
/stream_router/
/features/
/models/
/scoring/
/ranker/
/explain_llm/
/notifiers/
/storage/
/observability/
/scripts/
/tests/
/docs/   (PRD, ARCH, LIMITS, DATA_CONTRACTS)
/configs/
```

## 2) Language & tooling
- Python 3.11+
- Async-first (asyncio) for WS ingestion
- Type hints everywhere; `ruff` + `mypy`
- Tests: `pytest`
- Serialization: `orjson` or stdlib json (consistent)

## 3) Interfaces to implement (contracts are strict)
Implement exactly the JSON contracts in `docs/DATA_CONTRACTS.md`:
- `MarketEvent`
- `FeatureSnapshot`
- `PredictionSnapshot`
- `RankEvent`
- `LLM explain input/output`

All modules must accept/return these objects; no ad-hoc dicts.

## 4) Binance connector requirements
- WS market streams are the primary data source.
- Implement sharding across multiple WS connections:
  - each connection <= 800 streams (headroom under 1024)
  - enforce <= 10 incoming messages/sec per connection (throttle control msgs)
- Implement robust reconnect:
  - exponential backoff + jitter
  - max reconnect rate per IP
  - circuit breaker on repeated disconnects
- Implement REST bootstrap only:
  - exchangeInfo for symbol metadata
  - slow endpoints at low cadence (minutes)

## 5) Feature engine
- Ring buffers per symbol for windows 1s/10s/60s/5m
- Deterministic feature computation (same code used offline & online)
- Emit `FeatureSnapshot` on cadence (default 1s; configurable)

## 6) ML inference (v1)
- Start with LightGBM/CatBoost model artifacts loaded from `models/artifacts/`
- Provide a `ModelRunner` abstraction:
  - `predict(snapshot: FeatureSnapshot) -> PredictionSnapshot`
- Provide calibration adapters (isotonic/Platt) as separate artifacts.

If model artifacts absent, run in **baseline mode**:
- compute heuristic score from cost + volatility + flow (so system still works)

## 7) Toxicity model
- Provide `p_toxic` head and apply penalties/gates in scoring.

## 8) Ranker and anti-flicker
- Maintain top‑K with hysteresis:
  - enter threshold > exit threshold
  - min dwell time
- Emit RankEvent deltas rather than spamming full lists.

## 9) LLM explainability
- Provide a strict `ExplainLLM` interface.
- LLM input must only contain:
  - reason codes + numeric summary + allowed labels
- Validate LLM output against schema:
  - max length, no new numbers
  - required keys: headline, status_label
- Provide “LLM off” mode producing only deterministic reason codes.

## 10) Observability and safety
- Metrics: message rate, lag, stale ms, latency p95, reconnect count, 429/418 events
- Structured logs with correlation ids
- On any 429/418: reduce load and log an incident event.

## 11) Local dev commands (must work)
- `make test` or equivalent → runs pytest
- `python -m scripts.run_live` → live stream demo (read-only)
- `python -m scripts.run_replay` → replay recorded data
- `python -m scripts.train` → training (if data exists)

## 12) Definition of Done for each PR
- Tests pass
- Contracts validated
- Proof bundle attached
- SSOT files updated when necessary


---

## Progress reporting / Proof bundle
**Обязательное правило:**

* Любое “готово/сделано” без доказательств = **NOT DONE**.
* Каждый PR обязан включать **patch-level** доказательства и **сырой вывод** команд.

**Требуемый proof bundle для каждого PR**

1. **Git**

* `git status` (должно быть clean)
* `git show --stat <commit>`
* `git show <commit>` (полный патч) **или** `git diff <base>..<commit>`
* (если PR в GitHub/GitLab) ссылка на PR + target branch

2. **Toolchain versions**

* `python --version`
* `ruff --version`
* `mypy --version`
* `pytest --version`

3. **Quality gates (raw output, без сокращений)**

* `ruff check .`
* `mypy .`
* `pytest -q`

4. **Contracts**

* 1–2 **реальных** JSON из fixtures (не “пример руками”), пути к файлам
* доказательство валидации: имя теста/команды + raw output
* roundtrip тесты: `to_json/from_json` (минимум по одному на контракт)

5. **Determinism / Replay (если затронуто поведение)**

* команда запуска `run_replay`
* лог целиком
* sha256 **всех** fixture файлов (market/expected/manifest/…)
* доказательство сравнения **emitted vs expected** + digest

6. **LLM guardrails (если есть LLM контракты/валидаторы)**

* тесты на “no-new-numbers”
* тесты на enums (`status_label`)
* тест на `max_chars`
* fallback: тест “invalid → fallback”, и тест “fallback always valid”
* если меняется политика — **обязателен** `DECISIONS.md` + обновление доков

**Reporting format (MANDATORY, verbatim-only)**

1. Before claiming "done/ready", generate a proof packet and paste it verbatim into the PR body (and reviewer chat).
2. Preferred command:
   ```bash
   ./scripts/acceptance_packet.sh <PR_NUMBER>
   ```
3. Fallback command (if acceptance_packet is unavailable):
   ```bash
   ./scripts/proof_bundle.sh <PR_NUMBER>
   ```
4. Paste the **full output verbatim**, without truncation, from the first marker to the END marker.
5. **NO summaries, no tables, no paraphrasing.** Any "ready/done" without verbatim packet = **NOT DONE**.
6. If exit code ≠ 0: paste the failing output verbatim and fix until it passes.
7. If reviewer requests extra evidence: update `acceptance_packet.sh` (preferred) so the packet contains it, re-run, paste verbatim.

---

## Reporting format (every update)

Use `STATUS_UPDATE_TEMPLATE.md` (required).

---

## PR Automation (GitHub)

### Proof Bundle Standard v3 (CRITICAL — enforced by CI)

**PR не готов без raw proof bundle.** Никаких пересказов ("All passed!") вместо сырого вывода команд.

**Для каждого PR:**

1. Создай PR (получи номер)
2. Запусти `./scripts/proof_bundle.sh <PR_NUMBER>`
3. Вставь **весь raw output** в PR body (не редактируй маркеры!)

Скрипт включает: PR identity, PR checks, changed files, toolchain versions, git show, ruff/mypy/pytest.

**11 обязательных маркеров** (CI фейлит без них):
- `== PROOF_BUNDLE_FILE ==`
- `== PR URL ==`
- `== GH PR VIEW ==`
- `== GH PR CHECKS ==`
- `== CHANGED FILES ==`
- `== TOOLCHAIN VERSIONS ==`
- `== GIT SHOW --STAT ==`
- `== GIT SHOW ==`
- `== RUFF CHECK . ==`
- `== MYPY . ==`
- `== PYTEST -Q ==`

**Conditional требования:**

| Если PR затрагивает | Требуется |
|---------------------|-----------|
| contracts/events/models | 3 JSON примера + roundtrip/schema proof (raw) |
| runner/ranker/alerter/replay/run_live | sha256 fixture + digest + determinism proof (run #1 == run #2) |

### Available scripts

| Script | Purpose |
|--------|---------|
| `scripts/proof_bundle.sh <PR#>` | Prints full proof bundle (11 markers) + saves to `artifacts/` — paste into PR body |
| `scripts/proof_bundle_chat.sh <PR#>` | Prints compact chat report (4 blocks: identity/CI/git/gates) for reviewer message |
| `scripts/gen_proof_bundle.sh` | Generates proof bundle markdown with git, tools, quality gates, checksums, replay |
| `scripts/pr_create.sh` | Full PR cycle: branch → commit → push → proof → PR → auto-merge |

### Workflow for creating a PR

1. **Make changes** — implement feature/fix
2. **Run single command:**
   ```bash
   ./scripts/pr_create.sh "feature/pr-00XX-name" "PR#X: description"
   ```
3. **Script automatically:**
   - Creates/switches to branch
   - Commits all changes
   - Pushes to GitHub
   - Generates proof bundle (git, ruff, mypy, pytest, checksums, replay)
   - Creates PR with proof bundle as body
   - Enables auto-merge (squash) + delete-branch

### GitHub Actions (CI)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push to main, all PRs | Runs ruff, mypy, pytest, replay fixture |
| `proof_guard.yml` | PR opened/edited | Validates PR body has required proof bundle sections |

### Required proof bundle sections (enforced by proof_guard)

Always required:
- `## 0) Header`
- `## 1) Scope`
- `## 2) Files Changed`
- `## 3) Proof Artifacts`
- `## 4) Quality Gates`
- `## 9) Git Evidence`

Conditionally required (if PR touches `tests/fixtures/` or `scripts/run_replay.py`):
- `## 5) Replay Determinism`

### Pre-flight checklist (before running pr_create.sh)

1. Activate venv: `source .venv/bin/activate`
2. Ensure changes exist: `git status` shows modified files
3. Verify quality gates pass locally:
   ```bash
   ruff check . && mypy . && pytest -q
   ```

### Manual proof bundle generation

If you need proof bundle without creating PR:
```bash
./scripts/gen_proof_bundle.sh [fixture_dir] [output_file]
# Default: tests/fixtures/sample_run → proof_bundle.md
```

### Step-by-step PR creation algorithm

**This is the exact sequence Claude must follow for every PR:**

1. **Make changes** — implement feature/fix in code
2. **Verify quality gates locally:**
   ```bash
   source .venv/bin/activate
   ruff check . && mypy . && pytest -q
   ```
3. **Run PR creation script:**
   ```bash
   ./scripts/pr_create.sh "feature/pr-00XX-name" "PR#X: description"
   ```
4. **If push fails** (authentication error):
   ```bash
   gh auth setup-git
   git push -u origin <branch-name>
   mkdir -p proof
   ./scripts/gen_proof_bundle.sh tests/fixtures/sample_run proof/proof_<branch>.md
   gh pr create --base main --head <branch> --title "PR title" --body-file proof/proof_<branch>.md
   gh pr merge --auto --squash --delete-branch <pr-url>
   ```
5. **Verify PR created** — confirm URL returned
6. **Auto-merge enabled** — CI will run and merge if passing

**Key principle:** Never claim "done" without running these steps and providing the PR URL as proof.

---

## Required PR Proof Bundle Report (Template)

When reporting PR completion, Claude MUST use this exact format in a **single message**.

### 0) Header

| Field | Value |
|-------|-------|
| **PR** | `#<num>` — `<title>` |
| **Status** | OPEN / MERGED |
| **PR URL** | `<url>` |
| **Base** | `<branch> @ <hash>` |
| **Head** | `<hash>` |
| **Merge commit** | `<hash>` (if merged) |
| **Files** | `<count>` |
| **Lines** | `+<added> / -<deleted>` |
| **Tests** | `<before> → <after> (+<delta>)` |

### 1) Scope (max 6 bullets)

Bullet list only. No marketing. No "should". Pure facts: what changed.

### 2) Files Changed (table)

| File | Role | LOC |
|------|------|-----|
| `path/to/file.py` | Brief description (3–8 words) | +N |

Include **all** touched files.

### 3) Proof Artifacts (sha256)

```bash
# Patch file
<sha256>  proof/pr<id>.patch

# Contract examples
<sha256>  tests/contract_examples/<file>.json
...

# Fixtures
<sha256>  tests/fixtures/sample_run/<file>
...
```

### 4) Quality Gates (raw output)

Each tool in **separate code block**, verbatim terminal output:

```bash
$ ruff check .
<raw output>
```

```bash
$ mypy .
<raw output>
```

```bash
$ pytest -q
<raw output with test counts>
```

### 5) Replay Determinism (raw output)

**MUST include full verbose log** with these key lines visible:
- `RankEvent stream digest: <sha256>` (computed)
- `Expected digest: <sha256>`
- `✓ DETERMINISM CHECK PASSED: digests match` (or FAILED)

```bash
$ python -m scripts.run_replay --fixture <path> -v
2026-01-24 ... [INFO] Fixture: <path>
2026-01-24 ... [INFO]   market_events.jsonl sha256: <sha256>
2026-01-24 ... [INFO]   expected_rank_events.jsonl sha256: <sha256>
...
2026-01-24 ... [INFO] RankEvent stream digest: <sha256>
2026-01-24 ... [INFO] Expected digest: <sha256>
2026-01-24 ... [INFO] ✓ DETERMINISM CHECK PASSED: digests match

============================================================
REPLAY SUMMARY
============================================================
Fixture:        <path>
RankEvents:     N
Computed digest: <sha256>
Expected digest: <sha256>
Determinism:    PASSED
============================================================
```

### 6) Contract Validation Evidence

- Examples location: `tests/contract_examples/`
- Tests that validate:
  - Schema validation against Pydantic models
  - `extra="forbid"` rejects unknown fields
  - Roundtrip `to_json/from_json`
- If any contract changed: **must** reference SSOT update

### 7) SSOT Updates (mandatory section)

| File | Updated | Commit/Reason |
|------|---------|---------------|
| DECISIONS.md | yes/no | reason or commit hash |
| SPEC.md | yes/no | reason or commit hash |
| STATE.md | yes/no | reason or commit hash |
| CHANGELOG.md | yes/no | reason or commit hash |
| DATA_CONTRACTS.md | yes/no | reason or commit hash |
| LLM_INPUT_OUTPUT_SCHEMA.md | yes/no | reason or commit hash |

**Rule:** If report includes any "by design" deviation, relaxed rule, new default, new enum/value, new behavior → **DECISIONS.md must be "yes"** and relevant SSOT docs must be updated.

### 8) Risks / Deviations (max 3)

Each item must be one of:
- `Mismatch vs SSOT (needs DECISION)` — requires follow-up
- `Known limitation (explicitly accepted in DECISIONS)` — already documented
- `Follow-up required (tracked issue link)` — deferred

No operational chatter ("workflow failed once"). Only items that impact correctness or reproducibility.

### 9) Git Evidence

```bash
$ git status
<raw output — must be clean>

$ git log -5 --oneline
<raw output>

$ git show <head> --stat
<raw output>
```

---

## Self-Lint Rules (Claude MUST verify before sending)

| Rule | Check |
|------|-------|
| A1 | Header table present and complete |
| A2 | PR URL is valid GitHub PR link |
| B1 | Scope ≤ 6 bullets |
| B2 | No marketing/subjective words in Scope |
| C1 | Files table includes ALL touched files |
| D1 | All sha256 hashes are 64 hex chars |
| E1 | Quality gates have separate code blocks with `$` prefix |
| F1 | SSOT Updates: every "yes" has commit hash |
| F2 | SSOT Updates: no "yes" with "Need to..." |
| F3 | Risks reference DECISIONS.md or issue link |
| G | Footer line present and correctly formatted |

---

## Hard Reject Criteria

Report is **rejected** if any of:
- No raw outputs (summaries instead of terminal output)
- No sha256 for fixtures/examples/patch
- No replay determinism evidence (if fixtures touched)
- Any "by design" change without DECISIONS.md + SSOT doc update
- Risks/Deviations contradict SSOT Updates section
- Multiple messages instead of single consolidated report
- Missing or malformed footer line

### Git Evidence Requirements (CRITICAL)

**"ahead of origin by N commits" is NOT sufficient proof.**

Required git evidence:
1. **All commit hashes** (full 40 chars or minimum 12 chars)
2. `git log -N --oneline` (where N = number of new commits)
3. `git show <each_commit> --stat` for each commit
4. `git show <each_commit>` (full diff) OR `git diff <base>..<head>`
5. If PR exists: PR URL + "Files changed" link

**Why:** Without commit hashes and diffs, changes may be:
- Not actually committed
- Different from claimed
- Impossible to verify critical sections (validation, fallback, etc.)

### pyproject.toml Changes (CRITICAL for mypy/ruff ignores)

Any modification to `pyproject.toml` that adds ignores MUST include:
1. **Exact diff** of the change (`git diff pyproject.toml`)
2. **Justification**: why this ignore is needed
3. **Scope verification**: confirm ignore is point-targeted (e.g., `module = "anthropic"` only), not global amnesty

**Example of acceptable ignore:**
```toml
[[tool.mypy.overrides]]
module = "anthropic"  # Third-party, no stubs available
ignore_missing_imports = true
```

**Unacceptable:** Global `ignore_missing_imports = true` without module scope.

### Schema/Contract Proof (for LLM/validation changes)

When changing LLM contracts or validators, MUST include:
1. **Real JSON examples** (from fixtures or generated by tests):
   - 1x valid `LLMExplainInput` JSON
   - 1x valid `LLMExplainOutput` JSON
2. **Validation coverage proof**:
   - Test name that validates schema (Pydantic `model_validate`)
   - Test name that validates `extra="forbid"` rejects unknown fields
   - Test name that validates roundtrip `to_json/from_json`
3. **For AnthropicExplainer specifically**:
   - Proof that validation + fallback happen in ONE place (no invalid output escapes)
   - Test name: `test_validation_failure_uses_fallback` or equivalent

---

## Merge Gate Policy

1. **proof-guard.yml** validates PR body has required sections
2. **CI (ci.yml)** runs quality gates (ruff, mypy, pytest, replay)
3. **Auto-merge** enabled only after both pass
4. **Manual merge** blocked if proof-guard fails

---

## Ultra-Strict Footer (CI Parsing)

Every proof bundle report MUST end with exactly one of:

```
PROOF_BUNDLE_SELF_CHECK=PASS
```

or

```
PROOF_BUNDLE_SELF_CHECK=FAIL;RULES=A1,B2,...
```

**Rules:**
- Footer MUST be the last line of the message
- Never write anything after the footer line
- If any self-lint rule fails, list all failing rules after `RULES=`
- CI/proof-guard can parse this line to auto-reject malformed reports

---

## Reviewer Message Contract (MANDATORY)

When reporting progress to the reviewer in chat, summary-only messages are INVALID.

Every report MUST include raw evidence in the following 4 blocks (copy/paste output):

1) Identity
- PR number + URL
- state (OPEN/MERGED) + merge commit (if merged)

2) CI (raw)
- `gh pr checks <PR_NUMBER>`

3) Git evidence (raw)
- `git show <mergeCommit> --stat` (or `git show --stat` if not merged)
- Link to "Files changed"

4) Gates (raw, repo-scope)
- `ruff check .`
- `mypy .`
- `pytest -q`

**How to use:**
Run `./scripts/proof_bundle_chat.sh <PR_NUMBER>` and paste its output **verbatim** from `== CHAT PROOF: IDENTITY ==` through `== CHAT PROOF: END ==`.

**Do NOT add any commentary, summary, or extra lines.**

Any message that includes additional summary text after `== CHAT PROOF: END ==` is INVALID.

---

## Acceptance Packet (Mandatory for PR Review)

**CI is the source of truth.** The `Acceptance Packet` workflow automatically:
1. Runs `./scripts/acceptance_packet.sh <PR_NUMBER>` on every PR
2. Uploads artifact `acceptance_packet_pr<N>` with SHA256/size
3. Auto-updates PR body with proof (FULL VERBATIM or CI ARTIFACT mode)
4. Fails the check if acceptance_packet.sh exits non-zero

**Local runs are optional but allowed.** You can still run locally:
```bash
./scripts/acceptance_packet.sh <PR_NUMBER>
```

**Two proof modes in PR body:**
1. **FULL VERBATIM**: Complete packet with `diff --git` in body (auto-inserted by CI if size permits)
2. **CI ARTIFACT**: Reference block with RUN_URL, ARTIFACT_NAME, SHA256, BYTES, CHECK_NAME, HEAD_SHA (for large diffs)

**PENDING marker:**
- **What it is:** The string `== ACCEPTANCE PACKET: PENDING ==` in PR body
- **When it appears:** Immediately on PR creation, before CI has updated the body
- **Proof Guard behavior:** **ALWAYS FAILS** if PENDING marker is present (prevents merge without proof)
- **What to do:**
  1. Wait for `Acceptance Packet` workflow to complete and update PR body
  2. If workflow passed but body still has PENDING → re-run `Acceptance Packet` via workflow_dispatch
  3. Then re-run `Proof Guard` to re-validate the updated body
- **STRICT MODE:** If GitHub API fetch fails and PR body has PENDING or no valid proof markers → Proof Guard FAILS (no silent bypass)
- **Note:** Marker must be on its own line (trailing whitespace allowed). CI writes it correctly.

**Proof Guard validates:**
- FULL VERBATIM: all markers + `diff --git` present
- CI ARTIFACT: check-run `acceptance-packet` (job name) passed for HEAD_SHA via GitHub API

**Exit codes (acceptance_packet.sh):**
- `0` — All checks passed, ready for ACCEPT
- `1` — Some check failed (CI, gates, replay)
- `2` — Usage error

**Replay-required detection:**
PR requires replay proof if it touches ANY of:
- `scripts/run_replay.py`
- `scripts/run_record.py`
- `tests/replay/*`
- `tests/fixtures/*`

**Workflow:**
1. Create PR (get number)
2. CI runs automatically, updates PR body
3. If `acceptance-packet` check passes → ready for review
4. If check fails → fix issues, push, CI re-runs

**Terminology (important for debugging):**
- **Workflow name:** `Acceptance Packet` (visible in Actions tab)
- **Job/check name:** `acceptance-packet` (used by Proof Guard to validate via check-runs API)
- Proof Guard in CI ARTIFACT mode queries GitHub Check-Runs API for HEAD_SHA and looks for `name: "acceptance-packet"` with `conclusion: "success"`

**Manual paste is no longer required — CI handles everything.**

**Never edit proof blocks manually.** CI auto-updates PR body between `<!-- ACCEPTANCE_PACKET_START -->` and `<!-- ACCEPTANCE_PACKET_END -->` markers. Local runs are for diagnostics only; the source of truth is always the CI-generated packet. If you need to re-generate, use workflow_dispatch on `Acceptance Packet` workflow.

---

## Reviewer Message Generator (Convenience)

**For sending a ready-to-paste message to the reviewer chat:**

```bash
./scripts/reviewer_message.sh <PR_NUMBER>
```

This script:
1. Runs `acceptance_packet.sh` internally
2. Wraps output in a clean "copy from here" format
3. Shows status (ready/not ready) at the end

**Usage:**
1. Run `./scripts/reviewer_message.sh <PR_NUMBER>`
2. Copy everything between the dashed lines
3. Paste into reviewer chat

This is optional — you can always use `acceptance_packet.sh` directly.

---

## Stacked PRs and Merge Safety

**Stacked PRs** are PRs that target another feature branch instead of `main`. The acceptance packet automatically detects this and resolves the full PR chain.

### Merge Readiness Classification

The acceptance packet distinguishes two readiness states:

- `Ready for review: true/false` — all quality checks passed, PR can be reviewed
- `Ready for final merge: true/false` — PR can be merged to main right now

| merge_type | Ready for review | Ready for final merge |
|------------|------------------|----------------------|
| DIRECT     | ✓ (if checks pass) | ✓ (if checks pass) |
| STACKED    | ✓ (if checks pass) | ✗ (merge prerequisites first) |

### PR Chain Resolution

For stacked PRs, the acceptance packet walks up the branch tree to find all prerequisite PRs:

```
Prerequisite PR chain (merge in order):
  1. PR#47: Reviewer Message Generator (https://...)
  2. PR#46: Robust Replay Detection (https://...)
  ...
```

If a base branch has **no corresponding open PR**, the packet FAILS:
```
ERROR: Stacked base branch 'feature/xyz' has no corresponding open PR
STACKED_CHAIN_BROKEN: No open PR found for base branch 'feature/xyz'
```

### Workflow for Stacked PRs

1. Create stacked PR chain (each PR targets the previous feature branch)
2. Run `acceptance_packet.sh` on each PR — all should show `Ready for review: true`
3. Get review approval on all PRs
4. Merge PRs **in order** (starting from the one closest to main)
5. After merging prerequisites, rebase remaining PRs onto main
6. Run `acceptance_packet.sh` again to verify `Ready for final merge: true`

### Enforcing Main-Only Base

For the final PR in a chain (after prerequisites are merged):
```bash
./scripts/acceptance_packet.sh --require-main-base <PR_NUMBER>
```

This will **FAIL** if the PR base is not `main` or `master`.

### Best Practices

- Use stacked PRs for incremental review of large features
- Always check both `Ready for review` and `Ready for final merge` before merging
- Never merge a STACKED PR directly — merge prerequisites first
- Use `--require-main-base` for the final merge step
