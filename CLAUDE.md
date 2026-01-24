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

**Формат отчёта**

* Сначала: “Что изменено” (по файлам)
* Затем: “Что доказано” (артефакты выше)
* Затем: “Что не покрыто / риски”
* Затем: “Next PR scope”


---

## Reporting format (every update)

Use `STATUS_UPDATE_TEMPLATE.md` (required).

---

## PR Automation (GitHub)

### Available scripts

| Script | Purpose |
|--------|---------|
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
- `## 1) Git proof`
- `## 2) Tool versions`
- `## 3) Quality gates`

Conditionally required (if PR touches `tests/fixtures/` or `scripts/run_replay.py`):
- `## 4) Fixtures checksums`
- `## 5) Replay determinism`

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
