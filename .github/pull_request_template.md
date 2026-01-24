Ок, давай прям “как для первого раза”, по шагам, с кликами и точными файлами. Мы уже запушили репо — дальше делаем так, чтобы GitHub сам заставлял приносить пруфы и чтобы проверки бегали автоматически.

Ниже 2 пути: **через веб-интерфейс GitHub** (самый простой для новичка) и **через WSL** (быстрее, но чуть “терминальнее”). Я рекомендую начать с **веб-интерфейса**, чтобы ты не путался.

---

# Вариант 1 (РЕКОМЕНДУЮ): всё через GitHub в браузере

## Шаг 1 — Открой свой репозиторий

1. Открой браузер
2. Перейди на репо: `bnzr-hub/cryptoscreener`
3. Убедись, что ты на ветке **main** (слева над списком файлов переключатель веток)

---

## Шаг 2 — Создай PR Template (файл, который автоподставляется в PR)

### Что это такое?

Это обычный файл в репо. GitHub сам берёт его содержимое и вставляет в описание PR.

### Как создать

1. На главной странице репо нажми кнопку **Add file** (справа сверху над списком файлов)
2. Выбери **Create new file**
3. В поле имени файла (вверху) вставь **точно**:

```
.github/pull_request_template.md
```

> Важно: точка в `.github` обязательна. GitHub сам создаст папку.

4. В большое поле “Edit new file” вставь **весь текст шаблона ниже** (я даю готовый)

### Вставь это (PR Template целиком)

````md
# PR Proof Bundle (REQUIRED)

If any section is missing, the PR is NOT REVIEWABLE.

## Header
- PR / Change ID:
- Scope (SSOT refs): PRD: <...> | SPEC: <...> | DATA_CONTRACTS: <...> | (optional) BINANCE_LIMITS: <...>
- Out of scope additions: None / <list>
- Commit hash(es):
- Base ref (e.g., main@<sha> before changes):
- Target branch:

---

## 1) Git proof (PASTE RAW OUTPUTS)
```bash
git status
````

```bash
git log --oneline --decorate -n 10
```

```bash
git show --stat <COMMIT_HASH>
```

```bash
git show <COMMIT_HASH>
```

> If multiple commits: include `git show` for each OR squash to a single commit.

---

## 2) Tool versions (PASTE RAW OUTPUTS)

```bash
python --version
ruff --version
mypy --version
pytest --version
```

---

## 3) Quality gates (PASTE RAW OUTPUTS, NO TRUNCATION)

```bash
ruff check .
```

```bash
mypy .
```

```bash
pytest -q
```

---

## 4) Files changed (diff-based table)

| Path | Change summary | SSOT impacted? (yes/no) |
| ---- | -------------- | ----------------------- |
|      |                |                         |

---

## 5) Contracts proof (required if any JSON/contracts involved)

### 5.1 Fixtures used (paths)

* tests/fixtures/...

### 5.2 SHA256 for ALL fixture files (PASTE RAW OUTPUT)

```bash
sha256sum <file1> <file2> <file3> ...
```

### 5.3 Validation evidence (PASTE RAW OUTPUT)

* Roundtrip tests:

```bash
pytest -q tests/contracts/test_contracts.py::Test<Contract>::test_roundtrip_json -v
```

### 5.4 Sample payloads (MUST be copied from real fixture files)

* <paste 1–3 lines from files>

---

## 6) Replay determinism (required if any output stream exists)

### 6.1 Command

```bash
python -m scripts.run_replay --fixture tests/fixtures/<name>/ -v
```

### 6.2 Full log

```text
<paste full log here>
```

### 6.3 Emitted vs expected summary

* emitted: <n>, expected: <n>
* digest(actual): <sha>
* digest(expected): <sha>
* tolerance: exact / <rule>

### 6.4 Digest canonicalization (MUST state)

* sort_keys: true/false
* separators: <...>
* ensure_ascii: true/false
* newline policy: <...>

---

## 7) LLM guardrails (required if LLM touched)

* Policy: "no new numbers (stringwise subset), no conversions"
* Tests (PASTE RAW OUTPUT):

```bash
pytest -q tests/contracts/test_contracts.py::TestLLMOutputValidation::test_rejects_new_number_in_headline -v
pytest -q tests/contracts/test_contracts.py::TestLLMOutputValidation::test_invalid_status_label_rejected -v
pytest -q tests/contracts/test_contracts.py::TestLLMOutputValidation::test_headline_over_limit_rejected -v
pytest -q tests/contracts/test_contracts.py::TestLLMFallback::test_validate_or_fallback_returns_fallback_on_invalid -v
pytest -q tests/contracts/test_contracts.py::TestLLMFallback::test_fallback_output_is_valid -v
```

* DECISIONS.md updated? (yes/no) + summary if yes

---

## 8) Risks / open issues

| Issue | Severity | Mitigation / follow-up |
| ----- | -------- | ---------------------- |
|       |          |                        |

---

## 9) Next steps (next PR scope)

* PR#<n> scope:
* Non-goals:

---

## STOP CONDITIONS (PR will be rejected)

* Missing `git show <commit>` patch
* Missing raw outputs for ruff/mypy/pytest
* Missing sha256 for fixtures (if fixtures exist)
* Missing determinism proof (if output exists)
* Any SSOT policy/contract change without DECISIONS.md + doc updates

```
