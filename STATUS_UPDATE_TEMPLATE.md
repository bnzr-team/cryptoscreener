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

## Как обновить **STATUS_UPDATE_TEMPLATE.md** (чтобы он приносил ровно то, что нужно)

Сделай шаблон жёстким: без этих полей апдейт считается невалидным.

### STATUS_UPDATE_TEMPLATE.md (готовый блок)

**PR Title / ID:**
**Scope vs docs (SPEC/PRD refs):**
**Commit hash:**
**Base / target branch:**

### 1) Git proof (paste outputs)

```bash
git status
git show --stat <commit>
git show <commit>
```

### 2) Tool versions (paste outputs)

```bash
python --version
ruff --version
mypy --version
pytest --version
```

### 3) Quality gates (raw output)

```bash
ruff check .
mypy .
pytest -q
```

### 4) Files changed (table)

* path | purpose | contracts/spec touched (yes/no)

### 5) Contracts proof

* Fixtures used (paths):
* sha256 (ALL files):

```bash
sha256sum <list all fixture files>
```

* Validation command(s) + raw output:
* Roundtrip tests: name(s) + raw output:

### 6) Replay determinism (if any behavior / outputs exist)

* Command:
* Full log:
* Emitted vs expected comparison summary:
* Digest method (canonical JSON rules):

  * sort_keys:
  * separators:
  * newline policy:

### 7) LLM guardrails (if LLM touched)

* No-new-numbers tests:
* Enum/status tests:
* Max length tests:
* Fallback tests:
* Any DECISIONS.md changes? (link/summary)

### 8) Open issues / risks

* item | severity | mitigation

### 9) Next steps (next PR scope)

* …

---

## Маленький бонус: “stop conditions” (очень экономит время)

Добавь в оба файла коротко:

* Нет `git show <commit>` → **update rejected**
* Нет raw outputs → **update rejected**
* Нет sha256 fixture bundle (если есть fixtures) → **update rejected**
* Любое изменение контрактов/политик без `DECISIONS.md` → **update rejected**

---

Если хочешь, я могу ещё предложить **мини-чеклист для PR description** (чтобы Claude копипастил прямо в PR body), но уже эти две правки обычно решают проблему на 80–90%.
