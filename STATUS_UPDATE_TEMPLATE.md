# STATUS_UPDATE_TEMPLATE.md
Use this template for every Claude progress update. Keep it factual and reproducible.

---

## 1) Summary (what you claim is done)
- [ ] Bullet 1 (link to file(s))
- [ ] Bullet 2
- [ ] Bullet 3

**Scope check:** confirm you did NOT add features outside PRD/SPEC. If you did, list them and reference DECISIONS.md.

---

## 2) Git proof
- Commit hash: `<hash>`  (or paste `git diff` snippet if uncommitted)
- Files changed (high level):
  - `path/to/file.py` — what changed
  - `docs/...` — what changed

---

## 3) Commands + raw output (paste verbatim)

### 3.1 Lint
Command:
```bash
ruff check .
```
Output:
```text
<PASTE RAW OUTPUT>
```

### 3.2 Types
Command:
```bash
mypy .
```
Output:
```text
<PASTE RAW OUTPUT>
```

### 3.3 Tests
Command:
```bash
pytest -q
```
Output:
```text
<PASTE RAW OUTPUT>
```

---

## 4) Contracts proof (required if you touched any contracts/payloads)
### 4.1 Sample JSON payloads
Provide at least one example each (matching DATA_CONTRACTS exactly):
- MarketEvent
- FeatureSnapshot
- PredictionSnapshot
- RankEvent
- (If LLM) LLM explain input + output

Paste as JSON:
```json
{}
```

### 4.2 Schema validation
- Show how you validated (unit test or schema tool)
- Paste the validation output

---

## 5) Replay determinism proof (non-negotiable for pipeline changes)

### 5.1 Command
```bash
python -m scripts.run_replay --fixture tests/fixtures/sample_run/
```

### 5.2 Evidence
- Attach the relevant log excerpt
- Provide a digest:
  - number of RankEvents: `<n>`
  - sha256 of RankEvent stream (or stable summary): `<hash>`
  - tolerance used: `<tol>`

### 5.3 “Before vs after”
If behavior changed, provide:
- before hash + after hash
- explanation + DECISIONS.md reference (if intentional)

---

## 6) Binance safety proof (required when touching connectors/governor)
- WS sharding summary: connections=<n>, streams/conn=<max>, headroom=<%>
- Reconnect policy: base/cap/jitter and max reconnect/min
- 429/418 counters in a short test run
- Any rate-limit incidents (log excerpt)

---

## 7) LLM guardrails proof (required when touching LLM)
- Schema validation tests passing
- “No-new-numbers” adversarial test cases (list + outcome)
- Fallback behavior demo (LLM disabled or invalid output)

---

## 8) Metrics snapshot (at least once per milestone)
- e2e latency p50/p95/p99
- dropped_events_total
- stale_book_ms distribution summary
- top-K churn rate (events/min)

---

## 9) Open issues / risks / blockers
- Issue 1 + severity + proposed next step
- Issue 2

---

## 10) Next steps (explicit)
1. …
2. …
3. …
