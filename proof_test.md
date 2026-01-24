# Proof bundle

## 1) Git proof
```bash
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	proof_test.md
	scripts/gen_proof_bundle.sh

nothing added to commit but untracked files present (use "git add" to track)

b9e886d (HEAD -> main, origin/main) Merge pull request #2 from bnzr-team/docs/proof-bundle-status-template
3b42e8e (origin/docs/proof-bundle-status-template) docs: enforce proof bundle in status updates
f26a7be Merge pull request #1 from bnzr-team/ci/rename-job-checks
e477f62 (origin/ci/rename-job-checks) "ci: rename status check job to checks"
bac115f ci: add workflow
1b022e5 `chore: add PR template`
35b7764 Delete .github/pull_request_template.md
468f6ff chore: add CODEOWNERS
cf10bfc Create pull_request_template.md
d9c9751 feat(PR#1): scaffold + contracts + replay harness
```

Commit: `b9e886dbf9021c5e407cc2bd9ac746da95d808d6`

```bash
commit b9e886dbf9021c5e407cc2bd9ac746da95d808d6
Merge: f26a7be 3b42e8e
Author: bnzr-hub <benzar.evgeniy@gmail.com>
Date:   Sat Jan 24 04:38:18 2026 +0200

    Merge pull request #2 from bnzr-team/docs/proof-bundle-status-template
    
    docs: enforce proof bundle in status updates

 CLAUDE.md                 |  59 ++++++++++++++
 STATUS_UPDATE_TEMPLATE.md | 202 +++++++++++++++++++++++++---------------------
 2 files changed, 170 insertions(+), 91 deletions(-)
```

## 2) Tool versions
```bash
Python 3.12.3
ruff 0.14.14
mypy 1.19.1 (compiled: yes)
pytest 9.0.2
```

## 3) Quality gates (raw output)
```bash
All checks passed!

Success: no issues found in 10 source files

============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/benya/Project/cryptoscreener
configfile: pyproject.toml
testpaths: tests
plugins: cov-7.0.0
collected 48 items

tests/contracts/test_contracts.py ...................................... [ 79%]
..........                                                               [100%]

============================== 48 passed in 0.18s ==============================
```

## 4) Fixtures checksums
Fixture dir: `tests/fixtures/sample_run`
```bash
901a6cc399a2de563f55c1b3458edba8250b08a785978848ef890ca435e34335  expected_rank_events.jsonl
e76a70f59f268f4d4bbf40178f5a1b32ce8e685d0fdd222e584a77cb37f5f6f0  manifest.json
ba7d6e2018426517893ac4de3052a145e72b88f20d82f8d864558fca99eea277  market_events.jsonl
```

## 5) Replay determinism
```bash

============================================================
REPLAY SUMMARY
============================================================
Fixture:        tests/fixtures/sample_run
RankEvents:     3
Digest:         08f158e3d78b74e0b75318772bf0cd689859783de25c3b404ad501153efcd44d
Determinism:    PASSED
============================================================
```

## 6) Patch (git show)
```diff
commit b9e886dbf9021c5e407cc2bd9ac746da95d808d6
Merge: f26a7be 3b42e8e
Author: bnzr-hub <benzar.evgeniy@gmail.com>
Date:   Sat Jan 24 04:38:18 2026 +0200

    Merge pull request #2 from bnzr-team/docs/proof-bundle-status-template
    
    docs: enforce proof bundle in status updates

```
