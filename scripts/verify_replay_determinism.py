"""Double-run replay determinism gate.

Runs the replay harness twice on every fixture that has market_events.jsonl,
then asserts both runs produce identical RankEvent stream digests.

Exit code 0 = deterministic; 1 = mismatch or error.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.run_replay import run_replay


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    fixtures_dir = root / "tests" / "fixtures"

    # Find all fixture dirs with market_events.jsonl
    fixture_dirs = sorted(
        p.parent for p in fixtures_dir.rglob("market_events.jsonl")
    )

    if not fixture_dirs:
        print("ERROR: No fixtures with market_events.jsonl found")
        return 1

    all_ok = True
    for fdir in fixture_dirs:
        rel = fdir.relative_to(root)
        print(f"\n=== {rel} ===")

        # Run 1
        _, digest1, passed1 = run_replay(fdir, verify_expected=True)
        print(f"  Run 1 digest: {digest1}  (expected match: {passed1})")

        # Run 2 (fresh pipeline instance inside run_replay)
        _, digest2, passed2 = run_replay(fdir, verify_expected=True)
        print(f"  Run 2 digest: {digest2}  (expected match: {passed2})")

        if digest1 != digest2:
            print("  FAIL: digests differ between runs!")
            all_ok = False
        elif not passed1:
            print("  FAIL: digest does not match expected")
            all_ok = False
        else:
            print(f"  PASS: both runs match (digest={digest1[:16]}...)")

    print()
    if all_ok:
        print("ALL DIGESTS MATCH — determinism verified.")
        return 0
    else:
        print("DETERMINISM CHECK FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
