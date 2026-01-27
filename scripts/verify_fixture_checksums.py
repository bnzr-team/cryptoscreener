"""Verify SHA256 checksums of fixture files against their manifest.json.

Scans all directories under tests/fixtures/ and artifacts/ for manifest.json files,
then verifies:
1. Every file listed in the manifest exists and has correct SHA256 hash
2. No extra data files exist in the directory that are not tracked by the manifest
3. No symlinks or path traversal in artifact names

Supports two manifest formats:
- Simple: {"checksums": {"filename": "sha256hex"}}
- Registry: {"artifacts": [{"name": "...", "sha256": "...", "size_bytes": N}]}

Exit code 0 = all checksums match, no untracked files; 1 = error.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path, PurePath

# Files that are allowed in fixture dirs without being in manifest
IGNORED_FILES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "__init__.py",
        "__pycache__",
        ".gitkeep",
    }
)

# Extensions for auto-generated / support files (not data artifacts)
IGNORED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".pyc",
    }
)


def _sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_dir(manifest_path: Path) -> list[str]:
    """Verify checksums and detect untracked files. Returns list of errors."""
    errors: list[str] = []
    pkg_dir = manifest_path.parent
    pkg_dir_resolved = pkg_dir.resolve()

    with manifest_path.open() as f:
        data = json.load(f)

    # Build {filename: expected_sha256} from whichever format is present
    expected: dict[str, str] = {}

    if "checksums" in data and isinstance(data["checksums"], dict):
        expected = {k: v.lower() for k, v in data["checksums"].items()}
    elif "artifacts" in data and isinstance(data["artifacts"], list):
        for entry in data["artifacts"]:
            expected[entry["name"]] = entry["sha256"].lower()

    if not expected:
        errors.append(f"{manifest_path}: no checksums found in manifest")
        return errors

    # --- Validate each manifest entry ---
    for filename, want_hash in sorted(expected.items()):
        # Security: reject path traversal / absolute paths / subdirectories
        pure = PurePath(filename)
        if pure.is_absolute():
            errors.append(f"{filename}: absolute path in manifest (rejected)")
            continue
        if len(pure.parts) != 1 or ".." in pure.parts:
            errors.append(f"{filename}: path traversal in manifest (rejected)")
            continue

        filepath = pkg_dir / filename

        # Security: verify resolved path stays inside package directory
        try:
            filepath.resolve().relative_to(pkg_dir_resolved)
        except (ValueError, OSError):
            errors.append(f"{filename}: resolves outside package directory")
            continue

        if not filepath.exists():
            errors.append(f"{filepath}: file missing (expected by manifest)")
            continue

        # Security: reject symlinks
        if filepath.is_symlink():
            errors.append(f"{filepath}: is a symlink (not allowed)")
            continue

        got_hash = _sha256(filepath)
        if got_hash != want_hash:
            errors.append(
                f"{filepath}: SHA256 mismatch: expected {want_hash[:16]}..., got {got_hash[:16]}..."
            )
        else:
            print(f"  OK {filename} ({want_hash[:16]}...)")

    # --- Detect untracked data files ---
    tracked_names = set(expected.keys()) | IGNORED_FILES
    for child in sorted(pkg_dir.iterdir()):
        if child.name in IGNORED_FILES:
            continue
        if child.is_dir():
            # Skip directories (__pycache__, etc.)
            continue
        if child.suffix in IGNORED_SUFFIXES:
            continue
        if child.name not in tracked_names:
            errors.append(
                f"{child}: untracked data file not in manifest "
                f"(add to manifest or to IGNORED_FILES/IGNORED_SUFFIXES)"
            )

    return errors


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    search_dirs = [root / "tests" / "fixtures", root / "artifacts"]

    manifests: list[Path] = []
    for search_dir in search_dirs:
        if search_dir.is_dir():
            manifests.extend(search_dir.rglob("manifest.json"))

    if not manifests:
        print("ERROR: No manifest.json files found")
        return 1

    all_errors: list[str] = []
    for mp in sorted(manifests):
        print(f"\n=== {mp.relative_to(root)} ===")
        errs = _verify_dir(mp)
        all_errors.extend(errs)

    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s):")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print(f"PASSED: {len(manifests)} manifest(s) verified, all checksums match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
