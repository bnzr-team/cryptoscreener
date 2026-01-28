#!/usr/bin/env python3
"""DEC-034: Secret hygiene CI gate.

Scans tracked files for patterns that look like leaked secrets
(high-entropy strings, known key prefixes, env var assignments with
real-looking values).

Exit code 0 = clean, 1 = findings detected.

Usage:
    python -m scripts.secret_guard           # scan repo
    python -m scripts.secret_guard --verbose  # show matched lines
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Files / directories always excluded from scanning.
EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        "artifacts",
    }
)

# Files excluded from scanning (contain legitimate hashes/digests, not secrets).
EXCLUDE_FILES: frozenset[str] = frozenset(
    {
        "MANIFEST.md",
        "DECISIONS.md",
        "STATE.md",
        "CHANGELOG.md",
        "tests/test_secret_guard.py",
    }
)

# Path prefixes excluded (contain legitimate SHA256 fixture digests).
EXCLUDE_PREFIXES: tuple[str, ...] = (
    "tests/fixtures/",
    "docs/audits/",
)

EXCLUDE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".whl",
        ".egg-info",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".lock",
    }
)

# Binary / non-text files we never scan.
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".gz",
        ".tar",
        ".zip",
        ".bz2",
        ".xz",
        ".bin",
        ".exe",
        ".dll",
    }
)

# Regex patterns that strongly suggest a leaked secret.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # AWS
    ("AWS Access Key ID", re.compile(r"AKIA[0-9A-Z]{16}")),
    # Generic high-entropy hex (>= 40 chars, e.g. API keys)
    ("Long hex string", re.compile(r"[0-9a-fA-F]{40,}")),
    # Generic base64 secret assignment (KEY=<base64 value>)
    (
        "Env var with base64 value",
        re.compile(
            r"""(?:API_KEY|SECRET_KEY|SECRET|TOKEN|PASSWORD|PRIVATE_KEY)"""
            r"""[\s]*[=:]\s*["']?[A-Za-z0-9+/]{32,}={0,2}["']?"""
        ),
    ),
]

# Files that are *expected* to contain placeholder/template values.
# These get scanned but findings in them require the value to NOT be
# a placeholder like "REPLACE_ME", "changeme", "xxx", etc.
TEMPLATE_FILES: frozenset[str] = frozenset(
    {
        "k8s/secret.yaml",
    }
)

PLACEHOLDER_PATTERNS: re.Pattern[str] = re.compile(
    r"REPLACE_ME|changeme|xxx|TODO|FIXME|<your-|example\.com",
    re.IGNORECASE,
)

# Lines matching these patterns contain legitimate hex digests (not secrets).
DIGEST_CONTEXT: re.Pattern[str] = re.compile(
    r"sha256|digest|expected|checksum|commit|_sha\b",
    re.IGNORECASE,
)

# Minimum Shannon entropy (bits per char) to flag a string as suspicious.
ENTROPY_THRESHOLD: float = 4.0
ENTROPY_MIN_LENGTH: int = 20


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single secret-leak finding."""

    file: str
    line_no: int
    rule: str
    snippet: str


@dataclass
class ScanResult:
    """Aggregate scan result."""

    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0

    @property
    def clean(self) -> bool:
        return len(self.findings) == 0


def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy in bits per character."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def _is_template_file(path: str) -> bool:
    return any(path.endswith(t) or path == t for t in TEMPLATE_FILES)


def _is_placeholder(line: str) -> bool:
    return bool(PLACEHOLDER_PATTERNS.search(line))


def _get_tracked_files(repo_root: Path) -> list[Path]:
    """Get git-tracked files, falling back to walking the directory."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=True,
        )
        return [repo_root / f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: walk directory
        files = []
        for p in repo_root.rglob("*"):
            if p.is_file():
                files.append(p)
        return files


def _should_skip(path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root)
    rel_str = str(rel)
    parts = rel.parts
    if any(d in EXCLUDE_DIRS for d in parts):
        return True
    if rel_str in EXCLUDE_FILES or rel.name in EXCLUDE_FILES:
        return True
    if any(rel_str.startswith(p) for p in EXCLUDE_PREFIXES):
        return True
    suffix = path.suffix.lower()
    return suffix in EXCLUDE_EXTENSIONS or suffix in BINARY_EXTENSIONS


def scan_file(path: Path, repo_root: Path) -> list[Finding]:
    """Scan a single file for secret patterns."""
    findings: list[Finding] = []
    rel_path = str(path.relative_to(repo_root))
    is_template = _is_template_file(rel_path)

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return findings

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        # Skip comments and empty lines
        if not stripped or stripped.startswith("#"):
            continue

        for rule_name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                # In template files, skip placeholder values
                if is_template and _is_placeholder(line):
                    continue
                # Skip hex strings in digest/checksum contexts
                if rule_name == "Long hex string" and DIGEST_CONTEXT.search(line):
                    continue
                findings.append(
                    Finding(
                        file=rel_path,
                        line_no=line_no,
                        rule=rule_name,
                        snippet=stripped[:120],
                    )
                )

    return findings


def scan_repo(repo_root: Path) -> ScanResult:
    """Scan the entire repository for secret leaks."""
    result = ScanResult()
    files = _get_tracked_files(repo_root)

    for path in files:
        if _should_skip(path, repo_root):
            continue
        if not path.is_file():
            continue
        result.files_scanned += 1
        result.findings.extend(scan_file(path, repo_root))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DEC-034: Scan repo for leaked secrets.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: git root or cwd)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show matched lines",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root
    if repo_root is None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            )
            repo_root = Path(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            repo_root = Path.cwd()

    scan = scan_repo(repo_root)

    if scan.clean:
        print(f"secret_guard: PASS ({scan.files_scanned} files scanned, 0 findings)")
        return 0

    print(f"secret_guard: FAIL ({len(scan.findings)} finding(s) in {scan.files_scanned} files)")
    for f in scan.findings:
        msg = f"  {f.file}:{f.line_no} [{f.rule}]"
        if args.verbose:
            msg += f"  {f.snippet}"
        print(msg)

    return 1


if __name__ == "__main__":
    sys.exit(main())
