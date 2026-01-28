"""DEC-034: Unit tests for scripts/secret_guard.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from scripts.secret_guard import (
    main,
    scan_file,
    scan_repo,
    shannon_entropy,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal fake repo directory."""
    return tmp_path


def _write(repo: Path, name: str, content: str) -> Path:
    p = repo / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


class TestShannonEntropy:
    def test_empty_string(self) -> None:
        assert shannon_entropy("") == 0.0

    def test_single_char(self) -> None:
        assert shannon_entropy("aaaa") == 0.0

    def test_high_entropy(self) -> None:
        # Random-looking string should have high entropy
        s = "aB3$xZ9!mK2@pL7&"
        assert shannon_entropy(s) > 3.5


class TestScanFile:
    def test_clean_file(self, tmp_repo: Path) -> None:
        f = _write(tmp_repo, "clean.py", 'x = 42\nprint("hello")\n')
        findings = scan_file(f, tmp_repo)
        assert findings == []

    def test_detects_aws_key(self, tmp_repo: Path) -> None:
        f = _write(tmp_repo, "bad.py", 'key = "AKIAIOSFODNN7EXAMPLE"\n')
        findings = scan_file(f, tmp_repo)
        assert len(findings) >= 1
        assert any("AWS" in f.rule for f in findings)

    def test_detects_env_var_secret(self, tmp_repo: Path) -> None:
        f = _write(
            tmp_repo,
            "bad.env",
            "API_KEY=dGhpc2lzYXZlcnlsb25nYmFzZTY0c2VjcmV0a2V5dmFsdWU=\n",
        )
        findings = scan_file(f, tmp_repo)
        assert len(findings) >= 1

    def test_skips_comments(self, tmp_repo: Path) -> None:
        f = _write(tmp_repo, "safe.py", "# API_KEY=AKIAIOSFODNN7EXAMPLE\n")
        findings = scan_file(f, tmp_repo)
        assert findings == []

    def test_template_placeholder_skipped(self, tmp_repo: Path) -> None:
        # Simulate a template file with REPLACE_ME placeholder
        f = _write(
            tmp_repo,
            "k8s/secret.yaml",
            "  BINANCE_API_KEY: REPLACE_ME\n",
        )
        findings = scan_file(f, tmp_repo)
        assert findings == []

    def test_long_hex_detected(self, tmp_repo: Path) -> None:
        hex_str = "a" * 40
        f = _write(tmp_repo, "leak.txt", f"token={hex_str}\n")
        findings = scan_file(f, tmp_repo)
        assert len(findings) >= 1
        assert any("hex" in f.rule.lower() for f in findings)


class TestScanRepo:
    def test_clean_repo(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "main.py", "print('hello')\n")
        _write(tmp_repo, "README.md", "# My project\n")
        result = scan_repo(tmp_repo)
        assert result.clean
        assert result.files_scanned >= 1

    def test_repo_with_leak(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "config.py", 'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n')
        result = scan_repo(tmp_repo)
        assert not result.clean
        assert len(result.findings) >= 1

    def test_excludes_binary(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "data.gz", "AKIAIOSFODNN7EXAMPLE")
        _write(tmp_repo, "ok.py", "x = 1\n")
        result = scan_repo(tmp_repo)
        assert result.clean

    def test_excludes_pycache(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "__pycache__/mod.py", 'k = "AKIAIOSFODNN7EXAMPLE"\n')
        _write(tmp_repo, "ok.py", "x = 1\n")
        result = scan_repo(tmp_repo)
        assert result.clean


class TestCLI:
    def test_clean_exit_zero(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "main.py", "x = 1\n")
        rc = main(["--repo-root", str(tmp_repo)])
        assert rc == 0

    def test_dirty_exit_one(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "leak.py", 'key = "AKIAIOSFODNN7EXAMPLE"\n')
        rc = main(["--repo-root", str(tmp_repo)])
        assert rc == 1

    def test_verbose_flag(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write(tmp_repo, "leak.py", 'key = "AKIAIOSFODNN7EXAMPLE"\n')
        main(["--repo-root", str(tmp_repo), "--verbose"])
        captured = capsys.readouterr()
        assert "AKIA" in captured.out
