"""Script to create deterministic model and calibration fixtures.

This creates tiny artifacts for testing MLRunner with real model inference.
The model is deterministic - same inputs always produce same outputs.

Run once to generate the fixtures:
    python -m tests.fixtures.mlrunner_model.create_fixture
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from tests.fixtures.mlrunner_model.deterministic_model import DeterministicModel


def create_model_fixture(output_dir: Path) -> str:
    """Create deterministic model pickle file.

    Returns:
        SHA256 hash of the created file.
    """
    model = DeterministicModel(seed=42)
    model_path = output_dir / "model.pkl"

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    # Compute hash
    sha256 = hashlib.sha256()
    with model_path.open("rb") as f:
        sha256.update(f.read())

    return sha256.hexdigest()


def create_calibration_fixture(output_dir: Path) -> str:
    """Create calibration artifact JSON file.

    Returns:
        SHA256 hash of the created file.
    """
    calibration = {
        "metadata": {
            "schema_version": "1.0.0",
            "git_sha": "abc1234567890",
            "config_hash": "test_config_hash",
            "data_hash": "test_data_hash",
            "created_at": "2026-01-25T00:00:00Z",
        },
        "calibrators": {
            "p_inplay_30s": {"type": "platt", "a": -1.5, "b": 0.2},
            "p_inplay_2m": {"type": "platt", "a": -1.2, "b": 0.1},
            "p_inplay_5m": {"type": "platt", "a": -1.0, "b": 0.05},
            "p_toxic": {"type": "platt", "a": -2.0, "b": 0.3},
        },
    }

    cal_path = output_dir / "calibration.json"
    with cal_path.open("w") as f:
        json.dump(calibration, f, indent=2)

    # Compute hash
    sha256 = hashlib.sha256()
    with cal_path.open("rb") as f:
        sha256.update(f.read())

    return sha256.hexdigest()


def create_manifest(output_dir: Path, model_sha: str, cal_sha: str) -> None:
    """Create manifest with file hashes."""
    manifest = {
        "version": "1.0.0",
        "description": "Deterministic MLRunner test fixture",
        "files": {
            "model": "model.pkl",
            "calibration": "calibration.json",
        },
        "checksums": {
            "model.pkl": model_sha,
            "calibration.json": cal_sha,
        },
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    """Create all fixture files."""
    output_dir = Path(__file__).parent

    print(f"Creating fixtures in {output_dir}")

    model_sha = create_model_fixture(output_dir)
    print(f"Created model.pkl (sha256: {model_sha[:16]}...)")

    cal_sha = create_calibration_fixture(output_dir)
    print(f"Created calibration.json (sha256: {cal_sha[:16]}...)")

    create_manifest(output_dir, model_sha, cal_sha)
    print("Created manifest.json")

    print("\nFixture creation complete!")
    print(f"Model SHA256: {model_sha}")
    print(f"Calibration SHA256: {cal_sha}")


if __name__ == "__main__":
    main()
