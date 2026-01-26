"""MLRunner model fixture for determinism testing.

SECURITY: No pickle files are stored in the repository.
Model artifacts are generated on-the-fly from source code.
"""

from __future__ import annotations

import hashlib
import pickle
import tempfile
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent
CALIBRATION_PATH = FIXTURE_DIR / "calibration.json"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"

# Expected SHA256 for generated model (deterministic from seed=42)
# This ensures the generated model matches expected output
EXPECTED_MODEL_SHA256 = "25a8dddab9d992c6924263c2f5c3e1046930413f21599fb2d0d4b5c6e6ac9b3c"
CALIBRATION_SHA256 = "19d3d37c26ac98eed4c119a1e1a54f7072149cc3c88bba9e7f89931591581128"

# Cache for generated model path (per-process)
_model_cache: dict[str, Path] = {}


def get_model_path() -> Path:
    """Get path to generated model artifact.

    Generates model.pkl on-the-fly in a temp directory.
    The model is deterministic (seed=42) and cached per-process.

    Returns:
        Path to generated model.pkl file.

    Raises:
        AssertionError: If generated model SHA256 doesn't match expected.
    """
    if "model_path" in _model_cache:
        path = _model_cache["model_path"]
        if path.exists():
            return path

    # Import here to avoid circular imports
    from tests.fixtures.mlrunner_model.deterministic_model import DeterministicModel

    # Create model in temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="mlrunner_test_"))
    model_path = temp_dir / "model.pkl"

    # Generate deterministic model
    model = DeterministicModel(seed=42)
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    # Verify SHA256 matches expected (ensures determinism)
    with model_path.open("rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()

    assert actual_sha256 == EXPECTED_MODEL_SHA256, (
        f"Generated model SHA256 mismatch!\n"
        f"Expected: {EXPECTED_MODEL_SHA256}\n"
        f"Actual:   {actual_sha256}\n"
        "This indicates non-deterministic model generation."
    )

    _model_cache["model_path"] = model_path
    return model_path


def get_model_sha256() -> str:
    """Get SHA256 of the generated model.

    Returns:
        SHA256 hex digest of the model file.
    """
    return EXPECTED_MODEL_SHA256


# Convenience alias for backward compatibility
MODEL_SHA256 = EXPECTED_MODEL_SHA256
