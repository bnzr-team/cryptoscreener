"""Model Package End-to-End Smoke Tests.

Tests the full path: ModelPackage → MLRunner load → calibration → inference.

This is the acceptance test for DEC-021: Model Package E2E Smoke (Offline).

Package E2E Contract:
1. Package integrity: manifest valid, checksums match, required files exist
2. MLRunner loads: model + calibration loaded without fallback (PROD mode)
3. Determinism: double run produces identical digests

Artifacts are generated on-the-fly to avoid storing binaries in git.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from typing import TYPE_CHECKING

import pytest

from cryptoscreener.contracts.events import (
    DataHealth,
    Features,
    FeatureSnapshot,
    PredictionSnapshot,
    PredictionStatus,
    RankEvent,
    RegimeTrend,
    RegimeVol,
    compute_rank_events_digest,
)
from cryptoscreener.model_runner import (
    InferenceStrictness,
    MLRunner,
    MLRunnerConfig,
)
from cryptoscreener.ranker import Ranker, RankerConfig
from cryptoscreener.registry import (
    ArtifactEntry,
    Manifest,
    PackageValidationError,
    compute_file_sha256,
    generate_checksums_txt,
    load_package,
    validate_package,
)
from cryptoscreener.scoring.scorer import Scorer, ScorerConfig

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Package Builder (On-the-fly generation)
# =============================================================================


def create_deterministic_model(seed: int = 42) -> bytes:
    """Create deterministic model bytes.

    Uses the same DeterministicModel from mlrunner_model fixture.
    """
    from tests.fixtures.mlrunner_model.deterministic_model import DeterministicModel

    model = DeterministicModel(seed=seed)
    return pickle.dumps(model)


def create_calibration_json() -> str:
    """Create calibration.json content."""
    calibration = {
        "metadata": {
            "schema_version": "1.0.0",
            "git_sha": "smoke_test_abc",
            "config_hash": "smoke_config_hash",
            "data_hash": "smoke_data_hash",
            "calibration_timestamp": "2026-01-26T00:00:00+00:00",
            "method": "platt",
            "heads": ["p_inplay_30s", "p_inplay_2m", "p_inplay_5m", "p_toxic"],
            "n_samples": 1000,
            "metrics_before": {
                "p_inplay_30s": {"brier": 0.25, "ece": 0.15},
                "p_inplay_2m": {"brier": 0.22, "ece": 0.12},
                "p_inplay_5m": {"brier": 0.20, "ece": 0.10},
                "p_toxic": {"brier": 0.18, "ece": 0.08},
            },
            "metrics_after": {
                "p_inplay_30s": {"brier": 0.20, "ece": 0.05},
                "p_inplay_2m": {"brier": 0.18, "ece": 0.04},
                "p_inplay_5m": {"brier": 0.16, "ece": 0.03},
                "p_toxic": {"brier": 0.14, "ece": 0.02},
            },
        },
        "calibrators": {
            "p_inplay_30s": {"type": "platt", "a": 1.5, "b": 0.2, "head_name": "p_inplay_30s"},
            "p_inplay_2m": {"type": "platt", "a": 1.2, "b": 0.1, "head_name": "p_inplay_2m"},
            "p_inplay_5m": {"type": "platt", "a": 1.0, "b": 0.05, "head_name": "p_inplay_5m"},
            "p_toxic": {"type": "platt", "a": 2.0, "b": 0.3, "head_name": "p_toxic"},
        },
    }
    return json.dumps(calibration, indent=2)


def build_package_dir(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Build a complete model package directory.

    Returns:
        Tuple of (package_dir, hashes_dict) where hashes_dict maps filename to sha256.
    """
    # Create model.pkl
    model_bytes = create_deterministic_model(seed=42)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(model_bytes)
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()

    # Create calibration.json
    calibration_content = create_calibration_json()
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(calibration_content)
    calibration_sha256 = hashlib.sha256(calibration_content.encode()).hexdigest()

    # Create schema_version.json
    schema_data = {
        "schema_version": "1.0.0",
        "model_version": f"1.0.0+smoke123+20260126+{model_sha256[:8]}",
        "compatible_schemas": [],
    }
    schema_path = tmp_path / "schema_version.json"
    schema_path.write_text(json.dumps(schema_data, indent=2))

    # Create features.json
    features_data = {
        "features": [
            "spread_bps",
            "mid",
            "book_imbalance",
            "flow_imbalance",
            "natr_14_5m",
            "impact_bps_q",
        ],
        "version": "1.0.0",
    }
    features_path = tmp_path / "features.json"
    features_path.write_text(json.dumps(features_data, indent=2))

    # Use fixed timestamp for determinism
    created_at = "2026-01-26T00:00:00Z"
    model_version = str(schema_data["model_version"])

    # Collect all files for manifest (except checksums.txt)
    files_to_hash = [model_path, calibration_path, schema_path, features_path]
    artifacts = []
    for file_path in files_to_hash:
        artifacts.append(
            ArtifactEntry(
                name=file_path.name,
                sha256=compute_file_sha256(file_path),
                size_bytes=file_path.stat().st_size,
            )
        )

    # Create temp manifest for checksums.txt generation
    temp_manifest = Manifest(
        schema_version="1.0.0",
        model_version=model_version,
        created_at=created_at,
        artifacts=artifacts,
    )

    # Generate checksums.txt and write it
    checksums_content = generate_checksums_txt(temp_manifest)
    checksums_path = tmp_path / "checksums.txt"
    checksums_path.write_text(checksums_content)

    # Add checksums.txt to artifacts with its actual hash
    artifacts.append(
        ArtifactEntry(
            name="checksums.txt",
            sha256=compute_file_sha256(checksums_path),
            size_bytes=checksums_path.stat().st_size,
        )
    )

    # Create final manifest with all artifacts
    final_manifest = Manifest(
        schema_version="1.0.0",
        model_version=model_version,
        created_at=created_at,
        artifacts=artifacts,
    )

    # Write manifest.json only (don't regenerate checksums.txt via save_manifest)
    manifest_path = tmp_path / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(final_manifest.to_dict(), f, indent=2)

    hashes = {
        "model.pkl": model_sha256,
        "calibration.json": calibration_sha256,
    }

    return tmp_path, hashes


# =============================================================================
# Fixture for FeatureSnapshots (reuse pattern from DEC-019)
# =============================================================================

BASE_TS = 1767225600000  # 2026-01-01 00:00:00 UTC


def make_feature_snapshot(
    symbol: str,
    ts: int,
    spread_bps: float = 2.0,
    mid: float = 50000.0,
    book_imbalance: float = 0.3,
    flow_imbalance: float = 0.4,
    natr: float = 0.02,
    impact_bps: float = 5.0,
    regime_vol: RegimeVol = RegimeVol.HIGH,
    regime_trend: RegimeTrend = RegimeTrend.TREND,
) -> FeatureSnapshot:
    """Create a deterministic FeatureSnapshot."""
    return FeatureSnapshot(
        ts=ts,
        symbol=symbol,
        features=Features(
            spread_bps=spread_bps,
            mid=mid,
            book_imbalance=book_imbalance,
            flow_imbalance=flow_imbalance,
            natr_14_5m=natr,
            impact_bps_q=impact_bps,
            regime_vol=regime_vol,
            regime_trend=regime_trend,
        ),
        data_health=DataHealth(
            stale_book_ms=0,
            stale_trades_ms=0,
        ),
    )


SMOKE_FIXTURE: list[list[FeatureSnapshot]] = [
    # t=0: Initial state
    [
        make_feature_snapshot(
            symbol="BTCUSDT",
            ts=BASE_TS,
            spread_bps=1.5,
            book_imbalance=0.5,
            flow_imbalance=0.6,
            natr=0.025,
            impact_bps=3.0,
        ),
        make_feature_snapshot(
            symbol="ETHUSDT",
            ts=BASE_TS,
            spread_bps=2.0,
            book_imbalance=0.4,
            flow_imbalance=0.5,
            natr=0.02,
            impact_bps=4.0,
        ),
    ],
    # t=2000ms
    [
        make_feature_snapshot(
            symbol="BTCUSDT",
            ts=BASE_TS + 2000,
            spread_bps=1.8,
            book_imbalance=0.55,
            flow_imbalance=0.65,
            natr=0.024,
            impact_bps=3.5,
        ),
        make_feature_snapshot(
            symbol="ETHUSDT",
            ts=BASE_TS + 2000,
            spread_bps=1.9,
            book_imbalance=0.5,
            flow_imbalance=0.55,
            natr=0.022,
            impact_bps=3.8,
        ),
    ],
    # t=4000ms
    [
        make_feature_snapshot(
            symbol="BTCUSDT",
            ts=BASE_TS + 4000,
            spread_bps=1.6,
            book_imbalance=0.6,
            flow_imbalance=0.7,
            natr=0.026,
            impact_bps=3.2,
        ),
        make_feature_snapshot(
            symbol="ETHUSDT",
            ts=BASE_TS + 4000,
            spread_bps=1.8,
            book_imbalance=0.55,
            flow_imbalance=0.6,
            natr=0.023,
            impact_bps=3.5,
        ),
    ],
]


# =============================================================================
# Helper: Run MLRunner Pipeline from Package
# =============================================================================


def run_pipeline_from_package(
    package_dir: Path,
    hashes: dict[str, str],
    fixture: list[list[FeatureSnapshot]],
) -> tuple[list[RankEvent], list[PredictionSnapshot]]:
    """Run full MLRunner pipeline using package artifacts.

    Args:
        package_dir: Path to model package directory.
        hashes: Dict of artifact filename to sha256.
        fixture: List of frames, each containing FeatureSnapshots.

    Returns:
        Tuple of (all RankEvents, all PredictionSnapshots).
    """
    # Load package with validation
    package = load_package(package_dir, validate=True)

    # Configure MLRunner in PROD mode (no fallback)
    model_path = package_dir / "model.pkl"
    calibration_path = package_dir / "calibration.json"

    runner_config = MLRunnerConfig(
        strictness=InferenceStrictness.PROD,  # PROD mode: no fallback
        model_path=model_path,
        model_sha256=hashes["model.pkl"],
        calibration_path=calibration_path,
        calibration_sha256=hashes["calibration.json"],
        require_calibration=True,
        fallback_to_baseline=False,
        model_version=package.schema.model_version,
    )

    runner = MLRunner(runner_config)
    scorer = Scorer(ScorerConfig())
    ranker_config = RankerConfig(
        top_k=5,
        enter_ms=500,  # Fast timing for test fixture
        exit_ms=1000,
        min_dwell_ms=500,
        score_threshold=0.001,
    )
    ranker = Ranker(ranker_config, scorer)

    all_events: list[RankEvent] = []
    all_predictions: list[PredictionSnapshot] = []

    for frame in fixture:
        if not frame:
            continue

        ts = frame[0].ts
        predictions: dict[str, PredictionSnapshot] = {}

        for snapshot in frame:
            prediction = runner.predict(snapshot)
            predictions[snapshot.symbol] = prediction
            all_predictions.append(prediction)

        events = ranker.update(predictions, ts)
        all_events.extend(events)

    return all_events, all_predictions


def compute_predictions_digest(predictions: list[PredictionSnapshot]) -> str:
    """Compute SHA256 digest of predictions list."""
    content = b"".join(p.to_json() for p in predictions)
    return hashlib.sha256(content).hexdigest()


# =============================================================================
# Test Class 1: Package Integrity
# =============================================================================


class TestPackageIntegrity:
    """Tests for package integrity validation.

    Verifies that:
    1. Valid package passes validation
    2. Invalid manifest fails
    3. Checksum mismatch fails
    4. Missing required files fail
    """

    def test_valid_package_passes_validation(self, tmp_path: Path) -> None:
        """Valid package with all artifacts passes validation."""
        package_dir, _ = build_package_dir(tmp_path)

        errors = validate_package(package_dir)
        assert len(errors) == 0, f"Validation failed: {errors}"

    def test_valid_package_loads_successfully(self, tmp_path: Path) -> None:
        """Valid package loads without errors."""
        package_dir, _ = build_package_dir(tmp_path)

        package = load_package(package_dir, validate=True)

        assert package.manifest is not None
        assert package.schema.schema_version == "1.0.0"
        assert len(package.features.features) > 0
        # Model exists in manifest (we use model.pkl, not model.bin)
        model_artifact = package.manifest.get_artifact("model.pkl")
        assert model_artifact is not None
        assert (package_dir / "model.pkl").exists()

    def test_checksum_mismatch_fails(self, tmp_path: Path) -> None:
        """Package with wrong checksum fails validation."""
        package_dir, _ = build_package_dir(tmp_path)

        # Corrupt model.pkl
        model_path = package_dir / "model.pkl"
        model_path.write_bytes(b"corrupted data")

        errors = validate_package(package_dir)
        assert any("SHA256 mismatch" in e for e in errors), f"Expected SHA256 error: {errors}"

    def test_missing_required_file_fails(self, tmp_path: Path) -> None:
        """Package missing required file fails validation."""
        package_dir, _ = build_package_dir(tmp_path)

        # Remove features.json
        (package_dir / "features.json").unlink()

        errors = validate_package(package_dir)
        assert any("features.json" in e for e in errors), f"Expected missing file error: {errors}"

    def test_missing_checksums_fails(self, tmp_path: Path) -> None:
        """Package missing checksums.txt fails validation."""
        package_dir, _ = build_package_dir(tmp_path)

        # Remove checksums.txt
        (package_dir / "checksums.txt").unlink()

        errors = validate_package(package_dir)
        assert any("checksums.txt" in e for e in errors), (
            f"Expected missing checksums error: {errors}"
        )

    def test_load_with_validation_raises_on_invalid(self, tmp_path: Path) -> None:
        """load_package with validate=True raises on invalid package."""
        package_dir, _ = build_package_dir(tmp_path)

        # Corrupt model.pkl
        (package_dir / "model.pkl").write_bytes(b"bad")

        with pytest.raises(PackageValidationError) as exc_info:
            load_package(package_dir, validate=True)

        assert "SHA256 mismatch" in str(exc_info.value)


# =============================================================================
# Test Class 2: MLRunner Loads from Package
# =============================================================================


class TestMLRunnerLoadsFromPackage:
    """Tests that MLRunner can load artifacts from a validated package.

    Verifies that:
    1. MLRunner loads model and calibration from package
    2. Predictions are non-DATA_ISSUE (actual model inference)
    3. Model version reflects package manifest
    """

    def test_mlrunner_loads_from_package(self, tmp_path: Path) -> None:
        """MLRunner successfully loads model from package."""
        package_dir, hashes = build_package_dir(tmp_path)
        package = load_package(package_dir, validate=True)

        model_path = package_dir / "model.pkl"
        calibration_path = package_dir / "calibration.json"

        config = MLRunnerConfig(
            strictness=InferenceStrictness.PROD,
            model_path=model_path,
            model_sha256=hashes["model.pkl"],
            calibration_path=calibration_path,
            calibration_sha256=hashes["calibration.json"],
            require_calibration=True,
            fallback_to_baseline=False,
            model_version=package.schema.model_version,
        )
        runner = MLRunner(config)

        # Should not be using fallback
        assert not runner.is_using_fallback

    def test_predictions_not_data_issue(self, tmp_path: Path) -> None:
        """Predictions from package are not DATA_ISSUE (actual inference)."""
        package_dir, hashes = build_package_dir(tmp_path)

        _events, predictions = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)

        # At least some predictions should exist
        assert len(predictions) > 0

        # Check that predictions are not all DATA_ISSUE
        non_data_issue = [p for p in predictions if p.status != PredictionStatus.DATA_ISSUE]
        assert len(non_data_issue) > 0, (
            "All predictions are DATA_ISSUE - model not loaded correctly"
        )

    def test_model_version_matches_package(self, tmp_path: Path) -> None:
        """Prediction model_version matches package manifest version."""
        package_dir, hashes = build_package_dir(tmp_path)
        package = load_package(package_dir, validate=True)

        _, predictions = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)

        # Model version should contain package version
        expected_version = package.schema.model_version
        for pred in predictions:
            assert pred.model_version == expected_version, (
                f"Expected model_version '{expected_version}', got '{pred.model_version}'"
            )

    def test_calibration_applied(self, tmp_path: Path) -> None:
        """Calibration is applied (calibration_version set)."""
        package_dir, hashes = build_package_dir(tmp_path)

        _, predictions = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)

        # At least some predictions should show calibration effect
        has_calibration = any(
            pred.calibration_version and len(pred.calibration_version) > 0 for pred in predictions
        )
        assert has_calibration, "No predictions have calibration_version set"

    def test_produces_rank_events(self, tmp_path: Path) -> None:
        """Package-loaded MLRunner produces RankEvents (not empty)."""
        package_dir, hashes = build_package_dir(tmp_path)

        events, _ = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)

        # Should produce at least 1 event
        assert len(events) > 0, "No RankEvents produced"


# =============================================================================
# Test Class 3: Determinism (Double Run)
# =============================================================================


class TestPackageDeterminism:
    """Tests that package-loaded inference is deterministic.

    Verifies that:
    1. Two runs with same input produce identical output
    2. Digests match across runs
    """

    def test_same_input_same_output(self, tmp_path: Path) -> None:
        """Same input produces same output across two runs."""
        package_dir, hashes = build_package_dir(tmp_path)

        events1, preds1 = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)
        events2, preds2 = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)

        # Same number of outputs
        assert len(events1) == len(events2)
        assert len(preds1) == len(preds2)

        # Field-by-field comparison for predictions
        for p1, p2 in zip(preds1, preds2, strict=True):
            assert p1.symbol == p2.symbol
            assert p1.ts == p2.ts
            assert p1.status == p2.status
            assert p1.p_inplay_30s == p2.p_inplay_30s
            assert p1.p_inplay_2m == p2.p_inplay_2m
            assert p1.p_inplay_5m == p2.p_inplay_5m
            assert p1.p_toxic == p2.p_toxic
            assert p1.expected_utility_bps_2m == p2.expected_utility_bps_2m

    def test_rank_events_digest_stable(self, tmp_path: Path) -> None:
        """RankEvent digest is stable across runs."""
        package_dir, hashes = build_package_dir(tmp_path)

        events1, _ = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)
        events2, _ = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)

        digest1 = compute_rank_events_digest(events1)
        digest2 = compute_rank_events_digest(events2)

        assert digest1 == digest2, f"RankEvent digest mismatch: {digest1} != {digest2}"

    def test_predictions_digest_stable(self, tmp_path: Path) -> None:
        """Prediction digest is stable across runs."""
        package_dir, hashes = build_package_dir(tmp_path)

        _, preds1 = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)
        _, preds2 = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)

        digest1 = compute_predictions_digest(preds1)
        digest2 = compute_predictions_digest(preds2)

        assert digest1 == digest2, f"Prediction digest mismatch: {digest1} != {digest2}"

    def test_generate_replay_proof(self, tmp_path: Path) -> None:
        """Generate and print replay proof for verification.

        This test:
        1. Computes package artifact hashes
        2. Runs pipeline twice
        3. Computes output digests for both runs
        4. Prints proof for manual verification
        """
        package_dir, hashes = build_package_dir(tmp_path)
        package = load_package(package_dir, validate=True)

        # Run 1
        events1, preds1 = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)
        rank_digest1 = compute_rank_events_digest(events1)
        pred_digest1 = compute_predictions_digest(preds1)

        # Run 2
        events2, preds2 = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)
        rank_digest2 = compute_rank_events_digest(events2)
        pred_digest2 = compute_predictions_digest(preds2)

        # Print proof
        print("\n=== Model Package E2E Smoke Replay Proof ===")
        print(f"Package schema_version: {package.schema.schema_version}")
        print(f"Package model_version:  {package.schema.model_version}")
        print(f"Model SHA256:           {hashes['model.pkl'][:16]}...")
        print(f"Calibration SHA256:     {hashes['calibration.json'][:16]}...")
        print(f"RankEvent digest (r1):  {rank_digest1}")
        print(f"RankEvent digest (r2):  {rank_digest2}")
        print(f"Prediction digest (r1): {pred_digest1}")
        print(f"Prediction digest (r2): {pred_digest2}")
        print(f"Total RankEvents:       {len(events1)}")
        print(f"Total Predictions:      {len(preds1)}")
        print(
            f"Digests match:          {rank_digest1 == rank_digest2 and pred_digest1 == pred_digest2}"
        )

        # Assertions
        assert rank_digest1 == rank_digest2, "RankEvent digest mismatch"
        assert pred_digest1 == pred_digest2, "Prediction digest mismatch"
        assert len(events1) > 0, "No RankEvents produced"


# =============================================================================
# Test Class 4: JSON Examples
# =============================================================================


class TestJSONExamples:
    """Generate JSON examples for proof bundle."""

    def test_prediction_snapshot_json(self, tmp_path: Path) -> None:
        """Print example PredictionSnapshot JSON."""
        package_dir, hashes = build_package_dir(tmp_path)

        _, predictions = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)

        # Get first prediction
        pred = predictions[0]
        json_str = pred.to_json()

        print("\n=== Example PredictionSnapshot JSON ===")
        print(json_str)

        # Validate roundtrip
        loaded = PredictionSnapshot.from_json(json_str)
        assert loaded.symbol == pred.symbol
        assert loaded.ts == pred.ts
        assert loaded.status == pred.status

    def test_rank_event_json(self, tmp_path: Path) -> None:
        """Print example RankEvent JSON."""
        package_dir, hashes = build_package_dir(tmp_path)

        events, _ = run_pipeline_from_package(package_dir, hashes, SMOKE_FIXTURE)

        if not events:
            pytest.skip("No RankEvents generated")

        # Get first event
        event = events[0]
        json_str = event.to_json()

        print("\n=== Example RankEvent JSON ===")
        print(json_str)

        # Validate roundtrip
        loaded = RankEvent.from_json(json_str)
        assert loaded.symbol == event.symbol
        assert loaded.ts == event.ts
        assert loaded.event == event.event
