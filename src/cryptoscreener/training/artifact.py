"""Model artifact packaging (DEC-038).

Creates complete model packages for deployment:
- model.pkl (trained sklearn model)
- calibration.json (Platt calibrators)
- features.json (feature schema)
- schema_version.json (package metadata)
- checksums.txt (SHA256 hashes)
- manifest.json (machine-readable manifest)
- training_report.md (human-readable report)
"""

from __future__ import annotations

import pickle
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from cryptoscreener.registry.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ArtifactEntry,
    Manifest,
    compute_file_sha256,
    save_manifest,
)
from cryptoscreener.training.feature_schema import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    compute_feature_hash,
    save_features_json,
)

if TYPE_CHECKING:
    from cryptoscreener.calibration.artifact import CalibrationArtifact
    from cryptoscreener.training.trainer import HeadMetrics, TrainingConfig


class ArtifactBuildError(Exception):
    """Raised when artifact building fails."""


def _get_git_sha() -> str:
    """Get current git commit SHA.

    Returns:
        Git SHA or "unknown" if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate_model_version(
    major: int = 1,
    minor: int = 0,
    patch: int = 0,
    git_sha: str | None = None,
    feature_hash: str | None = None,
) -> str:
    """Generate model version string.

    Format: {major}.{minor}.{patch}+{git_sha}+{date}+{feature_hash[:8]}

    Args:
        major: Major version.
        minor: Minor version.
        patch: Patch version.
        git_sha: Git commit SHA. Auto-detected if not provided.
        feature_hash: Feature schema hash. Computed if not provided.

    Returns:
        Version string like "1.0.0+abc1234+20260125+a1b2c3d4".
    """
    if git_sha is None:
        git_sha = _get_git_sha()

    date_str = datetime.now(UTC).strftime("%Y%m%d")

    if feature_hash is None:
        feature_hash = compute_feature_hash(FEATURE_ORDER)

    return f"{major}.{minor}.{patch}+{git_sha}+{date_str}+{feature_hash[:8]}"


def _save_model_pkl(model: Any, path: Path) -> None:
    """Save sklearn model to pickle file.

    Args:
        model: Trained sklearn model.
        path: Output path.
    """
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_schema_version_json(
    path: Path,
    model_version: str,
    created_at: str,
) -> None:
    """Save schema_version.json.

    Args:
        path: Output path.
        model_version: Model version string.
        created_at: ISO timestamp.
    """
    data = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_version": model_version,
        "created_at": created_at,
    }
    with path.open("w") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode())
        f.write("\n")


def _generate_training_report(
    model_version: str,
    config: TrainingConfig,
    metrics: dict[str, HeadMetrics],
    metrics_calibrated: dict[str, HeadMetrics] | None,
    train_samples: int,
    val_samples: int,
) -> str:
    """Generate markdown training report.

    Args:
        model_version: Model version string.
        config: Training configuration.
        metrics: Raw model metrics per head.
        metrics_calibrated: Calibrated metrics per head (if available).
        train_samples: Number of training samples.
        val_samples: Number of validation samples.

    Returns:
        Markdown report content.
    """
    lines = [
        f"# Training Report: {model_version}",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Configuration",
        "",
        f"- **Model type**: {config.model_type}",
        f"- **Random seed**: {config.seed}",
        f"- **Profile**: {config.profile}",
    ]

    if config.model_type == "random_forest":
        lines.extend(
            [
                f"- **n_estimators**: {config.n_estimators}",
                f"- **max_depth**: {config.max_depth}",
            ]
        )
    else:
        lines.extend(
            [
                f"- **C**: {config.C}",
                f"- **max_iter**: {config.max_iter}",
            ]
        )

    lines.extend(
        [
            "",
            "## Dataset",
            "",
            f"- **Training samples**: {train_samples:,}",
            f"- **Validation samples**: {val_samples:,}",
            "",
            "## Feature Schema",
            "",
            f"- **Version**: {FEATURE_SCHEMA_VERSION}",
            f"- **Feature hash**: {compute_feature_hash(FEATURE_ORDER)}",
            f"- **Features**: {', '.join(FEATURE_ORDER)}",
            "",
            "## Metrics by Head",
            "",
        ]
    )

    for head, m in metrics.items():
        lines.append(f"### {head}")
        lines.append("")
        lines.append("| Metric | Raw | Calibrated |")
        lines.append("|--------|-----|------------|")

        cal_m = metrics_calibrated.get(head) if metrics_calibrated else None

        auc_cal = f"{cal_m.auc:.4f}" if cal_m else "N/A"
        pr_auc_cal = f"{cal_m.pr_auc:.4f}" if cal_m else "N/A"
        brier_cal = f"{cal_m.brier:.4f}" if cal_m else "N/A"
        ece_cal = f"{cal_m.ece:.4f}" if cal_m else "N/A"

        lines.append(f"| AUC | {m.auc:.4f} | {auc_cal} |")
        lines.append(f"| PR-AUC | {m.pr_auc:.4f} | {pr_auc_cal} |")
        lines.append(f"| Brier | {m.brier:.4f} | {brier_cal} |")
        lines.append(f"| ECE | {m.ece:.4f} | {ece_cal} |")
        lines.append(f"| Samples | {m.n_samples} | - |")
        lines.append(f"| Positives | {m.n_positives} | - |")
        lines.append("")

    lines.extend(
        [
            "## Artifact Checksums",
            "",
            "See `checksums.txt` for SHA256 hashes of all artifacts.",
            "",
        ]
    )

    return "\n".join(lines)


@dataclass
class ArtifactBuildResult:
    """Result of building a model artifact package.

    Attributes:
        output_dir: Path to output directory.
        model_version: Generated model version.
        manifest: Package manifest.
        checksums: File checksums.
    """

    output_dir: Path
    model_version: str
    manifest: Manifest
    checksums: dict[str, str]


def build_model_package(
    output_dir: Path,
    model: Any,
    config: TrainingConfig,
    metrics: dict[str, HeadMetrics],
    calibration: CalibrationArtifact | None = None,
    metrics_calibrated: dict[str, HeadMetrics] | None = None,
    train_samples: int = 0,
    val_samples: int = 0,
    model_version: str | None = None,
) -> ArtifactBuildResult:
    """Build complete model package.

    Creates the following files in output_dir:
    - model.pkl
    - calibration.json (if calibration provided)
    - features.json
    - schema_version.json
    - checksums.txt
    - manifest.json
    - training_report.md

    Args:
        output_dir: Output directory for artifacts.
        model: Trained sklearn model.
        config: Training configuration used.
        metrics: Raw model metrics per head.
        calibration: Optional calibration artifact.
        metrics_calibrated: Optional calibrated metrics.
        train_samples: Number of training samples.
        val_samples: Number of validation samples.
        model_version: Optional version string (auto-generated if not provided).

    Returns:
        ArtifactBuildResult with package info.

    Raises:
        ArtifactBuildError: If building fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(UTC).isoformat()

    # Generate version if not provided
    if model_version is None:
        model_version = generate_model_version()

    # 1. Save model.pkl
    model_path = output_dir / "model.pkl"
    _save_model_pkl(model, model_path)

    # 2. Save calibration.json (if available)
    calibration_path = output_dir / "calibration.json"
    if calibration is not None:
        from cryptoscreener.calibration.artifact import save_calibration_artifact

        save_calibration_artifact(calibration, calibration_path)

    # 3. Save features.json
    features_path = output_dir / "features.json"
    save_features_json(features_path)

    # 4. Save schema_version.json
    schema_path = output_dir / "schema_version.json"
    _save_schema_version_json(schema_path, model_version, created_at)

    # 5. Generate training report
    report_path = output_dir / "training_report.md"
    report_content = _generate_training_report(
        model_version=model_version,
        config=config,
        metrics=metrics,
        metrics_calibrated=metrics_calibrated,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    report_path.write_text(report_content)

    # 6. Compute checksums for all files
    artifacts: list[ArtifactEntry] = []
    checksums: dict[str, str] = {}

    files_to_hash = [
        model_path,
        features_path,
        schema_path,
        report_path,
    ]
    if calibration is not None:
        files_to_hash.append(calibration_path)

    for fpath in files_to_hash:
        sha256 = compute_file_sha256(fpath)
        size_bytes = fpath.stat().st_size
        checksums[fpath.name] = sha256
        artifacts.append(
            ArtifactEntry(
                name=fpath.name,
                sha256=sha256,
                size_bytes=size_bytes,
            )
        )

    # 7. Save checksums.txt
    checksums_path = output_dir / "checksums.txt"
    checksums_lines = [
        f"# checksums.txt - SHA256 hashes for {model_version}",
        f"# Generated: {created_at}",
        "",
    ]
    for name in sorted(checksums.keys()):
        checksums_lines.append(f"{checksums[name]}  {name}")
    checksums_path.write_text("\n".join(checksums_lines) + "\n")

    # Add checksums.txt to artifacts
    checksums_sha256 = compute_file_sha256(checksums_path)
    checksums_size = checksums_path.stat().st_size
    checksums[checksums_path.name] = checksums_sha256
    artifacts.append(
        ArtifactEntry(
            name="checksums.txt",
            sha256=checksums_sha256,
            size_bytes=checksums_size,
        )
    )

    # 8. Create and save manifest
    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        model_version=model_version,
        created_at=created_at,
        artifacts=artifacts,
        metadata={
            "model_type": config.model_type,
            "profile": config.profile,
            "seed": config.seed,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "train_samples": train_samples,
            "val_samples": val_samples,
        },
    )
    save_manifest(manifest, output_dir)

    # Update checksums with manifest.json
    manifest_sha256 = compute_file_sha256(output_dir / "manifest.json")
    checksums["manifest.json"] = manifest_sha256

    return ArtifactBuildResult(
        output_dir=output_dir,
        model_version=model_version,
        manifest=manifest,
        checksums=checksums,
    )
