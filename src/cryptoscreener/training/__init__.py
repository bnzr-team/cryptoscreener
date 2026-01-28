"""Training pipeline module (DEC-038).

Provides:
- Time-based train/val/test splits (no leakage)
- Dataset loading with schema validation
- Feature schema for training/inference compatibility
- Model training with sklearn
- Artifact packaging with checksums

Per PRD §11 Milestone 3: "ML v1 (GBDT) + calibration + ranker"
"""

from cryptoscreener.training.artifact import (
    ArtifactBuildError,
    ArtifactBuildResult,
    build_model_package,
    generate_model_version,
)
from cryptoscreener.training.feature_schema import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    HEAD_TO_LABEL,
    PREDICTION_HEADS,
    FeatureSchemaError,
    compute_feature_hash,
    get_label_column,
    load_features_json,
    save_features_json,
    validate_feature_compatibility,
)
from cryptoscreener.training.split import (
    SplitConfig,
    SplitMetadata,
    SplitResult,
    time_based_split,
)
from cryptoscreener.training.trainer import (
    HeadMetrics,
    Trainer,
    TrainingConfig,
    TrainingError,
    TrainingResult,
)

__all__ = [  # noqa: RUF022 - grouped by category for maintainability
    # Feature schema
    "FEATURE_ORDER",
    "FEATURE_SCHEMA_VERSION",
    "HEAD_TO_LABEL",
    "PREDICTION_HEADS",
    "FeatureSchemaError",
    "compute_feature_hash",
    "get_label_column",
    "load_features_json",
    "save_features_json",
    "validate_feature_compatibility",
    # Split
    "SplitConfig",
    "SplitMetadata",
    "SplitResult",
    "time_based_split",
    # Trainer
    "HeadMetrics",
    "Trainer",
    "TrainingConfig",
    "TrainingError",
    "TrainingResult",
    # Artifact
    "ArtifactBuildError",
    "ArtifactBuildResult",
    "build_model_package",
    "generate_model_version",
]
