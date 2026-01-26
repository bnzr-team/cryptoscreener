"""Training pipeline module.

Provides dataset splitting and preparation for ML training:
- Time-based train/val/test splits (no leakage)
- Dataset loading with schema validation
- Metadata tracking (git_sha, config_hash, data_hash)

Per PRD §11 Milestone 3: "ML v1 (GBDT) + calibration + ranker"
"""

from cryptoscreener.training.split import (
    SplitConfig,
    SplitMetadata,
    SplitResult,
    time_based_split,
)

__all__ = [
    "SplitConfig",
    "SplitMetadata",
    "SplitResult",
    "time_based_split",
]
