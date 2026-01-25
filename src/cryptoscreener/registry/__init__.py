"""Model registry and artifact management.

Implements MODEL_REGISTRY_VERSIONING.md specification:
- Unified manifest format (checksums.txt + manifest.json)
- Schema validation with fail-fast
- ModelPackage loader API
"""

from cryptoscreener.registry.manifest import (
    ArtifactEntry,
    Manifest,
    ManifestError,
    ManifestValidationError,
    compute_file_sha256,
    generate_checksums_txt,
    load_manifest,
    parse_checksums_txt,
    save_manifest,
    validate_manifest,
)
from cryptoscreener.registry.package import (
    ModelPackage,
    PackageError,
    PackageValidationError,
    load_package,
    validate_package,
)
from cryptoscreener.registry.version import (
    ModelVersion,
    parse_model_version,
)

__all__ = [
    "ArtifactEntry",
    "Manifest",
    "ManifestError",
    "ManifestValidationError",
    "ModelPackage",
    "ModelVersion",
    "PackageError",
    "PackageValidationError",
    "compute_file_sha256",
    "generate_checksums_txt",
    "load_manifest",
    "load_package",
    "parse_checksums_txt",
    "parse_model_version",
    "save_manifest",
    "validate_manifest",
    "validate_package",
]
