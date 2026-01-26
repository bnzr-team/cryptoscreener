"""Deterministic model for MLRunner testing.

This model produces deterministic output based on input hash.
It's used for replay determinism testing.

SECURITY NOTE: This module is used ONLY in tests. It is never loaded
in production code paths. The model is constructed in-memory (not from
pickle) to avoid arbitrary code execution risks.
"""

from __future__ import annotations

import hashlib
from typing import Any


class DeterministicModel:
    """Tiny deterministic model for testing.

    Implements predict_proba with deterministic output based on input hash.
    This ensures reproducible inference across runs.

    Output ranges are tuned to produce TRADEABLE predictions:
    - p_inplay heads (0-2): high values [0.6, 0.95] for TRADEABLE status
    - p_toxic head (3): low values [0.05, 0.3] to avoid TRAP status
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.version = "test-v1.0.0"

    def predict_proba(self, X: list[Any]) -> list[list[list[float]]]:
        """Return deterministic probabilities based on input.

        Uses hash of input to generate reproducible values.
        Returns multi-output format: list of [[neg_prob, pos_prob]] per head.

        Args:
            X: Input feature matrix (list of feature vectors).

        Returns:
            List of probability arrays per head (scikit-learn multi-output format).
        """
        n_samples = len(X)
        n_heads = 4

        # Compute deterministic probabilities for each sample and head
        all_probs = []
        for row in X:
            row_probs = []
            row_bytes = str(row).encode()
            h = hashlib.sha256(row_bytes + str(self.seed).encode()).hexdigest()

            for i in range(n_heads):
                # Take 8 hex chars, convert to float in [0, 1]
                val = int(h[i * 8 : (i + 1) * 8], 16) / (16**8)

                # p_inplay heads (0-2): high values [0.6, 0.95] for TRADEABLE
                # p_toxic head (3): low values [0.05, 0.3] to avoid TRAP
                prob = 0.6 + val * 0.35 if i < 3 else 0.05 + val * 0.25

                row_probs.append([1 - prob, prob])

            all_probs.append(row_probs)

        # Transpose to scikit-learn multi-output format:
        # [head1_all_samples, head2_all_samples, ...]
        output = []
        for head_idx in range(n_heads):
            head_probs = []
            for sample_idx in range(n_samples):
                head_probs.append(all_probs[sample_idx][head_idx])
            output.append(head_probs)

        return output
