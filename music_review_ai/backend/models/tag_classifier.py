"""
Placeholder multi-label tag classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TagClassifier:
    labels: List[str] | None = None
    projection_matrix: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.labels is None:
            self.labels = [
                "kick punchy",
                "snare crisp",
                "synth pad",
                "electric guitar",
                "acoustic guitar",
                "strings",
                "piano",
                "vocal airy",
                "bass warm",
                "ambient fx",
                "organ mellow",
                "lofi noise",
                "bright lead",
                "percussion dry",
                "choir lush",
            ]

    def _ensure_projection(self, dim: int) -> None:
        if self.projection_matrix is not None and self.projection_matrix.shape[1] == dim:
            return
        rng = np.random.default_rng(2024)
        self.projection_matrix = rng.normal(size=(len(self.labels), dim))

    def predict(self, embedding) -> Dict[str, float]:
        """
        Placeholder classifier: perform a deterministic projection + softmax to get pseudo probabilities.
        """

        if embedding is None:
            return {label: 0.0 for label in self.labels}

        emb = np.array(embedding, dtype=np.float32)
        self._ensure_projection(emb.shape[0])
        scores = self.projection_matrix @ emb
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / (exp_scores.sum() + 1e-8)

        return {label: float(prob) for label, prob in zip(self.labels, probs)}
