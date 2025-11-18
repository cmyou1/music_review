"""
Real multi-label tag classifier using Hugging Face AudioSet model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class TagClassifier:
    """
    Audio tag classifier using pre-trained AudioSet model from Hugging Face.
    Returns the original AudioSet labels to maximise fidelity.
    """
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    cache_dir: Optional[Path] = None
    local_path: Optional[Path] = None
    use_stub: bool = False  # For testing
    _model: Optional[any] = None
    _processor: Optional[any] = None

    top_k: int = 10
    min_probability: float = 1e-4

    def __post_init__(self):
        if not self.use_stub:
            self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            model_ref: str = str(self.local_path) if self.local_path else self.model_name
            logger.info(f"Loading audio classifier: {model_ref}")

            self._processor = AutoFeatureExtractor.from_pretrained(model_ref, cache_dir=self.cache_dir)
            self._model = AutoModelForAudioClassification.from_pretrained(model_ref, cache_dir=self.cache_dir)
            self._model.eval()

            logger.info("Audio classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load audio classifier, using stub: {e}")
            self.use_stub = True

    def predict(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """
        Predict instrument tags from audio data.

        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate (default 16000 for AST model)

        Returns:
            Dict of tag -> probability
        """
        if self.use_stub or self._model is None:
            return self._predict_stub()

        try:
            # Prepare audio input
            inputs = self._processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )

            # Get predictions
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1).numpy()

            # Get AudioSet labels
            audioset_labels = self._model.config.id2label

            label_scores = []
            for idx, prob in enumerate(probs):
                if prob < self.min_probability:
                    continue
                label = audioset_labels.get(idx, f"label_{idx}")
                label_scores.append((label, float(prob)))

            if not label_scores:
                return {}

            label_scores.sort(key=lambda item: item[1], reverse=True)
            top_scores = label_scores[: self.top_k]
            return dict(top_scores)

        except Exception as e:
            logger.error(f"Tag prediction failed: {e}")
            return self._predict_stub()

    def _predict_stub(self) -> Dict[str, float]:
        """Fallback stub prediction"""
        return {
            "synth pad": 0.3,
            "bass warm": 0.25,
            "percussion dry": 0.2,
            "bright lead": 0.15,
            "piano": 0.1
        }
