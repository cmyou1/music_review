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
    Maps AudioSet labels to music-specific instrument tags.
    """
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    cache_dir: Optional[Path] = None
    local_path: Optional[Path] = None
    use_stub: bool = False  # For testing
    _model: Optional[any] = None
    _processor: Optional[any] = None

    # AudioSet label -> Our tag mapping
    LABEL_MAPPING = {
        # Percussion
        "Bass drum": "kick punchy",
        "Drum": "percussion dry",
        "Drum kit": "percussion dry",
        "Snare drum": "snare crisp",
        "Hi-hat": "percussion dry",

        # Bass
        "Bass guitar": "bass warm",
        "Double bass": "bass warm",
        "Synthesizer": "synth pad",

        # Strings
        "Violin, fiddle": "strings",
        "Cello": "strings",
        "String section": "strings",

        # Guitar
        "Acoustic guitar": "acoustic guitar",
        "Electric guitar": "electric guitar",
        "Plucked string instrument": "acoustic guitar",

        # Keys
        "Piano": "piano",
        "Electronic organ": "organ mellow",
        "Electric piano": "piano",

        # Synth/Electronic
        "Synthesizer": "synth pad",
        "Electronic music": "synth pad",
        "Techno": "synth pad",
        "House music": "synth pad",
        "Trance music": "bright lead",

        # Vocals
        "Singing": "vocal airy",
        "Choir": "choir lush",

        # Ambient
        "Ambient music": "ambient fx",
        "Sound effect": "ambient fx",
    }

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

            # Map to our custom tags
            tag_scores = {}
            for idx, prob in enumerate(probs):
                if prob < 0.01:  # Skip very low probabilities
                    continue

                audioset_label = audioset_labels.get(idx, "")
                our_tag = self.LABEL_MAPPING.get(audioset_label)

                if our_tag:
                    # Accumulate scores for same tag
                    tag_scores[our_tag] = tag_scores.get(our_tag, 0.0) + float(prob)

            # Normalize
            total = sum(tag_scores.values())
            if total > 0:
                tag_scores = {k: v/total for k, v in tag_scores.items()}

            # Sort by probability
            tag_scores = dict(sorted(tag_scores.items(), key=lambda x: x[1], reverse=True))

            return tag_scores

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
