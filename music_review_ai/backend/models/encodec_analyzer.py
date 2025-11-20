"""
Meta EnCodec analyzer used to derive timbre/texture descriptors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - heavy optional dependency
    import torch
    from transformers import EncodecFeatureExtractor, EncodecModel
except Exception:  # pragma: no cover
    torch = None
    EncodecFeatureExtractor = None
    EncodecModel = None


def _safe_entropy(prob: np.ndarray) -> float:
    prob = prob[prob > 0]
    if not prob.size:
        return 0.0
    return float(-np.sum(prob * np.log2(prob)))


@dataclass
class EncodecAnalyzer:
    """
    Lightweight helper around `facebook/encodec_24khz` to expose musically meaningful
    timbre metrics instead of the raw codebooks.
    """

    model_name: str
    cache_dir: Optional[Path] = None
    local_path: Optional[Path] = None
    use_stub: bool = field(default=False, init=False)
    device: str = field(default="cpu", init=False)
    sample_rate: int = field(default=24000, init=False)
    codebook_size: int = field(default=1024, init=False)
    _model: Optional["EncodecModel"] = field(default=None, init=False, repr=False)
    _feature_extractor: Optional["EncodecFeatureExtractor"] = field(default=None, init=False, repr=False)

    @classmethod
    def from_pretrained(
        cls, model_name: str, cache_dir: Optional[Path] = None, local_path: Optional[Path] = None
    ) -> "EncodecAnalyzer":
        analyzer = cls(model_name=model_name, cache_dir=cache_dir, local_path=local_path)
        analyzer._maybe_load()
        return analyzer

    def _maybe_load(self) -> None:
        if self.use_stub or self._model is not None:
            return

        if EncodecModel is None or EncodecFeatureExtractor is None or torch is None:
            logger.warning("Encodec dependencies missing. Falling back to stub.")
            self.use_stub = True
            return

        try:
            target = self.local_path if self.local_path else self.model_name
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = EncodecModel.from_pretrained(target, cache_dir=self.cache_dir)
            self._feature_extractor = EncodecFeatureExtractor.from_pretrained(target, cache_dir=self.cache_dir)
            self._model.to(self.device).eval()
            if getattr(self._model.config, "sampling_rate", None):
                self.sample_rate = int(self._model.config.sampling_rate)
            if getattr(self._model.config, "codebook_size", None):
                self.codebook_size = int(self._model.config.codebook_size)
            logger.info("Loaded EnCodec model %s on %s", target, self.device)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load EnCodec model, using stub features: %s", exc)
            self.use_stub = True
            self._model = None
            self._feature_extractor = None

    def _describe_timbre(self, diversity: float, change_rate: float) -> str:
        if diversity > 0.6 and change_rate > 0.55:
            return "rich evolving texture"
        if diversity > 0.6 and change_rate <= 0.55:
            return "lush sustained tone"
        if diversity > 0.4 and change_rate > 0.6:
            return "percussive grainy texture"
        if diversity < 0.25 and change_rate < 0.35:
            return "focused smooth timbre"
        return "balanced detailed tone"

    def _summarize(self, codes: np.ndarray, scales: np.ndarray | None) -> Dict[str, float | str]:
        if codes.size == 0:
            return self._stub_summary()

        # Expected shape: (frames, batch, num_codebooks, chunk_length)
        batch_first = np.moveaxis(codes, 0, 1)  # (batch, frames, codebooks, chunk)
        batch_codes = batch_first[0]  # assume mono batch
        frames, num_codebooks, chunk_len = batch_codes.shape
        flattened = batch_codes.transpose(1, 0, 2).reshape(num_codebooks, -1)

        diversity_scores = []
        change_scores = []
        for row in flattened:
            unique = np.unique(row)
            diversity_scores.append(min(len(unique) / max(self.codebook_size, 1), 1.0))
            if row.size > 1:
                change_scores.append(float(np.mean(row[1:] != row[:-1])))
            else:
                change_scores.append(0.0)

        overall = flattened.reshape(-1)
        hist = np.bincount(overall, minlength=self.codebook_size).astype(np.float32)
        hist_sum = hist.sum()
        prob = hist / hist_sum if hist_sum else hist
        entropy = _safe_entropy(prob)
        normalized_entropy = float(entropy / np.log2(self.codebook_size)) if self.codebook_size > 1 else 0.0

        diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0
        change_rate = float(np.mean(change_scores)) if change_scores else 0.0

        scale_mean = 0.0
        scale_std = 0.0
        if isinstance(scales, np.ndarray) and scales.size:
            scale_mean = float(scales.mean())
            scale_std = float(scales.std())

        return {
            "codebook_diversity": round(diversity, 3),
            "transient_change_rate": round(change_rate, 3),
            "quantizer_entropy": round(normalized_entropy, 3),
            "scale_mean": round(scale_mean, 4),
            "scale_std": round(scale_std, 4),
            "descriptor": self._describe_timbre(diversity, change_rate),
            "num_codebooks": int(num_codebooks),
            "codebook_size": int(self.codebook_size),
        }

    def _stub_summary(self) -> Dict[str, float | str]:
        return {
            "codebook_diversity": 0.42,
            "transient_change_rate": 0.38,
            "quantizer_entropy": 0.55,
            "scale_mean": 0.07,
            "scale_std": 0.01,
            "descriptor": "balanced detailed tone",
            "num_codebooks": 8,
            "codebook_size": self.codebook_size,
        }

    def analyze(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, float | str]:
        if waveform.size == 0:
            return self._stub_summary()

        if self.use_stub or self._model is None or self._feature_extractor is None or torch is None:
            return self._stub_summary()

        target_sr = self.sample_rate or 24000
        if sample_rate != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr

        try:
            features = self._feature_extractor(
                raw_audio=[waveform],
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            features = features.to(self.device)
            input_values = features["input_values"]
            padding_mask = features.get("padding_mask")
            with torch.no_grad():
                encoded = self._model.encode(
                    input_values=input_values,
                    padding_mask=padding_mask,
                    return_dict=True,
                )
            codes = encoded.audio_codes.detach().cpu().numpy()
            scales = None
            if encoded.audio_scales is not None:
                if isinstance(encoded.audio_scales, (list, tuple)):
                    stacked = [scale.detach().cpu().numpy() for scale in encoded.audio_scales if scale is not None]
                    scales = np.stack(stacked) if stacked else None
                else:
                    scales = encoded.audio_scales.detach().cpu().numpy()
            return self._summarize(codes, scales)
        except Exception as exc:  # pragma: no cover
            logger.warning("Encodec analysis failed, returning stub values: %s", exc)
            self.use_stub = True
            return self._stub_summary()
