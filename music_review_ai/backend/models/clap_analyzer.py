"""
CLAP analyzer wrapping Hugging Face laion/clap-htsat-fused.

Provides helpers to compute audio/text embeddings and derive summary
information (mood hints, similarity to reference prompts) without
returning the raw vectors to callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - heavy optional dependency
    import torch
    from transformers import ClapModel, ClapProcessor
except Exception:  # pragma: no cover
    ClapModel = None
    ClapProcessor = None
    torch = None

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "glitchy bass with metallic noise bursts",
    "bit-crushed percussion and distorted snare hits",
    "dark techno groove with rolling sub bass",
    "dreamy lo-fi haze soaked in reverb",
    "melancholic piano and ambient textures",
    "uplifting synthwave with bright leads",
    "cinematic strings building emotional arcs",
    "minimal introspective soundscape",
    "euphoric festival EDM build with bright saw leads",
    "driving breakbeat with airy vocal chops",
]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _vector_stats(vec: np.ndarray) -> Dict[str, float]:
    if vec.size == 0:
        return {"l2_norm": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "l2_norm": float(np.linalg.norm(vec)),
        "mean": float(vec.mean()),
        "std": float(vec.std()),
        "max": float(vec.max()),
    }


@dataclass
class ClapAnalyzer:
    model_name: str
    cache_dir: Optional[Path] = None
    local_path: Optional[Path] = None
    use_stub: bool = field(default=False, init=False)
    device: str = field(default="cpu", init=False)
    _model: Optional["ClapModel"] = field(default=None, init=False, repr=False)
    _processor: Optional["ClapProcessor"] = field(default=None, init=False, repr=False)
    _text_cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_pretrained(
        cls, model_name: str, cache_dir: Optional[Path] = None, local_path: Optional[Path] = None
    ) -> "ClapAnalyzer":
        analyzer = cls(model_name=model_name, cache_dir=cache_dir, local_path=local_path)
        analyzer._maybe_load()
        return analyzer

    def _maybe_load(self) -> None:
        if self.use_stub or self._model is not None:
            return
        if ClapModel is None or ClapProcessor is None or torch is None:
            logger.warning("CLAP dependencies missing. Using statistical stub.")
            self.use_stub = True
            return
        try:
            target = self.local_path if self.local_path else self.model_name
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = ClapProcessor.from_pretrained(target, cache_dir=self.cache_dir)
            self._model = ClapModel.from_pretrained(target, cache_dir=self.cache_dir)
            self._model.to(self.device).eval()
            logger.info("Loaded CLAP model %s on %s", self.model_name, self.device)
        except Exception as exc:  # pragma: no cover
            logger.warning("CLAP 모델 로드 실패, stub 사용: %s", exc)
            self.use_stub = True
            self._model = None
            self._processor = None

    def _stub_embedding(self, dim: int = 512) -> np.ndarray:
        rng = np.random.default_rng(99)
        return rng.standard_normal(dim)

    def get_audio_embedding(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.use_stub or self._model is None or self._processor is None:
            return self._stub_embedding()
        audio = waveform.astype(np.float32)
        inputs = self._processor(
            audios=[audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            features = self._model.get_audio_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
        return features.squeeze(0).cpu().numpy()

    def _encode_texts(self, texts: Iterable[str]) -> Dict[str, np.ndarray]:
        embeddings: Dict[str, np.ndarray] = {}
        missing = [text for text in texts if text not in self._text_cache]
        if missing and not self.use_stub and self._processor and self._model:
            inputs = self._processor(text=missing, return_tensors="pt", padding=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            for text, feature in zip(missing, feats):
                self._text_cache[text] = feature.cpu().numpy()

        for text in texts:
            if text in self._text_cache:
                embeddings[text] = self._text_cache[text]
            else:
                embeddings[text] = self._stub_embedding()
        return embeddings

    def compare_with_prompts(self, audio_embedding: np.ndarray, prompts: List[str]) -> List[Dict[str, float]]:
        prompt_embeddings = self._encode_texts(prompts)
        audio_norm = _normalize(audio_embedding)
        scores: List[Tuple[str, float]] = []
        for text, emb in prompt_embeddings.items():
            sim = float(np.dot(audio_norm, _normalize(emb)))
            scores.append((text, sim))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [{"text": text, "score": round(score, 3)} for text, score in scores]

    def analyze(self, waveform: np.ndarray, sample_rate: int, prompts: Optional[List[str]] = None) -> Dict[str, object]:
        prompts = prompts or DEFAULT_PROMPTS
        audio_embedding = self.get_audio_embedding(waveform, sample_rate)
        matches = self.compare_with_prompts(audio_embedding, prompts)
        primary = matches[0] if matches else {"text": "unknown", "score": 0.0}

        rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
        energy = float(np.clip(rms * 10.0, 0.0, 1.0))

        return {
            "primary_prompt": primary["text"],
            "prompt_score": primary["score"],
            "top_matches": matches[:4],
            "audio_embedding_stats": _vector_stats(audio_embedding),
            "energy_level": energy,
        }

    def text_similarity(self, text: str, audio_embedding: np.ndarray) -> float:
        prompt_embedding = self._encode_texts([text])[text]
        return float(np.dot(_normalize(audio_embedding), _normalize(prompt_embedding)))
