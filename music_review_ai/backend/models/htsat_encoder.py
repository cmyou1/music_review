"""
Lightweight wrapper for loading an HTSAT encoder (MERT).

원래는 MVP 환경에서 무거운 체크포인트를 생략하기 위한 스텁을 제공했지만,
지금은 Hugging Face에서 실제 모델을 내려받아 로딩할 수 있도록 구성되어 있다.
torch/transformers 의존성이 없거나 모델을 찾지 못하는 경우에만 stub을 사용한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
except Exception:  # pragma: no cover - optional heavy deps
    AutoFeatureExtractor = None
    AutoModel = None
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class HTSATEncoder:
    model_name: str
    cache_dir: Optional[Path] = None
    local_path: Optional[Path] = None
    embedding_dim: int = 1024
    sample_rate: Optional[int] = None
    use_stub: bool = field(default=False, init=False)
    device: str = field(default="cpu", init=False)
    _model: Optional["AutoModel"] = field(default=None, init=False, repr=False)
    _feature_extractor: Optional["AutoFeatureExtractor"] = field(default=None, init=False, repr=False)

    @classmethod
    def from_pretrained(
        cls, model_name: str, cache_dir: Optional[Path] = None, local_path: Optional[Path] = None
    ) -> "HTSATEncoder":
        encoder = cls(model_name=model_name, cache_dir=cache_dir, local_path=local_path)
        encoder._maybe_load()
        return encoder

    def _maybe_load(self) -> None:
        if self.use_stub or self._model is not None:
            return
        if AutoModel is None or AutoFeatureExtractor is None or torch is None:
            logger.warning("Transformers/torch 미설치 상태이므로 임베딩을 무작위로 대체합니다.")
            self.use_stub = True
            return
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("HTSAT encoder is using CUDA for inference.")
            else:
                self.device = "cpu"
                logger.info("HTSAT encoder is using CPU for inference.")
            pretrained_ref = self.local_path if self.local_path else self.model_name
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_ref, cache_dir=self.cache_dir, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                pretrained_ref, cache_dir=self.cache_dir, trust_remote_code=True
            )
            self.embedding_dim = getattr(self._model.config, "hidden_size", self.embedding_dim)
            self.sample_rate = getattr(self._feature_extractor, "sampling_rate", None)
            self._model.to(self.device).eval()
        except Exception as exc:  # pragma: no cover
            logger.warning("HTSAT 모델을 불러오지 못했습니다. 임시 임베딩을 사용합니다: %s", exc)
            self.use_stub = True
            self._model = None
            self._feature_extractor = None

    def _stub_embedding(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(self.embedding_dim)

    def get_required_sample_rate(self) -> Optional[int]:
        return self.sample_rate

    def get_embedding(self, waveform, sample_rate: int):
        """
        Convert waveform into a fixed-size embedding.
        """

        if self.use_stub or self._model is None or self._feature_extractor is None:
            return self._stub_embedding()

        try:
            with torch.no_grad():
                inputs = self._feature_extractor(
                    waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self._model(**inputs)
                hidden = outputs.last_hidden_state  # (batch, seq, hidden)
                embedding = hidden.mean(dim=1).squeeze(0)
                return embedding.cpu().numpy()
        except Exception as exc:  # pragma: no cover
            logger.warning("HTSAT 임베딩 계산 실패, 스텁으로 대체합니다: %s", exc)
            self.use_stub = True
            return self._stub_embedding()
