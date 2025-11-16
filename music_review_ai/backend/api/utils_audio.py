"""
Audio utility helpers for the review API.
"""

from __future__ import annotations

import logging
import os
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

import librosa

from backend.config import settings
from ..models.htsat_encoder import HTSATEncoder
from ..models.tag_classifier import TagClassifier

logger = logging.getLogger(__name__)

_encoder: Optional[HTSATEncoder] = None
_tagger: Optional[TagClassifier] = None
HTSAT_MODEL_NAME = "htsat_stub_model"


def _get_encoder() -> HTSATEncoder:
    global _encoder
    if _encoder is None:
        model_name = settings.htsat_model_name or HTSAT_MODEL_NAME
        _encoder = HTSATEncoder.from_pretrained(
            model_name, cache_dir=settings.htsat_cache_dir, local_path=settings.htsat_model_path
        )
    return _encoder


def _get_tagger() -> TagClassifier:
    global _tagger
    if _tagger is None:
        _tagger = TagClassifier()
    return _tagger


async def extract_features(upload_file) -> Dict:
    """
    Load an uploaded audio file and compute core features required for review.

    Returns a dictionary that can be fed into the downstream LLM prompt.
    """

    with NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp.write(await upload_file.read())
        tmp_path = tmp.name

    try:
        encoder = _get_encoder()
        target_sr = encoder.get_required_sample_rate() or settings.audio_sample_rate or 48000
        waveform, sample_rate = librosa.load(tmp_path, sr=target_sr, mono=True)
    finally:
        os.unlink(tmp_path)

    tagger = _get_tagger()
    embedding = encoder.get_embedding(waveform, sample_rate)

    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
    duration = librosa.get_duration(y=waveform, sr=sample_rate)

    tag_scores = tagger.predict(embedding)
    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    top_tags = [name for name, _ in sorted_tags[:5]]

    logger.info(
        "추출된 특징 - bpm: %.2f duration: %.2f genre: %s tags: %s",
        tempo,
        duration,
        top_tags[0] if top_tags else "unknown",
        top_tags,
    )

    return {
        "bpm": float(tempo),
        "embedding": embedding.tolist(),
        "duration": float(duration),
        "genre": top_tags[0] if top_tags else "unknown",
        "tags": top_tags,
    }
