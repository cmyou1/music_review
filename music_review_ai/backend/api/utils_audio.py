"""
Audio utility helpers for the review API.
"""

from __future__ import annotations

import logging
import os
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

import librosa
import numpy as np

from backend.config import settings
from ..models.clap_analyzer import ClapAnalyzer
from ..models.htsat_encoder import HTSATEncoder
from ..models.tag_classifier import TagClassifier
from ..models.embedding_analyzer import EmbeddingAnalyzer

logger = logging.getLogger(__name__)

_encoder: Optional[HTSATEncoder] = None
_tagger: Optional[TagClassifier] = None
_clap: Optional[ClapAnalyzer] = None
_embedding_analyzer: Optional[EmbeddingAnalyzer] = None
HTSAT_MODEL_NAME = "htsat_stub_model"
CLAP_MODEL_NAME = "laion/clap-htsat-fused"


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
        _tagger = TagClassifier(
            model_name=settings.tag_model_name,
            cache_dir=settings.tag_cache_dir,
            local_path=settings.tag_model_path,
        )
    return _tagger


def _get_clap() -> ClapAnalyzer:
    global _clap
    if _clap is None:
        model_name = settings.clap_model_name or CLAP_MODEL_NAME
        _clap = ClapAnalyzer.from_pretrained(
            model_name,
            cache_dir=settings.clap_cache_dir,
            local_path=settings.clap_model_path,
        )
    return _clap


def _get_embedding_analyzer() -> EmbeddingAnalyzer:
    global _embedding_analyzer
    if _embedding_analyzer is None:
        # CLAP analyzer를 전달해서 text encoder 사용
        clap = _get_clap()
        _embedding_analyzer = EmbeddingAnalyzer(clap_analyzer=clap)
    return _embedding_analyzer


def _summarize_embedding(vector: np.ndarray) -> Dict[str, float]:
    if vector.size == 0:
        return {"l2_norm": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}
    norm = float(np.linalg.norm(vector))
    return {
        "l2_norm": round(norm, 3),
        "mean": round(float(vector.mean()), 4),
        "std": round(float(vector.std()), 4),
        "max": round(float(vector.max()), 4),
    }


def _compute_spectral_summary(waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
    if not waveform.size:
        return {}

    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)
    rms = librosa.feature.rms(y=waveform)
    zcr = librosa.feature.zero_crossing_rate(y=waveform)

    def _avg(metric):
        return float(np.mean(metric)) if metric.size else 0.0

    centroid_hz = _avg(centroid)
    rolloff_hz = _avg(rolloff)
    bw_hz = _avg(bandwidth)
    rms_level = _avg(rms)
    zcr_ratio = _avg(zcr)

    brightness = float(np.clip(centroid_hz / 6000.0, 0.0, 1.0))
    warmth = float(np.clip(1.0 - brightness, 0.0, 1.0))
    sharpness = float(np.clip(rolloff_hz / 9000.0, 0.0, 1.0))
    airiness = float(np.clip(zcr_ratio * 2.0, 0.0, 1.0))

    return {
        "centroid_hz": round(centroid_hz, 2),
        "rolloff_hz": round(rolloff_hz, 2),
        "bandwidth_hz": round(bw_hz, 2),
        "rms": round(rms_level, 4),
        "zcr": round(zcr_ratio, 4),
        "brightness": round(brightness, 3),
        "warmth": round(warmth, 3),
        "sharpness": round(sharpness, 3),
        "airiness": round(airiness, 3),
    }


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
    clap = _get_clap()
    emb_analyzer = _get_embedding_analyzer()
    embedding = np.asarray(encoder.get_embedding(waveform, sample_rate), dtype=np.float32)

    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
    duration = librosa.get_duration(y=waveform, sr=sample_rate)
    spectral = _compute_spectral_summary(waveform, sample_rate)

    # Tag prediction using real audio (resample to 16kHz for AST model)
    tag_waveform = waveform
    if sample_rate != 16000:
        tag_waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    tag_scores = tagger.predict(tag_waveform, sample_rate=16000)
    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    top_tags = [name for name, _ in sorted_tags[:5]]
    genre = top_tags[0] if top_tags else "unknown"
    # 상위 악기 확률 저장 (LLM 프롬프트용)
    top_instruments = {name: round(score, 3) for name, score in sorted_tags[:5]}

    # HTSAT 임베딩 분석 (reference-based)
    embedding_characteristics = emb_analyzer.analyze(embedding)
    embedding_features = emb_analyzer.get_musical_features(embedding)
    clap_waveform = waveform
    clap_sr = sample_rate
    if clap and clap.device and sample_rate != 48000:
        clap_waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=48000)
        clap_sr = 48000
    clap_summary = clap.analyze(clap_waveform, clap_sr)

    logger.info("추출특징 - bpm: %.2f duration: %.2f genre: %s tags: %s", tempo, duration, genre, top_tags)

    return {
        "bpm": float(tempo),
        "duration": float(duration),
        "genre": genre,
        "tags": top_tags,
        "instruments": top_instruments,
        "embedding_stats": _summarize_embedding(embedding),
        "embedding_characteristics": embedding_characteristics,
        "embedding_features": embedding_features,
        "spectral": spectral,
        "clap": clap_summary,
    }
