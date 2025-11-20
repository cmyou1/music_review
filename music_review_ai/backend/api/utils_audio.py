"""
Audio utility helpers for the review API.
"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

import librosa
import numpy as np

from backend.config import settings
from ..models.clap_analyzer import ClapAnalyzer
from ..models.encodec_analyzer import EncodecAnalyzer
from ..models.htsat_encoder import HTSATEncoder
from ..models.tag_classifier import TagClassifier
from ..models.embedding_analyzer import EmbeddingAnalyzer

logger = logging.getLogger(__name__)

_encoder: Optional[HTSATEncoder] = None
_tagger: Optional[TagClassifier] = None
_clap: Optional[ClapAnalyzer] = None
_embedding_analyzer: Optional[EmbeddingAnalyzer] = None
_encodec_analyzer: Optional[EncodecAnalyzer] = None
HTSAT_MODEL_NAME = "htsat_stub_model"
CLAP_MODEL_NAME = "laion/clap-htsat-fused"
ENCODEC_MODEL_NAME = "facebook/encodec_24khz"


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


def _get_encodec_analyzer() -> EncodecAnalyzer:
    global _encodec_analyzer
    if _encodec_analyzer is None:
        model_name = settings.encodec_model_name or ENCODEC_MODEL_NAME
        _encodec_analyzer = EncodecAnalyzer.from_pretrained(
            model_name,
            cache_dir=settings.encodec_cache_dir,
            local_path=settings.encodec_model_path,
        )
    return _encodec_analyzer


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


def _generate_windows(waveform: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    window_sec = settings.audio_max_duration or 30.0
    hop_sec = getattr(settings, "audio_window_hop", None) or max(window_sec / 2, 1.0)
    if window_sec <= 0:
        return [waveform]
    window_samples = int(window_sec * sample_rate)
    hop_samples = max(1, int(hop_sec * sample_rate))
    if waveform.size <= window_samples:
        return [waveform]

    windows: list[np.ndarray] = []
    last_start = max(0, waveform.size - window_samples)
    start = 0
    while True:
        if start > last_start:
            start = last_start
        end = start + window_samples
        segment = waveform[start:end]
        if segment.size < window_samples:
            segment = waveform[last_start:last_start + window_samples]
        windows.append(segment.copy())
        if start >= last_start:
            break
        start += hop_samples
    return windows


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


def _compute_advanced_dsp_metrics(waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
    if not waveform.size:
        return {}

    stft = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=512))
    if stft.size == 0:
        return {}

    # Spectral flux
    flux = np.diff(stft, axis=1)
    flux = np.maximum(flux, 0.0)
    spectral_flux = float(np.mean(np.sqrt((flux ** 2).sum(axis=0))))

    # High-band energy ratio (>8kHz)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    high_band_mask = freqs >= 8000
    total_energy = np.sum(stft, axis=0) + 1e-9
    high_energy = np.sum(stft[high_band_mask], axis=0)
    high_band_ratio = float(np.mean(high_energy / total_energy))

    # Crest factor
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    crest_factor = float(peak / (rms + 1e-9))

    # Transient density via onset detection
    onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
    onset_peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate, units="time")
    duration = librosa.get_duration(y=waveform, sr=sample_rate) or 1.0
    transient_density = float(len(onset_peaks) / duration)

    return {
        "spectral_flux": round(spectral_flux, 4),
        "high_band_ratio": round(np.clip(high_band_ratio, 0.0, 1.0), 4),
        "crest_factor": round(crest_factor, 3),
        "transient_density": round(transient_density, 3),
    }


async def extract_features(upload_file) -> Dict:
    """Sliding-window audio analysis feeding the LLM."""

    with NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp.write(await upload_file.read())
        tmp_path = tmp.name

    try:
        encoder = _get_encoder()
        target_sr = encoder.get_required_sample_rate() or settings.audio_sample_rate or 48000
        waveform, sample_rate = librosa.load(tmp_path, sr=target_sr, mono=True)
    finally:
        os.unlink(tmp_path)

    duration = librosa.get_duration(y=waveform, sr=sample_rate)
    windows = _generate_windows(waveform, sample_rate)
    if not windows:
        windows = [waveform]
    window_count = len(windows)

    tagger = _get_tagger()
    clap = _get_clap()
    emb_analyzer = _get_embedding_analyzer()
    timbre_analyzer = _get_encodec_analyzer()

    embeddings = []
    spectral_totals = defaultdict(float)
    tempos = []
    tag_totals = defaultdict(float)
    clap_match_totals = defaultdict(float)
    clap_energy_total = 0.0
    clap_audio_totals = defaultdict(float)
    timbre_totals = defaultdict(float)
    timbre_descriptors = Counter()

    for segment in windows:
        embedding_vec = np.asarray(encoder.get_embedding(segment, sample_rate), dtype=np.float32)
        embeddings.append(embedding_vec)

        tempo, _ = librosa.beat.beat_track(y=segment, sr=sample_rate)
        tempos.append(float(tempo))

        spectral = _compute_spectral_summary(segment, sample_rate)
        advanced = _compute_advanced_dsp_metrics(segment, sample_rate)
        for key, value in {**spectral, **advanced}.items():
            spectral_totals[key] += value

        seg_waveform = segment
        if sample_rate != 16000:
            seg_waveform = librosa.resample(segment, orig_sr=sample_rate, target_sr=16000)
        tag_scores = tagger.predict(seg_waveform, sample_rate=16000)
        for name, score in tag_scores.items():
            tag_totals[name] += score

        clap_waveform = segment
        clap_sr = sample_rate
        if clap and clap.device and sample_rate != 48000:
            clap_waveform = librosa.resample(segment, orig_sr=sample_rate, target_sr=48000)
            clap_sr = 48000
        clap_summary = clap.analyze(clap_waveform, clap_sr)
        clap_energy_total += clap_summary.get("energy_level", 0.0)
        for match in clap_summary.get("top_matches", []):
            clap_match_totals[match["text"]] += match["score"]
        for key, value in (clap_summary.get("audio_embedding_stats") or {}).items():
            clap_audio_totals[key] += value

        timbre_summary = timbre_analyzer.analyze(segment, sample_rate)
        for key, value in timbre_summary.items():
            if isinstance(value, (int, float)):
                timbre_totals[key] += value
        desc = timbre_summary.get("descriptor")
        if desc:
            timbre_descriptors[desc] += 1

    embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    embedding_characteristics = emb_analyzer.analyze(embedding)
    embedding_features = emb_analyzer.get_musical_features(embedding)

    tempo = float(np.mean(tempos)) if tempos else 0.0
    spectral_avg = {key: round(value / window_count, 4) for key, value in spectral_totals.items()}

    sorted_tags = sorted(tag_totals.items(), key=lambda x: x[1], reverse=True)
    top_tags = [name for name, _ in sorted_tags[:5]]
    genre = top_tags[0] if top_tags else "unknown"
    top_instruments = {name: round(score, 3) for name, score in sorted_tags[:5]}

    top_clap_matches = sorted(clap_match_totals.items(), key=lambda x: x[1], reverse=True)
    if top_clap_matches:
        primary_prompt, primary_score = top_clap_matches[0]
    else:
        primary_prompt, primary_score = "unknown", 0.0
    clap_summary_agg = {
        "primary_prompt": primary_prompt,
        "prompt_score": round(primary_score / window_count, 3),
        "top_matches": [
            {"text": text, "score": round(score / window_count, 3)}
            for text, score in top_clap_matches[:4]
        ],
        "audio_embedding_stats": {
            key: round(value / window_count, 4) for key, value in clap_audio_totals.items()
        },
        "energy_level": round(clap_energy_total / window_count, 3),
    }

    timbre_descriptor = timbre_descriptors.most_common(1)[0][0] if timbre_descriptors else "unknown"
    timbre_summary = {
        key: round(value / window_count, 3) for key, value in timbre_totals.items()
    }
    timbre_summary["descriptor"] = timbre_descriptor

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
        "spectral": spectral_avg,
        "clap": clap_summary_agg,
        "timbre": timbre_summary,
    }
