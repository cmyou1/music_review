"""
Simple recommendation engine using tag vectors and BPM similarity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

LIB_PATH = Path(__file__).with_name("library.json")

if LIB_PATH.exists():
    LIBRARY: List[Dict] = json.loads(LIB_PATH.read_text(encoding="utf-8"))
else:  # pragma: no cover
    LIBRARY = []

TAG_VOCAB = sorted({tag for item in LIBRARY for tag in item.get("tags", [])})
TAG_INDEX = {tag: idx for idx, tag in enumerate(TAG_VOCAB)}


def _tag_vector(tags) -> np.ndarray:
    vec = np.zeros(len(TAG_VOCAB), dtype=float)
    for tag in tags or []:
        idx = TAG_INDEX.get(tag)
        if idx is not None:
            vec[idx] = 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


for track in LIBRARY:
    track["_tag_vector"] = _tag_vector(track.get("tags"))


def _similarity(features: Dict, candidate: Dict) -> float:
    cand_vec = candidate.get("_tag_vector")
    input_vec = _tag_vector(features.get("tags"))
    tag_score = float(input_vec @ cand_vec) if cand_vec is not None and TAG_VOCAB else 0.0

    bpm_in = float(features.get("bpm") or 0.0)
    bpm_candidate = float(candidate.get("bpm") or 0.0)
    bpm_score = math.exp(-abs(bpm_in - bpm_candidate) / 80.0)

    genre_score = 1.0 if candidate.get("genre") == features.get("genre") else 0.0
    return 0.6 * tag_score + 0.25 * bpm_score + 0.15 * genre_score


def recommend(features: Dict, top_k: int = 3) -> List[Dict]:
    scored = []
    for track in LIBRARY:
        score = _similarity(features, track)
        scored.append((score, track))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, track in scored[:top_k]:
        overlap = set(features.get("tags", [])) & set(track.get("tags", []))
        reason = f"{len(overlap)}개의 태그와 BPM 유사도 기반 (score={score:.2f})"
        results.append(
            {
                "title": track["title"],
                "artist": track["artist"],
                "match_reason": reason,
                "preview_url": track.get("preview_url", ""),
                "tags": track.get("tags", []),
                "genre": track.get("genre", ""),
            }
        )
    return results

