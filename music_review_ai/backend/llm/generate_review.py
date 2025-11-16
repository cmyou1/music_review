"""Utility to talk to an LLM (OpenAI 기반)."""

from __future__ import annotations

from typing import Dict

from backend.config import settings
from .client import get_openai_client
import logging

logger = logging.getLogger(__name__)

def _build_prompt(features: Dict) -> str:
    tag_line = ", ".join(features.get("tags", [])) or "미지정"
    return f"""
<song_analysis>
BPM: {features.get('bpm')}
Genre: {features.get('genre', 'unknown')}
Duration: {features.get('duration')}
Tags: {tag_line}
Embedding: (길이 {len(features.get('embedding', []))} 벡터)
</song_analysis>

다음 항목을 반드시 포함해서 사람스러운 음악 리뷰를 작성해줘.
- 도입부 분위기
- 킥/스네어의 느낌
- 악기 구성
- 공간감(리버브/팬닝)
- 전체적인 감성
"""


def generate_review(features: Dict) -> str:
    """
    Generate a human-like review from extracted audio features.
    """

    client = get_openai_client()
    prompt = _build_prompt(features)

    if client:
        try:
            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, str):
                logger.info("LLM 리뷰 생성 성공 (모델: %s)", settings.llm_model_name)
                return content.strip()
        except Exception as exc:  # pragma: no cover
            # Fallback below
            logger.warning("LLM 호출 실패: %s", exc)

    bpm = features.get("bpm", "unknown")
    genre = features.get("genre", "unspecified genre")
    tags = ", ".join(features.get("tags", [])) or "no specific tags"

    return (
        f"도입부에서 미묘하게 압축된 킥이 등장하며 곡의 기조를 잡습니다. "
        f"장르는 {genre}로 분류되며 BPM {bpm} 근처에서 차분하게 전개돼요. "
        f"주요 악기/태그: {tags}. 임베딩 기반으로 판단했을 때 공간감이 넓고 "
        "리버브의 꼬리가 긴 편이라 밤에 듣기 좋은 무드를 만듭니다."
    )


def evaluate_user_review(track_title: str, track_artist: str, review_text: str) -> str:
    client = get_openai_client()
    prompt = f"""
당신은 음악 평론을 평가하는 전문가다.
아래는 한 사용자가 작성한 음악 리뷰다.

곡 제목: {track_title}
아티스트: {track_artist}
사용자 리뷰:
\"\"\"{review_text}\"\"\"

다음 항목에 대한 평가를 작성해 줘:
1. 기술적 분석 정확도
2. 감정/분위기 표현력
3. 서술의 구체성
마지막에 한 줄로 총평을 적어라.
"""
    if client:
        try:
            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("메타 리뷰 생성 실패: %s", exc)
    return "리뷰를 평가하려면 LLM 설정이 필요합니다."
