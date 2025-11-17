"""LLM prompt helpers and retry-aware caller."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from backend.config import settings
from .client import get_openai_client

try:  # pragma: no cover - optional httpx import
    from httpx import TimeoutException
except Exception:  # pragma: no cover
    TimeoutException = Exception  # type: ignore

try:  # pragma: no cover - optional OpenAI deps
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
except Exception:  # pragma: no cover
    APIConnectionError = APIStatusError = APITimeoutError = RateLimitError = Exception  # type: ignore

logger = logging.getLogger(__name__)


def _log_messages(messages: List[Dict[str, str]]) -> None:
    """
    Log the outbound LLM messages for debugging prompt content.
    """

    if not logger.isEnabledFor(logging.DEBUG):
        return

    divider = "=" * 20 + " LLM PROMPT " + "=" * 20
    logger.debug(divider)
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        logger.debug("%s: %s", role, content)
    logger.debug("=" * len(divider))


def _build_prompt(features: Dict) -> str:
    tag_line = ", ".join(features.get("tags", [])) or "no-tag"
    stats = features.get("embedding_stats") or {}
    spectral = features.get("spectral") or {}
    clap = features.get("clap") or {}

    emb_line = f"L2={stats.get('l2_norm', 'n/a')}, std={stats.get('std', 'n/a')}, peak={stats.get('max', 'n/a')}"
    spectral_line = (
        f"brightness={spectral.get('brightness', 'n/a')}, warmth={spectral.get('warmth', 'n/a')}, "
        f"sharpness={spectral.get('sharpness', 'n/a')}, rms={spectral.get('rms', 'n/a')}"
    )
    clap_line = f"mood={clap.get('primary_mood', 'unknown')}, prompt_hint={clap.get('prompt_hint', 'n/a')}"

    return f"""
<song_analysis>
BPM: {features.get('bpm')}
Genre: {features.get('genre', 'unknown')}
Duration: {features.get('duration')}
Tags: {tag_line}
Embedding summary: {emb_line}
Spectral summary: {spectral_line}
CLAP mood: {clap_line}
</song_analysis>

Write a natural Korean music review that covers:
- intro energy & development
- instruments used
- spatial / reverb feel
- overall emotion
- mood/imagery hinted by CLAP analysis
"""


def _call_llm(messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int] = None) -> Optional[str]:
    client = get_openai_client()
    if not client:
        return None

    timeout = max(1.0, float(settings.llm_request_timeout))
    max_attempts = max(1, int(settings.llm_max_retries) + 1)
    retryable_errors = (TimeoutException, APITimeoutError, APIConnectionError, APIStatusError, RateLimitError)

    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("LLM request success in %.1f ms (model=%s)", elapsed_ms, settings.llm_model_name)
            message = response.choices[0].message
            content = getattr(message, "content", None)
            return content.strip() if isinstance(content, str) else None
        except retryable_errors as exc:  # pragma: no cover - depends on runtime
            if attempt >= max_attempts:
                logger.warning("LLM call failed after retries: %s", exc)
                break
            delay = max(0.5, settings.llm_retry_backoff * attempt)
            logger.info("LLM call failed (%s/%s): %s -> retry in %.1fs", attempt, max_attempts - 1, exc, delay)
            time.sleep(delay)
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM call aborted: %s", exc)
            break
    return None


def generate_review(features: Dict) -> Tuple[str, bool]:
    prompt = _build_prompt(features)
    messages = [{"role": "user", "content": prompt}]
    _log_messages(messages)
    content = _call_llm(messages, temperature=0.7, max_tokens=450)
    if content:
        return content, False

    bpm = features.get("bpm", "unknown")
    genre = features.get("genre", "unspecified genre")
    tags = ", ".join(features.get("tags", [])) or "no specific tags"
    fallback = (
        f"[Fallback] This {genre} track flows with a calm energy around BPM {bpm}. "
        f"Key tags: {tags}. Smooth ambience and reverb keep the mood intimate."
    )
    logger.warning("LLM review generation failed; returning fallback text (model=%s)", settings.llm_model_name)
    return fallback, True


def evaluate_user_review(track_title: str, track_artist: str, review_text: str) -> Tuple[str, bool]:
    prompt = f"""
당신은 음악 평론가입니다. 아래 사용자 리뷰를 짧게 평가해 주세요.

곡 제목: {track_title}
아티스트: {track_artist}
리뷰 원문:
\"\"\"{review_text}\"\"\"

1) 기술적 분석 정확도
2) 감정/분위기 표현력
3) 묘사의 구체성
그리고 마지막에 한 줄 총평.
"""
    content = _call_llm([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=350)
    if content:
        return content, False
    logger.warning("LLM 메타 리뷰 폴백 사용 (모델: %s)", settings.llm_model_name)
    return "LLM 호출이 안정화된 뒤 다시 시도해 주세요.", True
