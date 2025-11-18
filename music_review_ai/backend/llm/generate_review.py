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

import re


def _remove_numbers_from_review(text: str) -> str:
    """
    Post-process review to remove sentences containing numbers/scores.

    Strategy: Remove entire sentences that contain numeric references.
    """
    # Split into sentences using Korean sentence endings
    # Don't split on periods inside numbers (0.49)
    # Split on '. ' (period + space), '! ', '? ' or period at end of text
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)

    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if sentence contains problematic patterns
        has_numbers = False

        # BPM mentions
        if re.search(r'\b\d+\.?\d*\s*BPM|BPM\s*\d+', sentence, re.IGNORECASE):
            has_numbers = True
        # Percentages
        if re.search(r'\d+\.?\d*\s*%', sentence):
            has_numbers = True
        # Score mentions
        if re.search(r'점수[가는이를]?\s*\d+\.?\d*|\d+\.?\d*[으로]*\s*점수', sentence):
            has_numbers = True
        # Duration in seconds
        if re.search(r'\d+\.?\d*초', sentence):
            has_numbers = True
        # Feature names with numbers
        if re.search(r'(에너지|밀도|복잡성|다이나믹|텍스처|따뜻함|공기감|airiness|brightness|warmth)[^.!?]*\d+\.?\d*', sentence, re.IGNORECASE):
            has_numbers = True
        # Ratio/percentage descriptions
        if re.search(r'\d+\.?\d*\s*[으로의와에]?\s*(비율|높|낮|차지)', sentence):
            has_numbers = True

        if not has_numbers:
            cleaned_sentences.append(sentence)

    result = ' '.join(cleaned_sentences)
    return result.strip()


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
    spectral = features.get("spectral") or {}
    clap = features.get("clap") or {}
    instruments = features.get("instruments") or {}
    emb_char = features.get("embedding_characteristics") or {}
    emb_feat = features.get("embedding_features") or {}

    # Spectral features (음악적으로 의미 있는 특성)
    spectral_line = (
        f"brightness={spectral.get('brightness', 'n/a')}, "
        f"warmth={spectral.get('warmth', 'n/a')}, "
        f"sharpness={spectral.get('sharpness', 'n/a')}, "
        f"airiness={spectral.get('airiness', 'n/a')}"
    )

    # CLAP mood (primary_prompt 사용)
    clap_primary = clap.get('primary_prompt', 'unknown')
    clap_score = clap.get('prompt_score', 0.0)
    clap_energy = clap.get('energy_level', 0.0)

    # 악기 확률 (상위 3개만, 비율 제외)
    instrument_lines = []
    for inst, prob in list(instruments.items())[:3]:
        if prob > 0.1:  # 10% 이상만 표시
            instrument_lines.append(inst)
    instruments_str = ", ".join(instrument_lines) if instrument_lines else "not detected"

    # HTSAT 임베딩 특성 (reference-based)
    primary_char = emb_char.get('primary_characteristic', 'unknown')
    char_score = emb_char.get('characteristic_score', 0.0)

    # 음악적 feature 점수 (0.0-1.0)
    energy = emb_feat.get('energy', 0.5)
    complexity = emb_feat.get('complexity', 0.5)
    texture = emb_feat.get('texture_smoothness', 0.5)
    dynamism = emb_feat.get('dynamism', 0.5)
    density = emb_feat.get('density', 0.5)

    musical_features_line = (
        f"energy={energy:.2f}, complexity={complexity:.2f}, "
        f"texture_smoothness={texture:.2f}, dynamism={dynamism:.2f}, density={density:.2f}"
    )

    # Duration 표현 (범주만)
    duration_secs = features.get('duration', 0)
    if duration_secs < 90:
        duration_desc = "short"
    elif duration_secs < 180:
        duration_desc = "medium length"
    elif duration_secs < 300:
        duration_desc = "longer"
    else:
        duration_desc = "extended"

    # BPM 표현 (범주만)
    bpm = features.get('bpm', 0)
    if bpm > 140:
        tempo_desc = "fast"
    elif bpm > 120:
        tempo_desc = "upbeat"
    elif bpm > 100:
        tempo_desc = "moderate"
    elif bpm > 80:
        tempo_desc = "relaxed"
    else:
        tempo_desc = "slow"

    return f"""다음은 한 음악 트랙의 분석 결과입니다.
각 수치는 청각적 특성을 나타내며, 리뷰에는 수치를 그대로 언급하지 말고
음악적으로 자연스럽게 해석하여 표현해주세요.

Instruments detected: {instruments_str}
Overall mood: {clap_primary}

Audio features (interpret naturally):
- Spectral: brightness={spectral.get('brightness', 0.5):.2f}, warmth={spectral.get('warmth', 0.5):.2f}, airiness={spectral.get('airiness', 0.5):.2f}
- Musical: energy={energy:.2f}, complexity={complexity:.2f}, texture_smoothness={texture:.2f}
- Development: dynamism={dynamism:.2f}, density={density:.2f}

Track info:
- Tempo feel: {tempo_desc}
- Duration: {duration_desc}

위 정보를 바탕으로 프로듀서/엔지니어 관점의 한국어 리뷰(약 200-250단어)를 작성해주세요.

작성 가이드:
- 수치를 직접 언급하지 마세요 (BPM, 퍼센트, 정확한 Hz값 등 금지)
- 프로덕션 기술 용어를 자연스럽게 사용하세요 (EQ, 컴프레션, 리버브, 레이어링, 마스킹, 스테레오 이미징 등)
- 믹싱 밸런스와 주파수 분리에 대해 언급하세요
- 사운드 디자인과 악기 선택의 적절성을 평가하세요
- 감지된 악기만 언급하고, 없는 악기를 만들어내지 마세요
- 과도한 비유나 추상적 표현 대신 구체적인 사운드 특성을 설명하세요

초점:
1. 프로덕션 퀄리티 (믹싱, 마스터링)
2. 사운드 디자인과 악기 처리
3. 주파수 밸런스와 공간감
4. 장르 특성과 전개 방식
5. 개선 가능한 점이 있다면 건설적으로 제시
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

    # Few-shot example: LLM interprets numeric features creatively
    example_prompt = """다음은 한 음악 트랙의 분석 결과입니다.

Instruments detected: synth pad, bass warm, kick punchy
Overall mood: uplifting synthwave with bright leads

Audio features (interpret naturally):
- Spectral: brightness=0.82, warmth=0.71, airiness=0.68
- Musical: energy=0.88, complexity=0.45, texture_smoothness=0.79
- Development: dynamism=0.74, density=0.52

Track info:
- Tempo feel: upbeat
- Duration: medium length

위 정보를 기반으로 자연스러운 한국어 음악 리뷰를 작성해주세요. 수치를 직접 언급하지 말고 창의적으로 해석하세요."""

    example_review = """신스패드 레이어링이 깔끔하게 처리된 신스웨이브 트랙이다. 하이패스 필터로 정리된 패드가 상단을 채우고, 서브 베이스는 로우엔드를 단단하게 잡아준다. 킥 드럼의 어택이 선명하고, 사이드체인 컴프레션이 적절히 걸려 있어 그루브가 살아있다.

믹싱 밸런스가 좋다. 신스와 베이스가 주파수 대역을 잘 나눠 가졌고, 서로 마스킹 없이 깔끔하게 분리된다. 리버브는 은은하게 걸려 있어 공간감을 주면서도 뭉개지지 않는다. 스테레오 이미징도 잘 활용했다. 패드는 넓게 퍼지고, 킥과 베이스는 센터에 단단히 자리 잡았다.

다이나믹 전개가 자연스럽다. 빌드업과 브레이크다운이 예측 가능하지만 효과적이고, 에너지 레벨이 잘 조절되어 있다. 전형적인 신스웨이브 사운드팔레트를 사용하면서도 프로덕션 퀄리티로 차별화를 둔 트랙이다."""

    messages = [
        {"role": "system", "content": "You are an experienced music producer and audio engineer reviewing tracks for a professional music production magazine. Write technical but accessible reviews focusing on production quality, sound design, mixing, instrumentation, and genre characteristics. Use industry terminology naturally. Be specific about what works and what could be improved. NEVER mention exact numbers, BPM values, percentages, or statistics."},
        {"role": "user", "content": example_prompt},
        {"role": "assistant", "content": example_review},
        {"role": "user", "content": prompt}
    ]
    _log_messages(messages)
    content = _call_llm(messages, temperature=0.8, max_tokens=500)
    if content:
        # Post-process to remove any numbers that slipped through
        cleaned_content = _remove_numbers_from_review(content)
        if cleaned_content:
            return cleaned_content, False
        # If cleaning removed everything, return original
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
