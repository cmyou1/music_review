"""
테스트 스크립트: Feature Extraction 출력 확인

실제 음원으로 다음을 출력:
1. HTSAT embedding output
2. CLAP tag output
3. TagClassifier 결과
4. LLM에 넘어가는 prompt
5. 최종 LLM 리뷰 출력
"""

import json
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
import numpy as np
from fastapi import UploadFile

# 백엔드 모듈 임포트
from backend.api.utils_audio import extract_features, _get_encoder, _get_clap, _get_tagger
from backend.llm.generate_review import _build_prompt, generate_review

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """섹션 구분자 출력"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def test_feature_extraction(audio_path: str):
    """음원 파일로 feature extraction 테스트"""

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {audio_path}")
        return

    print(f"🎵 테스트 음원: {audio_file.name}")

    # UploadFile 객체 생성
    with open(audio_file, "rb") as f:
        file_content = f.read()

    class MockUploadFile:
        def __init__(self, filename: str, content: bytes, content_type: str):
            self.filename = filename
            self._content = content
            self.content_type = content_type

        async def read(self):
            return self._content

    upload_file = MockUploadFile(
        filename=audio_file.name,
        content=file_content,
        content_type="audio/wav"
    )

    # =================================================================
    # 1단계: Feature Extraction 실행
    # =================================================================
    print_section("1️⃣  FEATURE EXTRACTION 실행 중...")

    features = await extract_features(upload_file)

    # =================================================================
    # 2단계: HTSAT Output 확인
    # =================================================================
    print_section("2️⃣  HTSAT EMBEDDING OUTPUT")

    encoder = _get_encoder()
    print(f"✓ Model: {encoder.model_name}")
    print(f"✓ Device: {encoder.device}")
    print(f"✓ Use Stub: {encoder.use_stub}")
    print(f"✓ Embedding Dim: {encoder.embedding_dim}")

    embedding_stats = features.get("embedding_stats", {})
    print("\n📊 Embedding Statistics:")
    print(json.dumps(embedding_stats, indent=2, ensure_ascii=False))

    # =================================================================
    # 3단계: CLAP Output 확인
    # =================================================================
    print_section("3️⃣  CLAP TAG OUTPUT")

    clap = _get_clap()
    print(f"✓ Model: {clap.model_name}")
    print(f"✓ Device: {clap.device}")
    print(f"✓ Use Stub: {clap.use_stub}")

    clap_output = features.get("clap", {})
    print("\n🎭 CLAP Analysis:")
    print(json.dumps(clap_output, indent=2, ensure_ascii=False))

    # =================================================================
    # 4단계: TagClassifier 결과 확인
    # =================================================================
    print_section("4️⃣  TAG CLASSIFIER 결과")

    tagger = _get_tagger()
    # TagClassifier는 predict() 호출 시 동적으로 레이블을 반환
    # labels 속성은 없으므로 features에서 직접 태그 정보를 가져옴

    tags = features.get("tags", [])
    print("\n🏷️  Top Tags:")
    for i, tag in enumerate(tags, 1):
        print(f"  {i}. {tag}")

    # TagClassifier의 실제 확률값 출력 (features에는 top5만 있음)
    # 실제 raw scores를 보려면 다시 계산 필요
    print("\n📊 모든 Tag Scores (재계산):")
    import librosa as lb

    # 음원 다시 로드
    target_sr = encoder.get_required_sample_rate() or 48000
    waveform_reloaded, sample_rate_reloaded = lb.load(audio_path, sr=target_sr, mono=True)
    embedding_reloaded = np.asarray(encoder.get_embedding(waveform_reloaded, sample_rate_reloaded), dtype=np.float32)

    tag_scores = tagger.predict(embedding_reloaded)
    sorted_tag_scores = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)

    for tag, score in sorted_tag_scores[:10]:
        print(f"  {tag}: {score:.4f}")

    # =================================================================
    # 5단계: LLM에 넘어가는 Prompt 확인
    # =================================================================
    print_section("5️⃣  LLM PROMPT (실제 전달되는 내용)")

    prompt = _build_prompt(features)
    print(prompt)

    # =================================================================
    # 6단계: 전체 Features JSON 출력
    # =================================================================
    print_section("6️⃣  전체 FEATURES (JSON)")

    print(json.dumps(features, indent=2, ensure_ascii=False, default=str))

    # =================================================================
    # 7단계: LLM 리뷰 생성
    # =================================================================
    print_section("7️⃣  최종 LLM 리뷰 출력")

    review, is_fallback = generate_review(features)

    if is_fallback:
        print("⚠️  [FALLBACK] LLM 호출 실패, fallback 텍스트 사용\n")
    else:
        print("✅ LLM 리뷰 생성 성공\n")

    print(review)

    # =================================================================
    # 분석 결과
    # =================================================================
    print_section("🔍 분석 결과")

    print("✅ 정상 작동:")
    if features.get("bpm"):
        print(f"  ✓ BPM 추출: {features['bpm']:.2f}")
    if features.get("spectral"):
        print(f"  ✓ Spectral features: {len(features['spectral'])} 항목")
    if features.get("tags"):
        print(f"  ✓ Tag 분류: {len(features['tags'])} 개")

    print("\n❌ 문제점:")

    # CLAP 키 불일치 확인
    if clap_output:
        if "primary_mood" not in clap_output and "primary_prompt" in clap_output:
            print("  ⚠️  CLAP 키 불일치: prompt에서 'primary_mood' 찾지만 실제로는 'primary_prompt'")
        if "prompt_hint" not in clap_output:
            print("  ⚠️  CLAP 키 불일치: prompt에서 'prompt_hint' 찾지만 실제 키 없음")

    # TagClassifier 결과가 prompt에 있는지 확인
    if "kick punchy" not in prompt and "synth pad" not in prompt:
        print("  ⚠️  TagClassifier의 악기 확률이 LLM prompt에 포함되지 않음")

    # 임베딩 통계가 prompt에 있는지 확인
    if "L2=" in prompt or "std=" in prompt:
        print("  ⚠️  임베딩 통계값(L2, std 등)이 prompt에 포함됨 - 음악적 의미 없음")


if __name__ == "__main__":
    # Windows 콘솔 인코딩 설정
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    # 테스트 음원 경로
    audio_path = "data/sample_songs/tone.wav"

    # 명령줄 인자로 경로 지정 가능
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]

    print("Music Review AI - Feature Extraction Test")
    print("=" * 80)

    # asyncio 실행
    asyncio.run(test_feature_extraction(audio_path))

    print("\n" + "=" * 80)
    print("✅ 테스트 완료")
    print("=" * 80)
