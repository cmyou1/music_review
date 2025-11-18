"""
실제 음악 파일로 API 테스트

사용법:
    python test_with_real_music.py <음악_파일_경로>
    python test_with_real_music.py "C:\Music\song.mp3"
"""

import sys
import json
import requests
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def test_music_file(file_path: str, api_url: str = "http://127.0.0.1:8888"):
    """음악 파일로 리뷰 API 테스트"""

    file = Path(file_path)

    if not file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    print("=" * 80)
    print(f"🎵 테스트 음원: {file.name}")
    print(f"📁 파일 경로: {file}")
    print(f"📊 파일 크기: {file.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 80)
    print()

    # API 호출
    print("⏳ API 요청 중...")

    with open(file, 'rb') as f:
        files = {'file': (file.name, f, 'audio/mpeg')}

        try:
            response = requests.post(
                f"{api_url}/api/review",
                files=files,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()

                print("\n" + "=" * 80)
                print("✅ 리뷰 생성 성공!")
                print("=" * 80)

                # Features 출력
                features = result.get("features", {})

                print("\n📊 음원 분석 결과:")
                print("-" * 80)
                print(f"  BPM: {features.get('bpm', 'N/A'):.1f}")
                print(f"  Duration: {features.get('duration', 'N/A'):.1f}초")
                print(f"  Genre: {features.get('genre', 'N/A')}")

                # 악기
                instruments = features.get('instruments', {})
                if instruments:
                    print(f"\n  🎸 감지된 악기 (Top 3):")
                    for inst, prob in list(instruments.items())[:3]:
                        if prob > 0.01:
                            print(f"    - {inst}: {prob:.1%}")

                # Spectral
                spectral = features.get('spectral', {})
                if spectral:
                    print(f"\n  🎛️ 음향 특성:")
                    print(f"    - Brightness: {spectral.get('brightness', 0):.3f}")
                    print(f"    - Warmth: {spectral.get('warmth', 0):.3f}")
                    print(f"    - Airiness: {spectral.get('airiness', 0):.3f}")

                # CLAP
                clap = features.get('clap', {})
                if clap:
                    print(f"\n  🎭 CLAP 분위기 분석:")
                    print(f"    - Primary mood: \"{clap.get('primary_prompt', 'N/A')}\"")
                    print(f"    - Confidence: {clap.get('prompt_score', 0):.3f}")
                    print(f"    - Energy: {clap.get('energy_level', 0):.3f}")

                    top_matches = clap.get('top_matches', [])
                    if len(top_matches) > 1:
                        print(f"\n    Top moods:")
                        for match in top_matches[:3]:
                            print(f"      {match.get('score', 0):.3f} - {match.get('text', 'N/A')}")

                # Embedding characteristics
                emb_char = features.get('embedding_characteristics', {})
                if emb_char:
                    print(f"\n  🎚️ 프로덕션 스타일:")
                    print(f"    - \"{emb_char.get('primary_characteristic', 'N/A')}\"")
                    print(f"    - Confidence: {emb_char.get('characteristic_score', 0):.3f}")

                # Musical features
                emb_feat = features.get('embedding_features', {})
                if emb_feat:
                    print(f"\n  📈 음악적 특성:")
                    print(f"    - Energy: {emb_feat.get('energy', 0):.2f}")
                    print(f"    - Complexity: {emb_feat.get('complexity', 0):.2f}")
                    print(f"    - Dynamism: {emb_feat.get('dynamism', 0):.2f}")
                    print(f"    - Texture smoothness: {emb_feat.get('texture_smoothness', 0):.2f}")
                    print(f"    - Density: {emb_feat.get('density', 0):.2f}")

                # 리뷰
                review = result.get("review", "")
                is_fallback = result.get("review_is_fallback", False)

                print("\n" + "=" * 80)
                if is_fallback:
                    print("⚠️  LLM 리뷰 (Fallback)")
                else:
                    print("✨ AI 생성 리뷰")
                print("=" * 80)
                print()
                print(review)
                print()

                # JSON 저장
                output_file = file.with_suffix('.review.json')
                with open(output_file, 'w', encoding='utf-8') as out:
                    json.dump(result, out, ensure_ascii=False, indent=2)

                print("\n" + "=" * 80)
                print(f"💾 전체 결과가 저장되었습니다: {output_file}")
                print("=" * 80)

            else:
                print(f"❌ API 오류: {response.status_code}")
                print(response.text)

        except requests.exceptions.Timeout:
            print("❌ 요청 타임아웃 (120초 초과)")
        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 실패: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_with_real_music.py <음악_파일_경로>")
        print()
        print("예시:")
        print('  python test_with_real_music.py "C:\\Music\\song.mp3"')
        print('  python test_with_real_music.py data/sample_songs/tone.wav')
        sys.exit(1)

    music_file = sys.argv[1]
    test_music_file(music_file)
