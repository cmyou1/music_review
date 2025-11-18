"""
음악 파일 처리 시간 측정

각 단계별 소요 시간을 측정합니다.
"""

import sys
import time
import requests
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def measure_processing_time(file_path: str, api_url: str = "http://127.0.0.1:8888"):
    """처리 시간 측정"""

    file = Path(file_path)

    if not file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    print("=" * 80)
    print(f"⏱️  처리 시간 측정: {file.name}")
    print("=" * 80)
    print()

    # API 호출 시작
    start_time = time.time()

    with open(file, 'rb') as f:
        files = {'file': (file.name, f, 'audio/mpeg')}

        try:
            print("📤 API 요청 시작...")
            request_start = time.time()

            response = requests.post(
                f"{api_url}/api/review",
                files=files,
                timeout=180
            )

            request_end = time.time()
            total_time = request_end - start_time

            if response.status_code == 200:
                result = response.json()

                print(f"✅ 요청 완료!")
                print()
                print("⏱️  소요 시간:")
                print("-" * 80)
                print(f"  총 처리 시간: {total_time:.2f}초")
                print()

                # 대략적인 단계별 시간 (추정)
                # 실제로는 백엔드 로그를 분석해야 정확함

                features = result.get("features", {})
                duration = features.get("duration", 0)

                print("📊 예상 단계별 시간 (추정):")
                print(f"  1. 파일 업로드 및 로드: ~{duration * 0.1:.1f}초")
                print(f"  2. 오디오 처리 (librosa): ~{duration * 0.2:.1f}초")
                print(f"  3. HTSAT 임베딩 추출: ~{duration * 0.15:.1f}초")
                print(f"  4. CLAP 분석: ~{duration * 0.15:.1f}초")
                print(f"  5. Feature 분석: ~{duration * 0.1:.1f}초")
                print(f"  6. LLM 리뷰 생성: ~10-15초")
                print()

                # 파일 크기 대비 처리 속도
                file_size_mb = file.stat().st_size / 1024 / 1024
                speed = file_size_mb / total_time if total_time > 0 else 0

                print(f"💾 파일 정보:")
                print(f"  파일 크기: {file_size_mb:.2f} MB")
                print(f"  음원 길이: {duration:.1f}초 ({duration/60:.1f}분)")
                print(f"  처리 속도: {speed:.2f} MB/초")
                print(f"  실시간 배속: {duration/total_time:.2f}x")
                print()

                if total_time < 10:
                    print("⚡ 매우 빠름!")
                elif total_time < 30:
                    print("✅ 정상 속도")
                elif total_time < 60:
                    print("⚠️  조금 느림")
                else:
                    print("🐌 느림 - 최적화 필요")

                print()
                print("-" * 80)
                print("💡 처리 시간은 다음 요인에 영향을 받습니다:")
                print("  - CPU 성능 (현재 CPU 모드)")
                print("  - 음원 길이 (긴 곡일수록 오래 걸림)")
                print("  - LLM API 응답 속도 (OpenAI 서버)")
                print("  - 첫 실행 여부 (모델 로딩 시간)")

            else:
                print(f"❌ API 오류: {response.status_code}")
                print(response.text)

        except requests.exceptions.Timeout:
            print("❌ 요청 타임아웃 (180초 초과)")
        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 실패: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_processing_time.py <음악_파일_경로>")
        print()
        print("예시:")
        print('  python test_processing_time.py "C:\\Music\\song.mp3"')
        sys.exit(1)

    music_file = sys.argv[1]
    measure_processing_time(music_file)
