# 🎵 Music Review AI (MVP)

음원 파일을 업로드하면 장르·분위기·BPM·악기 구성과 사람스러운 리뷰를 자동으로 생성하는 FastAPI + Next.js 기반 MVP입니다. 오디오 임베딩은 MERT(HTSAT) 모델, 리뷰 생성은 OpenAI LLM을 사용합니다.

## 폴더 구조
```
music_review_ai/
├── backend/          # FastAPI, 오디오 분석, LLM 연동
│   ├── api/          # 엔드포인트 및 오디오 유틸
│   ├── models/       # Python 모델 래퍼, 태그 분류기
│   ├── llm/          # LLM 클라이언트/프롬프트
│   ├── recommendations/ # 추천 라이브러리 + 엔진
│   └── feedback_logs/   # 피드백 JSONL/CSV
├── frontend/         # Next.js + Tailwind 업로드 UI
├── models/           # HTSAT/MERT 모델 파일 (.pt, .bin)
├── scripts/          # 데이터/분석/다운로드 유틸
└── data/             # 샘플 음원 및 임시 데이터
```

## 백엔드 실행
1. `.env.example`을 복사해 `.env`를 만들고 OpenAI 키와 MERT 모델 정보를 설정합니다. `scripts/download_htsat.py`로 `m-a-p/MERT-v1-95M`을 내려받았다면 `HTSAT_MODEL_PATH=models/htsat`을 지정하세요.
2. 의존성 설치
   ```bash
   cd music_review_ai/backend
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. 서버 기동
   ```bash
   uvicorn backend.api.main:app --reload
   ```
4. 주요 엔드포인트
   - `POST /api/review`: 음원 업로드 → BPM/임베딩/LLM 리뷰 반환
   - `POST /api/recommendations`: 음원 업로드 → 태그/BPM/장르 기반 추천 리스트 반환
   - `POST /api/feedback/review`, `/api/feedback/recommendation`: 사용자 피드백 수집(JSONL/CSV 로그 저장)

### 로컬 테스트 루틴
```bash
cd music_review_ai
python scripts/generate_tone.py --output data/sample_songs/tone.wav
python scripts/call_review_api.py --file data/sample_songs/tone.wav --api-base http://localhost:8000
```
LLM 연결 확인을 위해서는 `OPENAI_API_KEY`를 설정한 뒤 `curl`이나 위 스크립트로 결과를 확인하면 됩니다.

## 프런트엔드 실행
1. `.env.local.example`을 복사해 `.env.local`을 만든 뒤 `NEXT_PUBLIC_API_BASE_URL`을 FastAPI 주소로 설정합니다.
2. Next.js 의존성 설치 및 실행
   ```bash
   cd music_review_ai/frontend
   npm install
   npm run dev
   ```
3. http://localhost:3000 에 접속하면 업로드 폼과 LLM 리뷰, 추천, 피드백 UI를 확인할 수 있습니다.

## 추천/피드백
- `backend/recommendations/library.json`에 샘플 곡 메타데이터가 들어 있으며, 태그 벡터 + BPM 근접도로 유사도를 계산합니다. 실제 곡 데이터나 벡터 DB로 쉽게 교체할 수 있습니다.
- 피드백은 `backend/feedback_logs/*.jsonl`에 저장되며, `scripts/export_feedback.py` + `scripts/analyze_feedback.py`로 CSV 변환 및 간단한 통계를 확인할 수 있습니다.

## 다음 단계 아이디어
1. 실데이터/벡터 검색을 이용한 추천 고도화
2. 피드백 로그를 DB/대시보드로 연동해 모델 개선에 활용
3. PANNs/HTSAT 태깅 모델을 붙여 정확한 악기/장르 추정 구현
4. 커뮤니티 기능(리뷰 공유, 추천 피드백)을 위한 UI/백엔드 확장


## Feature stack (Late Fusion)
- **HTSAT encoder**: 악기/음향 이벤트를 포착한 태그 임베딩을 생성합니다.
- **CLAP analyzer (stub)**: 장르+BPM+spectral brightness를 이용한 분위기/에너지 벡터를 텍스트 매칭 형태로 제공합니다.
- **Librosa BPM & spectral stats**: tempo/키감각과 spectral centroid·rolloff·bandwidth·ZCR을 추출해 밝기/무게감/다이나믹을 정량화합니다.
- 이 4가지 신호를 extract_features()에서 하나의 JSON으로 병합, LLM 프롬프트에 직접 넣어주어 Late Fusion 효과를 냅니다.

