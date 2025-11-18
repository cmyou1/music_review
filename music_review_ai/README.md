# 🎵 Music Review AI (MVP)

음원 파일을 업로드하면 자동으로 장르/분위기/BPM/악기 구성과 사람스러운 리뷰를 생성하는 MVP 스캐폴드입니다. 오디오 임베딩은 HTSAT, 리뷰 생성은 LLM 연동을 목표로 하며 FastAPI 백엔드와 Next.js 프런트엔드 구조를 미리 잡아두었습니다.

## 폴더 구조
```
music_review_ai/
├── backend/          # FastAPI + 오디오 분석
│   ├── api/          # 엔드포인트 및 오디오 유틸
│   ├── llm/          # 리뷰 생성 헬퍼
│   ├── models/       # HTSAT 래퍼 등
│   └── requirements.txt
├── frontend/         # Next.js + Tailwind (아직 비어 있음)
├── data/             # 샘플 음원/임베딩/데이터셋
└── scripts/          # 데이터 전처리/학습 스크립트 자리
```

## 백엔드 실행 방법
1. 환경 변수 설정  
   `.env.example`을 복사해 `.env`를 만들고 OpenAI 키, HTSAT(MERT) 모델명을 지정합니다.  
   `scripts/download_htsat.py`로 모델을 내려받은 경우 `HTSAT_MODEL_PATH`에 해당 폴더를 적으면 오프라인으로도 동작합니다. 기본 샘플레이트는 모델의 feature extractor에서 자동으로 읽어옵니다.
2. 가상 환경 생성 후 패키지 설치
   ```bash
   cd music_review_ai/backend
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. FastAPI 서버 기동
   ```bash
   uvicorn backend.api.main:app --reload
   ```
4. `/api/review`로 음원을 등록하면 BPM/임베딩/리뷰가, `/api/recommendations`로 업로드하면 태그+장르+BPM 유사도 기반 추천 리스트가 JSON으로 반환됩니다.

### LLM 연결 확인
```bash
set OPENAI_API_KEY=sk-xxx  # Windows PowerShell 예시
curl -X POST http://localhost:8000/api/review ^
  -F "file=@data/sample_songs/demo.wav" ^
  | jq
```
정상 동작 시 `review` 필드에 OpenAI가 생성한 한국어 리뷰가 표시됩니다. 키가 없으면 템플릿 리뷰가 반환됩니다.

### 로컬 테스트 루틴
1. **샘플 오디오 생성**
   ```bash
   cd music_review_ai
   python scripts/generate_tone.py --output data/sample_songs/tone.wav --duration 4 --freq 220
   ```
2. **API 호출 스크립트**
   ```bash
   python scripts/call_review_api.py --file data/sample_songs/tone.wav --api-base http://localhost:8000
   ```
   FastAPI 서버가 켜져 있다면 BPM/태그/리뷰가 JSON으로 출력됩니다.

## 프런트엔드 실행
1. `cd music_review_ai/frontend`
2. `.env.local.example`을 복사해 `.env.local`을 만들고 `NEXT_PUBLIC_API_BASE_URL`을 조정합니다.
3. `npm install` (Next.js 초기화 후)
4. `npm run dev`로 로컬 서버 구동
5. http://localhost:3000 접속 후 파일 업로드 폼으로 FastAPI(`NEXT_PUBLIC_API_BASE_URL`)에 요청할 수 있습니다.

## HTSAT & LLM 연동 팁
- 기본 임베딩 모델은 `m-a-p/MERT-v1-95M` (오디오 전용)으로 변경했습니다. 다른 모델을 쓰고 싶다면 `.env`의 `HTSAT_MODEL_NAME`을 바꿔 주세요.
- `scripts/download_htsat.py`를 실행하면 HuggingFace 허브에서 모델을 내려받아 오프라인으로 사용할 수 있습니다. `.env`의 `HTSAT_MODEL_PATH`를 다운로드된 폴더로 지정하면 네트워크 없이도 임베딩을 계산합니다.
- `backend/llm/generate_review.py`는 OpenAI SDK를 기본으로 사용하며 키가 없으면 템플릿 리뷰로 자동 폴백합니다.
- `backend/models/tag_classifier.py`는 임베딩을 고정된 직교 행렬에 투영하고 softmax로 확률을 만들어 상위 태그 5개를 뽑는 자리표시자입니다. 추후 PANNs/HTSAT 태깅 모델을 이 인터페이스에 맞춰 교체하면 됩니다.
- `/api/recommendations` 엔드포인트는 `backend/recommendations/library.json`에 담긴 샘플 곡들과 입력한 태그/장르/BPM을 비교해 코사인 유사도 형태의 점수를 계산합니다. 라이브러리에 실제 데이터를 추가하거나 벡터 검색으로 교체하면 바로 확장 가능합니다.

## 다음 작업 아이디어
1. `extract_features()`에 장르/악기 태깅 모델(PANNs/HTSAT tagger) 추가
2. 프런트엔드 업로드 UI 작성 → FastAPI와 연결
3. 사용자 리뷰/추천/커뮤니티 기능 단계별 확장

## Performance knobs
- LLM_REQUEST_TIMEOUT / LLM_MAX_RETRIES / LLM_RETRY_BACKOFF: configure OpenAI timeout + retries.
- API_WEB_CONCURRENCY: number of uvicorn workers when reload is disabled (e.g. API_WEB_CONCURRENCY=6 python -m backend.api.main).
- API_RELOAD: keep false outside development so worker scaling is enabled.
- Audio endpoints now ship lightweight embedding_stats instead of the raw vector, preventing prompt token blow-ups.

