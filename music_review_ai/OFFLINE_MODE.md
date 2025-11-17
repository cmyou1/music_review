# Offline model setup

이 프로젝트는 HTSAT, CLAP, AudioSet AST 태거 3가지 사전 학습 모델을 사용합니다.  
`HTSAT`는 기본 `.env`에 로컬 경로가 지정되어 있으므로 `scripts/download_htsat.py`만 실행하면 됩니다.  
아래 단계를 따르면 나머지 모델들도 네트워크 없이 동작합니다.

1. **사전 준비**
   ```bash
   cd music_review_ai
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```
2. **모델 내려받기**
   ```bash
   python scripts/download_clap.py    # models/clap 폴더로 Hugging Face CLAP 체크포인트 저장
   python scripts/download_tagger.py  # models/tagger 폴더로 AudioSet AST(태그 분류기) 저장
   ```
3. **환경 변수 설정(.env)**
   ```
   CLAP_MODEL_NAME=laion/clap-htsat-fused
   CLAP_MODEL_PATH=models/clap
   CLAP_CACHE_DIR=.cache/clap
   TAG_MODEL_NAME=MIT/ast-finetuned-audioset-10-10-0.4593
   TAG_MODEL_PATH=models/tagger
   TAG_CACHE_DIR=.cache/tagger
   ```
4. **FastAPI 실행**
   ```bash
   cd backend
   uvicorn backend.api.main:app --reload
   ```

이후 FastAPI가 뜰 때 세 모델 모두 로컬 체크포인트에서 로드되며, 인터넷 연결이 끊겨 있어도 동일한 품질로 리뷰/분석 파이프라인을 돌릴 수 있습니다.
