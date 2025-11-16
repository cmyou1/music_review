"""
Routers for music review operations.
"""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..llm.generate_review import evaluate_user_review, generate_review
from ..recommendations.engine import recommend
from .utils_audio import extract_features
from .schemas import FeedbackPayload, RecommendationFeedbackPayload, UserReviewPayload, MetaReviewRequest

router = APIRouter(prefix="/api", tags=["review"])

LOG_DIR = Path(__file__).resolve().parents[1] / "feedback_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_LOG = LOG_DIR / "review_feedback.jsonl"
REC_LOG = LOG_DIR / "recommendation_feedback.jsonl"
USER_REVIEW_LOG = Path(__file__).resolve().parents[1] / "reviews" / "review_submissions.jsonl"
META_REVIEW_LOG = Path(__file__).resolve().parents[1] / "reviews" / "meta_reviews.jsonl"
USER_REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)


def _append_log(path: Path, payload: dict):
    payload = payload.copy()
    payload["timestamp"] = datetime.utcnow().isoformat()
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


@router.post("/review")
async def review_music(file: UploadFile = File(...)):
    """
    Accept an uploaded audio file and return analysis + natural language review.
    """

    if file.content_type not in {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 오디오 형식입니다.")

    features = await extract_features(file)
    review = generate_review(features)
    recommendations = recommend(features)

    return {"features": features, "review": review, "recommendations": recommendations}


@router.post("/recommendations")
async def recommend_tracks(file: UploadFile = File(...)):
    """
    Placeholder endpoint returning mock recommendations based on tags/genre.
    """

    if file.content_type not in {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 오디오 형식입니다.")

    features = await extract_features(file)
    recommendations = recommend(features)

    return {"features": features, "recommendations": recommendations}


@router.post("/feedback/review")
async def submit_review_feedback(payload: FeedbackPayload):
    _append_log(REVIEW_LOG, payload.dict())
    return {"status": "ok", "message": "리뷰 피드백이 접수되었습니다."}


@router.post("/feedback/recommendation")
async def submit_recommendation_feedback(payload: RecommendationFeedbackPayload):
    _append_log(REC_LOG, payload.dict())
    return {"status": "ok", "message": "추천 피드백이 접수되었습니다."}


@router.post("/reviews/submit")
async def submit_user_review(payload: UserReviewPayload):
    """
    Save user-authored review submissions for later analysis.
    """

    _append_log(USER_REVIEW_LOG, payload.dict())
    return {"status": "ok", "message": "리뷰 제출이 완료되었어요."}


@router.post("/reviews/meta")
async def meta_review(request: MetaReviewRequest):
    """
    Provide an expert-style evaluation of a user-submitted review.
    """

    meta_review_text = evaluate_user_review(
        track_title=request.track_title,
        track_artist=request.track_artist,
        review_text=request.review_text,
    )
    log_entry = request.dict()
    log_entry["meta_review"] = meta_review_text
    _append_log(META_REVIEW_LOG, log_entry)
    return {"meta_review": meta_review_text}
