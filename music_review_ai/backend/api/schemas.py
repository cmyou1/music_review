from __future__ import annotations

from pydantic import BaseModel, Field


class FeedbackPayload(BaseModel):
    review: str = Field(..., description="LLM 리뷰 전문")
    liked: bool = Field(..., description="사용자가 리뷰를 좋아했는지 여부")
    comment: str | None = Field(default=None, description="선택 메모")


class RecommendationFeedbackPayload(BaseModel):
    title: str
    artist: str
    liked: bool
    comment: str | None = None


class UserReviewPayload(BaseModel):
    track_title: str
    track_artist: str
    review_text: str
    mood: str | None = Field(default=None, description="선택: 사용자가 느낀 분위기")


class MetaReviewRequest(BaseModel):
    track_title: str
    track_artist: str
    review_text: str
