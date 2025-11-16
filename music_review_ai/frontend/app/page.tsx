"use client";

import { FormEvent, useState } from "react";

type Recommendation = {
  title: string;
  artist: string;
  match_reason: string;
  preview_url: string;
  tags?: string[];
  genre?: string;
};

type ReviewResponse = {
  review: string;
  features: {
    bpm: number;
    duration: number;
    genre: string;
    tags: string[];
  };
  recommendations?: Recommendation[];
};

const formatSeconds = (seconds: number) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}분 ${secs}초`;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReviewResponse | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [reviewLiked, setReviewLiked] = useState<boolean | null>(null);
  const [reviewComment, setReviewComment] = useState("");
  const [recFeedback, setRecFeedback] = useState<Record<string, boolean | null>>({});
  const [reviewFeedbackSubmitting, setReviewFeedbackSubmitting] = useState(false);
  const [reviewFeedbackMessage, setReviewFeedbackMessage] = useState<string | null>(null);
  const [userReview, setUserReview] = useState({
    trackTitle: "",
    trackArtist: "",
    reviewText: "",
    mood: "",
  });
  const [metaReview, setMetaReview] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      setError("먼저 파일을 선택하세요.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setRecommendations([]);
    setReviewLiked(null);
    setReviewComment("");
    setReviewFeedbackMessage(null);
    setReviewFeedbackSubmitting(false);
    setRecFeedback({});
    setMetaReview(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const endpoint = `${API_BASE_URL.replace(/\/$/, "")}/api/review`;
      const res = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.detail ?? "업로드 실패");
      }

      const data = (await res.json()) as ReviewResponse;
      setResult(data);
      setRecommendations(data.recommendations ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const submitReviewFeedback = async () => {
    if (!result || reviewLiked === null || reviewFeedbackSubmitting) return;
    setReviewFeedbackMessage(null);
    setReviewFeedbackSubmitting(true);
    try {
      const res = await fetch(`${API_BASE_URL.replace(/\/$/, "")}/api/feedback/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          review: result.review,
          liked: reviewLiked,
          comment: reviewComment,
        }),
      });
      if (!res.ok) {
        throw new Error("피드백 전송에 실패했어요.");
      }
      setReviewFeedbackMessage("피드백이 전송되었습니다. 감사합니다!");
    } catch (err) {
      setReviewFeedbackMessage(err instanceof Error ? err.message : "피드백 전송 중 오류가 발생했어요.");
    } finally {
      setReviewFeedbackSubmitting(false);
    }
  };

  const submitRecommendationFeedback = async (title: string, artist: string, liked: boolean) => {
    setRecFeedback((prev) => ({ ...prev, [title]: liked }));
    await fetch(`${API_BASE_URL.replace(/\/$/, "")}/api/feedback/recommendation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, artist, liked }),
    });
  };

  const submitUserReview = async () => {
    if (!userReview.trackTitle || !userReview.reviewText) {
      setError("곡 제목과 리뷰 내용을 입력해주세요.");
      return;
    }

    try {
      await fetch(`${API_BASE_URL.replace(/\/$/, "")}/api/reviews/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          track_title: userReview.trackTitle,
          track_artist: userReview.trackArtist,
          review_text: userReview.reviewText,
          mood: userReview.mood || undefined,
        }),
      });

      const metaRes = await fetch(`${API_BASE_URL.replace(/\/$/, "")}/api/reviews/meta`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          track_title: userReview.trackTitle,
          track_artist: userReview.trackArtist,
          review_text: userReview.reviewText,
        }),
      });

      if (!metaRes.ok) {
        throw new Error("메타 리뷰 생성에 실패했어요.");
      }

      const data = await metaRes.json();
      setMetaReview(data.meta_review);
    } catch (err) {
      setError(err instanceof Error ? err.message : "리뷰 제출 중 오류가 발생했어요.");
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white px-4 py-10">
      <div className="max-w-4xl mx-auto space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold">Music Review AI</h1>
          <p className="text-slate-300">
            음원을 업로드하면 FastAPI 백엔드가 BPM·임베딩·태그를 추출하고 OpenAI LLM 리뷰를 생성합니다.
          </p>
        </header>

        <section className="grid gap-6 md:grid-cols-2">
          <form
            onSubmit={handleSubmit}
            className="space-y-4 bg-slate-900/70 p-6 rounded-xl border border-slate-800"
          >
            <p className="text-sm text-slate-400">mp3/wav/flac 파일을 올려 주세요.</p>
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500"
            />
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-lg font-semibold bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50"
            >
              {loading ? "분석 중..." : "리뷰 생성"}
            </button>
            {error && <p className="text-red-400 text-sm">{error}</p>}
            {file && !loading && !error && (
              <p className="text-xs text-slate-500">선택된 파일: {file.name}</p>
            )}
          </form>

          <article className="bg-slate-900/70 p-6 rounded-xl border border-slate-800 space-y-3">
            <h2 className="text-xl font-semibold">상태</h2>
            <dl className="space-y-2 text-sm text-slate-300">
              <div className="flex justify-between">
                <dt>서버</dt>
                <dd className="font-mono">{API_BASE_URL}</dd>
              </div>
              <div className="flex justify-between">
                <dt>업로드 상태</dt>
                <dd>{loading ? "LLM 응답 대기 중..." : "대기"}</dd>
              </div>
              <div className="flex justify-between">
                <dt>파일 선택</dt>
                <dd>{file ? file.name : "없음"}</dd>
              </div>
            </dl>
          </article>
          </section>

        {result && (
          <section className="space-y-6">
            <article className="bg-slate-900/80 p-6 rounded-xl border border-slate-800 space-y-4">
              <div>
                <h2 className="text-2xl font-semibold">LLM 리뷰</h2>
                <p className="text-slate-200 whitespace-pre-line mt-2">{result.review}</p>
              </div>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 text-sm text-slate-300">
                <div className="bg-slate-900/60 rounded-lg p-4">
                  <p className="text-slate-400 text-xs">BPM</p>
                  <p className="text-xl font-semibold">{result.features.bpm.toFixed(1)}</p>
                </div>
                <div className="bg-slate-900/60 rounded-lg p-4">
                  <p className="text-slate-400 text-xs">길이</p>
                  <p className="text-xl font-semibold">{formatSeconds(result.features.duration)}</p>
                </div>
                <div className="bg-slate-900/60 rounded-lg p-4">
                  <p className="text-slate-400 text-xs">예상 장르</p>
                  <p className="text-xl font-semibold">{result.features.genre}</p>
                </div>
                <div className="bg-slate-900/60 rounded-lg p-4">
                  <p className="text-slate-400 text-xs">태그</p>
                  <p className="text-sm">{result.features.tags.join(", ") || "없음"}</p>
                </div>
              </div>
            </article>
            <div className="bg-slate-900/70 p-6 rounded-xl border border-slate-800 space-y-4">
              <h3 className="text-xl font-semibold">리뷰 피드백</h3>
              <div className="flex gap-2">
                <button
                  type="button"
                  className={`px-4 py-2 rounded-lg ${reviewLiked === true ? "bg-green-600" : "bg-slate-800"}`}
                  onClick={() => setReviewLiked(true)}
                >
                  👍 좋았어요
                </button>
                <button
                  type="button"
                  className={`px-4 py-2 rounded-lg ${reviewLiked === false ? "bg-red-600" : "bg-slate-800"}`}
                  onClick={() => setReviewLiked(false)}
                >
                  👎 아쉬웠어요
                </button>
              </div>
              <textarea
                placeholder="느낀 점을 간단히 적어 주세요"
                value={reviewComment}
                onChange={(e) => setReviewComment(e.target.value)}
                className="w-full rounded-lg bg-slate-950 border border-slate-700 text-sm p-3"
              />
              <div className="space-y-2">
                <button
                  type="button"
                  disabled={reviewLiked === null || reviewFeedbackSubmitting}
                  onClick={submitReviewFeedback}
                  className="px-4 py-2 rounded-lg bg-indigo-600 disabled:opacity-50"
                >
                  {reviewFeedbackSubmitting ? "전송 중..." : "피드백 보내기"}
                </button>
                {reviewFeedbackMessage && (
                  <p className="text-sm text-slate-300">{reviewFeedbackMessage}</p>
                )}
              </div>
            </div>
          </section>
        )}

        {recommendations.length > 0 && (
          <section className="space-y-4">
            <h2 className="text-2xl font-semibold">비슷한 곡 추천</h2>
            <div className="grid gap-4 md:grid-cols-2">
              {recommendations.map((item) => (
                <article
                  key={`${item.title}-${item.artist}`}
                  className="bg-slate-900/70 p-4 rounded-xl border border-slate-800 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-lg font-semibold">{item.title}</p>
                      <p className="text-slate-400 text-sm">
                        {item.artist} {item.genre ? `· ${item.genre}` : ""}
                      </p>
                    </div>
                    <a
                      href={item.preview_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-indigo-400 text-sm hover:underline"
                    >
                      미리듣기
                    </a>
                  </div>
                  <p className="text-slate-300 text-sm">{item.match_reason}</p>
                  {item.tags && item.tags.length > 0 && (
                    <div className="flex flex-wrap gap-2 text-xs">
                      {item.tags.map((tag) => (
                        <span
                          key={tag}
                          className="bg-slate-800 border border-slate-700 text-slate-200 px-2 py-0.5 rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="flex gap-2">
                    <button
                      type="button"
                      className={`px-3 py-1 rounded-lg text-sm ${recFeedback[item.title] === true ? "bg-green-600" : "bg-slate-800"}`}
                      onClick={() => submitRecommendationFeedback(item.title, item.artist, true)}
                    >
                      👍
                    </button>
                    <button
                      type="button"
                      className={`px-3 py-1 rounded-lg text-sm ${recFeedback[item.title] === false ? "bg-red-600" : "bg-slate-800"}`}
                      onClick={() => submitRecommendationFeedback(item.title, item.artist, false)}
                    >
                      👎
                    </button>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}

        <section className="space-y-4 bg-slate-900/70 p-6 rounded-xl border border-slate-800">
          <h2 className="text-2xl font-semibold">내 리뷰 제출 & 메타 리뷰</h2>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-4">
              <input
                className="w-full rounded-lg bg-slate-950 border border-slate-700 p-3 text-sm"
                placeholder="곡 제목"
                value={userReview.trackTitle}
                onChange={(e) => setUserReview((prev) => ({ ...prev, trackTitle: e.target.value }))}
              />
              <input
                className="w-full rounded-lg bg-slate-950 border border-slate-700 p-3 text-sm"
                placeholder="아티스트"
                value={userReview.trackArtist}
                onChange={(e) => setUserReview((prev) => ({ ...prev, trackArtist: e.target.value }))}
              />
              <input
                className="w-full rounded-lg bg-slate-950 border border-slate-700 p-3 text-sm"
                placeholder="느낀 분위기 (선택)"
                value={userReview.mood}
                onChange={(e) => setUserReview((prev) => ({ ...prev, mood: e.target.value }))}
              />
              <textarea
                className="w-full rounded-lg bg-slate-950 border border-slate-700 p-3 text-sm"
                placeholder="사람 리뷰를 입력하세요"
                rows={5}
                value={userReview.reviewText}
                onChange={(e) => setUserReview((prev) => ({ ...prev, reviewText: e.target.value }))}
              />
              <button
                type="button"
                className="w-full py-2 rounded-lg bg-indigo-600"
                onClick={submitUserReview}
              >
                리뷰 제출 및 메타 리뷰 요청
              </button>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg min-h-[200px]">
              <h3 className="text-lg font-semibold mb-2">메타 리뷰</h3>
              <p className="text-sm text-slate-200 whitespace-pre-line">
                {metaReview ?? "리뷰를 제출하면 AI 평가가 표시됩니다."}
              </p>
            </div>
          </div>
        </section>

      </div>
    </main>
  );
}
