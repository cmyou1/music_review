"""
Simple analyzer for feedback CSVs.
"""

from __future__ import annotations

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "backend" / "feedback_logs"


def summarize_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    total = len(rows)
    if not total:
        return {"total": 0}
    liked = sum(1 for row in rows if row.get("liked", "").lower() == "true")
    return {
        "total": total,
        "liked_ratio": liked / total if total else 0.0,
        "latest": rows[-1],
    }


def main():
    review_stats = summarize_csv(LOG_DIR / "review_feedback.csv")
    rec_stats = summarize_csv(LOG_DIR / "recommendation_feedback.csv")
    print("Review Feedback:", review_stats)
    print("Recommendation Feedback:", rec_stats)


if __name__ == "__main__":
    main()
