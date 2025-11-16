"""
Aggregate feedback JSONL logs into CSV summaries.
"""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "backend" / "feedback_logs"
OUTPUT_DIR = BASE_DIR / "backend" / "feedback_logs"


def export_jsonl_to_csv(jsonl_path: Path, csv_path: Path):
    rows = []
    if not jsonl_path.exists():
        return
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    export_jsonl_to_csv(
        LOG_DIR / "review_feedback.jsonl", OUTPUT_DIR / "review_feedback.csv"
    )
    export_jsonl_to_csv(
        LOG_DIR / "recommendation_feedback.jsonl",
        OUTPUT_DIR / "recommendation_feedback.csv",
    )
    print("CSV export complete.")


if __name__ == "__main__":
    main()

