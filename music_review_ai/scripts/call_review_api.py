"""
CLI helper to upload a file to the FastAPI /api/review endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def request_review(file_path: Path, api_base: str = "http://localhost:8000"):
    endpoint = f"{api_base.rstrip('/')}/api/review"
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "audio/wav")}
        response = requests.post(endpoint, files=files, timeout=120)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI 리뷰 엔드포인트 호출")
    parser.add_argument("--file", type=Path, default=Path("data/sample_songs/tone.wav"))
    parser.add_argument("--api-base", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    result = request_review(args.file, args.api_base)
    print(json.dumps(result, ensure_ascii=False, indent=2))

