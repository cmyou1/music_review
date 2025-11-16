"""
LLM client helpers (OpenAI by default).
"""

from __future__ import annotations

from typing import Optional

from backend.config import settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

_client: Optional["OpenAI"] = None


def get_openai_client() -> Optional["OpenAI"]:
    global _client

    if settings.openai_api_key is None or OpenAI is None:
        return None

    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client

