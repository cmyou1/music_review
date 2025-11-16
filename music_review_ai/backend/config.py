"""
Centralised configuration for the backend service.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # HTSAT / audio
    htsat_model_name: str = Field("m-a-p/MERT-v1-95M", env="HTSAT_MODEL_NAME")
    htsat_cache_dir: Optional[Path] = Field(default=None, env="HTSAT_CACHE_DIR")
    htsat_model_path: Optional[Path] = Field(default=None, env="HTSAT_MODEL_PATH")
    audio_sample_rate: Optional[int] = Field(default=None, env="AUDIO_SAMPLE_RATE")

    # LLM
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    llm_model_name: str = Field("gpt-4o-mini", env="LLM_MODEL_NAME")

    # API
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
