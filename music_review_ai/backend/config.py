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
    audio_max_duration: Optional[float] = Field(default=30.0, env="AUDIO_MAX_DURATION")
    audio_window_hop: Optional[float] = Field(default=15.0, env="AUDIO_WINDOW_HOP")

    # LLM
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    llm_model_name: str = Field("gpt-4o-mini", env="LLM_MODEL_NAME")
    llm_request_timeout: float = Field(15.0, env="LLM_REQUEST_TIMEOUT")
    llm_max_retries: int = Field(2, env="LLM_MAX_RETRIES")
    llm_retry_backoff: float = Field(1.2, env="LLM_RETRY_BACKOFF")

    # CLAP
    clap_model_name: str = Field("laion/clap-htsat-fused", env="CLAP_MODEL_NAME")
    clap_cache_dir: Optional[Path] = Field(default=None, env="CLAP_CACHE_DIR")
    clap_model_path: Optional[Path] = Field(default=None, env="CLAP_MODEL_PATH")

    # EnCodec timbre encoder
    encodec_model_name: str = Field("facebook/encodec_24khz", env="ENCODEC_MODEL_NAME")
    encodec_cache_dir: Optional[Path] = Field(default=None, env="ENCODEC_CACHE_DIR")
    encodec_model_path: Optional[Path] = Field(default=None, env="ENCODEC_MODEL_PATH")

    # Tag classifier (AudioSet AST)
    tag_model_name: str = Field("MIT/ast-finetuned-audioset-10-10-0.4593", env="TAG_MODEL_NAME")
    tag_cache_dir: Optional[Path] = Field(default=None, env="TAG_CACHE_DIR")
    tag_model_path: Optional[Path] = Field(default=None, env="TAG_MODEL_PATH")

    # API
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_reload: bool = Field(False, env="API_RELOAD")
    api_web_concurrency: int = Field(2, env="API_WEB_CONCURRENCY")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
