"""Application configuration (full SRS Appendix B coverage).

All settings load from `.env`/environment via pydantic-settings.
Secrets (OLLAMA_API_KEY, DATABASE_URL) are SecretStr — never logged,
never in repr (NFR-5). Every key is typed and validated; every
Appendix B key is present with its spec default.
"""

from typing import Literal

from pydantic import Field, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All SRS Appendix B keys with spec defaults and validation."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Ollama (two separate endpoints) ---
    OLLAMA_LOCAL_URL: str = "http://localhost:11434"
    OLLAMA_CLOUD_URL: str = "https://ollama.com"
    OLLAMA_API_KEY: SecretStr = Field(description="Ollama Cloud API key (never log)")
    AGENT_MODEL: str = "glm-5.3-flash"
    OCR_MODEL: str = "glm-ocr"
    EMBED_MODEL: str = "qwen3-embedding:0.6b"
    AGENT_TIMEOUT_SECONDS: PositiveInt = 300
    OCR_TIMEOUT_SECONDS: PositiveInt = 120
    AGENT_MAX_OUTPUT_TOKENS: PositiveInt | None = None

    # --- Database (credential-bearing: SecretStr, never logged) ---
    DATABASE_URL: SecretStr = Field(description="PostgreSQL DSN incl. password (never log)")

    # --- Storage ---
    CHROMA_PATH: str = "./chroma"
    KEEP_WORKSPACE_ON_SUCCESS: bool = False
    MAX_CLEANUP_RETRIES: PositiveInt = 3

    # --- Server ---
    BIND_HOST: str = "127.0.0.1"
    PORT: int = Field(default=8000, ge=1, le=65535)

    # --- Pipeline defaults ---
    RENDER_DPI: PositiveInt = 200
    ROLLING_CONTEXT_PAGES: PositiveInt = 3
    COVERAGE_THRESHOLD: int = Field(default=95, ge=0, le=100)
    COVERAGE_FLOOR_TOKENS: int = Field(default=80, ge=0, le=100)
    COVERAGE_FLOOR_MIN_OCR_TOKENS: PositiveInt = 30
    MAX_PAGE_RETRIES: int = Field(default=2, ge=0)
    MAX_PDF_MB: PositiveInt = 500
    MAX_PDF_PAGES: PositiveInt = 1000
    HYBRID_ROUTING: bool = False
    THINKING_EFFORT_TRANSCRIBE: Literal["low", "high"] = "low"
    THINKING_EFFORT_QA: Literal["low", "high"] = "high"
    TOC_ENABLED: bool = True
    FIG_DETAILS_BLOCKS: bool = True


def load_settings() -> Settings:
    """Load settings from .env/environment.

    Raises:
        ValidationError: with a clear message when a required key
            (e.g. OLLAMA_API_KEY, DATABASE_URL) is missing or invalid.
    """
    return Settings()  # type: ignore[call-arg]


__all__ = ["Settings", "load_settings"]
