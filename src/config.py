"""Application configuration (SRS Appendix B coverage).

All settings load from `.env`/environment via pydantic-settings.
Secrets (OLLAMA_API_KEY, DATABASE_URL) are SecretStr — never logged,
never in repr (NFR-5). Every key is typed and validated; every
Appendix B key is present with its spec default.

The primary capability is diagram→Mermaid reinterpretation; its knobs
(`DIAGRAM_*`) are config-gated and on by default.
"""

from typing import Literal

from pydantic import Field, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Conservative default allowlist: the Mermaid diagram types a PDF figure
# realistically maps to and that GitHub renders. Includes the newer
# GitHub-supported beta types (xychart/sankey/architecture/radar/kanban).
DEFAULT_DIAGRAM_TYPES = (
    "flowchart",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "gantt",
    "mindmap",
    "timeline",
    "journey",
    "pie",
    "gitGraph",
    "quadrantChart",
    "xychart-beta",
    "sankey-beta",
    "architecture-beta",
    "radar-beta",
    "kanban",
)


class Settings(BaseSettings):
    """All SRS Appendix B keys with spec defaults and validation."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Ollama (two separate endpoints) ---
    OLLAMA_LOCAL_URL: str = "http://localhost:11434"
    OLLAMA_CLOUD_URL: str = "https://ollama.com"
    OLLAMA_API_KEY: SecretStr = Field(description="Ollama Cloud API key (never log)")
    AGENT_MODEL: str = "glm-5.3-flash"
    OCR_MODEL: str = "glm-ocr"
    AGENT_TIMEOUT_SECONDS: PositiveInt = 300
    OCR_TIMEOUT_SECONDS: PositiveInt = 120
    AGENT_MAX_OUTPUT_TOKENS: PositiveInt | None = None

    # --- Database (credential-bearing: SecretStr, never logged) ---
    DATABASE_URL: SecretStr = Field(description="PostgreSQL DSN incl. password (never log)")

    # --- Storage ---
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
    # Per-page reference routing: a page with a substantial native text layer
    # uses it as the character reference (exact, zero model calls); OCR runs
    # only for scanned/sparse pages and as the retry escalation.
    NATIVE_TEXT_FIRST: bool = True
    NATIVE_TEXT_MIN_WORDS: PositiveInt = 20
    # Master switch for the local OCR stage. False = never call glm-ocr:
    # scanned/sparse pages transcribe vision-only (empty reference), native
    # pages still use their text layer. Recorded in the fingerprint so the
    # two arms of an ablation can never be confused.
    OCR_ENABLED: bool = True
    THINKING_EFFORT_TRANSCRIBE: Literal["low", "high"] = "low"
    THINKING_EFFORT_DIAGRAM: Literal["low", "high"] = "high"
    TOC_ENABLED: bool = True
    FIG_DETAILS_BLOCKS: bool = True

    # --- Diagram -> Mermaid (primary capability) ---
    DIAGRAM_TO_MERMAID: bool = True
    # Comma-separated Mermaid type allowlist (see DEFAULT_DIAGRAM_TYPES).
    DIAGRAM_ALLOWED_TYPES: str = ",".join(DEFAULT_DIAGRAM_TYPES)
    # Minimum verifier confidence (0-100) for a Mermaid reinterpretation.
    DIAGRAM_MIN_CONFIDENCE: int = Field(default=80, ge=0, le=100)
    DIAGRAM_VERIFY: bool = True
    # Fallback when a figure is not (or cannot be) converted:
    #   "image" -> image + alt/caption
    #   "table" -> OCR-grounded data table + image
    #   "both"  -> data table for charts, image for everything else
    DIAGRAM_FALLBACK: Literal["image", "table", "both"] = "both"
    # Keep the original figure collapsibly beneath a converted Mermaid.
    DIAGRAM_KEEP_IMAGE: bool = True

    def allowed_diagram_types(self) -> frozenset[str]:
        """Parsed Mermaid allowlist from the comma-separated setting."""
        return frozenset(
            item.strip() for item in self.DIAGRAM_ALLOWED_TYPES.split(",") if item.strip()
        )


def load_settings() -> Settings:
    """Load settings from .env/environment.

    Raises:
        ValidationError: with a clear message when a required key
            (e.g. OLLAMA_API_KEY, DATABASE_URL) is missing or invalid.
    """
    return Settings()  # type: ignore[call-arg]


__all__ = ["DEFAULT_DIAGRAM_TYPES", "Settings", "load_settings"]
