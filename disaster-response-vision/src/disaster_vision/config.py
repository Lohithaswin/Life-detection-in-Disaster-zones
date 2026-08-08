"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for disaster-response-vision."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Paths
    weights_dir: Path = Field(default=Path("weights"), alias="WEIGHTS_DIR")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    runs_dir: Path = Field(default=Path("runs"), alias="RUNS_DIR")

    # Detection defaults
    default_model: str = Field(default="yolov8n", alias="DEFAULT_MODEL")
    confidence_threshold: float = Field(default=0.25, alias="CONFIDENCE_THRESHOLD")

    # Database (SQLAlchemy — fully wired in Phase 3)
    database_url: str = Field(
        default="sqlite:///./data/disaster_vision.db",
        alias="DATABASE_URL",
    )

    # Email / SMTP
    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    alert_from: str = Field(default="", alias="ALERT_FROM")
    alert_to: str = Field(default="", alias="ALERT_TO")

    # Alert deduplication window in seconds
    alert_dedup_seconds: int = Field(default=300, alias="ALERT_DEDUP_SECONDS")

    # API server
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    def resolve_model_path(self, model_name: str | None = None) -> Path:
        """Return the on-disk path for a model weights file."""
        name = model_name or self.default_model
        if not name.endswith(".pt"):
            name = f"{name}.pt"
        return self.weights_dir / name

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not exist."""
        for directory in (self.weights_dir, self.data_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
