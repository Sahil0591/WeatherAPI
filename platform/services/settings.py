from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "WeatherAPI"
    APP_VERSION: str = "0.1.0"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Data/model/caching
    ARTIFACTS_DIR: str = "ml_engine/artifacts"
    REDIS_URL: Optional[str] = None
    CACHE_TTL_SECONDS: int = 300
    WEATHER_HOURS_BACK: int = 72

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="APP_",  # e.g. APP_ARTIFACTS_DIR
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()