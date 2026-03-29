"""Configuration loading from environment variables."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # TradeStation API Credentials
    tradestation_client_id: str = Field(..., alias="TRADESTATION_CLIENT_ID")
    tradestation_client_secret: str = Field(..., alias="TRADESTATION_CLIENT_SECRET")
    tradestation_redirect_uri: str = Field(
        default="http://localhost:8080/callback",
        alias="TRADESTATION_REDIRECT_URI",
    )
    tradestation_refresh_token: str = Field(
        default="",
        alias="TRADESTATION_REFRESH_TOKEN",
    )

    # App Configuration
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Trading Configuration
    streaming_symbols: list[str] = Field(
        default=["MNQH26"],
        alias="STREAMING_SYMBOLS",
    )

    @field_validator("streaming_symbols")
    @classmethod
    def validate_streaming_symbols(cls, v: list[str]) -> list[str]:
        """Validate streaming symbols list is not empty."""
        if not v or len(v) == 0:
            raise ValueError("streaming_symbols cannot be empty")
        if not all(isinstance(s, str) and len(s) > 0 for s in v):
            raise ValueError("all symbols must be non-empty strings")
        return v


def load_settings() -> Settings:
    """Load application settings from environment.

    Returns:
        Settings object with all configuration loaded

    Raises:
        ConfigurationError: If required settings are missing
    """
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as e:
        from .exceptions import ConfigurationError

        raise ConfigurationError(f"Failed to load settings: {e}") from e
