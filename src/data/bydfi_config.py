"""
BYDFI Configuration Loader

Loads configuration settings from environment variables for BYDFI API integration.
Based on BYDFI API documentation: https://developers.bydfi.com/en/public
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_bydfi_settings() -> "BYDFISettings":
    """
    Load BYDFI settings from environment variables.

    Environment variables are loaded from .env.bydfi file.

    Returns:
        BYDFISettings: Validated settings object

    Raises:
        ValidationError: If required settings are missing or invalid
    """
    return BYDFISettings()


class BYDFISettings(BaseSettings):
    """
    BYDFI configuration settings.

    Attributes:
        bydfi_api_key: BYDFI API key for authentication
        bydfi_api_secret: BYDFI API secret key for signature generation
        bydfi_trading_symbol: Trading symbol (e.g., BTC-USDT)
        bydfi_environment: Trading environment (production or testnet)
        base_url: Base URL for BYDFI API
        websocket_url: WebSocket URL for real-time data
    """

    model_config = SettingsConfigDict(
        env_file=".env.bydfi",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Credentials
    bydfi_api_key: str = Field(
        ...,
        description="BYDFI API key from API Management",
        min_length=10,
    )

    bydfi_api_secret: str = Field(
        ...,
        description="BYDFI API secret key for signature generation",
        min_length=10,
    )

    # Trading Configuration
    bydfi_trading_symbol: str = Field(
        default="BTC-USDT",
        description="Trading symbol (BYDFI format: BTC-USDT)",
        pattern=r"^[A-Z]+-[A-Z]+$",
    )

    bydfi_environment: Literal["production", "testnet"] = Field(
        default="production",
        description="Trading environment",
    )

    # API Endpoints (from https://developers.bydfi.com/en/domainName)
    @property
    def base_url(self) -> str:
        """Get base URL based on environment."""
        urls = {
            "production": "https://api.bydfi.com/api",
            "testnet": "https://api.bydfi.com/api",  # BYDFI uses same domain
        }
        return urls[self.bydfi_environment]

    @property
    def websocket_url(self) -> str:
        """Get WebSocket URL based on environment."""
        urls = {
            "production": "wss://stream.bydfi.com/v1/public/fapi",
            "testnet": "wss://stream.bydfi.com/v1/public/fapi",
        }
        return urls[self.bydfi_environment]

    # Risk Management Configuration
    daily_reset_time_utc: str = Field(
        default="00:00",
        description="Daily reset time for risk limits (UTC, HH:MM format)",
    )

    position_close_time_utc: str = Field(
        default="21:00",
        description="Position closing time (UTC, HH:MM format)",
    )

    allow_weekend_trading: bool = Field(
        default=True,
        description="Allow trading on weekends (crypto markets are 24/7)",
    )

    # Dollar Bar Configuration
    crypto_dollar_bar_threshold: int = Field(
        default=10000000,
        description="Dollar bar threshold for crypto (USD notional value)",
    )

    # Position Sizing
    crypto_position_size_multiplier: float = Field(
        default=0.3,
        description="Position size multiplier for crypto volatility (0.3 = 30% of futures)",
        ge=0.0,
        le=1.0,
    )

    # Volatility Filter
    atr_percentile_threshold: float = Field(
        default=75.0,
        description="ATR percentile threshold for volatility filter (0-100)",
        ge=0.0,
        le=100.0,
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )


@lru_cache
def get_bydfi_settings() -> BYDFISettings:
    """
    Cached getter for BYDFI settings.

    Returns:
        BYDFISettings: Cached settings object
    """
    return load_bydfi_settings()
