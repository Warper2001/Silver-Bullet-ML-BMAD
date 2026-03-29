"""KuCoin-specific configuration loading from environment variables."""

from datetime import time
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KucoinExchangeSettings(BaseSettings):
    """KuCoin exchange settings loaded from environment variables.

    This configuration is for KuCoin API integration and crypto-specific
    trading parameters. All times are stored in UTC for 24/7 markets.

    KuCoin is US-friendly and supports BTC-USDT trading.

    Attributes:
        kucoin_api_key: KuCoin API key for authentication
        kucoin_api_secret: KuCoin API secret key for request signing
        kucoin_api_passphrase: KuCoin API passphrase (required for all requests)
        kucoin_environment: Environment selection (sandbox/production)
        kucoin_trading_symbols: List of trading symbols (e.g., ["BTC-USDT"])
        daily_reset_time_utc: Daily reset time for risk limits (UTC, default "00:00")
        position_close_time_utc: Time to close all positions (UTC, default "21:00" = 5pm ET)
        allow_weekend_trading: Allow trading on weekends (default true for 24/7 crypto)
        kucoin_dollar_bar_threshold: Notional value threshold for dollar bar aggregation
        kucoin_position_size_multiplier: Position size multiplier for crypto volatility
        atr_percentile_threshold: ATR percentile threshold for volatility filter
        app_env: Application environment (development/staging/production)
        log_level: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """

    model_config = SettingsConfigDict(
        env_file=".env.kucoin",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # KuCoin API Credentials
    kucoin_api_key: str = Field(
        ...,
        alias="KUCOIN_API_KEY",
        description="KuCoin API key for authentication",
        min_length=24,
        max_length=256,
    )
    kucoin_api_secret: str = Field(
        ...,
        alias="KUCOIN_API_SECRET",
        description="KuCoin API secret key for HMAC SHA256 signature",
        min_length=36,
        max_length=256,
    )
    kucoin_api_passphrase: str = Field(
        ...,
        alias="KUCOIN_API_PASSPHRASE",
        description="KuCoin API passphrase (created when generating API key)",
        min_length=1,
        max_length=128,
    )

    # Exchange Environment
    kucoin_environment: str = Field(
        default="sandbox",
        alias="KUCOIN_ENVIRONMENT",
        description="Exchange environment: sandbox or production",
    )

    # Trading Symbols
    kucoin_trading_symbols: list[str] = Field(
        default=["BTC-USDT"],
        alias="KUCOIN_TRADING_SYMBOLS",
        description="List of trading symbols (e.g., BTC-USDT, ETH-USDT)",
    )

    # Time-Based Configuration (All UTC)
    daily_reset_time_utc: str = Field(
        default="00:00",
        alias="DAILY_RESET_TIME_UTC",
        description="Daily reset time for risk limits in UTC (HH:MM format)",
    )
    position_close_time_utc: str = Field(
        default="21:00",
        alias="POSITION_CLOSE_TIME_UTC",
        description="Time to close all positions in UTC (default 21:00 = 5pm ET)",
    )

    # Weekend Trading
    allow_weekend_trading: bool = Field(
        default=True,
        alias="ALLOW_WEEKEND_TRADING",
        description="Allow trading on weekends (crypto markets are 24/7)",
    )

    # Dollar Bar Configuration
    kucoin_dollar_bar_threshold: float = Field(
        default=10_000_000.0,  # $10M notional (vs $50M for futures)
        alias="KUCOIN_DOLLAR_BAR_THRESHOLD",
        description="Notional value threshold for dollar bar aggregation in USD",
        gt=0,
    )

    # Position Sizing
    kucoin_position_size_multiplier: float = Field(
        default=0.3,  # 30% of futures size due to higher volatility
        alias="KUCOIN_POSITION_SIZE_MULTIPLIER",
        description="Position size multiplier for crypto volatility adjustment",
        gt=0,
        le=1.0,
    )

    # Volatility Filter
    atr_percentile_threshold: float = Field(
        default=75.0,  # 75th percentile (from research)
        alias="ATR_PERCENTILE_THRESHOLD",
        description="ATR percentile threshold for volatility filter (0-100)",
        ge=0,
        le=100,
    )

    # App Configuration
    app_env: str = Field(
        default="development",
        alias="APP_ENV",
        description="Application environment",
    )
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level",
    )

    @field_validator("kucoin_environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate exchange environment is sandbox or production."""
        v_lower = v.lower()
        if v_lower not in ["sandbox", "production"]:
            raise ValueError(
                'kucoin_environment must be either "sandbox" or "production"'
            )
        return v_lower

    @field_validator("kucoin_trading_symbols")
    @classmethod
    def validate_trading_symbols(cls, v: list[str]) -> list[str]:
        """Validate trading symbols list is not empty and properly formatted."""
        if not v or len(v) == 0:
            raise ValueError("kucoin_trading_symbols cannot be empty")
        if not all(isinstance(s, str) and len(s) > 0 for s in v):
            raise ValueError("all symbols must be non-empty strings")
        # Validate symbol format (e.g., BTC-USDT, ETH-USDT)
        for symbol in v:
            if not symbol.isupper() or "-" not in symbol or len(symbol) < 7:
                raise ValueError(
                    f'invalid symbol format "{symbol}": '
                    "must be uppercase with dash (e.g., BTC-USDT)"
                )
        return v

    @field_validator("daily_reset_time_utc", "position_close_time_utc")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate time format is HH:MM in 24-hour format."""
        try:
            hour, minute = v.split(":")
            hour_int = int(hour)
            minute_int = int(minute)
            if not (0 <= hour_int <= 23):
                raise ValueError("hour must be between 00 and 23")
            if not (0 <= minute_int <= 59):
                raise ValueError("minute must be between 00 and 59")
            return v
        except ValueError as e:
            raise ValueError(
                f'time must be in HH:MM format (24-hour), got "{v}": {e}'
            ) from e

    @field_validator("kucoin_api_key", "kucoin_api_secret", "kucoin_api_passphrase")
    @classmethod
    def validate_credentials_length(cls, v: str, info) -> str:  # type: ignore[no-untyped-def]
        """Validate credential length meets KuCoin requirements."""
        field_name = str(info.field_name)
        if field_name == "kucoin_api_key" and len(v) < 24:
            raise ValueError(f"{field_name} must be at least 24 characters")
        if field_name == "kucoin_api_secret" and len(v) < 36:
            raise ValueError(f"{field_name} must be at least 36 characters")
        if field_name == "kucoin_api_passphrase" and len(v) < 1:
            raise ValueError(f"{field_name} cannot be empty")
        return v

    @property
    def base_url(self) -> str:
        """Get KuCoin API base URL based on environment.

        Returns:
            Base URL for KuCoin REST API
        """
        if self.kucoin_environment == "sandbox":
            return "https://openapi-sandbox.kucoin.com"
        return "https://api.kucoin.com"

    @property
    def websocket_base_url(self) -> str:
        """Get KuCoin WebSocket base URL based on environment.

        Returns:
            Base URL for KuCoin WebSocket connections
        """
        if self.kucoin_environment == "sandbox":
            return "wss://ws-api-sandbox.kucoin.com"
        return "wss://ws-api.kucoin.com"

    @property
    def daily_reset_time(self) -> time:
        """Parse daily reset time string into time object.

        Returns:
            Time object for daily reset
        """
        hour, minute = map(int, self.daily_reset_time_utc.split(":"))
        return time(hour=hour, minute=minute)

    @property
    def position_close_time(self) -> time:
        """Parse position close time string into time object.

        Returns:
            Time object for position closing
        """
        hour, minute = map(int, self.position_close_time_utc.split(":"))
        return time(hour=hour, minute=minute)


def load_kucoin_settings() -> KucoinExchangeSettings:
    """Load KuCoin exchange settings from environment.

    Returns:
        KucoinExchangeSettings object with all configuration loaded

    Raises:
        ConfigurationError: If required settings are missing or invalid
    """
    try:
        return KucoinExchangeSettings()  # type: ignore[call-arg]
    except Exception as e:
        from .exceptions import ConfigurationError

        raise ConfigurationError(f"Failed to load kucoin settings: {e}") from e
