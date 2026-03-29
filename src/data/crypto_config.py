"""Crypto-specific configuration loading from environment variables."""

from datetime import time
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CryptoExchangeSettings(BaseSettings):
    """Crypto exchange settings loaded from environment variables.

    This configuration is for Binance API integration and crypto-specific
    trading parameters. All times are stored in UTC for 24/7 markets.

    Attributes:
        crypto_exchange_api_key: Binance API key for authentication
        crypto_exchange_api_secret: Binance API secret key for request signing
        crypto_exchange_environment: Environment selection (testnet/production)
        crypto_trading_symbols: List of trading symbols (e.g., ["BTCUSDT"])
        daily_reset_time_utc: Daily reset time for risk limits (UTC, default "00:00")
        position_close_time_utc: Time to close all positions (UTC, default "21:00" = 5pm ET)
        allow_weekend_trading: Allow trading on weekends (default true for 24/7 crypto)
        crypto_dollar_bar_threshold: Notional value threshold for dollar bar aggregation
        crypto_position_size_multiplier: Position size multiplier for crypto volatility
        atr_percentile_threshold: ATR percentile threshold for volatility filter
        app_env: Application environment (development/staging/production)
        log_level: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """

    model_config = SettingsConfigDict(
        env_file=".env.crypto",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Binance API Credentials
    crypto_exchange_api_key: str = Field(
        ...,
        alias="CRYPTO_EXCHANGE_API_KEY",
        description="Binance API key for authentication",
        min_length=64,
        max_length=256,
    )
    crypto_exchange_api_secret: str = Field(
        ...,
        alias="CRYPTO_EXCHANGE_API_SECRET",
        description="Binance API secret key for HMAC SHA256 signature",
        min_length=64,
        max_length=256,
    )

    # Exchange Environment
    crypto_exchange_environment: str = Field(
        default="testnet",
        alias="CRYPTO_EXCHANGE_ENVIRONMENT",
        description="Exchange environment: testnet or production",
    )

    # Trading Symbols
    crypto_trading_symbols: list[str] = Field(
        default=["BTCUSDT"],
        alias="CRYPTO_TRADING_SYMBOLS",
        description="List of trading symbols (e.g., BTCUSDT, ETHUSDT)",
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
    crypto_dollar_bar_threshold: float = Field(
        default=10_000_000.0,  # $10M notional (vs $50M for futures)
        alias="CRYPTO_DOLLAR_BAR_THRESHOLD",
        description="Notional value threshold for dollar bar aggregation in USD",
        gt=0,
    )

    # Position Sizing
    crypto_position_size_multiplier: float = Field(
        default=0.3,  # 30% of futures size due to higher volatility
        alias="CRYPTO_POSITION_SIZE_MULTIPLIER",
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

    @field_validator("crypto_exchange_environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate exchange environment is testnet or production."""
        v_lower = v.lower()
        if v_lower not in ["testnet", "production"]:
            raise ValueError(
                'crypto_exchange_environment must be either "testnet" or "production"'
            )
        return v_lower

    @field_validator("crypto_trading_symbols")
    @classmethod
    def validate_trading_symbols(cls, v: list[str]) -> list[str]:
        """Validate trading symbols list is not empty and properly formatted."""
        if not v or len(v) == 0:
            raise ValueError("crypto_trading_symbols cannot be empty")
        if not all(isinstance(s, str) and len(s) > 0 for s in v):
            raise ValueError("all symbols must be non-empty strings")
        # Validate symbol format (e.g., BTCUSDT, ETHUSDT)
        for symbol in v:
            if not symbol.isupper() or len(symbol) < 6:
                raise ValueError(
                    f'invalid symbol format "{symbol}": '
                    "must be uppercase and at least 6 characters (e.g., BTCUSDT)"
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

    @field_validator("crypto_exchange_api_key", "crypto_exchange_api_secret")
    @classmethod
    def validate_credentials_length(cls, v: str, info) -> str:  # type: ignore[no-untyped-def]
        """Validate credential length meets Binance requirements."""
        if len(v) < 64:
            raise ValueError(
                f"{info.field_name} must be at least 64 characters (Binance requirement)"
            )
        return v

    @property
    def base_url(self) -> str:
        """Get Binance API base URL based on environment.

        Returns:
            Base URL for Binance REST API
        """
        if self.crypto_exchange_environment == "testnet":
            return "https://testnet.binance.vision"
        return "https://api.binance.com"

    @property
    def websocket_base_url(self) -> str:
        """Get Binance WebSocket base URL based on environment.

        Returns:
            Base URL for Binance WebSocket connections
        """
        if self.crypto_exchange_environment == "testnet":
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"

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


def load_crypto_settings() -> CryptoExchangeSettings:
    """Load crypto exchange settings from environment.

    Returns:
        CryptoExchangeSettings object with all configuration loaded

    Raises:
        ConfigurationError: If required settings are missing or invalid
    """
    try:
        return CryptoExchangeSettings()  # type: ignore[call-arg]
    except Exception as e:
        from .exceptions import ConfigurationError

        raise ConfigurationError(f"Failed to load crypto settings: {e}") from e
