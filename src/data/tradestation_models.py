"""Pydantic models for TradeStation API responses.

This module defines data models for TradeStation OAuth and market data API responses.
All models follow Pydantic v2 patterns with field validators.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class OAuthTokenResponse(BaseModel):
    """OAuth 2.0 token response from TradeStation.

    Returned when exchanging authorization code or refresh token for access token.
    """

    access_token: str = Field(..., description="Bearer access token for API requests")
    refresh_token: str = Field(..., description="Long-lived token for obtaining new access tokens")
    expires_in: int = Field(..., gt=0, description="Token lifetime in seconds")
    token_type: str = Field(default="Bearer", description="Token type (always Bearer)")
    scope: str = Field(default="", description="Granted OAuth scopes")
    expires_at: Optional[datetime] = Field(default=None, description="Calculated expiration datetime")

    @field_validator("expires_at", mode="before")
    @classmethod
    def calculate_expires_at(
        cls, v: None | datetime, info: ValidationInfo
    ) -> datetime:  # type: ignore[no-untyped-def]
        """Calculate token expiration datetime from expires_in seconds."""
        if v is not None:
            return v

        expires_in = info.data.get("expires_in")
        if expires_in is None:
            raise ValueError("expires_in required to calculate expires_at")

        # Calculate expiration with 60-second buffer
        return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))

    @property
    def is_valid(self) -> bool:
        """Check if access token is currently valid.

        Returns:
            True if token has not expired with buffer, False otherwise
        """
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) < (self.expires_at)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of token response
        """
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_in": self.expires_in,
            "token_type": self.token_type,
            "scope": self.scope,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class TokenCache(BaseModel):
    """Cached token data persisted to disk."""

    access_token: str = Field(..., description="Cached access token")
    refresh_token: str = Field(..., description="Cached refresh token")
    expires_at: datetime = Field(..., description="Token expiration datetime (UTC)")
    cached_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When token was cached (UTC)"
    )

    @field_validator("expires_at")
    @classmethod
    def expires_at_must_be_future(
        cls, v: datetime, info: ValidationInfo
    ) -> datetime:  # type: ignore[no-untyped-def]
        """Validate expires_at is in the future."""
        if v < datetime.now(timezone.utc):
            raise ValueError("expires_at must be in the future")
        return v

    @property
    def is_valid(self) -> bool:
        """Check if cached token is still valid.

        Returns:
            True if token not expired with 60-second buffer, False otherwise
        """
        buffer_seconds = 60
        return datetime.now(timezone.utc) < (self.expires_at)

    def to_token_response(self) -> OAuthTokenResponse:
        """Convert cache to OAuthTokenResponse.

        Returns:
            OAuthTokenResponse instance
        """
        expires_in = int((self.expires_at - datetime.now(timezone.utc)).total_seconds())
        return OAuthTokenResponse(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            expires_in=expires_in,
            expires_at=self.expires_at,
        )


class RefreshTokenRequest(BaseModel):
    """Request body for refresh token grant."""

    grant_type: str = Field(default="refresh_token", description="OAuth grant type")
    refresh_token: str = Field(..., description="Refresh token")

    def to_form_data(self) -> dict:
        """Convert to form-encoded data for HTTP request.

        Returns:
            Dictionary for form-encoded POST body
        """
        return {
            "grant_type": self.grant_type,
            "refresh_token": self.refresh_token,
        }


class BarData(BaseModel):
    """Historical bar data from TradeStation API.

    Represents OHLCV data for a single time interval.
    """

    symbol: str = Field(..., description="Trading symbol (e.g., 'MNQH26')")
    timestamp: datetime = Field(..., description="Bar timestamp (timezone-aware)")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: int = Field(..., ge=0, description="Trade volume")

    @field_validator("high")
    @classmethod
    def high_gte_low(
        cls, v: float, info: ValidationInfo
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate high price is >= low price."""
        low_val = info.data.get("low")
        if low_val is not None and v < low_val:
            raise ValueError("high must be >= low")
        return v

    @field_validator("close")
    @classmethod
    def close_within_high_low(
        cls, v: float, info: ValidationInfo
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate close is within high and low."""
        high_val = info.data.get("high")
        low_val = info.data.get("low")

        if high_val is not None and v > high_val:
            raise ValueError("close must be <= high")
        if low_val is not None and v < low_val:
            raise ValueError("close must be >= low")
        return v

    def to_dollar_bar(self, contract_multiplier: float = 0.5) -> "DollarBar":
        """Convert to DollarBar model for storage.

        Args:
            contract_multiplier: Futures contract multiplier (MNQ = 0.5)

        Returns:
            DollarBar instance with notional value calculated
        """
        # Import here to avoid circular dependency
        from .models import DollarBar

        # Calculate notional value: close * volume * multiplier
        notional_value = self.close * self.volume * contract_multiplier

        return DollarBar(
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            notional_value=notional_value,
            is_forward_filled=False,  # Historical data is not forward-filled
        )
