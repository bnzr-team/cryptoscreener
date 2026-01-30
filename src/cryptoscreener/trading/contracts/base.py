"""Base configuration for trading contracts.

All trading contracts inherit from TradingContractBase which enforces:
- schema_version and session_id are required
- Extra fields are forbidden
- Decimal serialization as strings
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Schema version for all v2 trading contracts
SCHEMA_VERSION = "1.0.0"


def decimal_serializer(value: Decimal) -> str:
    """Serialize Decimal to string for JSON."""
    return str(value)


# Type alias for Decimal fields that accept string input
DecimalStr = Annotated[
    Decimal,
    Field(description="Decimal value serialized as string"),
]


class TradingContractBase(BaseModel):
    """Base class for all trading contracts.

    Enforces:
    - schema_version and session_id required
    - Extra fields forbidden
    - Decimal serialized as strings
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        ser_json_bytes="base64",
    )

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Contract schema version",
    )
    session_id: str = Field(
        description="Trading session identifier (UUID or date-based)",
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, v: Any) -> str:
        """Ensure schema_version is provided."""
        if v is None:
            raise ValueError("schema_version is required")
        return str(v)

    @field_validator("session_id", mode="before")
    @classmethod
    def validate_session_id(cls, v: Any) -> str:
        """Ensure session_id is provided."""
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("session_id is required and cannot be empty")
        return str(v)


def parse_decimal(v: Any) -> Decimal:
    """Parse value to Decimal safely.

    Accepts:
    - Decimal (passthrough)
    - str (parsed to Decimal)
    - int (converted via string to avoid precision loss)
    - float (NOT recommended, but converted via string)
    """
    if isinstance(v, Decimal):
        return v
    if isinstance(v, str):
        return Decimal(v)
    if isinstance(v, int):
        return Decimal(str(v))
    if isinstance(v, float):
        # Convert via string to preserve representation
        return Decimal(str(v))
    raise ValueError(f"Cannot convert {type(v).__name__} to Decimal")
