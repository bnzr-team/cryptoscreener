"""
Data contracts for CryptoScreener-X pipeline.

All contracts follow DATA_CONTRACTS.md specification exactly.
These are the canonical schemas used between modules.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MarketEventType(str, Enum):
    """Type of market event from exchange."""

    TRADE = "trade"
    BOOK = "book"
    KLINE = "kline"
    MARK = "mark"
    OI = "oi"
    FUNDING = "funding"


class MarketEvent(BaseModel):
    """
    Canonical market event from exchange.

    Attributes:
        ts: Event timestamp from exchange (milliseconds).
        source: Data source identifier (e.g., "binance_usdm").
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        type: Type of market event.
        payload: Event-specific data.
        recv_ts: Local receive timestamp (milliseconds) for latency metrics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: int = Field(..., ge=0, description="Event timestamp from exchange (ms)")
    source: str = Field(..., min_length=1, description="Data source identifier")
    symbol: str = Field(..., min_length=1, description="Trading pair symbol")
    type: MarketEventType = Field(..., description="Type of market event")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event-specific data")
    recv_ts: int = Field(..., ge=0, description="Local receive timestamp (ms)")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    def to_json(self) -> bytes:
        """Serialize to JSON bytes using orjson."""
        return orjson.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: bytes | str) -> MarketEvent:
        """Deserialize from JSON."""
        if isinstance(data, str):
            data = data.encode()
        return cls.model_validate(orjson.loads(data))


class RegimeVol(str, Enum):
    """Volatility regime classification."""

    LOW = "low"
    HIGH = "high"


class RegimeTrend(str, Enum):
    """Trend regime classification."""

    TREND = "trend"
    CHOP = "chop"


class Features(BaseModel):
    """Core feature set for a symbol snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spread_bps: float = Field(..., description="Spread in basis points")
    mid: float = Field(..., gt=0, description="Mid price")
    book_imbalance: float = Field(..., ge=-1, le=1, description="Order book imbalance [-1, 1]")
    flow_imbalance: float = Field(..., ge=-1, le=1, description="Trade flow imbalance [-1, 1]")
    natr_14_5m: float = Field(..., ge=0, description="Normalized ATR (14-period, 5m)")
    impact_bps_q: float = Field(..., ge=0, description="Price impact in bps for reference qty")
    regime_vol: RegimeVol = Field(..., description="Volatility regime")
    regime_trend: RegimeTrend = Field(..., description="Trend regime")


class Windows(BaseModel):
    """Rolling window aggregates at different timescales."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    w1s: dict[str, Any] = Field(default_factory=dict, description="1-second window features")
    w10s: dict[str, Any] = Field(default_factory=dict, description="10-second window features")
    w60s: dict[str, Any] = Field(default_factory=dict, description="60-second window features")


class DataHealth(BaseModel):
    """Data health metrics for staleness detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stale_book_ms: int = Field(default=0, ge=0, description="Milliseconds since last book update")
    stale_trades_ms: int = Field(
        default=0, ge=0, description="Milliseconds since last trade update"
    )
    missing_streams: list[str] = Field(
        default_factory=list, description="List of missing data streams"
    )


class FeatureSnapshot(BaseModel):
    """
    Feature snapshot for a symbol at a point in time.

    Published by feature engine at fixed cadence (e.g., every 250ms).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: int = Field(..., ge=0, description="Snapshot timestamp (ms)")
    symbol: str = Field(..., min_length=1, description="Trading pair symbol")
    features: Features = Field(..., description="Core feature values")
    windows: Windows = Field(default_factory=Windows, description="Rolling window aggregates")
    data_health: DataHealth = Field(default_factory=DataHealth, description="Data health metrics")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    def to_json(self) -> bytes:
        """Serialize to JSON bytes using orjson."""
        return orjson.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: bytes | str) -> FeatureSnapshot:
        """Deserialize from JSON."""
        if isinstance(data, str):
            data = data.encode()
        return cls.model_validate(orjson.loads(data))


class ReasonCode(BaseModel):
    """
    Reason code explaining a prediction factor.

    Used for SHAP/feature attribution explanations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., min_length=1, description="Reason code identifier (e.g., RC_FLOW_SURGE)")
    value: float = Field(..., description="Numeric value (e.g., z-score)")
    unit: str = Field(..., description="Unit of measurement (e.g., 'z', 'bps', '%')")
    evidence: str = Field(..., description="Human-readable evidence string")


class PredictionStatus(str, Enum):
    """Trading status classification."""

    TRADEABLE = "TRADEABLE"
    WATCH = "WATCH"
    TRAP = "TRAP"
    DEAD = "DEAD"
    DATA_ISSUE = "DATA_ISSUE"


class ExecutionProfile(str, Enum):
    """Execution profile type."""

    A = "A"
    B = "B"
    COMBINED = "COMBINED"


class PredictionSnapshot(BaseModel):
    """
    Prediction snapshot from ML inference.

    Contains probabilities, expected utility, and reasons for the prediction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: int = Field(..., ge=0, description="Prediction timestamp (ms)")
    symbol: str = Field(..., min_length=1, description="Trading pair symbol")
    profile: ExecutionProfile = Field(..., description="Execution profile used")
    p_inplay_30s: float = Field(..., ge=0, le=1, description="P(in-play) at 30s horizon")
    p_inplay_2m: float = Field(..., ge=0, le=1, description="P(in-play) at 2m horizon")
    p_inplay_5m: float = Field(..., ge=0, le=1, description="P(in-play) at 5m horizon")
    expected_utility_bps_2m: float = Field(..., description="Expected utility in bps at 2m")
    p_toxic: float = Field(..., ge=0, le=1, description="Probability of toxic flow")
    status: PredictionStatus = Field(..., description="Trading status classification")
    reasons: list[ReasonCode] = Field(default_factory=list, description="Explanation reasons")
    model_version: str = Field(..., min_length=1, description="Model version (semver+gitsha)")
    calibration_version: str = Field(..., min_length=1, description="Calibration version")
    data_health: DataHealth = Field(default_factory=DataHealth, description="Data health metrics")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    def to_json(self) -> bytes:
        """Serialize to JSON bytes using orjson."""
        return orjson.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: bytes | str) -> PredictionSnapshot:
        """Deserialize from JSON."""
        if isinstance(data, str):
            data = data.encode()
        return cls.model_validate(orjson.loads(data))


class RankEventType(str, Enum):
    """Type of ranking event."""

    SYMBOL_ENTER = "SYMBOL_ENTER"
    SYMBOL_EXIT = "SYMBOL_EXIT"
    ALERT_TRADABLE = "ALERT_TRADABLE"
    ALERT_TRAP = "ALERT_TRAP"
    DATA_ISSUE = "DATA_ISSUE"


class RankEventPayload(BaseModel):
    """Payload for a rank event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction: dict[str, Any] = Field(
        default_factory=dict, description="PredictionSnapshot as dict"
    )
    llm_text: str = Field(default="", description="LLM-generated explanation text")


class RankEvent(BaseModel):
    """
    Ranking event emitted by ranker module.

    Represents state transitions in the top-K ranking.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: int = Field(..., ge=0, description="Event timestamp (ms)")
    event: RankEventType = Field(..., description="Type of ranking event")
    symbol: str = Field(..., min_length=1, description="Trading pair symbol")
    rank: int = Field(..., ge=0, description="Current rank (0-indexed)")
    score: float = Field(..., ge=0, le=1, description="Ranking score [0, 1]")
    payload: RankEventPayload = Field(default_factory=RankEventPayload, description="Event payload")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    def to_json(self) -> bytes:
        """Serialize to JSON bytes using orjson."""
        return orjson.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: bytes | str) -> RankEvent:
        """Deserialize from JSON."""
        if isinstance(data, str):
            data = data.encode()
        return cls.model_validate(orjson.loads(data))

    def digest(self) -> str:
        """
        Compute deterministic digest of this event for replay verification.

        Returns:
            SHA256 hex digest of canonical JSON representation.
        """
        canonical = orjson.dumps(
            self.model_dump(mode="json"),
            option=orjson.OPT_SORT_KEYS,
        )
        return hashlib.sha256(canonical).hexdigest()


# LLM Explain Contract (strict)


class NumericSummary(BaseModel):
    """Numeric summary passed to LLM (read-only for LLM)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spread_bps: float = Field(..., description="Spread in basis points")
    impact_bps: float = Field(..., description="Price impact in bps")
    p_toxic: float = Field(..., ge=0, le=1, description="Toxicity probability")
    regime: str = Field(..., description="Regime description (e.g., 'high-vol trend')")


class LLMStyle(BaseModel):
    """Style configuration for LLM output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tone: Literal["friendly", "neutral", "technical"] = Field(
        default="friendly", description="Output tone"
    )
    max_chars: int = Field(default=180, ge=50, le=500, description="Max characters for headline")


class LLMExplainInput(BaseModel):
    """
    Input contract for LLM explain module.

    LLM MUST NOT output numbers different from inputs.
    It can only rephrase or omit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(..., min_length=1, description="Trading pair symbol")
    timeframe: str = Field(..., description="Prediction timeframe (e.g., '2m')")
    status: PredictionStatus = Field(..., description="Prediction status")
    score: float = Field(..., ge=0, le=1, description="Ranking score")
    reasons: list[ReasonCode] = Field(default_factory=list, description="Reason codes")
    numeric_summary: NumericSummary = Field(..., description="Numeric summary (read-only for LLM)")
    style: LLMStyle = Field(default_factory=LLMStyle, description="Style configuration")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    def to_json(self) -> bytes:
        """Serialize to JSON bytes using orjson."""
        return orjson.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: bytes | str) -> LLMExplainInput:
        """Deserialize from JSON."""
        if isinstance(data, str):
            data = data.encode()
        return cls.model_validate(orjson.loads(data))


class LLMExplainOutput(BaseModel):
    """
    Output contract for LLM explain module.

    LLM MUST NOT output numbers different from inputs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    headline: str = Field(..., min_length=1, max_length=500, description="Main headline")
    subtext: str = Field(default="", max_length=500, description="Secondary explanation")
    status_label: str = Field(..., min_length=1, max_length=50, description="UI status label")
    tooltips: dict[str, str] = Field(default_factory=dict, description="Field tooltips")

    def to_json(self) -> bytes:
        """Serialize to JSON bytes using orjson."""
        return orjson.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: bytes | str) -> LLMExplainOutput:
        """Deserialize from JSON."""
        if isinstance(data, str):
            data = data.encode()
        return cls.model_validate(orjson.loads(data))


def compute_rank_events_digest(events: list[RankEvent]) -> str:
    """
    Compute deterministic digest of a sequence of RankEvents.

    Used for replay determinism verification.

    Args:
        events: List of RankEvent objects.

    Returns:
        SHA256 hex digest of the concatenated canonical representations.
    """
    hasher = hashlib.sha256()
    for event in events:
        canonical = orjson.dumps(
            event.model_dump(mode="json"),
            option=orjson.OPT_SORT_KEYS,
        )
        hasher.update(canonical)
    return hasher.hexdigest()
