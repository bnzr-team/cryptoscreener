"""Trading v2 contract types and enums.

All enums are strict string enums matching docs/trading/03_CONTRACTS.md.
"""

from enum import Enum


class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_MARKET = "STOP_MARKET"


class TimeInForce(str, Enum):
    """Time in force."""

    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Post Only (maker only)


class OrderPriority(str, Enum):
    """Order priority for OrderGovernor."""

    NORMAL = "NORMAL"
    KILL = "KILL"  # Emergency close, bypasses rate limits
    EMERGENCY = "EMERGENCY"  # Highest priority


class OrderStatus(str, Enum):
    """Order status from exchange."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(str, Enum):
    """Position side."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"
    BOTH = "BOTH"  # For fills in one-way mode


class MarginType(str, Enum):
    """Margin type."""

    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


class SessionStateEnum(str, Enum):
    """Trading session state machine states."""

    INITIALIZING = "INITIALIZING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    KILLED = "KILLED"


class BreachType(str, Enum):
    """Risk breach types."""

    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    POSITION_SIZE_LIMIT = "POSITION_SIZE_LIMIT"
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"
    ORDER_RATE_LIMIT = "ORDER_RATE_LIMIT"
    CONSECUTIVE_LOSS = "CONSECUTIVE_LOSS"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"


class BreachSeverity(str, Enum):
    """Risk breach severity."""

    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"


class RiskAction(str, Enum):
    """Automated action taken on risk breach."""

    NONE = "NONE"
    PAUSE_SYMBOL = "PAUSE_SYMBOL"
    PAUSE_SESSION = "PAUSE_SESSION"
    KILL_SESSION = "KILL_SESSION"
    FLATTEN_ALL = "FLATTEN_ALL"
