"""
Base sink protocol (DEC-039).

Abstract base for all delivery sinks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cryptoscreener.delivery.formatter import FormattedMessage


@dataclass
class DeliveryResult:
    """Result of a delivery attempt."""

    success: bool
    sink_name: str
    error: str | None = None
    status_code: int | None = None
    retry_after_s: float | None = None  # For rate limit responses


class DeliverySink(ABC):
    """Abstract base class for delivery sinks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this sink."""
        ...

    @property
    @abstractmethod
    def sink_type(self) -> Literal["telegram", "slack", "webhook"]:
        """Type of this sink."""
        ...

    @abstractmethod
    async def send(self, message: FormattedMessage) -> DeliveryResult:
        """
        Send a formatted message to this sink.

        Args:
            message: Formatted message with text/html/markdown variants

        Returns:
            DeliveryResult indicating success or failure
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close any resources held by this sink."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
