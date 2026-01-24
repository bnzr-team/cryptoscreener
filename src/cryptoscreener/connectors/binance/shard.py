"""
WebSocket shard - single connection handler for Binance streams.

Per BINANCE_LIMITS.md:
- Max 1024 streams per connection (we use 800)
- Max 10 incoming messages per second per connection
- Exponential backoff + jitter for reconnects
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

import aiohttp

from cryptoscreener.connectors.backoff import (
    BackoffConfig,
    BackoffState,
    CircuitBreaker,
    compute_backoff_delay,
)
from cryptoscreener.connectors.binance.types import (
    ConnectionState,
    ConnectorConfig,
    RawMessage,
    ShardConfig,
    ShardMetrics,
    StreamSubscription,
)

logger = logging.getLogger(__name__)

# Type alias for message callback
MessageCallback = Callable[[RawMessage], Coroutine[Any, Any, None]]


class WebSocketShard:
    """
    A single WebSocket connection handling a subset of streams.

    Responsible for:
    - Connection lifecycle management
    - Subscription management
    - Reconnection with exponential backoff
    - Message dispatch to callback
    """

    def __init__(
        self,
        shard_id: int,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        on_message: MessageCallback,
        on_state_change: Callable[[int, ConnectionState], None] | None = None,
    ) -> None:
        """
        Initialize the WebSocket shard.

        Args:
            shard_id: Unique identifier for this shard.
            config: Connector configuration.
            circuit_breaker: Shared circuit breaker for protection.
            on_message: Async callback for received messages.
            on_state_change: Optional callback for state changes.
        """
        self._shard_id = shard_id
        self._config = config
        self._shard_config: ShardConfig = config.shard_config
        self._circuit_breaker = circuit_breaker
        self._on_message = on_message
        self._on_state_change = on_state_change

        self._subscriptions: set[StreamSubscription] = set()
        self._active_streams: set[str] = set()
        self._state = ConnectionState.DISCONNECTED

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._ping_task: asyncio.Task[None] | None = None

        self._backoff_config = BackoffConfig()
        self._backoff_state = BackoffState()

        self._metrics = ShardMetrics(shard_id=shard_id)
        self._last_pong_ts: int = 0
        self._request_id: int = 0
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def shard_id(self) -> int:
        """Get shard ID."""
        return self._shard_id

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def stream_count(self) -> int:
        """Get number of subscribed streams."""
        return len(self._active_streams)

    @property
    def can_add_streams(self) -> bool:
        """Check if shard can accept more streams."""
        return len(self._active_streams) < self._shard_config.max_streams

    @property
    def available_capacity(self) -> int:
        """Get number of streams that can be added."""
        return self._shard_config.max_streams - len(self._active_streams)

    def get_metrics(self) -> ShardMetrics:
        """Get current shard metrics."""
        self._metrics.stream_count = len(self._active_streams)
        self._metrics.state = self._state
        return self._metrics

    def _set_state(self, state: ConnectionState) -> None:
        """Update connection state and notify callback."""
        if self._state != state:
            old_state = self._state
            self._state = state
            self._metrics.state = state
            logger.debug(
                "Shard state changed",
                extra={
                    "shard_id": self._shard_id,
                    "old_state": old_state.value,
                    "new_state": state.value,
                },
            )
            if self._on_state_change:
                self._on_state_change(self._shard_id, state)

    def _next_request_id(self) -> int:
        """Generate next request ID for WS commands."""
        self._request_id += 1
        return self._request_id

    def _build_ws_url(self) -> str:
        """Build WebSocket URL with combined streams."""
        if not self._active_streams:
            return f"{self._config.base_ws_url}/ws"

        streams = "/".join(sorted(self._active_streams))
        return f"{self._config.base_ws_url}/stream?streams={streams}"

    async def connect(self) -> None:
        """
        Establish WebSocket connection.

        Raises:
            Exception: If circuit breaker is open or connection fails.
        """
        if not self._circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, cannot connect",
                extra={"shard_id": self._shard_id},
            )
            raise ConnectionError("Circuit breaker open")

        self._set_state(ConnectionState.CONNECTING)

        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            url = self._build_ws_url()
            logger.info(
                "Connecting to WebSocket",
                extra={"shard_id": self._shard_id, "streams": len(self._active_streams)},
            )

            self._ws = await self._session.ws_connect(
                url,
                heartbeat=self._shard_config.ping_interval_ms / 1000,
            )

            self._set_state(ConnectionState.CONNECTED)
            self._circuit_breaker.record_success()
            self._backoff_state.reset()
            self._last_pong_ts = int(time.time() * 1000)

            # Start receive and ping tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._ping_task = asyncio.create_task(self._ping_loop())

            logger.info(
                "WebSocket connected",
                extra={"shard_id": self._shard_id, "streams": len(self._active_streams)},
            )

        except Exception as e:
            logger.error(
                "Failed to connect",
                extra={"shard_id": self._shard_id, "error": str(e)},
            )
            self._set_state(ConnectionState.DISCONNECTED)
            self._circuit_breaker.record_failure()
            raise

    async def disconnect(self) -> None:
        """Gracefully disconnect WebSocket."""
        self._set_state(ConnectionState.CLOSING)

        # Cancel tasks
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
            self._receive_task = None

        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ping_task
            self._ping_task = None

        # Close WebSocket
        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None

        # Close session
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        self._set_state(ConnectionState.CLOSED)
        logger.info("WebSocket disconnected", extra={"shard_id": self._shard_id})

    async def reconnect(self) -> None:
        """
        Reconnect with exponential backoff.

        Per BINANCE_LIMITS.md: exponential backoff + jitter for reconnects.
        """
        self._set_state(ConnectionState.RECONNECTING)
        self._metrics.reconnect_count += 1

        # Close existing connection
        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None

        # Compute backoff delay
        self._backoff_state.record_error()
        delay_ms = compute_backoff_delay(self._backoff_config, self._backoff_state)

        logger.info(
            "Reconnecting with backoff",
            extra={
                "shard_id": self._shard_id,
                "delay_ms": delay_ms,
                "attempt": self._backoff_state.attempt,
            },
        )

        await asyncio.sleep(delay_ms / 1000)

        try:
            await self.connect()
        except Exception as e:
            logger.error(
                "Reconnect failed",
                extra={"shard_id": self._shard_id, "error": str(e)},
            )
            # Schedule another reconnect
            if self._backoff_state.attempt < self._backoff_config.max_retries:
                task = asyncio.create_task(self.reconnect())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                logger.error(
                    "Max reconnect attempts reached",
                    extra={"shard_id": self._shard_id},
                )
                self._set_state(ConnectionState.DISCONNECTED)

    async def add_subscriptions(
        self,
        subscriptions: list[StreamSubscription],
    ) -> list[StreamSubscription]:
        """
        Add stream subscriptions.

        Args:
            subscriptions: List of subscriptions to add.

        Returns:
            List of subscriptions that were actually added.
        """
        added = []
        for sub in subscriptions:
            if len(self._active_streams) >= self._shard_config.max_streams:
                break
            stream_name = sub.to_stream_name()
            if stream_name not in self._active_streams:
                self._subscriptions.add(sub)
                self._active_streams.add(stream_name)
                added.append(sub)

        if added and self._state == ConnectionState.CONNECTED:
            await self._send_subscribe([s.to_stream_name() for s in added])

        return added

    async def remove_subscriptions(
        self,
        subscriptions: list[StreamSubscription],
    ) -> list[StreamSubscription]:
        """
        Remove stream subscriptions.

        Args:
            subscriptions: List of subscriptions to remove.

        Returns:
            List of subscriptions that were actually removed.
        """
        removed = []
        for sub in subscriptions:
            stream_name = sub.to_stream_name()
            if stream_name in self._active_streams:
                self._subscriptions.discard(sub)
                self._active_streams.discard(stream_name)
                removed.append(sub)

        if removed and self._state == ConnectionState.CONNECTED:
            await self._send_unsubscribe([s.to_stream_name() for s in removed])

        return removed

    async def _send_subscribe(self, streams: list[str]) -> None:
        """Send SUBSCRIBE command to WebSocket."""
        if not self._ws or self._ws.closed:
            return

        # Batch subscriptions
        for i in range(0, len(streams), self._shard_config.subscribe_batch_size):
            batch = streams[i : i + self._shard_config.subscribe_batch_size]
            msg = {
                "method": "SUBSCRIBE",
                "params": batch,
                "id": self._next_request_id(),
            }
            await self._ws.send_json(msg)
            logger.debug(
                "Sent SUBSCRIBE",
                extra={"shard_id": self._shard_id, "streams": len(batch)},
            )

    async def _send_unsubscribe(self, streams: list[str]) -> None:
        """Send UNSUBSCRIBE command to WebSocket."""
        if not self._ws or self._ws.closed:
            return

        msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": self._next_request_id(),
        }
        await self._ws.send_json(msg)
        logger.debug(
            "Sent UNSUBSCRIBE",
            extra={"shard_id": self._shard_id, "streams": len(streams)},
        )

    async def _receive_loop(self) -> None:
        """Main loop for receiving WebSocket messages."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    recv_ts = int(time.time() * 1000)
                    try:
                        data = json.loads(msg.data)
                        self._metrics.messages_received += 1
                        self._metrics.last_message_ts = recv_ts

                        # Handle combined stream format
                        if "stream" in data and "data" in data:
                            raw_msg = RawMessage(
                                data=data["data"],
                                recv_ts=recv_ts,
                                shard_id=self._shard_id,
                            )
                        else:
                            raw_msg = RawMessage(
                                data=data,
                                recv_ts=recv_ts,
                                shard_id=self._shard_id,
                            )

                        await self._on_message(raw_msg)

                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse message",
                            extra={"shard_id": self._shard_id},
                        )

                elif msg.type == aiohttp.WSMsgType.PONG:
                    self._last_pong_ts = int(time.time() * 1000)

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info(
                        "WebSocket closed",
                        extra={"shard_id": self._shard_id},
                    )
                    break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        "WebSocket error",
                        extra={
                            "shard_id": self._shard_id,
                            "error": str(self._ws.exception()),
                        },
                    )
                    break

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "Error in receive loop",
                extra={"shard_id": self._shard_id, "error": str(e)},
            )

        # Trigger reconnect if not intentionally closing
        if self._state not in (ConnectionState.CLOSING, ConnectionState.CLOSED):
            task = asyncio.create_task(self.reconnect())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _ping_loop(self) -> None:
        """Ping loop to keep connection alive and detect stale connections."""
        ping_interval = self._shard_config.ping_interval_ms / 1000
        ping_timeout = self._shard_config.ping_timeout_ms / 1000

        try:
            while self._state == ConnectionState.CONNECTED:
                await asyncio.sleep(ping_interval)

                if not self._ws or self._ws.closed:
                    break

                # Check if last pong is stale
                now_ms = int(time.time() * 1000)
                if self._last_pong_ts > 0:
                    stale_ms = now_ms - self._last_pong_ts
                    if stale_ms > self._shard_config.ping_timeout_ms * 2:
                        logger.warning(
                            "Connection stale, triggering reconnect",
                            extra={
                                "shard_id": self._shard_id,
                                "stale_ms": stale_ms,
                            },
                        )
                        break

                # Send ping
                try:
                    await asyncio.wait_for(
                        self._ws.ping(),
                        timeout=ping_timeout,
                    )
                except TimeoutError:
                    logger.warning(
                        "Ping timeout",
                        extra={"shard_id": self._shard_id},
                    )
                    break

        except asyncio.CancelledError:
            raise

        # Trigger reconnect if loop exited unexpectedly
        if self._state == ConnectionState.CONNECTED:
            task = asyncio.create_task(self.reconnect())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
