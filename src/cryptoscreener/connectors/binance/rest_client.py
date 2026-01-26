"""
REST client for Binance USD-M Futures API bootstrap.

Per BINANCE_LIMITS.md:
- REST only for bootstrap (exchangeInfo)
- 2,400 requests per minute per IP limit
- Use WS for live updates, not REST polling
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import aiohttp

from cryptoscreener.connectors.backoff import (
    BackoffConfig,
    BackoffState,
    CircuitBreaker,
    GovernorDroppedError,
    GovernorTimeoutError,
    RateLimitError,
    RestGovernor,
    compute_backoff_delay,
    handle_error_response,
)
from cryptoscreener.connectors.binance.types import (
    ConnectorConfig,
    ExchangeInfo,
    SymbolInfo,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class BinanceRestClient:
    """
    Async REST client for Binance USD-M Futures bootstrap operations.

    This client is used only for initial data fetching (exchangeInfo).
    All live data should come from WebSocket streams.
    """

    def __init__(
        self,
        config: ConnectorConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        governor: RestGovernor | None = None,
    ) -> None:
        """
        Initialize the REST client.

        Args:
            config: Connector configuration.
            circuit_breaker: Shared circuit breaker for rate limit protection.
            governor: Optional RestGovernor for budget/queue/concurrency control.
                      DEC-023d: If provided, replaces manual circuit breaker checks
                      and adds budget-based rate limiting with bounded queue.
        """
        self._config = config or ConnectorConfig()
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._governor = governor
        self._backoff_config = BackoffConfig()
        self._backoff_state = BackoffState()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_ms / 1000)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, str] | None = None,
    ) -> Any:
        """
        Make an HTTP request with retry and circuit breaker logic.

        DEC-023d: If governor is configured, uses permit() for budget/queue/concurrency
        control. Governor integrates with circuit breaker internally.

        Args:
            method: HTTP method.
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            JSON response data (dict or list depending on endpoint).

        Raises:
            RateLimitError: If rate limited and retries exhausted, or circuit breaker open.
            GovernorDroppedError: If governor queue is full (DEC-023d).
            GovernorTimeoutError: If timeout waiting in governor queue (DEC-023d).
            aiohttp.ClientError: On network errors.
        """
        url = f"{self._config.base_rest_url}{endpoint}"

        while self._backoff_state.attempt <= self._backoff_config.max_retries:
            # DEC-023d: Use governor if configured, otherwise fall back to direct circuit breaker
            if self._governor is not None:
                # Governor handles circuit breaker check, budget, queue, and concurrency
                # Exceptions (GovernorDroppedError, GovernorTimeoutError, RateLimitError)
                # propagate up - no retry for these governance errors
                try:
                    return await self._request_with_governor(method, endpoint, url, params)
                except (GovernorDroppedError, GovernorTimeoutError):
                    # Governance errors are not retryable
                    raise
                except RateLimitError:
                    # Circuit breaker open via governor - not retryable
                    raise
                except aiohttp.ClientError as e:
                    # Network errors - record and potentially retry
                    self._backoff_state.record_error()
                    logger.warning(
                        "Request failed (governed)",
                        extra={"error": str(e), "attempt": self._backoff_state.attempt},
                    )
                    if self._backoff_state.attempt > self._backoff_config.max_retries:
                        raise
                    # Apply backoff before retry
                    delay_ms = compute_backoff_delay(self._backoff_config, self._backoff_state)
                    if delay_ms > 0:
                        await asyncio.sleep(delay_ms / 1000)
                    continue
            else:
                # Legacy path: direct circuit breaker check (no governor)
                result = await self._request_legacy(method, endpoint, url, params)
                if result is not None:
                    return result
                # None means retry (rate limit or server error)
                continue

        raise RateLimitError(
            f"Max retries ({self._backoff_config.max_retries}) exhausted",
        )

    async def _request_with_governor(
        self,
        method: str,
        endpoint: str,
        url: str,
        params: dict[str, str] | None,
    ) -> Any:
        """
        Execute request with governor controlling budget/queue/concurrency.

        DEC-023d: Governor.permit() handles:
        - Circuit breaker check (fail-fast if OPEN)
        - Budget consumption (token bucket)
        - Queue management (bounded FIFO with drop-new)
        - Concurrency limiting (semaphore)
        - Automatic slot release on exit

        Args:
            method: HTTP method.
            endpoint: API endpoint path (already normalized, no query string).
            url: Full URL.
            params: Query parameters.

        Returns:
            JSON response data.

        Raises:
            RateLimitError: If circuit breaker is OPEN or rate limit hit.
            GovernorDroppedError: If queue is full.
            GovernorTimeoutError: If timeout waiting in queue.
            aiohttp.ClientError: On network errors.
        """
        assert self._governor is not None  # Type narrowing

        # DEC-023d: Normalize endpoint at client boundary using URL path
        # This guarantees clean path even if endpoint or url contains query strings
        endpoint_key = urlsplit(url).path
        weight = self._governor.config.get_endpoint_weight(endpoint_key)
        timeout_ms = self._config.request_timeout_ms

        async with self._governor.permit(endpoint_key, weight=weight, timeout_ms=timeout_ms):
            session = await self._get_session()
            async with session.request(method, url, params=params) as response:
                # Parse Retry-After header if present
                retry_after_ms = None
                if "Retry-After" in response.headers:
                    with contextlib.suppress(ValueError):
                        retry_after_ms = int(response.headers["Retry-After"]) * 1000

                # Check for rate limit errors
                if response.status in (429, 418):
                    error_code = None
                    try:
                        data = await response.json()
                        error_code = data.get("code")
                    except Exception:
                        pass

                    rate_limit_error = handle_error_response(
                        response.status,
                        error_code=error_code,
                        retry_after_ms=retry_after_ms,
                    )
                    if rate_limit_error:
                        # Record failure in circuit breaker (governor's or standalone)
                        self._circuit_breaker.record_failure(
                            is_rate_limit=True,
                            is_ip_ban=rate_limit_error.is_ip_ban,
                        )
                        self._backoff_state.record_error()
                        logger.warning(
                            "Rate limit hit (governed)",
                            extra={
                                "status": response.status,
                                "error_code": error_code,
                                "retry_after_ms": retry_after_ms,
                            },
                        )
                        raise rate_limit_error

                # Handle other errors
                if response.status >= 400:
                    self._backoff_state.record_error()
                    text = await response.text()
                    logger.error(
                        "HTTP error (governed)",
                        extra={"status": response.status, "body": text[:500]},
                    )
                    if response.status >= 500:
                        # Server error - raise to trigger retry in _request()
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=f"Server error: {text[:200]}",
                        )
                    # Client error (4xx non-rate-limit) - don't retry
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=text[:200],
                    )

                # Success
                self._circuit_breaker.record_success()
                self._backoff_state.reset()
                result: dict[str, Any] = await response.json()
                return result

    async def _request_legacy(
        self,
        method: str,
        endpoint: str,
        url: str,
        params: dict[str, str] | None,
    ) -> Any | None:
        """
        Execute request with legacy circuit breaker (no governor).

        Pre-DEC-023d path for backward compatibility.

        Args:
            method: HTTP method.
            endpoint: API endpoint path.
            url: Full URL.
            params: Query parameters.

        Returns:
            JSON response data on success, None if should retry.

        Raises:
            RateLimitError: If circuit breaker is open.
            aiohttp.ClientError: On non-retryable client errors.
        """
        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, blocking request",
                extra={"endpoint": endpoint},
            )
            raise RateLimitError(
                "Circuit breaker open",
                retry_after_ms=self._circuit_breaker.recovery_timeout_ms,
            )

        # Apply backoff delay
        delay_ms = compute_backoff_delay(
            self._backoff_config,
            self._backoff_state,
        )
        if delay_ms > 0:
            logger.debug(
                "Backing off before request",
                extra={"delay_ms": delay_ms, "attempt": self._backoff_state.attempt},
            )
            await asyncio.sleep(delay_ms / 1000)

        try:
            session = await self._get_session()
            async with session.request(method, url, params=params) as response:
                # Parse Retry-After header if present
                retry_after_ms = None
                if "Retry-After" in response.headers:
                    with contextlib.suppress(ValueError):
                        retry_after_ms = int(response.headers["Retry-After"]) * 1000

                # Check for rate limit errors
                if response.status in (429, 418):
                    error_code = None
                    try:
                        data = await response.json()
                        error_code = data.get("code")
                    except Exception:
                        pass

                    rate_limit_error = handle_error_response(
                        response.status,
                        error_code=error_code,
                        retry_after_ms=retry_after_ms,
                    )
                    if rate_limit_error:
                        self._circuit_breaker.record_failure(
                            is_rate_limit=True,
                            is_ip_ban=rate_limit_error.is_ip_ban,
                        )
                        self._backoff_state.record_error()
                        logger.warning(
                            "Rate limit hit",
                            extra={
                                "status": response.status,
                                "error_code": error_code,
                                "retry_after_ms": retry_after_ms,
                            },
                        )
                        return None  # Signal retry

                # Handle other errors
                if response.status >= 400:
                    self._backoff_state.record_error()
                    text = await response.text()
                    logger.error(
                        "HTTP error",
                        extra={"status": response.status, "body": text[:500]},
                    )
                    if response.status >= 500:
                        # Server error, retry
                        return None
                    # Client error, don't retry
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=text[:200],
                    )

                # Success
                self._circuit_breaker.record_success()
                self._backoff_state.reset()
                result: dict[str, Any] = await response.json()
                return result

        except aiohttp.ClientError as e:
            self._backoff_state.record_error()
            logger.warning(
                "Request failed",
                extra={"error": str(e), "attempt": self._backoff_state.attempt},
            )
            if self._backoff_state.attempt > self._backoff_config.max_retries:
                raise
            return None  # Signal retry

    async def get_exchange_info(self) -> ExchangeInfo:
        """
        Fetch exchange information from Binance.

        Returns:
            ExchangeInfo with symbols, server time, and rate limits.
        """
        logger.info("Fetching exchangeInfo from Binance")
        data = await self._request("GET", "/fapi/v1/exchangeInfo")

        return ExchangeInfo(
            symbols=data.get("symbols", []),
            server_time=data.get("serverTime", int(time.time() * 1000)),
            rate_limits=data.get("rateLimits", []),
        )

    async def get_tradeable_symbols(
        self,
        quote_asset: str = "USDT",
        contract_type: str = "PERPETUAL",
    ) -> Sequence[SymbolInfo]:
        """
        Get list of tradeable perpetual symbols.

        Args:
            quote_asset: Quote asset filter (default: USDT).
            contract_type: Contract type filter (default: PERPETUAL).

        Returns:
            List of SymbolInfo for tradeable symbols.
        """
        exchange_info = await self.get_exchange_info()

        symbols = []
        for raw in exchange_info.symbols:
            if raw.get("status") != "TRADING":
                continue
            if raw.get("quoteAsset") != quote_asset:
                continue
            if raw.get("contractType") != contract_type:
                continue

            try:
                symbols.append(SymbolInfo.from_raw(raw))
            except KeyError as e:
                logger.warning(
                    "Failed to parse symbol",
                    extra={"symbol": raw.get("symbol"), "error": str(e)},
                )

        logger.info(
            "Fetched tradeable symbols",
            extra={"count": len(symbols), "quote_asset": quote_asset},
        )
        return symbols

    async def get_24h_tickers(self) -> dict[str, float]:
        """
        Get 24h quote volume for all symbols.

        Returns:
            Dict mapping symbol to 24h quote volume (in USDT).
        """
        logger.info("Fetching 24h tickers from Binance")
        # Note: This endpoint returns a list of ticker objects
        data = await self._request("GET", "/fapi/v1/ticker/24hr")

        volumes: dict[str, float] = {}
        if not isinstance(data, list):
            logger.warning("Unexpected response type from ticker endpoint")
            return volumes

        for ticker in data:
            if not isinstance(ticker, dict):
                continue
            symbol = str(ticker.get("symbol", ""))
            quote_volume = ticker.get("quoteVolume", "0")
            try:
                volumes[symbol] = float(str(quote_volume))
            except (ValueError, TypeError):
                volumes[symbol] = 0.0

        logger.info(
            "Fetched 24h tickers",
            extra={"count": len(volumes)},
        )
        return volumes

    async def get_server_time(self) -> int:
        """
        Get Binance server time.

        Returns:
            Server timestamp in milliseconds.
        """
        data = await self._request("GET", "/fapi/v1/time")
        server_time = data.get("serverTime")
        if isinstance(server_time, int):
            return server_time
        return int(time.time() * 1000)
