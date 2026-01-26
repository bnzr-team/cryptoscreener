"""Tests for Binance REST client."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from cryptoscreener.connectors.backoff import (
    CircuitBreaker,
    GovernorDroppedError,
    RateLimitError,
    RestGovernor,
    RestGovernorConfig,
)
from cryptoscreener.connectors.binance.rest_client import BinanceRestClient
from cryptoscreener.connectors.binance.types import ConnectorConfig


class TestBinanceRestClient:
    """Tests for BinanceRestClient."""

    @pytest.fixture
    def client(self) -> BinanceRestClient:
        """Create REST client instance."""
        return BinanceRestClient()

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Create mock aiohttp response."""
        response = MagicMock()
        response.status = 200
        response.headers = {}
        response.json = AsyncMock(return_value={})
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    @pytest.mark.asyncio
    async def test_get_exchange_info_success(
        self,
        client: BinanceRestClient,
        mock_response: MagicMock,
    ) -> None:
        """Successfully fetch exchangeInfo."""
        mock_response.json = AsyncMock(
            return_value={
                "symbols": [{"symbol": "BTCUSDT"}],
                "serverTime": 1234567890,
                "rateLimits": [{"type": "REQUEST_WEIGHT"}],
            }
        )

        with patch.object(aiohttp.ClientSession, "request", return_value=mock_response):
            info = await client.get_exchange_info()

        assert len(info.symbols) == 1
        assert info.server_time == 1234567890
        assert len(info.rate_limits) == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_get_tradeable_symbols(
        self,
        client: BinanceRestClient,
        mock_response: MagicMock,
    ) -> None:
        """Filter tradeable perpetual USDT symbols."""
        mock_response.json = AsyncMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "baseAsset": "BTC",
                        "quoteAsset": "USDT",
                        "pricePrecision": 2,
                        "quantityPrecision": 3,
                        "contractType": "PERPETUAL",
                        "status": "TRADING",
                    },
                    {
                        "symbol": "BTCBUSD",
                        "baseAsset": "BTC",
                        "quoteAsset": "BUSD",
                        "pricePrecision": 2,
                        "quantityPrecision": 3,
                        "contractType": "PERPETUAL",
                        "status": "TRADING",
                    },
                    {
                        "symbol": "BTCUSDT_230929",
                        "baseAsset": "BTC",
                        "quoteAsset": "USDT",
                        "pricePrecision": 2,
                        "quantityPrecision": 3,
                        "contractType": "CURRENT_QUARTER",
                        "status": "TRADING",
                    },
                    {
                        "symbol": "XYZUSDT",
                        "baseAsset": "XYZ",
                        "quoteAsset": "USDT",
                        "pricePrecision": 2,
                        "quantityPrecision": 3,
                        "contractType": "PERPETUAL",
                        "status": "BREAK",
                    },
                ],
                "serverTime": 1234567890,
                "rateLimits": [],
            }
        )

        with patch.object(aiohttp.ClientSession, "request", return_value=mock_response):
            symbols = await client.get_tradeable_symbols()

        # Only BTCUSDT matches: USDT, PERPETUAL, TRADING
        assert len(symbols) == 1
        assert symbols[0].symbol == "BTCUSDT"

        await client.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_request(
        self,
        client: BinanceRestClient,
    ) -> None:
        """Circuit breaker blocks requests when open."""
        # Force circuit breaker open
        client._circuit_breaker.force_open(60000, "test")

        with pytest.raises(RateLimitError) as exc_info:
            await client.get_exchange_info()

        assert "Circuit breaker open" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_rate_limit_429_opens_circuit_breaker(
        self,
        client: BinanceRestClient,
        mock_response: MagicMock,
    ) -> None:
        """429 response opens circuit breaker immediately (per BINANCE_LIMITS.md)."""
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "1"}
        mock_response.json = AsyncMock(return_value={"code": -1003})

        with patch.object(aiohttp.ClientSession, "request", return_value=mock_response):
            # First request hits 429, circuit breaker opens
            # Second attempt is blocked by circuit breaker
            with pytest.raises(RateLimitError) as exc_info:
                await client.get_exchange_info()

            assert "Circuit breaker open" in str(exc_info.value)

        # Circuit breaker should be open
        assert not client._circuit_breaker.can_execute()

        await client.close()

    @pytest.mark.asyncio
    async def test_rate_limit_418_triggers_ip_ban(
        self,
        client: BinanceRestClient,
        mock_response: MagicMock,
    ) -> None:
        """418 response indicates IP ban."""
        mock_response.status = 418
        mock_response.json = AsyncMock(return_value={})

        with (
            patch.object(aiohttp.ClientSession, "request", return_value=mock_response),
            pytest.raises(RateLimitError),
        ):
            # Max retries will be exhausted
            client._backoff_config.max_retries = 1
            await client.get_exchange_info()

        # Circuit breaker should be open due to IP ban
        assert client._circuit_breaker.is_banned

        await client.close()

    @pytest.mark.asyncio
    async def test_get_server_time(
        self,
        client: BinanceRestClient,
        mock_response: MagicMock,
    ) -> None:
        """Get server time endpoint."""
        mock_response.json = AsyncMock(return_value={"serverTime": 1234567890})

        with patch.object(aiohttp.ClientSession, "request", return_value=mock_response):
            server_time = await client.get_server_time()

        assert server_time == 1234567890

        await client.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, client: BinanceRestClient) -> None:
        """Close can be called multiple times."""
        await client.close()
        await client.close()  # Should not raise


class TestBinanceRestClientConfig:
    """Tests for REST client configuration."""

    def test_custom_config(self) -> None:
        """Custom configuration is used."""
        config = ConnectorConfig(
            base_rest_url="https://testnet.binance.com",
            request_timeout_ms=5000,
        )
        client = BinanceRestClient(config=config)

        assert client._config.base_rest_url == "https://testnet.binance.com"
        assert client._config.request_timeout_ms == 5000

    def test_shared_circuit_breaker(self) -> None:
        """Shared circuit breaker can be provided."""
        cb = CircuitBreaker()
        client = BinanceRestClient(circuit_breaker=cb)

        assert client._circuit_breaker is cb

    def test_governor_can_be_provided(self) -> None:
        """RestGovernor can be provided (DEC-023d)."""
        cb = CircuitBreaker()
        governor = RestGovernor(circuit_breaker=cb)
        client = BinanceRestClient(circuit_breaker=cb, governor=governor)

        assert client._governor is governor


# =============================================================================
# DEC-023d: Governor Integration Tests
# =============================================================================


class TestBinanceRestClientWithGovernor:
    """Integration tests for REST client with RestGovernor (DEC-023d)."""

    @pytest.fixture
    def circuit_breaker(self) -> CircuitBreaker:
        """Create circuit breaker for testing."""
        return CircuitBreaker()

    @pytest.fixture
    def governor(self, circuit_breaker: CircuitBreaker) -> RestGovernor:
        """Create governor with small limits for testing."""
        config = RestGovernorConfig(
            budget_weight_per_minute=100,  # Small budget for testing
            max_queue_depth=3,  # Small queue for testing
            max_concurrent_requests=2,  # Small concurrency for testing
            default_timeout_ms=5000,
        )
        return RestGovernor(config=config, circuit_breaker=circuit_breaker)

    @pytest.fixture
    def client(self, circuit_breaker: CircuitBreaker, governor: RestGovernor) -> BinanceRestClient:
        """Create REST client with governor."""
        return BinanceRestClient(circuit_breaker=circuit_breaker, governor=governor)

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Create mock aiohttp response for successful request."""
        response = MagicMock()
        response.status = 200
        response.headers = {}
        response.json = AsyncMock(return_value={"serverTime": 1234567890})
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    @pytest.mark.asyncio
    async def test_governor_permits_normal_request(
        self,
        client: BinanceRestClient,
        governor: RestGovernor,
        mock_response: MagicMock,
    ) -> None:
        """Normal request goes through governor successfully."""
        with patch.object(aiohttp.ClientSession, "request", return_value=mock_response):
            result = await client.get_server_time()

        assert result == 1234567890
        assert governor.metrics.requests_allowed == 1
        assert governor.metrics.requests_dropped == 0

        await client.close()

    @pytest.mark.asyncio
    async def test_governor_breaker_open_fails_fast(
        self,
        client: BinanceRestClient,
        circuit_breaker: CircuitBreaker,
        governor: RestGovernor,
    ) -> None:
        """Circuit breaker OPEN via governor → fail-fast (DEC-023d)."""
        # Force circuit breaker open
        circuit_breaker.force_open(60000, "test")

        with pytest.raises(RateLimitError) as exc_info:
            await client.get_server_time()

        assert "Circuit breaker OPEN" in str(exc_info.value)
        # Verify fail-fast: only 1 attempt, no retries
        assert governor.metrics.requests_failed_breaker == 1
        assert governor.metrics.requests_allowed == 0
        assert governor.metrics.requests_deferred == 0

        await client.close()

    @pytest.mark.asyncio
    async def test_governor_queue_full_drops_request(
        self,
        client: BinanceRestClient,
        governor: RestGovernor,
        mock_response: MagicMock,
    ) -> None:
        """Queue full → drop-new policy rejects request (DEC-023d)."""
        # First, exhaust budget to force queuing
        # Budget is 100, default weight is 10, so 10 requests would exhaust
        # But we also have concurrency=2, so only 2 can be in-flight
        # We need to fill the queue (max_queue_depth=3)

        # Simulate a slow response that blocks concurrency slots
        slow_response = MagicMock()
        slow_response.status = 200
        slow_response.headers = {}
        slow_response.text = AsyncMock(return_value="")
        slow_response.__aenter__ = AsyncMock(return_value=slow_response)
        slow_response.__aexit__ = AsyncMock(return_value=None)

        # This will block on json() call, simulating slow response
        slow_event = asyncio.Event()

        async def slow_json() -> dict[str, int]:
            await slow_event.wait()
            return {"serverTime": 1}

        slow_response.json = slow_json

        tasks: list[asyncio.Task[int]] = []

        with patch.object(aiohttp.ClientSession, "request", return_value=slow_response):
            # Start max_concurrent requests (2) - they will block
            for _ in range(governor.config.max_concurrent_requests):
                task = asyncio.create_task(client.get_server_time())
                tasks.append(task)
                # Give time for task to acquire slot
                await asyncio.sleep(0.01)

            # Queue up max_queue_depth requests (3)
            for _ in range(governor.config.max_queue_depth):
                task = asyncio.create_task(client.get_server_time())
                tasks.append(task)
                await asyncio.sleep(0.01)

            # Next request should be dropped
            with pytest.raises(GovernorDroppedError) as exc_info:
                await client.get_server_time()

            assert "Queue full" in str(exc_info.value)
            assert governor.metrics.drop_reason_queue_full > 0

            # Cleanup: signal slow responses to complete
            slow_event.set()
            for task in tasks:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(task, timeout=1.0)

        await client.close()

    @pytest.mark.asyncio
    async def test_governor_concurrency_cap_respected(
        self,
        client: BinanceRestClient,
        governor: RestGovernor,
    ) -> None:
        """Concurrency cap is respected during concurrent requests (DEC-023d)."""
        max_concurrent = governor.config.max_concurrent_requests
        concurrency_samples: list[int] = []
        proceed_event = asyncio.Event()

        # Create a slow mock response that blocks inside the governor-protected section
        def make_slow_response() -> MagicMock:
            response = MagicMock()
            response.status = 200
            response.headers = {}
            response.text = AsyncMock(return_value="")
            response.__aenter__ = AsyncMock(return_value=response)
            response.__aexit__ = AsyncMock(return_value=None)

            async def slow_json() -> dict[str, int]:
                # Sample governor's concurrent count while we're holding a slot
                count = await governor.get_concurrent_count()
                concurrency_samples.append(count)
                # Wait until signaled to complete
                await proceed_event.wait()
                return {"serverTime": 1}

            response.json = slow_json
            return response

        slow_response = make_slow_response()

        with patch.object(aiohttp.ClientSession, "request", return_value=slow_response):
            # Start more requests than concurrency allows
            num_requests = max_concurrent + 3
            tasks = [asyncio.create_task(client.get_server_time()) for _ in range(num_requests)]

            # Let tasks start and block on slow_json
            await asyncio.sleep(0.1)

            # Check that only max_concurrent are in-flight in the governor
            current_concurrent = await governor.get_concurrent_count()
            assert current_concurrent <= max_concurrent, (
                f"Concurrent count {current_concurrent} exceeds max {max_concurrent}"
            )

            # Let requests complete
            proceed_event.set()
            await asyncio.gather(*tasks)

        # Verify max concurrency was never exceeded (samples taken inside permit())
        max_observed = max(concurrency_samples) if concurrency_samples else 0
        assert max_observed <= max_concurrent, (
            f"Max observed concurrency {max_observed} exceeds cap {max_concurrent}"
        )

        await client.close()

    @pytest.mark.asyncio
    async def test_governor_metrics_grow_under_load(
        self,
        client: BinanceRestClient,
        governor: RestGovernor,
        mock_response: MagicMock,
    ) -> None:
        """Metrics accumulate correctly under load (DEC-023d)."""
        initial_allowed = governor.metrics.requests_allowed
        num_requests = 5

        with patch.object(aiohttp.ClientSession, "request", return_value=mock_response):
            for _ in range(num_requests):
                await client.get_server_time()

        assert governor.metrics.requests_allowed == initial_allowed + num_requests
        # Verify budget was consumed
        status = await governor.get_status_async()
        # Each /fapi/v1/time has weight 1 (per DEFAULT_ENDPOINT_WEIGHTS)
        # 5 requests = 5 weight consumed from 100 budget
        assert status["budget_tokens"] < 100  # Some budget consumed

        await client.close()

    @pytest.mark.asyncio
    async def test_governed_request_records_rate_limit_in_breaker(
        self,
        client: BinanceRestClient,
        circuit_breaker: CircuitBreaker,
        mock_response: MagicMock,
    ) -> None:
        """Rate limit (429) via governed request opens circuit breaker (DEC-023d)."""
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "1"}
        mock_response.json = AsyncMock(return_value={"code": -1003})

        # Limit retries for faster test
        client._backoff_config.max_retries = 1

        with (
            patch.object(aiohttp.ClientSession, "request", return_value=mock_response),
            pytest.raises(RateLimitError),
        ):
            await client.get_server_time()

        # Circuit breaker should be open
        assert not circuit_breaker.can_execute()

        await client.close()


class TestBinanceRestClientGovernorBackwardCompatibility:
    """Tests verifying backward compatibility without governor (DEC-023d)."""

    @pytest.fixture
    def client_no_governor(self) -> BinanceRestClient:
        """Create REST client without governor (legacy mode)."""
        return BinanceRestClient()

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Create mock aiohttp response."""
        response = MagicMock()
        response.status = 200
        response.headers = {}
        response.json = AsyncMock(return_value={"serverTime": 1234567890})
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    @pytest.mark.asyncio
    async def test_legacy_mode_still_works(
        self,
        client_no_governor: BinanceRestClient,
        mock_response: MagicMock,
    ) -> None:
        """Client without governor still works (backward compatibility)."""
        assert client_no_governor._governor is None

        with patch.object(aiohttp.ClientSession, "request", return_value=mock_response):
            result = await client_no_governor.get_server_time()

        assert result == 1234567890

        await client_no_governor.close()

    @pytest.mark.asyncio
    async def test_legacy_circuit_breaker_blocks(
        self,
        client_no_governor: BinanceRestClient,
    ) -> None:
        """Legacy circuit breaker blocking still works."""
        client_no_governor._circuit_breaker.force_open(60000, "test")

        with pytest.raises(RateLimitError) as exc_info:
            await client_no_governor.get_server_time()

        # Legacy error message format
        assert "Circuit breaker open" in str(exc_info.value)

        await client_no_governor.close()


class TestBinanceRestClientGovernorWeightTimeout:
    """Tests for governor weight and timeout wiring semantics (DEC-023d-wiring).

    These tests verify that:
    1. Endpoint weight is correctly looked up and passed to governor.permit()
    2. Timeout from client config is passed to governor.permit()
    3. Budget consumption matches expected endpoint weights
    """

    @pytest.fixture
    def circuit_breaker(self) -> CircuitBreaker:
        """Create circuit breaker for testing."""
        return CircuitBreaker()

    @pytest.mark.asyncio
    async def test_endpoint_weight_passed_to_governor(
        self,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Verify correct endpoint weight is passed to governor.permit() (DEC-023d-wiring).

        Tests that:
        - /fapi/v1/time uses weight=1
        - /fapi/v1/exchangeInfo uses weight=40
        - Budget consumption matches expected weights
        """
        # Use small budget to make weight consumption visible
        config = RestGovernorConfig(
            budget_weight_per_minute=100,
            max_queue_depth=10,
            max_concurrent_requests=10,
        )
        governor = RestGovernor(config=config, circuit_breaker=circuit_breaker)
        client = BinanceRestClient(circuit_breaker=circuit_breaker, governor=governor)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Initial budget
        initial_status = governor.get_status()
        initial_budget = initial_status["budget_tokens"]
        assert initial_budget == 100.0

        with patch.object(aiohttp.ClientSession, "request", return_value=mock_response):
            # Request to /fapi/v1/time (weight=1)
            mock_response.json = AsyncMock(return_value={"serverTime": 123})
            await client.get_server_time()

            status_after_time = governor.get_status()
            budget_after_time = status_after_time["budget_tokens"]
            # Should consume weight=1
            assert budget_after_time == 99.0, f"Expected 99.0, got {budget_after_time}"

            # Request to /fapi/v1/exchangeInfo (weight=40)
            mock_response.json = AsyncMock(
                return_value={"symbols": [], "serverTime": 123, "rateLimits": []}
            )
            await client.get_exchange_info()

            status_after_exchange = governor.get_status()
            budget_after_exchange = status_after_exchange["budget_tokens"]
            # Should consume additional weight=40
            assert budget_after_exchange == 59.0, f"Expected 59.0, got {budget_after_exchange}"

        await client.close()

    @pytest.mark.asyncio
    async def test_client_timeout_passed_to_governor(
        self,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Verify client timeout config is passed to governor.permit() (DEC-023d-wiring).

        Tests that the request_timeout_ms from ConnectorConfig is used as the
        timeout_ms parameter when calling governor.permit().
        """
        # Custom timeout in config
        custom_timeout_ms = 15000
        client_config = ConnectorConfig(request_timeout_ms=custom_timeout_ms)

        # Governor with short default timeout (different from client's)
        governor_config = RestGovernorConfig(
            budget_weight_per_minute=1000,
            max_queue_depth=10,
            max_concurrent_requests=10,
            default_timeout_ms=5000,  # Governor default is different
        )
        governor = RestGovernor(config=governor_config, circuit_breaker=circuit_breaker)
        client = BinanceRestClient(
            config=client_config,
            circuit_breaker=circuit_breaker,
            governor=governor,
        )

        # Track what parameters were passed to permit()
        permit_calls: list[tuple[str, int | None, int | None]] = []
        original_permit = governor.permit

        @contextlib.asynccontextmanager
        async def tracking_permit(
            endpoint: str,
            weight: int | None = None,
            timeout_ms: int | None = None,
        ):
            permit_calls.append((endpoint, weight, timeout_ms))
            async with original_permit(endpoint, weight, timeout_ms):
                yield

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value={"serverTime": 123})
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(governor, "permit", tracking_permit),
            patch.object(aiohttp.ClientSession, "request", return_value=mock_response),
        ):
            await client.get_server_time()

        # Verify permit was called with correct parameters
        assert len(permit_calls) == 1
        endpoint, weight, timeout_ms = permit_calls[0]
        assert endpoint == "/fapi/v1/time"
        assert weight == 1  # Known weight for this endpoint
        assert timeout_ms == custom_timeout_ms  # Client's timeout, not governor's default

        await client.close()

    @pytest.mark.asyncio
    async def test_ticker_endpoint_weight_40(
        self,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Verify /fapi/v1/ticker/24hr uses weight=40 (DEC-023d-wiring).

        Tests that the ticker endpoint correctly consumes its documented weight.
        """
        config = RestGovernorConfig(
            budget_weight_per_minute=100,
            max_queue_depth=10,
            max_concurrent_requests=10,
        )
        governor = RestGovernor(config=config, circuit_breaker=circuit_breaker)
        client = BinanceRestClient(circuit_breaker=circuit_breaker, governor=governor)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value=[{"symbol": "BTCUSDT", "quoteVolume": "1000"}])
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        initial_budget = governor.get_status()["budget_tokens"]

        with patch.object(aiohttp.ClientSession, "request", return_value=mock_response):
            # get_24h_tickers calls /fapi/v1/ticker/24hr which has weight=40
            await client.get_24h_tickers()

        # Verify known weight was used (40 for /fapi/v1/ticker/24hr)
        after_budget = governor.get_status()["budget_tokens"]
        consumed = initial_budget - after_budget
        assert consumed == 40, f"Expected weight 40, consumed {consumed}"

        await client.close()
