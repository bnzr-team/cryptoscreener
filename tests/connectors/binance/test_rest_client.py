"""Tests for Binance REST client."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from cryptoscreener.connectors.backoff import CircuitBreaker, RateLimitError
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

        with patch.object(
            aiohttp.ClientSession, "request", return_value=mock_response
        ):
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

        with patch.object(
            aiohttp.ClientSession, "request", return_value=mock_response
        ):
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

        with patch.object(
            aiohttp.ClientSession, "request", return_value=mock_response
        ):
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

        with patch.object(
            aiohttp.ClientSession, "request", return_value=mock_response
        ):
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
