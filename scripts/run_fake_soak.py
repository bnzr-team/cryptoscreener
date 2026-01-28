#!/usr/bin/env python3
"""
DEC-032: Run an offline soak test using FakeWSServer.

Starts a local fake WS server that streams synthetic aggTrade messages
indefinitely, then runs the live pipeline against it for --duration-s seconds.

Usage:
    python -m scripts.run_fake_soak --duration-s 60 --summary-json baseline.json
    python -m scripts.run_fake_soak --duration-s 60 --summary-json overload.json \
        --fault-slow-consumer-ms 50

No outbound network required. Suitable for CI.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import aiohttp.web

logger = logging.getLogger(__name__)


class ContinuousFakeWSServer:
    """Fake WS server that sends aggTrade messages continuously (no disconnect)."""

    def __init__(self, *, send_interval_ms: int = 10, num_symbols: int = 5) -> None:
        self.send_interval_s = send_interval_ms / 1000
        self.num_symbols = num_symbols
        self.total_messages_sent = 0
        self._runner: aiohttp.web.AppRunner | None = None
        self.port: int = 0
        self._symbols = [f"SYMBOL{i}USDT" for i in range(num_symbols)]

    async def _ws_handler(self, request: aiohttp.web.Request) -> aiohttp.web.WebSocketResponse:
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)
        logger.info("FakeWSServer: client connected")

        idx = 0
        try:
            while not ws.closed:
                symbol = self._symbols[idx % len(self._symbols)]
                msg = {
                    "stream": f"{symbol.lower()}@aggTrade",
                    "data": {
                        "e": "aggTrade",
                        "E": int(time.time() * 1000),
                        "s": symbol,
                        "p": f"{50000 + idx % 100:.2f}",
                        "q": "0.100",
                        "T": int(time.time() * 1000),
                        "m": idx % 2 == 0,
                    },
                }
                await ws.send_json(msg)
                self.total_messages_sent += 1
                idx += 1
                await asyncio.sleep(self.send_interval_s)
        except (ConnectionResetError, asyncio.CancelledError):
            pass

        return ws

    async def start(self) -> None:
        app = aiohttp.web.Application()
        app.router.add_get("/ws", self._ws_handler)
        app.router.add_get("/stream", self._ws_handler)
        # Combined stream endpoint pattern used by BinanceStreamManager
        app.router.add_get("/stream/{streams:.*}", self._ws_handler)
        self._runner = aiohttp.web.AppRunner(app)
        await self._runner.setup()
        site = aiohttp.web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        assert self._runner.addresses
        self.port = self._runner.addresses[0][1]
        logger.info("FakeWSServer started on port %d", self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        logger.info("FakeWSServer stopped (sent %d messages)", self.total_messages_sent)

    @property
    def base_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"


async def run_soak(
    duration_s: int,
    summary_json: Path,
    *,
    fault_slow_consumer_ms: int | None = None,
    fault_drop_ws_every_s: int | None = None,
    metrics_port: int = 0,
    send_interval_ms: int = 10,
    num_symbols: int = 5,
    top_n: int = 5,
    cadence_ms: int = 1000,
) -> int:
    """Start FakeWSServer and run pipeline against it."""
    from scripts.run_live import LivePipelineConfig, run_pipeline

    server = ContinuousFakeWSServer(
        send_interval_ms=send_interval_ms,
        num_symbols=num_symbols,
    )
    await server.start()

    try:
        config = LivePipelineConfig(
            symbols=[f"SYMBOL{i}USDT" for i in range(num_symbols)],
            top_n=top_n,
            snapshot_cadence_ms=cadence_ms,
            duration_s=duration_s,
            metrics_port=metrics_port,
            summary_json=summary_json,
            fault_slow_consumer_ms=fault_slow_consumer_ms,
            fault_drop_ws_every_s=fault_drop_ws_every_s,
            ws_url=server.base_url,
        )
        return await run_pipeline(config)
    finally:
        await server.stop()


def main() -> int:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="DEC-032: Run offline soak test with FakeWSServer.",
    )
    parser.add_argument(
        "--duration-s",
        type=int,
        required=True,
        help="Soak duration in seconds",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Output path for soak summary JSON",
    )
    parser.add_argument(
        "--fault-slow-consumer-ms",
        type=int,
        default=None,
        help="Slow consumer fault injection (ms delay per event)",
    )
    parser.add_argument(
        "--fault-drop-ws-every-s",
        type=int,
        default=None,
        help="Force WS disconnect every N seconds",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=0,
        help="Prometheus metrics port (0 = disabled, default: 0)",
    )
    parser.add_argument(
        "--send-interval-ms",
        type=int,
        default=10,
        help="FakeWSServer message interval in ms (default: 10)",
    )
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=5,
        help="Number of fake symbols (default: 5)",
    )
    parser.add_argument(
        "--cadence-ms",
        type=int,
        default=1000,
        help="Snapshot cadence in ms (default: 1000)",
    )
    args = parser.parse_args()

    import os
    os.environ["ALLOW_FAULTS"] = "1"

    return asyncio.run(
        run_soak(
            duration_s=args.duration_s,
            summary_json=args.summary_json,
            fault_slow_consumer_ms=args.fault_slow_consumer_ms,
            fault_drop_ws_every_s=args.fault_drop_ws_every_s,
            metrics_port=args.metrics_port,
            send_interval_ms=args.send_interval_ms,
            num_symbols=args.num_symbols,
            cadence_ms=args.cadence_ms,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
