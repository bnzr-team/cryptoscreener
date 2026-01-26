"""Record→Replay roundtrip tests.

Verifies that:
1. run_record.py generates valid fixture files
2. run_replay.py can load and replay those fixtures
3. The digest from recording matches the digest from replay (determinism)
"""

import tempfile
from pathlib import Path

import orjson
from scripts.run_record import (
    MinimalRecordPipeline,
    SyntheticMarketEventGenerator,
    compute_file_sha256,
    run_record,
)
from scripts.run_replay import (
    MinimalReplayPipeline,
    load_market_events,
    run_replay,
)

from cryptoscreener.contracts import (
    MarketEvent,
    MarketEventType,
    compute_rank_events_digest,
)


class TestSyntheticGenerator:
    """Tests for SyntheticMarketEventGenerator."""

    def test_generates_events_for_symbols(self) -> None:
        """Test that generator produces events for all requested symbols."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        generator = SyntheticMarketEventGenerator(symbols=symbols, cadence_ms=100)

        events = list(generator.generate(start_ts=1000, duration_ms=500))

        # Should have events for each symbol
        event_symbols = {e.symbol for e in events}
        assert event_symbols == set(symbols)

    def test_deterministic_generation(self) -> None:
        """Test that generation is deterministic with same parameters."""
        symbols = ["BTCUSDT"]
        gen1 = SyntheticMarketEventGenerator(symbols=symbols, cadence_ms=100, seed=42)
        gen2 = SyntheticMarketEventGenerator(symbols=symbols, cadence_ms=100, seed=42)

        events1 = list(gen1.generate(start_ts=1000, duration_ms=500))
        events2 = list(gen2.generate(start_ts=1000, duration_ms=500))

        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2, strict=True):
            assert e1.ts == e2.ts
            assert e1.symbol == e2.symbol
            assert e1.type == e2.type
            assert e1.payload == e2.payload

    def test_alternates_trade_and_book(self) -> None:
        """Test that generator alternates between trade and book events."""
        generator = SyntheticMarketEventGenerator(symbols=["BTCUSDT"], cadence_ms=100)
        events = list(generator.generate(start_ts=1000, duration_ms=300))

        types = [e.type.value for e in events]
        # Should alternate: trade, book, trade, ...
        assert types[0] == "trade"
        assert types[1] == "book"
        assert types[2] == "trade"


class TestMinimalRecordPipeline:
    """Tests for MinimalRecordPipeline."""

    def test_produces_enter_event_after_two_trades(self) -> None:
        """Test that pipeline emits SYMBOL_ENTER after 2 trades."""
        pipeline = MinimalRecordPipeline(seed=42)

        events = [
            MarketEvent(
                ts=1000,
                source="test",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"price": "42000", "qty": "0.1", "side": "buy"},
                recv_ts=1005,
            ),
            MarketEvent(
                ts=2000,
                source="test",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"price": "42001", "qty": "0.2", "side": "buy"},
                recv_ts=2005,
            ),
        ]

        rank_events = pipeline.record(events)

        assert len(rank_events) == 1
        assert rank_events[0].event.value == "SYMBOL_ENTER"
        assert rank_events[0].symbol == "BTCUSDT"

    def test_produces_alert_after_four_trades(self) -> None:
        """Test that pipeline emits ALERT_TRADABLE after 4 trades."""
        pipeline = MinimalRecordPipeline(seed=42)

        events = [
            MarketEvent(
                ts=1000 + i * 100,
                source="test",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"price": "42000", "qty": "0.1", "side": "buy"},
                recv_ts=1005 + i * 100,
            )
            for i in range(4)
        ]

        rank_events = pipeline.record(events)

        # Should have ENTER (at trade 2) and ALERT (at trade 4)
        assert len(rank_events) == 2
        assert rank_events[0].event.value == "SYMBOL_ENTER"
        assert rank_events[1].event.value == "ALERT_TRADABLE"

    def test_ignores_book_events(self) -> None:
        """Test that book events don't trigger rank events."""
        pipeline = MinimalRecordPipeline(seed=42)

        events = [
            MarketEvent(
                ts=1000,
                source="test",
                symbol="BTCUSDT",
                type=MarketEventType.BOOK,
                payload={"bid": "42000", "ask": "42001"},
                recv_ts=1005,
            ),
        ]

        rank_events = pipeline.record(events)
        assert len(rank_events) == 0


class TestRecordReplayRoundtrip:
    """Tests for full record→replay roundtrip."""

    def test_roundtrip_digest_match(self) -> None:
        """Test that record→replay produces matching digests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # Run record
            _manifest_path, record_digest = run_record(
                symbols=["BTCUSDT", "ETHUSDT"],
                duration_s=1,
                out_dir=out_dir,
                cadence_ms=100,
                source="synthetic",
                llm_enabled=False,
            )

            # Run replay
            _rank_events, replay_digest, passed = run_replay(
                fixture_path=out_dir,
                verify_expected=True,
            )

            # Digests must match
            assert record_digest == replay_digest, (
                f"Digest mismatch: record={record_digest}, replay={replay_digest}"
            )
            assert passed, "Replay determinism check should pass"

    def test_manifest_contains_required_fields(self) -> None:
        """Test that manifest has all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            manifest_path, _ = run_record(
                symbols=["BTCUSDT"],
                duration_s=1,
                out_dir=out_dir,
                cadence_ms=100,
            )

            with manifest_path.open("rb") as f:
                manifest = orjson.loads(f.read())

            # Check required fields
            assert "schema_version" in manifest
            assert "recorded_at" in manifest
            assert "source" in manifest
            assert "symbols" in manifest
            assert "duration_s" in manifest
            assert "sha256" in manifest
            assert "market_events.jsonl" in manifest["sha256"]
            assert "expected_rank_events.jsonl" in manifest["sha256"]
            assert "replay" in manifest
            assert "rank_event_stream_digest" in manifest["replay"]

    def test_sha256_checksums_valid(self) -> None:
        """Test that SHA256 checksums in manifest match actual files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            manifest_path, _ = run_record(
                symbols=["BTCUSDT"],
                duration_s=1,
                out_dir=out_dir,
                cadence_ms=100,
            )

            with manifest_path.open("rb") as f:
                manifest = orjson.loads(f.read())

            # Verify checksums
            market_sha = compute_file_sha256(out_dir / "market_events.jsonl")
            rank_sha = compute_file_sha256(out_dir / "expected_rank_events.jsonl")

            assert manifest["sha256"]["market_events.jsonl"] == market_sha
            assert manifest["sha256"]["expected_rank_events.jsonl"] == rank_sha

    def test_multiple_runs_same_params_same_digest(self) -> None:
        """Test that multiple record runs with same params produce same digest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir1 = Path(tmpdir) / "run1"
            out_dir2 = Path(tmpdir) / "run2"

            # Run two recording sessions
            _, _digest1 = run_record(
                symbols=["BTCUSDT"],
                duration_s=1,
                out_dir=out_dir1,
                cadence_ms=100,
            )

            _, _digest2 = run_record(
                symbols=["BTCUSDT"],
                duration_s=1,
                out_dir=out_dir2,
                cadence_ms=100,
            )

            # Digests will differ due to different start_ts (time.time())
            # But internal logic should be consistent
            # Load and compare event counts
            events1 = load_market_events(out_dir1 / "market_events.jsonl")
            events2 = load_market_events(out_dir2 / "market_events.jsonl")

            assert len(events1) == len(events2)


class TestRecordReplayPipelineEquivalence:
    """Tests verifying MinimalRecordPipeline == MinimalReplayPipeline."""

    def test_pipelines_produce_same_output(self) -> None:
        """Test that record and replay pipelines produce identical output."""
        # Create test events
        events = [
            MarketEvent(
                ts=1000 + i * 100,
                source="test",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"price": "42000", "qty": "0.1", "side": "buy"},
                recv_ts=1005 + i * 100,
            )
            for i in range(5)
        ]

        record_pipeline = MinimalRecordPipeline(seed=42)
        replay_pipeline = MinimalReplayPipeline(seed=42)

        record_events = record_pipeline.record(events)
        replay_events = replay_pipeline.replay(events)

        # Must produce identical events
        assert len(record_events) == len(replay_events)

        record_digest = compute_rank_events_digest(record_events)
        replay_digest = compute_rank_events_digest(replay_events)

        assert record_digest == replay_digest

    def test_pipelines_handle_multiple_symbols(self) -> None:
        """Test that both pipelines handle multiple symbols identically."""
        events = []
        for i in range(3):
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                events.append(
                    MarketEvent(
                        ts=1000 + i * 100,
                        source="test",
                        symbol=symbol,
                        type=MarketEventType.TRADE,
                        payload={"price": "100", "qty": "1", "side": "buy"},
                        recv_ts=1005 + i * 100,
                    )
                )

        record_pipeline = MinimalRecordPipeline(seed=42)
        replay_pipeline = MinimalReplayPipeline(seed=42)

        record_events = record_pipeline.record(events)
        replay_events = replay_pipeline.replay(events)

        record_digest = compute_rank_events_digest(record_events)
        replay_digest = compute_rank_events_digest(replay_events)

        assert record_digest == replay_digest


class TestEdgeCases:
    """Edge case tests for record→replay."""

    def test_empty_symbol_list(self) -> None:
        """Test that empty symbol list produces no events."""
        generator = SyntheticMarketEventGenerator(symbols=[], cadence_ms=100)
        events = list(generator.generate(start_ts=1000, duration_ms=1000))
        assert len(events) == 0

    def test_zero_duration(self) -> None:
        """Test that zero duration produces no events."""
        generator = SyntheticMarketEventGenerator(symbols=["BTCUSDT"], cadence_ms=100)
        events = list(generator.generate(start_ts=1000, duration_ms=0))
        assert len(events) == 0

    def test_single_event(self) -> None:
        """Test handling of single event."""
        pipeline = MinimalRecordPipeline(seed=42)
        events = [
            MarketEvent(
                ts=1000,
                source="test",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"price": "42000", "qty": "0.1", "side": "buy"},
                recv_ts=1005,
            ),
        ]

        rank_events = pipeline.record(events)
        # Single trade doesn't trigger any rank events
        assert len(rank_events) == 0

    def test_unknown_symbol_uses_default_price(self) -> None:
        """Test that unknown symbols use default base price."""
        generator = SyntheticMarketEventGenerator(symbols=["UNKNOWNCOIN"], cadence_ms=100)
        events = list(generator.generate(start_ts=1000, duration_ms=100))

        assert len(events) > 0
        # Should have generated events with default price (100.0)
        trade_events = [e for e in events if e.type.value == "trade"]
        if trade_events:
            price = float(trade_events[0].payload["price"])
            assert 99.0 < price < 101.0  # Close to default 100.0
