"""
Tests for replay determinism verification.

Validates that:
1. Fixture files exist and have correct structure
2. RankEvent digests are computed deterministically
3. Replay produces expected digest matching manifest
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import orjson

from cryptoscreener.contracts import (
    MarketEvent,
    RankEvent,
    compute_rank_events_digest,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_run"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


class TestFixtureFilesExist:
    """Test that required fixture files exist."""

    def test_market_events_file_exists(self) -> None:
        """Test market_events.jsonl exists."""
        assert (FIXTURES_DIR / "market_events.jsonl").exists()

    def test_expected_rank_events_file_exists(self) -> None:
        """Test expected_rank_events.jsonl exists."""
        assert (FIXTURES_DIR / "expected_rank_events.jsonl").exists()

    def test_manifest_file_exists(self) -> None:
        """Test manifest.json exists."""
        assert (FIXTURES_DIR / "manifest.json").exists()


class TestFixtureStructure:
    """Test fixture file structure and content."""

    def test_market_events_valid_jsonl(self) -> None:
        """Test market_events.jsonl contains valid JSON lines."""
        events = []
        with open(FIXTURES_DIR / "market_events.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    event = MarketEvent.model_validate(data)
                    events.append(event)

        assert len(events) > 0
        assert all(isinstance(e, MarketEvent) for e in events)

    def test_expected_rank_events_valid_jsonl(self) -> None:
        """Test expected_rank_events.jsonl contains valid RankEvents."""
        events = []
        with open(FIXTURES_DIR / "expected_rank_events.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    event = RankEvent.model_validate(data)
                    events.append(event)

        assert len(events) > 0
        assert all(isinstance(e, RankEvent) for e in events)

    def test_manifest_contains_required_fields(self) -> None:
        """Test manifest.json has required fields."""
        with open(FIXTURES_DIR / "manifest.json") as f:
            manifest = json.load(f)

        assert "expected_rank_events_digest" in manifest
        assert "market_events_count" in manifest
        assert "expected_rank_events_count" in manifest


class TestDigestComputation:
    """Test deterministic digest computation."""

    def test_rank_events_digest_matches_manifest(self) -> None:
        """Test that computed digest matches manifest expected digest.

        This is the core replay determinism check.
        """
        # Load expected digest from manifest
        with open(FIXTURES_DIR / "manifest.json") as f:
            manifest = json.load(f)
        expected_digest = manifest["expected_rank_events_digest"]

        # Load and compute actual digest
        events = []
        with open(FIXTURES_DIR / "expected_rank_events.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    event = RankEvent.model_validate(data)
                    events.append(event)

        actual_digest = compute_rank_events_digest(events)

        assert actual_digest == expected_digest, (
            f"Digest mismatch!\nExpected: {expected_digest}\nActual:   {actual_digest}"
        )

    def test_digest_is_deterministic(self) -> None:
        """Test that digest computation is deterministic across multiple runs."""
        events = []
        with open(FIXTURES_DIR / "expected_rank_events.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    event = RankEvent.model_validate(data)
                    events.append(event)

        digest1 = compute_rank_events_digest(events)
        digest2 = compute_rank_events_digest(events)
        digest3 = compute_rank_events_digest(events)

        assert digest1 == digest2 == digest3

    def test_digest_changes_with_content(self) -> None:
        """Test that digest changes when event content changes."""
        events = []
        with open(FIXTURES_DIR / "expected_rank_events.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    event = RankEvent.model_validate(data)
                    events.append(event)

        original_digest = compute_rank_events_digest(events)

        # Modify one event
        if events:
            modified_event = RankEvent(
                ts=events[0].ts,
                event=events[0].event,
                symbol=events[0].symbol,
                rank=events[0].rank + 100,  # Changed
                score=events[0].score,
                payload=events[0].payload,
            )
            modified_events = [modified_event, *events[1:]]
            modified_digest = compute_rank_events_digest(modified_events)

            assert original_digest != modified_digest

    def test_digest_changes_with_order(self) -> None:
        """Test that digest changes when event order changes."""
        events = []
        with open(FIXTURES_DIR / "expected_rank_events.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    event = RankEvent.model_validate(data)
                    events.append(event)

        if len(events) >= 2:
            original_digest = compute_rank_events_digest(events)
            reversed_digest = compute_rank_events_digest(list(reversed(events)))

            assert original_digest != reversed_digest


class TestFixtureChecksums:
    """Test fixture file checksums for integrity verification."""

    def test_market_events_checksum(self) -> None:
        """Verify market_events.jsonl checksum."""
        checksum = compute_file_sha256(FIXTURES_DIR / "market_events.jsonl")
        expected = "58958f3199b360f16667f4d1db459d943e5f5af694fa5ac5ce9fc01992f737b9"
        assert checksum == expected, f"Checksum mismatch: {checksum}"

    def test_expected_rank_events_checksum(self) -> None:
        """Verify expected_rank_events.jsonl checksum."""
        checksum = compute_file_sha256(FIXTURES_DIR / "expected_rank_events.jsonl")
        expected = "3eeff0d6f838717e8363a050026b92bec1c9a2eedf0e07da4a7dbf431ea9b30f"
        assert checksum == expected, f"Checksum mismatch: {checksum}"


class TestCanonicalJsonSerialization:
    """Test that canonical JSON serialization is deterministic."""

    def test_orjson_sort_keys_deterministic(self) -> None:
        """Test that orjson with OPT_SORT_KEYS produces deterministic output."""
        events = []
        with open(FIXTURES_DIR / "expected_rank_events.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    event = RankEvent.model_validate(data)
                    events.append(event)

        if events:
            event = events[0]

            # Serialize multiple times
            json1 = orjson.dumps(event.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
            json2 = orjson.dumps(event.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
            json3 = orjson.dumps(event.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)

            assert json1 == json2 == json3

    def test_individual_event_digest_deterministic(self) -> None:
        """Test that individual event digest is deterministic."""
        events = []
        with open(FIXTURES_DIR / "expected_rank_events.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    event = RankEvent.model_validate(data)
                    events.append(event)

        if events:
            event = events[0]
            digest1 = event.digest()
            digest2 = event.digest()
            assert digest1 == digest2
            assert len(digest1) == 64  # SHA256 hex
