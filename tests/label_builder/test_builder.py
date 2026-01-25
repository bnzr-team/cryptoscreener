"""Tests for label builder module."""

from cryptoscreener.cost_model.calculator import Profile
from cryptoscreener.label_builder import (
    Horizon,
    LabelBuilder,
    LabelBuilderConfig,
    ToxicityConfig,
)
from cryptoscreener.label_builder.builder import HORIZON_MS, PricePoint


class TestHorizonConstants:
    """Tests for horizon constants."""

    def test_horizon_ms_values(self) -> None:
        """Test horizon millisecond values."""
        assert HORIZON_MS[Horizon.H_30S] == 30_000
        assert HORIZON_MS[Horizon.H_2M] == 120_000
        assert HORIZON_MS[Horizon.H_5M] == 300_000


class TestLabelBuilderConfig:
    """Tests for LabelBuilderConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LabelBuilderConfig()
        assert config.x_bps_30s_a == 5.0
        assert config.x_bps_30s_b == 8.0
        assert config.x_bps_2m_a == 10.0
        assert config.x_bps_2m_b == 15.0
        assert config.x_bps_5m_a == 15.0
        assert config.x_bps_5m_b == 20.0
        assert config.spread_max_bps == 10.0
        assert config.impact_max_bps == 20.0

    def test_get_x_bps(self) -> None:
        """Test get_x_bps method."""
        config = LabelBuilderConfig()
        assert config.get_x_bps(Horizon.H_30S, Profile.A) == 5.0
        assert config.get_x_bps(Horizon.H_30S, Profile.B) == 8.0
        assert config.get_x_bps(Horizon.H_2M, Profile.A) == 10.0
        assert config.get_x_bps(Horizon.H_5M, Profile.B) == 20.0


class TestMFECalculation:
    """Tests for Maximum Favorable Excursion calculation."""

    def test_mfe_basic(self) -> None:
        """Test basic MFE calculation."""
        builder = LabelBuilder()
        entry_price = 100.0
        future_prices = [
            PricePoint(ts=1000, mid=100.5),
            PricePoint(ts=2000, mid=101.0),
            PricePoint(ts=3000, mid=100.8),
        ]

        # Max price = 101.0, MFE = (101 - 100) / 100 * 10000 = 100 bps
        mfe = builder.compute_mfe_bps(entry_price, future_prices, Horizon.H_30S, entry_ts=0)
        assert 99.0 < mfe < 101.0

    def test_mfe_respects_horizon(self) -> None:
        """Test that MFE respects horizon window."""
        builder = LabelBuilder()
        entry_price = 100.0
        future_prices = [
            PricePoint(ts=10_000, mid=101.0),  # Within 30s
            PricePoint(ts=50_000, mid=105.0),  # Beyond 30s
        ]

        # MFE at 30s should only see first price
        mfe_30s = builder.compute_mfe_bps(entry_price, future_prices, Horizon.H_30S, entry_ts=0)
        # MFE at 2m should see both
        mfe_2m = builder.compute_mfe_bps(entry_price, future_prices, Horizon.H_2M, entry_ts=0)

        assert mfe_30s == 100.0  # (101 - 100) / 100 * 10000
        assert mfe_2m == 500.0   # (105 - 100) / 100 * 10000

    def test_mfe_no_favorable_move(self) -> None:
        """Test MFE when price only goes down."""
        builder = LabelBuilder()
        entry_price = 100.0
        future_prices = [
            PricePoint(ts=1000, mid=99.0),
            PricePoint(ts=2000, mid=98.0),
        ]

        mfe = builder.compute_mfe_bps(entry_price, future_prices, Horizon.H_30S, entry_ts=0)
        assert mfe == 0.0

    def test_mfe_empty_future_prices(self) -> None:
        """Test MFE with no future prices."""
        builder = LabelBuilder()
        mfe = builder.compute_mfe_bps(100.0, [], Horizon.H_30S, entry_ts=0)
        assert mfe == 0.0


class TestMAECalculation:
    """Tests for Maximum Adverse Excursion calculation."""

    def test_mae_basic(self) -> None:
        """Test basic MAE calculation."""
        builder = LabelBuilder()
        entry_price = 100.0
        future_prices = [
            PricePoint(ts=1000, mid=99.5),
            PricePoint(ts=2000, mid=99.0),
            PricePoint(ts=3000, mid=99.2),
        ]

        # Min price = 99.0, MAE = (100 - 99) / 100 * 10000 = 100 bps
        mae = builder.compute_mae_bps(entry_price, future_prices, Horizon.H_30S, entry_ts=0)
        assert 99.0 < mae < 101.0

    def test_mae_no_adverse_move(self) -> None:
        """Test MAE when price only goes up."""
        builder = LabelBuilder()
        entry_price = 100.0
        future_prices = [
            PricePoint(ts=1000, mid=101.0),
            PricePoint(ts=2000, mid=102.0),
        ]

        mae = builder.compute_mae_bps(entry_price, future_prices, Horizon.H_30S, entry_ts=0)
        assert mae == 0.0


class TestToxicityLabel:
    """Tests for toxicity label calculation."""

    def test_toxic_event(self) -> None:
        """Test detection of toxic event."""
        config = LabelBuilderConfig(
            toxicity=ToxicityConfig(tau_ms=30_000, threshold_bps=10.0)
        )
        builder = LabelBuilder(config)

        entry_price = 100.0
        # Price drops by 15 bps within tau window
        future_prices = [
            PricePoint(ts=10_000, mid=99.85),  # 15 bps adverse
        ]

        y_toxic, severity = builder.compute_toxicity_label(
            entry_price, future_prices, entry_ts=0
        )
        assert y_toxic == 1
        assert 14.0 < severity < 16.0

    def test_non_toxic_event(self) -> None:
        """Test non-toxic event (small adverse move)."""
        config = LabelBuilderConfig(
            toxicity=ToxicityConfig(tau_ms=30_000, threshold_bps=10.0)
        )
        builder = LabelBuilder(config)

        entry_price = 100.0
        # Price drops by only 5 bps
        future_prices = [
            PricePoint(ts=10_000, mid=99.95),  # 5 bps adverse
        ]

        y_toxic, severity = builder.compute_toxicity_label(
            entry_price, future_prices, entry_ts=0
        )
        assert y_toxic == 0
        assert 4.0 < severity < 6.0

    def test_toxic_outside_window(self) -> None:
        """Test that toxic move outside tau window is ignored."""
        config = LabelBuilderConfig(
            toxicity=ToxicityConfig(tau_ms=30_000, threshold_bps=10.0)
        )
        builder = LabelBuilder(config)

        entry_price = 100.0
        # Price drops significantly but after tau window
        future_prices = [
            PricePoint(ts=10_000, mid=99.98),   # 2 bps within window
            PricePoint(ts=50_000, mid=95.0),    # Big drop but outside tau
        ]

        y_toxic, _severity = builder.compute_toxicity_label(
            entry_price, future_prices, entry_ts=0
        )
        assert y_toxic == 0


class TestTradeabilityLabel:
    """Tests for tradeability label calculation."""

    def test_tradeable_label(self) -> None:
        """Test generation of tradeable label."""
        config = LabelBuilderConfig(
            x_bps_30s_a=5.0,
            spread_max_bps=20.0,  # Wide gate
            impact_max_bps=50.0,
        )
        builder = LabelBuilder(config)

        # bid=100, ask=100.1 -> mid = 100.05
        # future price = 100.25 -> MFE = (100.25 - 100.05) / 100.05 * 10000 ≈ 20 bps
        future_prices = [
            PricePoint(ts=10_000, mid=100.25),
        ]

        row = builder.build_label_row(
            ts=0,
            symbol="BTCUSDT",
            bid=100.0,
            ask=100.1,  # ~10 bps spread
            future_prices=future_prices,
        )

        # Check 30s Profile A
        label = row.tradeability[(Horizon.H_30S, Profile.A)]
        # mid = 100.05, future = 100.25
        # MFE = (100.25 - 100.05) / 100.05 * 10000 ≈ 20 bps
        # cost = spread(10) + fees(2) = 12 bps
        # net_edge = 20 - 12 = 8 bps > x_bps(5)
        assert 19.0 < label.mfe_bps < 21.0  # Approximately 20 bps
        assert label.gates_passed  # spread < 20 bps gate
        assert label.i_tradeable == 1

    def test_not_tradeable_insufficient_edge(self) -> None:
        """Test non-tradeable due to insufficient net edge."""
        config = LabelBuilderConfig(
            x_bps_30s_a=15.0,  # High threshold
            spread_max_bps=20.0,
        )
        builder = LabelBuilder(config)

        # Small MFE of 10 bps
        future_prices = [
            PricePoint(ts=10_000, mid=100.1),
        ]

        row = builder.build_label_row(
            ts=0,
            symbol="BTCUSDT",
            bid=100.0,
            ask=100.1,
            future_prices=future_prices,
        )

        label = row.tradeability[(Horizon.H_30S, Profile.A)]
        # MFE = 10 bps, cost = ~12 bps, net_edge = -2 bps < x_bps(15)
        assert label.i_tradeable == 0

    def test_not_tradeable_gate_fail(self) -> None:
        """Test non-tradeable due to gate failure."""
        config = LabelBuilderConfig(
            x_bps_30s_a=5.0,
            spread_max_bps=5.0,  # Tight gate
        )
        builder = LabelBuilder(config)

        future_prices = [
            PricePoint(ts=10_000, mid=102.0),  # Big MFE
        ]

        row = builder.build_label_row(
            ts=0,
            symbol="BTCUSDT",
            bid=100.0,
            ask=100.2,  # 20 bps spread > 5 bps gate
            future_prices=future_prices,
        )

        label = row.tradeability[(Horizon.H_30S, Profile.A)]
        assert not label.gates_passed
        assert "SPREAD_GATE" in label.gate_failures
        assert label.i_tradeable == 0  # Gate fail overrides edge


class TestLabelRowConversion:
    """Tests for LabelRow to flat dict conversion."""

    def test_flat_dict_keys(self) -> None:
        """Test that flat dict has all expected keys."""
        builder = LabelBuilder()
        future_prices = [PricePoint(ts=10_000, mid=100.5)]

        row = builder.build_label_row(
            ts=0,
            symbol="BTCUSDT",
            bid=100.0,
            ask=100.1,
            future_prices=future_prices,
        )

        flat = builder.label_row_to_flat_dict(row)

        # Base keys
        assert "ts" in flat
        assert "symbol" in flat
        assert "mid_price" in flat
        assert "spread_bps" in flat
        assert "y_toxic" in flat
        assert "severity_toxic_bps" in flat

        # Tradeability keys for each horizon/profile
        for horizon in ["30s", "2m", "5m"]:
            for profile in ["a", "b"]:
                prefix = f"{horizon}_{profile}"
                assert f"i_tradeable_{prefix}" in flat
                assert f"mfe_bps_{prefix}" in flat
                assert f"cost_bps_{prefix}" in flat
                assert f"net_edge_bps_{prefix}" in flat
                assert f"gates_passed_{prefix}" in flat

    def test_flat_dict_values(self) -> None:
        """Test that flat dict values are correct types."""
        builder = LabelBuilder()
        future_prices = [PricePoint(ts=10_000, mid=100.5)]

        row = builder.build_label_row(
            ts=1234567890,
            symbol="ETHUSDT",
            bid=100.0,
            ask=100.1,
            future_prices=future_prices,
        )

        flat = builder.label_row_to_flat_dict(row)

        assert flat["ts"] == 1234567890
        assert flat["symbol"] == "ETHUSDT"
        assert isinstance(flat["mid_price"], float)
        assert isinstance(flat["y_toxic"], int)
        assert flat["y_toxic"] in (0, 1)


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_entry_price(self) -> None:
        """Test with zero entry price."""
        builder = LabelBuilder()
        mfe = builder.compute_mfe_bps(0.0, [PricePoint(ts=1000, mid=100)], Horizon.H_30S, 0)
        assert mfe == 0.0

    def test_negative_entry_price(self) -> None:
        """Test with negative entry price."""
        builder = LabelBuilder()
        mfe = builder.compute_mfe_bps(-100.0, [PricePoint(ts=1000, mid=100)], Horizon.H_30S, 0)
        assert mfe == 0.0

    def test_label_row_zero_prices(self) -> None:
        """Test label row with zero bid/ask."""
        builder = LabelBuilder()
        row = builder.build_label_row(
            ts=0,
            symbol="TEST",
            bid=0.0,
            ask=0.0,
            future_prices=[],
        )
        assert row.mid_price == 0.0
        assert row.spread_bps == 0.0
