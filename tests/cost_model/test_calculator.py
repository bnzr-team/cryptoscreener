"""Tests for cost model calculator."""

from cryptoscreener.cost_model import (
    CostCalculator,
    CostModelConfig,
    compute_impact_bps,
    compute_spread_bps,
)
from cryptoscreener.cost_model.calculator import (
    OrderbookLevel,
    OrderbookSnapshot,
    Profile,
)


class TestComputeSpreadBps:
    """Tests for compute_spread_bps function."""

    def test_basic_spread(self) -> None:
        """Test basic spread calculation."""
        # Mid = 100.05, spread = 0.1, spread_bps = 0.1/100.05*10000 ≈ 9.995
        result = compute_spread_bps(bid=100.0, ask=100.1)
        assert 9.9 < result < 10.1

    def test_tight_spread(self) -> None:
        """Test tight spread (1 bps)."""
        # Mid = 100.005, spread = 0.01
        result = compute_spread_bps(bid=100.0, ask=100.01)
        assert 0.9 < result < 1.1

    def test_wide_spread(self) -> None:
        """Test wide spread."""
        # Mid = 100.5, spread = 1.0, spread_bps = 1.0/100.5*10000 ≈ 99.5
        result = compute_spread_bps(bid=100.0, ask=101.0)
        assert 99.0 < result < 100.0

    def test_zero_bid(self) -> None:
        """Test with zero bid."""
        result = compute_spread_bps(bid=0.0, ask=100.0)
        assert result == 0.0

    def test_zero_ask(self) -> None:
        """Test with zero ask."""
        result = compute_spread_bps(bid=100.0, ask=0.0)
        assert result == 0.0

    def test_negative_prices(self) -> None:
        """Test with negative prices."""
        result = compute_spread_bps(bid=-100.0, ask=100.0)
        assert result == 0.0


class TestComputeImpactBps:
    """Tests for compute_impact_bps function."""

    def test_no_impact_small_order(self) -> None:
        """Test that small orders have minimal impact."""
        book = OrderbookSnapshot(
            bids=[OrderbookLevel(99.9, 100), OrderbookLevel(99.8, 100)],
            asks=[OrderbookLevel(100.1, 100), OrderbookLevel(100.2, 100)],
            mid=100.0,
        )
        # Small order fills entirely at first level
        result = compute_impact_bps(book, clip_size_usd=100, side="buy")
        # Avg price ≈ 100.1, slippage from mid = 0.1, impact = 0.1/100*10000 = 10 bps
        assert 9.0 < result < 11.0

    def test_impact_larger_order(self) -> None:
        """Test that larger orders walk through more levels."""
        book = OrderbookSnapshot(
            bids=[OrderbookLevel(99.9, 10), OrderbookLevel(99.8, 10)],
            asks=[OrderbookLevel(100.1, 10), OrderbookLevel(100.2, 10)],
            mid=100.0,
        )
        # Order of $1500 needs to walk through multiple levels
        # First level: 10 * 100.1 = $1001
        # Second level: remaining fills at 100.2
        result = compute_impact_bps(book, clip_size_usd=1500, side="buy")
        # Should be higher than single level
        assert result > 10.0

    def test_impact_sell_side(self) -> None:
        """Test impact for sell orders."""
        book = OrderbookSnapshot(
            bids=[OrderbookLevel(99.9, 100), OrderbookLevel(99.8, 100)],
            asks=[OrderbookLevel(100.1, 100), OrderbookLevel(100.2, 100)],
            mid=100.0,
        )
        result = compute_impact_bps(book, clip_size_usd=100, side="sell")
        # Selling at 99.9, slippage = 100 - 99.9 = 0.1
        assert 9.0 < result < 11.0

    def test_empty_orderbook(self) -> None:
        """Test with empty orderbook."""
        book = OrderbookSnapshot(bids=[], asks=[], mid=100.0)
        result = compute_impact_bps(book, clip_size_usd=100, side="buy")
        assert result == 100.0  # Returns max_bps

    def test_zero_clip_size(self) -> None:
        """Test with zero clip size."""
        book = OrderbookSnapshot(
            bids=[OrderbookLevel(99.9, 100)],
            asks=[OrderbookLevel(100.1, 100)],
            mid=100.0,
        )
        result = compute_impact_bps(book, clip_size_usd=0, side="buy")
        assert result == 0.0

    def test_max_bps_clipping(self) -> None:
        """Test that impact is clipped to max_bps."""
        book = OrderbookSnapshot(
            bids=[OrderbookLevel(90.0, 1)],  # Very thin, far from mid
            asks=[OrderbookLevel(110.0, 1)],
            mid=100.0,
        )
        result = compute_impact_bps(book, clip_size_usd=1000, side="buy", max_bps=50.0)
        assert result <= 50.0


class TestCostCalculator:
    """Tests for CostCalculator class."""

    def test_default_config(self) -> None:
        """Test calculator with default config."""
        calc = CostCalculator()
        assert calc.get_fees_bps(Profile.A) == 2.0
        assert calc.get_fees_bps(Profile.B) == 4.0

    def test_custom_config(self) -> None:
        """Test calculator with custom config."""
        config = CostModelConfig(fees_bps_a=1.5, fees_bps_b=3.5)
        calc = CostCalculator(config)
        assert calc.get_fees_bps(Profile.A) == 1.5
        assert calc.get_fees_bps(Profile.B) == 3.5

    def test_compute_clip_size_scalping(self) -> None:
        """Test clip size calculation for scalping style."""
        calc = CostCalculator()
        # k=0.01 for scalping
        clip = calc.compute_clip_size_usd(usd_volume_60s=100000, style="scalping")
        assert clip == 1000.0  # 0.01 * 100000

    def test_compute_clip_size_intraday(self) -> None:
        """Test clip size calculation for intraday style."""
        calc = CostCalculator()
        # k=0.03 for intraday
        clip = calc.compute_clip_size_usd(usd_volume_60s=100000, style="intraday")
        assert clip == 3000.0  # 0.03 * 100000

    def test_compute_costs_basic(self) -> None:
        """Test basic cost computation."""
        calc = CostCalculator()
        costs = calc.compute_costs(
            bid=100.0,
            ask=100.1,
            profile=Profile.A,
        )

        # Spread ≈ 10 bps
        assert 9.0 < costs.spread_bps < 11.0
        # Fees = 2.0 bps for Profile A
        assert costs.fees_bps == 2.0
        # No orderbook, no impact
        assert costs.impact_bps == 0.0
        # Total = spread + fees
        assert 11.0 < costs.total_bps < 13.0

    def test_compute_costs_with_orderbook(self) -> None:
        """Test cost computation with orderbook."""
        calc = CostCalculator()
        book = OrderbookSnapshot(
            bids=[OrderbookLevel(99.9, 100)],
            asks=[OrderbookLevel(100.1, 100)],
            mid=100.0,
        )

        costs = calc.compute_costs(
            bid=100.0,
            ask=100.1,
            profile=Profile.B,
            orderbook=book,
            usd_volume_60s=100000,
            style="scalping",
        )

        # Spread ≈ 10 bps
        assert 9.0 < costs.spread_bps < 11.0
        # Fees = 4.0 bps for Profile B
        assert costs.fees_bps == 4.0
        # Impact should be calculated
        assert costs.impact_bps > 0.0
        # Clip size = 0.01 * 100000 = 1000
        assert costs.clip_size_usd == 1000.0

    def test_compute_costs_both_profiles(self) -> None:
        """Test computing costs for both profiles."""
        calc = CostCalculator()
        costs_dict = calc.compute_costs_both_profiles(
            bid=100.0,
            ask=100.1,
        )

        assert Profile.A in costs_dict
        assert Profile.B in costs_dict

        # Profile B should have higher fees
        assert costs_dict[Profile.B].fees_bps > costs_dict[Profile.A].fees_bps


class TestEdgeCases:
    """Edge case tests."""

    def test_very_small_spread(self) -> None:
        """Test with very small spread (sub-bps)."""
        result = compute_spread_bps(bid=100.0, ask=100.001)
        assert 0.0 < result < 0.1

    def test_large_price_values(self) -> None:
        """Test with large price values (like BTC)."""
        result = compute_spread_bps(bid=50000.0, ask=50010.0)
        # Spread = 10/50005 * 10000 ≈ 2 bps
        assert 1.9 < result < 2.1

    def test_small_price_values(self) -> None:
        """Test with small price values (like SHIB)."""
        result = compute_spread_bps(bid=0.00001, ask=0.0000101)
        # Spread should be calculable
        assert result > 0.0

    def test_equal_bid_ask(self) -> None:
        """Test with equal bid and ask (zero spread)."""
        result = compute_spread_bps(bid=100.0, ask=100.0)
        assert result == 0.0
