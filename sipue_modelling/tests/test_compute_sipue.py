import math
import pytest
from main import compute_sipue

pytestmark = pytest.mark.unit


@pytest.fixture
def base_args():
    """Common arguments for compute_sipue tests."""
    return dict(
        E_FAC=1000.0,
        E_IT=800.0,
        credit_eur=50000.0,
        heat_rev_eur=30000.0,
        flex_rev_eur=20000.0,
        avg_heat_price=40.0,
        avg_flex_price=25.0,
        co2_cost_eur=5000.0,
        el_cost=60000.0,
        el_settlement_eur=55000.0,
        E_GRID=950.0,
        p_el_ref=[60.0] * 24,
        dc_load_series=[42.0] * 24,
        settle_lmp_series=[58.0] * 24,
        normalization="settle_lmp_dc",
        credit_price_mode="lmp",
        clip_nonnegative=True,
        price_floor_eur_per_mwh=5.0,
        include_co2_penalty=True,
        co2_scale=1.0,
        co2_ref_price_eur_per_mwh=120.0,
    )


class TestComputeSipueReturnShape:
    def test_returns_8_tuple(self, base_args):
        result = compute_sipue(**base_args)
        assert len(result) == 8

    def test_return_types(self, base_args):
        siPUE, credit_mwh, credit_heat, credit_flex, co2_mwh, price_used, price_source, floor_applied = compute_sipue(**base_args)
        assert isinstance(siPUE, float)
        assert isinstance(credit_mwh, float)
        assert isinstance(credit_heat, float)
        assert isinstance(credit_flex, float)
        assert isinstance(co2_mwh, float)
        assert isinstance(price_used, float)
        assert isinstance(price_source, str)
        assert isinstance(floor_applied, int)


class TestComputeSipueNormalization:
    def test_settle_lmp_dc(self, base_args):
        result = compute_sipue(**base_args)
        assert result[0] >= 0.0
        assert "settlement LMP" in result[6]

    def test_avg_paid(self, base_args):
        base_args["normalization"] = "avg_paid"
        result = compute_sipue(**base_args)
        assert "avg_paid" in result[6]
        # price_used = el_cost / E_FAC = 60000/1000 = 60
        assert result[5] == pytest.approx(60.0)

    def test_net_grid_cost(self, base_args):
        base_args["normalization"] = "net_grid_cost"
        result = compute_sipue(**base_args)
        assert "net_grid_cost" in result[6]
        # price_used = el_settlement_eur / E_GRID = 55000/950
        assert result[5] == pytest.approx(55000.0 / 950.0)

    def test_ref_price(self, base_args):
        base_args["normalization"] = "ref_price"
        result = compute_sipue(**base_args)
        assert "ref_price" in result[6]
        # price_used = avg(p_el_ref) = 60.0
        assert result[5] == pytest.approx(60.0)

    def test_invalid_normalization_raises(self, base_args):
        base_args["normalization"] = "invalid"
        with pytest.raises(ValueError):
            compute_sipue(**base_args)

    def test_settle_lmp_dc_missing_series_raises(self, base_args):
        base_args["dc_load_series"] = None
        with pytest.raises(ValueError):
            compute_sipue(**base_args)


class TestComputeSipueCreditMode:
    def test_lmp_mode(self, base_args):
        result = compute_sipue(**base_args)
        price_used = result[5]
        credit_heat = result[2]
        credit_flex = result[3]
        # In LMP mode, credits use price_used
        assert credit_heat == pytest.approx(30000.0 / price_used)
        assert credit_flex == pytest.approx(20000.0 / price_used)

    def test_service_price_mode(self, base_args):
        base_args["credit_price_mode"] = "service_price"
        result = compute_sipue(**base_args)
        credit_heat = result[2]
        credit_flex = result[3]
        # In service_price mode, credits use avg service prices
        assert credit_heat == pytest.approx(30000.0 / 40.0)
        assert credit_flex == pytest.approx(20000.0 / 25.0)


class TestComputeSipueClipping:
    def test_clip_nonnegative_true(self, base_args):
        base_args["heat_rev_eur"] = 500000.0
        base_args["flex_rev_eur"] = 500000.0
        result = compute_sipue(**base_args)
        assert result[0] >= 0.0

    def test_clip_nonnegative_false(self, base_args):
        base_args["heat_rev_eur"] = 500000.0
        base_args["flex_rev_eur"] = 500000.0
        base_args["clip_nonnegative"] = False
        result = compute_sipue(**base_args)
        assert isinstance(result[0], float)


class TestComputeSipueCo2:
    def test_co2_penalty_included(self, base_args):
        result_with = compute_sipue(**base_args)
        base_args["include_co2_penalty"] = False
        result_without = compute_sipue(**base_args)
        assert result_with[4] > 0.0
        assert result_without[4] == 0.0

    def test_co2_increases_sipue(self, base_args):
        base_args["include_co2_penalty"] = True
        sipue_with = compute_sipue(**base_args)[0]
        base_args["include_co2_penalty"] = False
        sipue_without = compute_sipue(**base_args)[0]
        # CO2 penalty adds to E_adj -> increases siPUE
        assert sipue_with >= sipue_without

    def test_price_floor_applied(self, base_args):
        base_args["normalization"] = "net_grid_cost"
        base_args["el_settlement_eur"] = 1.0  # Very small -> price below floor
        base_args["E_GRID"] = 1000.0
        result = compute_sipue(**base_args)
        assert result[7] == 1  # price_floor_applied


class TestComputeSipueDeterminism:
    def test_same_inputs_same_outputs(self, base_args):
        r1 = compute_sipue(**base_args)
        r2 = compute_sipue(**base_args)
        for a, b in zip(r1, r2):
            if isinstance(a, float):
                assert a == b
            else:
                assert a == b
