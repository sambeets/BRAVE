import pytest

from main import compute_sipue

pytestmark = pytest.mark.unit


def test_zero_system_benefit_recovers_pue():
    sipue, system_credit_mwh = compute_sipue(
        E_FAC=120.0,
        E_IT=100.0,
        net_system_benefit_eur=0.0,
        reference_price_eur_per_mwh=120.0,
    )

    assert sipue == pytest.approx(1.2)
    assert system_credit_mwh == pytest.approx(0.0)


def test_positive_system_benefit_lowers_sipue():
    sipue, system_credit_mwh = compute_sipue(
        E_FAC=120.0,
        E_IT=100.0,
        net_system_benefit_eur=2400.0,
        reference_price_eur_per_mwh=120.0,
    )

    assert system_credit_mwh == pytest.approx(20.0)
    assert sipue == pytest.approx(1.0)


def test_delay_factor_one_reproduces_current_sipue():
    sipue, system_credit_mwh = compute_sipue(
        E_FAC=120.0,
        E_IT=100.0,
        net_system_benefit_eur=2400.0,
        reference_price_eur_per_mwh=120.0,
        benefit_availability_factor=1.0,
    )

    assert system_credit_mwh == pytest.approx(20.0)
    assert sipue == pytest.approx(1.0)


def test_delay_factor_zero_removes_system_credit():
    sipue, system_credit_mwh = compute_sipue(
        E_FAC=120.0,
        E_IT=100.0,
        net_system_benefit_eur=2400.0,
        reference_price_eur_per_mwh=120.0,
        benefit_availability_factor=0.0,
    )

    assert system_credit_mwh == pytest.approx(0.0)
    assert sipue == pytest.approx(1.2)


def test_lower_delay_factor_weakens_sipue_improvement_monotonically():
    full, _ = compute_sipue(
        E_FAC=120.0,
        E_IT=100.0,
        net_system_benefit_eur=2400.0,
        reference_price_eur_per_mwh=120.0,
        benefit_availability_factor=1.0,
    )
    half, _ = compute_sipue(
        E_FAC=120.0,
        E_IT=100.0,
        net_system_benefit_eur=2400.0,
        reference_price_eur_per_mwh=120.0,
        benefit_availability_factor=0.5,
    )
    none, _ = compute_sipue(
        E_FAC=120.0,
        E_IT=100.0,
        net_system_benefit_eur=2400.0,
        reference_price_eur_per_mwh=120.0,
        benefit_availability_factor=0.0,
    )

    assert full < half < none


def test_negative_system_benefit_raises_sipue():
    sipue, system_credit_mwh = compute_sipue(
        E_FAC=120.0,
        E_IT=100.0,
        net_system_benefit_eur=-1200.0,
        reference_price_eur_per_mwh=120.0,
    )

    assert system_credit_mwh == pytest.approx(-10.0)
    assert sipue == pytest.approx(1.3)


def test_clip_nonnegative_prevents_negative_adjusted_energy():
    sipue, system_credit_mwh = compute_sipue(
        E_FAC=50.0,
        E_IT=100.0,
        net_system_benefit_eur=12000.0,
        reference_price_eur_per_mwh=120.0,
        clip_nonnegative=True,
    )

    assert system_credit_mwh == pytest.approx(100.0)
    assert sipue == pytest.approx(0.0)


def test_without_clipping_adjusted_energy_can_be_negative():
    sipue, system_credit_mwh = compute_sipue(
        E_FAC=50.0,
        E_IT=100.0,
        net_system_benefit_eur=12000.0,
        reference_price_eur_per_mwh=120.0,
        clip_nonnegative=False,
    )

    assert system_credit_mwh == pytest.approx(100.0)
    assert sipue == pytest.approx(-0.5)


def test_requires_positive_it_energy():
    with pytest.raises(ValueError, match="E_IT must be positive"):
        compute_sipue(
            E_FAC=120.0,
            E_IT=0.0,
            net_system_benefit_eur=0.0,
            reference_price_eur_per_mwh=120.0,
        )


def test_requires_positive_reference_price():
    with pytest.raises(ValueError, match="reference_price_eur_per_mwh must be positive"):
        compute_sipue(
            E_FAC=120.0,
            E_IT=100.0,
            net_system_benefit_eur=0.0,
            reference_price_eur_per_mwh=0.0,
        )
