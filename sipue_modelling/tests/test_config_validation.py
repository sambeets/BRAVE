import pytest
from main import Config, _validate_locked_calibration, LOCKED_CALIBRATION

pytestmark = pytest.mark.unit


def test_default_config_passes_validation():
    cfg = Config()
    _validate_locked_calibration(cfg)  # Should not raise


def test_bad_value_always_raises():
    """_validate_locked_calibration always validates regardless of enforce flag."""
    cfg = Config(enforce_locked_calibration=False, capex_HX=99999.0)
    with pytest.raises(ValueError, match="Locked calibration mismatch"):
        _validate_locked_calibration(cfg)


@pytest.mark.parametrize("field, bad_value", [
    ("capex_HX", 99999.0),
    ("capex_FLEX", 0.0),
    ("capex_BAT_E", 1.0),
    ("capex_BAT_P", 1.0),
    ("fixed_eff_reduction", 0.99),
    ("it_price_multiplier", 1.0),
    ("heat_price_multiplier", 0.99),
    ("flex_price_multiplier", 0.99),
    ("sipue_credit_normalization", "bad_mode"),
    ("sipue_credit_price_mode", "bad_mode"),
])
def test_locked_calibration_rejects_bad_values(field, bad_value):
    cfg = Config(**{field: bad_value})
    with pytest.raises(ValueError, match="Locked calibration mismatch"):
        _validate_locked_calibration(cfg)


def test_locked_keys_exist_in_config():
    """All keys in LOCKED_CALIBRATION should be valid Config fields."""
    cfg = Config()
    for key in LOCKED_CALIBRATION:
        assert hasattr(cfg, key), f"LOCKED_CALIBRATION key '{key}' not found in Config"
