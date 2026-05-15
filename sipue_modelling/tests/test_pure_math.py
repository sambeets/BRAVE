import math
import pytest
from main import nanmean, spearman_rank_corr, _pretty_type_label, _fs, PLOT_FONT_SCALE

pytestmark = pytest.mark.unit


# ---------- nanmean ----------

class TestNanmean:
    def test_basic(self):
        assert nanmean([1.0, 2.0, 3.0]) == 2.0

    def test_with_nan(self):
        assert nanmean([1.0, float("nan"), 3.0]) == 2.0

    def test_all_nan(self):
        assert nanmean([float("nan"), float("nan")]) == 0.0

    def test_empty(self):
        assert nanmean([]) == 0.0

    def test_single(self):
        assert nanmean([42.0]) == 42.0

    def test_negative(self):
        assert nanmean([-2.0, 4.0]) == pytest.approx(1.0)


# ---------- spearman_rank_corr ----------

class TestSpearmanRankCorr:
    def test_perfect_positive(self):
        assert spearman_rank_corr([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative(self):
        assert spearman_rank_corr([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0, abs=1e-9)

    def test_empty(self):
        assert math.isnan(spearman_rank_corr([], []))

    def test_mismatched_length(self):
        assert math.isnan(spearman_rank_corr([1, 2], [1, 2, 3]))

    def test_ties(self):
        result = spearman_rank_corr([1, 1, 2, 3], [1, 2, 3, 4])
        assert -1.0 <= result <= 1.0

    def test_constant_input(self):
        # den_a is 0 -> division leads to nan
        result = spearman_rank_corr([5, 5, 5], [1, 2, 3])
        assert math.isnan(result) or result == 0.0


# ---------- _pretty_type_label ----------

class TestPrettyTypeLabel:
    @pytest.mark.parametrize("input_type, expected", [
        ("hyperscale", "Hyperscale"),
        ("ai", "AI"),
        ("streaming", "Streaming"),
        ("colocation", "Colocation"),
    ])
    def test_known_types(self, input_type, expected):
        assert _pretty_type_label(input_type) == expected

    def test_unknown_type_fallback(self):
        assert _pretty_type_label("some_long_name") == "Some Long Name"


# ---------- _fs ----------

class TestFs:
    def test_basic(self):
        assert _fs(10) == int(round(10 * PLOT_FONT_SCALE))

    def test_returns_int(self):
        assert isinstance(_fs(13.5), int)
