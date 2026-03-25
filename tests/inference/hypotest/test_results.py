"""Tests for hypothesis testing result containers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from everwillow.hypotest.results import (
    BandValues,
    ExpectedBands,
    HypoTestResult,
    ToyResult,
)
from everwillow.hypotest.results import (
    TestStatResult as TSResult,
)
from everwillow.hypotest.utils import cl_s, significance

# =============================================================================
# ExpectedBands Tests
# =============================================================================

BAND_NAMES = [
    "minus_2sigma",
    "minus_1sigma",
    "median",
    "plus_1sigma",
    "plus_2sigma",
]


def _make_bands(pnulls, palts):
    """Build ExpectedBands from parallel lists of pnull/palt values."""
    return ExpectedBands(
        null_pvalue=BandValues(**{n: jnp.array(p) for n, p in zip(BAND_NAMES, pnulls, strict=False)}),
        alt_pvalue=BandValues(**{n: jnp.array(p) for n, p in zip(BAND_NAMES, palts, strict=False)}),
        cl_s=BandValues(
            **{n: cl_s(jnp.array(pn), jnp.array(pa)) for n, pn, pa in zip(BAND_NAMES, pnulls, palts, strict=False)}
        ),
        null_sig=BandValues(**{n: significance(jnp.array(p)) for n, p in zip(BAND_NAMES, pnulls, strict=False)}),
        alt_sig=BandValues(**{n: significance(jnp.array(p)) for n, p in zip(BAND_NAMES, palts, strict=False)}),
    )


class TestExpectedBands:
    """Tests for ExpectedBands CLs and significance bands.

    Uses known QMu p-values for μ=2, σ=1, q_A=4.

    Band   | pnull      | palt     | CLs        | Z_null | Z_alt
    -2σ    | 3.167e-5   | 0.02275  | 0.001392   | 4.0    | 2.0
    -1σ    | 0.00135    | 0.15866  | 0.008510   | 3.0    | 1.0
    median | 0.02275    | 0.5      | 0.04550    | 2.0    | 0.0
    +1σ    | 0.15866    | 0.84134  | 0.18858    | 1.0    | -1.0
    +2σ    | 0.5        | 0.97725  | 0.51163    | 0.0    | -2.0
    """

    @pytest.fixture
    def qmu_bands(self) -> ExpectedBands:
        """ExpectedBands with known QMu p-values (μ=2, σ=1, q_A=4)."""
        pnulls = [3.167e-5, 0.00135, 0.02275, 0.15866, 0.5]
        palts = [0.02275, 0.15866, 0.5, 0.84134, 0.97725]
        return _make_bands(pnulls, palts)

    @pytest.mark.parametrize(
        "band_name",
        ["minus_2sigma", "minus_1sigma", "median", "plus_1sigma", "plus_2sigma"],
    )
    def test_cls_bands_qmu_at_expected_upper_limit(self, band_name: str):
        """CLs = 0.05 at each band's expected upper limit.

        QMu expected p-values at μ_up(N) = Φ⁻¹(1 - α·Φ(N)) + N:
            Band N | μ_up  | pnull = 1-Φ(μ_up-N) | palt = Φ(N)  | CLs
            -2     | 1.052 | 1-Φ(3.052)=0.001138 | Φ(-2)=0.02275| 0.05
            -1     | 1.412 | 1-Φ(2.412)=0.00793  | Φ(-1)=0.15866| 0.05
             0     | 1.960 | 1-Φ(1.960)=0.02500  | Φ(0) =0.50000| 0.05
            +1     | 2.727 | 1-Φ(1.727)=0.04213  | Φ(1) =0.84134| 0.05
            +2     | 3.656 | 1-Φ(1.656)=0.04883  | Φ(2) =0.97725| 0.05
        """
        pnulls = [0.001138, 0.00793, 0.02500, 0.04213, 0.04883]
        palts = [0.02275, 0.15866, 0.50000, 0.84134, 0.97725]
        bands = _make_bands(pnulls, palts)

        # abs=1e-3 accounts for rounding in the hardcoded p-values above
        assert float(bands.cl_s[band_name]) == pytest.approx(0.05, abs=1e-3)

    @pytest.mark.parametrize(
        ("band_name", "expected_z"),
        [
            ("minus_2sigma", 4.0),
            ("minus_1sigma", 3.0),
            ("median", 2.0),
            ("plus_1sigma", 1.0),
            ("plus_2sigma", 0.0),
        ],
    )
    def test_null_significance_bands(self, qmu_bands: ExpectedBands, band_name: str, expected_z: float):
        """Z_null at each band: 4.0, 3.0, 2.0, 1.0, 0.0."""
        assert float(qmu_bands.null_sig[band_name]) == pytest.approx(expected_z, abs=0.01)

    @pytest.mark.parametrize(
        ("band_name", "expected_z"),
        [
            ("minus_2sigma", 2.0),
            ("minus_1sigma", 1.0),
            ("median", 0.0),
            ("plus_1sigma", -1.0),
            ("plus_2sigma", -2.0),
        ],
    )
    def test_alt_significance_bands(self, qmu_bands: ExpectedBands, band_name: str, expected_z: float):
        """Z_alt at each band: 2.0, 1.0, 0.0, -1.0, -2.0."""
        assert float(qmu_bands.alt_sig[band_name]) == pytest.approx(expected_z, abs=0.01)


# =============================================================================
# BandValues iteration / indexing tests
# =============================================================================


class TestBandValues:
    """Tests for BandValues __iter__, __getitem__, and __len__."""

    @pytest.fixture
    def bv(self) -> BandValues:
        return BandValues(
            minus_2sigma=jnp.array(1.0),
            minus_1sigma=jnp.array(2.0),
            median=jnp.array(3.0),
            plus_1sigma=jnp.array(4.0),
            plus_2sigma=jnp.array(5.0),
        )

    def test_iter_yields_name_value_pairs_in_order(self, bv: BandValues):
        """__iter__ yields (name, value) pairs in _NAMES order."""
        expected_names = [
            "minus_2sigma",
            "minus_1sigma",
            "median",
            "plus_1sigma",
            "plus_2sigma",
        ]
        expected_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for (name, value), exp_name, exp_val in zip(bv, expected_names, expected_values, strict=True):
            assert name == exp_name
            assert float(value) == exp_val

    def test_getitem_returns_correct_value(self, bv: BandValues):
        """bv["median"] returns the median value."""
        assert float(bv["median"]) == 3.0
        assert float(bv["minus_2sigma"]) == 1.0
        assert float(bv["plus_2sigma"]) == 5.0

    def test_getitem_invalid_key_raises_keyerror(self, bv: BandValues):
        """Invalid key raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            bv["nonexistent"]

    def test_len(self, bv: BandValues):
        """len(bv) == 5."""
        assert len(bv) == 5

    def test_dict_roundtrip(self, bv: BandValues):
        """dict(bv) produces {name: value} mapping that roundtrips."""
        d = dict(bv)
        assert list(d.keys()) == list(BandValues._NAMES)
        reconstructed = BandValues(**d)
        for (_, orig), (_, recon) in zip(bv, reconstructed, strict=True):
            assert float(orig) == float(recon)

    def test_zip_two_bandvalues(self, bv: BandValues):
        """zip of two BandValues yields paired (name, value) tuples."""
        bv2 = BandValues(
            minus_2sigma=jnp.array(10.0),
            minus_1sigma=jnp.array(20.0),
            median=jnp.array(30.0),
            plus_1sigma=jnp.array(40.0),
            plus_2sigma=jnp.array(50.0),
        )
        for (n1, v1), (n2, v2) in zip(bv, bv2, strict=True):
            assert n1 == n2
            assert float(v2) == float(v1) * 10.0


# =============================================================================
# TestStatResult Tests
# =============================================================================


class TestTSResult:
    """Tests for TestStatResult container."""

    def test_required_fields(self):
        """Construction with value and test stores both fields."""
        result = TSResult(value=jnp.array(3.5), test=jnp.array(1.0))
        assert float(result.value) == 3.5
        assert float(result.test) == 1.0

    def test_q_asimov_defaults_none(self):
        """q_asimov defaults to None when not provided."""
        result = TSResult(value=jnp.array(1.0), test=jnp.array(0.5))
        assert result.q_asimov is None

    def test_q_asimov_stored(self):
        """q_asimov is stored when provided."""
        result = TSResult(
            value=jnp.array(1.0),
            test=jnp.array(0.5),
            q_asimov=jnp.array(4.0),
        )
        assert float(result.q_asimov) == 4.0

    def test_extras_defaults_empty(self):
        """extras defaults to empty dict."""
        result = TSResult(value=jnp.array(1.0), test=jnp.array(0.5))
        assert result.extras == {}

    def test_extras_stored(self):
        """extras dict is stored and accessible."""
        result = TSResult(
            value=jnp.array(1.0),
            test=jnp.array(0.5),
            extras={"mu_hat": 0.7, "nll_min": 12.3},
        )
        assert result.extras["mu_hat"] == 0.7
        assert result.extras["nll_min"] == 12.3


# =============================================================================
# ToyResult Tests
# =============================================================================


class TestToyResult:
    """Tests for ToyResult container."""

    def test_fields_stored(self):
        """q_alt and q_null arrays are stored."""
        q_alt = jnp.array([1.0, 2.0, 3.0])
        q_null = jnp.array([0.1, 0.2, 0.3])
        result = ToyResult(q_alt=q_alt, q_null=q_null)

        assert result.q_alt.shape == (3,)
        assert result.q_null.shape == (3,)
        assert float(result.q_alt[0]) == pytest.approx(1.0)
        assert float(result.q_null[2]) == pytest.approx(0.3)


# =============================================================================
# HypoTestResult Tests
# =============================================================================


class TestHypoTestResult:
    """Tests for HypoTestResult container."""

    def test_with_all_fields(self):
        """Construction with all p-values."""
        ts_result = TSResult(value=jnp.array(3.0), test=jnp.array(1.0))
        result = HypoTestResult(
            q_obs=jnp.array(3.0),
            pnull=jnp.array(0.02),
            palt=jnp.array(0.85),
            test_stat_result=ts_result,
        )
        assert float(result.q_obs) == pytest.approx(3.0)
        assert float(result.pnull) == pytest.approx(0.02)
        assert float(result.palt) == pytest.approx(0.85)
        assert result.test_stat_result is ts_result

    def test_none_pvalues(self):
        """pnull and palt can be None."""
        ts_result = TSResult(value=jnp.array(1.0), test=jnp.array(1.0))
        result = HypoTestResult(
            q_obs=jnp.array(1.0),
            pnull=None,
            palt=None,
            test_stat_result=ts_result,
        )
        assert result.pnull is None
        assert result.palt is None
