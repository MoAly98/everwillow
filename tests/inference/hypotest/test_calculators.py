"""Tests for hypothesis test calculators.

Parameterized across all calculator types (HypoTestCalculator, AsymptoticCalculator,
ToyCalculator) to verify the calculator behavior shared by all types.

Counting model: S=10, B=5, Poisson NLL.
Synthetic limit model: CLs = exp(-poi) with known closed-form solutions.
"""

from __future__ import annotations

from unittest import mock

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import everwillow.statelib as sl
from everwillow._src.inference.hypotest.calculators import (
    AsymptoticCalculator,
    HypoTestCalculator,
    ToyCalculator,
)
from everwillow._src.inference.hypotest.distributions import (
    Distribution,
    QMuAsymptotic,
    SimpleEmpiricalDistribution,
    TMuAsymptotic,
)
from everwillow._src.inference.hypotest.limit_solvers import (
    BisectionLimitSolver,
    GridScanLimitSolver,
    LimitSolver,
    RootFindingLimitSolver,
    StochasticLimitSolver,
)
from everwillow._src.inference.hypotest.results import (
    BandValues,
    ExpectedBands,
    HypoTestResult,
)
from everwillow._src.inference.hypotest.results import (
    TestStatResult as TSResult,
)
from everwillow._src.inference.hypotest.test_statistics import QMu, QTilde, TestStatistic, TMu
from everwillow._src.inference.hypotest.toys import ToyGenerator
from everwillow._src.inference.hypotest.utils import cl_s, significance

from ._counting_model import (
    create_observation,
    create_observation_2poi,
    create_params,
    create_params_2poi,
    poisson_nll,
    poisson_nll_2poi,
    predict_fn,
    predict_fn_2poi,
)

# =============================================================================
# Helpers for limit tests (synthetic model with known closed-form solutions)
# =============================================================================


def _dummy_nll(params, observation):
    """No-op NLL for tests that bypass fitting."""
    return 0.0


_DUMMY_PARAMS = sl.State.from_pytree({"mu": 0.0})
_DUMMY_OBS = {}


class _IdentityTestStat(TestStatistic):
    """Returns the tested POI value as the test stat value (no fitting)."""

    def _compute(self, nll_fn, params, observation, poi_test, **kwargs):
        (value,) = poi_test.values()
        return jnp.asarray(value), {}


def _make_expected_bands(pnulls, palts):
    """Build ExpectedBands from parallel lists of pnull/palt values."""
    band_names = [
        "minus_2sigma",
        "minus_1sigma",
        "median",
        "plus_1sigma",
        "plus_2sigma",
    ]
    return ExpectedBands(
        null_pvalue=BandValues(**dict(zip(band_names, pnulls, strict=False))),
        alt_pvalue=BandValues(**dict(zip(band_names, palts, strict=False))),
        cl_s=BandValues(**{n: cl_s(pn, pa) for n, pn, pa in zip(band_names, pnulls, palts, strict=False)}),
        null_sig=BandValues(**{n: significance(pn) for n, pn in zip(band_names, pnulls, strict=False)}),
        alt_sig=BandValues(**{n: significance(pa) for n, pa in zip(band_names, palts, strict=False)}),
    )


class _ExponentialCLsDist(Distribution):
    """Deterministic distribution where CLs = exp(-poi).

    Observed: pnull = exp(-poi)*0.5, palt = 0.5 → CLs = exp(-poi).
    Upper limit at CLs=0.05: poi = ln(20) = 2.99573.

    Expected bands: CLs_N = exp(-rate_N * poi) with rates [0.5, 0.6, 0.8, 1.0, 1.2].
    Band limits at CLs=0.05: poi_N = ln(20)/rate_N.
    """

    def null_pval(self, result):
        (poi,) = result.test.values()
        return jnp.exp(-poi) * 0.5

    def alt_pval(self, result):
        return jnp.array(0.5)

    def pvalue_bands(self, result):
        (poi,) = result.test.values()
        palt = jnp.array(0.5)
        rates = [0.5, 0.6, 0.8, 1.0, 1.2]
        pnulls = [jnp.exp(-r * poi) * palt for r in rates]
        palts = [palt] * 5
        return _make_expected_bands(pnulls, palts)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(
    params=[
        pytest.param("asimov_arg", id="asimov-observation-per-call"),
        pytest.param("predict_fn", id="predict-fn-field"),
    ]
)
def counting_calc(request):
    """Counting model calc (S=10, B=5, n_obs=10) producing full p-values.

    Returns (calc, test_kwargs) where test_kwargs are extra args for test()
    to ensure q_asimov is computed. Both flavors are AsymptoticCalculator
    (the fixed-distribution calculator); they differ in how the Asimov
    dataset is supplied: per test() call, or generated via the predict_fn
    field.
    """
    params = create_params(mu_init=1.0)
    observed = create_observation(10.0)
    asimov = create_observation(5.0)  # Asimov at mu=0: n = 0*10 + 5 = 5

    if request.param == "asimov_arg":
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=params,
            observation=observed,
            poi_key="mu",
        )
        return calc, {"asimov_observation": asimov}
    # "predict_fn"
    calc = AsymptoticCalculator(
        nll_fn=poisson_nll,
        params=params,
        observation=observed,
        poi_key="mu",
        predict_fn=predict_fn,
    )
    return calc, {}


@pytest.fixture
def calc_factory():
    """Factory building a fixed-distribution calculator on the synthetic limit model.

    Keyword overrides replace parts of the default setup. ToyCalculator is
    not included: it is inherently stochastic and its differing contract is
    tested in its own sections.
    """

    def make(**overrides):
        calc_kwargs = {
            "nll_fn": _dummy_nll,
            "params": _DUMMY_PARAMS,
            "observation": _DUMMY_OBS,
            "poi_key": "mu",
            "test_statistic": _IdentityTestStat(),
            "distribution": _ExponentialCLsDist(),
        }
        calc_kwargs.update(overrides)
        return AsymptoticCalculator(**calc_kwargs)

    return make


# =============================================================================
# Calculator base class behavior
# =============================================================================


class TestCalculatorBase:
    """HypoTestCalculator is the abstract core; AsymptoticCalculator is the
    fixed-distribution concrete calculator."""

    def test_abc_cannot_instantiate(self):
        """HypoTestCalculator has an abstract test() and cannot be instantiated."""
        with pytest.raises(TypeError):
            HypoTestCalculator(
                nll_fn=poisson_nll,
                params=create_params(),
                observation=create_observation(1.0),
                poi_key="mu",
            )

    def test_default_test_statistic(self):
        """Default test statistic is QTilde."""
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(),
            observation=create_observation(1.0),
            poi_key="mu",
        )

        assert isinstance(calc.test_statistic, QTilde)

    def test_qmu_without_asimov(self):
        """QMu + QMuAsymptotic: pnull works without Asimov, palt is None.

        n_obs=10, mu_test=1: q_mu = 1.8907.
        pnull = 1 - Φ(√1.8907) = 1 - Φ(1.375) = 0.08456.
        """
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(10.0),
            poi_key="mu",
            test_statistic=QMu(),
            distribution=QMuAsymptotic(),
        )
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            result = calc.test(poi_test=1.0)

        assert result.q_obs == pytest.approx(1.8907, rel=1e-3)
        assert float(result.pnull) == pytest.approx(0.08456, rel=1e-3)
        assert result.palt is None


# =============================================================================
# test() — parameterized across calculator types
# =============================================================================


class TestCalculatorTest:
    """test() returns correct q_obs, pnull, palt.

    Counting model S=10, B=5, n_obs=10, poi_test=1.0.
    MLE: mu_hat = (10-5)/10 = 0.5.
    q_obs = 2*(15 - 10 - 10*ln(15/10)) = 2*(5 - 10*ln(1.5)) = 1.8907
    q_asimov = 2*(15 - 5 - 5*ln(3)) = 9.0139
    sqrt(q_obs)=1.375, sqrt(q_asimov)=3.002.
    pnull = 1 - Φ(1.375) = 0.08456
    palt = 1 - Φ(1.375 - 3.002) = 1 - Φ(-1.627) = 0.94816
    """

    def test_q_obs(self, counting_calc):
        calc, kwargs = counting_calc
        result = calc.test(1.0, **kwargs)

        assert result.q_obs == pytest.approx(1.8907, rel=1e-3)

    def test_pnull(self, counting_calc):
        calc, kwargs = counting_calc
        result = calc.test(1.0, **kwargs)

        assert float(result.pnull) == pytest.approx(0.08456, rel=1e-3)

    def test_palt(self, counting_calc):
        calc, kwargs = counting_calc
        result = calc.test(1.0, **kwargs)

        assert float(result.palt) == pytest.approx(0.94816, rel=1e-3)


# =============================================================================
# cls() — parameterized across calculator types
# =============================================================================


class TestCalculatorCls:
    """cls() returns CLs = pnull/palt.

    CLs = 0.08456/0.94816 = 0.08919.
    """

    def test_cls(self, counting_calc):
        calc, kwargs = counting_calc
        result = calc.test(1.0, **kwargs)

        assert float(calc.cls(result)) == pytest.approx(0.08919, rel=1e-2)

    def test_cls_none_when_palt_none(self):
        """cls() returns None when palt is None."""
        dist = SimpleEmpiricalDistribution(q_null=jnp.array([1.0, 2.0, 3.0]))

        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(),
            observation=create_observation(1.0),
            poi_key="mu",
            distribution=dist,
        )
        ts = TSResult(value=jnp.array(1.0), test=jnp.array(1.0))
        with pytest.warns(UserWarning, match="cannot be performed without q_alt"):
            palt = dist.alt_pval(ts)
        result = HypoTestResult(
            q_obs=ts.value,
            pnull=dist.null_pval(ts),
            palt=palt,
            test_stat_result=ts,
            distribution=dist,
        )

        assert calc.cls(result) is None


# =============================================================================
# expected() — parameterized across calculator types
# =============================================================================


class TestCalculatorPvalueBands:
    """expected() returns correct expected p-value bands.

    QTildeAsymptotic at poi_test=1.0, q_asimov=9.0139, sqrt_qA = 3.002.

    CLs(N) = (1 - Φ(sqrt_qA - N)) / (1 - Φ(-N)):
    Both the standard (N≥0) and boundary (N<0) piecewise regions of the
    QTilde CDF simplify to this single formula.
    """

    @pytest.mark.parametrize(
        ("band_name", "expected_cls"),
        [
            # CLs(-2) = (1-Φ(5.002))/(1-Φ(2)) = 2.84e-7/0.02275 ≈ 1.25e-5
            ("minus_2sigma", 1.25e-5),
            # CLs(-1) = (1-Φ(4.002))/(1-Φ(1)) = 3.14e-5/0.1587 ≈ 1.98e-4
            ("minus_1sigma", 1.98e-4),
            # CLs(0) = (1-Φ(3.002))/(1-Φ(0)) = 0.00134/0.5 = 0.00269
            ("median", 0.00269),
            # CLs(+1) = (1-Φ(2.002))/(1-Φ(-1)) = 0.02263/0.8413 = 0.02690
            ("plus_1sigma", 0.02690),
            # CLs(+2) = (1-Φ(1.002))/(1-Φ(-2)) = 0.1581/0.9772 = 0.1618
            ("plus_2sigma", 0.1618),
        ],
    )
    def test_expected_cls_band(self, counting_calc, band_name, expected_cls):
        calc, kwargs = counting_calc
        result = calc.test(1.0, **kwargs)
        bands = calc.pvalue_bands(result)

        assert float(bands.cl_s[band_name]) == pytest.approx(expected_cls, rel=0.1)

    def test_expected_raises_when_unsupported(self):
        """TMuAsymptotic does not implement pvalue_bands → NotImplementedError."""
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(15.0),
            poi_key="mu",
            predict_fn=predict_fn,
            distribution=TMuAsymptotic(),
        )
        result = calc.test(poi_test=1.0)

        with pytest.raises(NotImplementedError):
            calc.pvalue_bands(result)


# =============================================================================
# Type-specific: AsymptoticCalculator Asimov handling
# =============================================================================


class TestAsymptoticCalculatorAsimov:
    """AsymptoticCalculator-specific Asimov dataset tests."""

    def test_q_obs_zero_at_mle(self):
        """At MLE (n_obs=15 → mu_hat=1.0), q_obs=0 and pnull=0.5.

        q_asimov = 2*(15-5-5*ln(3)) = 9.0139.
        palt = 1-Φ(0 - 3.002) = 1-Φ(-3.002) = 0.99866.
        """
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(15.0),
            poi_key="mu",
            predict_fn=predict_fn,
        )
        result = calc.test(poi_test=1.0)

        assert result.q_obs == pytest.approx(0.0, abs=1e-5)
        assert float(result.pnull) == pytest.approx(0.5, rel=1e-3)
        assert float(result.palt) == pytest.approx(0.99866, rel=1e-3)
        assert result.test_stat_result.q_asimov == pytest.approx(9.0139, rel=1e-3)

    def test_q_asimov_with_predict_fn(self):
        """predict_fn at poi_asimov=0 generates Asimov n=5.

        q_asimov = 2*(15-5-5*ln(3)) = 9.0139.
        """
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(10.0),
            poi_key="mu",
            predict_fn=predict_fn,
        )
        result = calc.test(poi_test=1.0)

        assert result.test_stat_result.q_asimov == pytest.approx(9.0139, rel=1e-3)

    def test_asimov_observation_takes_precedence(self):
        """asimov_observation overrides predict_fn.

        predict_fn at mu=0 → n=5 → q_asimov=9.0139.
        Explicit asimov at mu=1 → n=15 → q_asimov=0.
        """
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(10.0),
            poi_key="mu",
            predict_fn=predict_fn,
            asimov_observation=create_observation(15.0),
        )
        result = calc.test(poi_test=1.0)

        assert result.test_stat_result.q_asimov == pytest.approx(0.0, abs=1e-4)

    def test_without_predict_fn_pvalues_none(self):
        """Without predict_fn, q_asimov is None and p-values are None."""
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(15.0),
            poi_key="mu",
        )
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            result = calc.test(poi_test=1.0)

        assert result.test_stat_result.q_asimov is None
        assert result.pnull is None
        assert result.palt is None

    def test_q_asimov_at_different_mu_test(self):
        """Asimov is always at poi_asimov=0, regardless of mu_test.

        At mu_test=0: Asimov at mu=0 (n=5), testing at 0 → q_asimov=0.
        At mu_test=2: Asimov at mu=0 (n=5), testing at 2:
          q_asimov = 2*(25-5-5*ln(5)) = 2*(20-8.047) = 23.906.
        """
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(15.0),
            poi_key="mu",
            predict_fn=predict_fn,
        )

        result_0 = calc.test(poi_test=0.0)
        assert result_0.test_stat_result.q_asimov == pytest.approx(0.0, abs=1e-4)

        result_2 = calc.test(poi_test=2.0)
        assert result_2.test_stat_result.q_asimov == pytest.approx(23.906, rel=1e-3)


# =============================================================================
# criterion error contract — parameterized across calculator types
# =============================================================================


class TestLimitCriterionErrors:
    """Limit methods raise ValueError when the criterion produces None."""

    def test_default_criterion_palt_none_raises(self, calc_factory):
        """Default CLs criterion needs palt; a q_null-only empirical distribution has none."""
        calc = calc_factory(distribution=SimpleEmpiricalDistribution(q_null=jnp.array([1.0, 2.0, 3.0])))

        with pytest.warns(UserWarning, match="without q_alt"), pytest.raises(ValueError, match="criterion"):
            calc.upper_limit(RootFindingLimitSolver(bounds=(0.0, 1.0)), level=0.05)

    def test_custom_criterion_returning_none_raises(self, calc_factory):
        """A user criterion returning None is rejected with a clear error."""
        calc = calc_factory()

        with pytest.raises(ValueError, match="criterion"):
            calc.upper_limit(RootFindingLimitSolver(bounds=(0.0, 1.0)), level=0.05, criterion=lambda result: None)

    def test_bands_default_criterion_none_raises(self, calc_factory):
        """QMuAsymptotic without q_asimov: bands() is None -> clear error."""
        calc = calc_factory(distribution=QMuAsymptotic())

        with pytest.warns(UserWarning, match="Asimov"), pytest.raises(ValueError, match="criterion"):
            calc.upper_limit_bands(RootFindingLimitSolver(bounds=(0.0, 1.0)), level=0.05)

    def test_gridscan_criterion_none_raises(self, calc_factory):
        """GridScan applies the same criterion error contract."""
        calc = calc_factory(distribution=SimpleEmpiricalDistribution(q_null=jnp.array([1.0, 2.0, 3.0])))

        with pytest.warns(UserWarning, match="without q_alt"), pytest.raises(ValueError, match="criterion"):
            calc.upper_limit(GridScanLimitSolver(scan=jnp.linspace(0.01, 8.0, 50)), level=0.05)


# =============================================================================
# fit_kwargs forwarding — parameterized across calculator types
# =============================================================================


class TestFitKwargsForwarding:
    """fit_kwargs reach the test statistic through the composed objective
    (recording solver evaluates the objective once; mock spy on compute)."""

    def test_upper_limit_fit_kwargs(self, calc_factory):
        calc = calc_factory()
        record = []

        with mock.patch.object(
            _IdentityTestStat, "compute", autospec=True, side_effect=_IdentityTestStat.compute
        ) as spy:
            calc.upper_limit(_RecordingSolver(record=record), level=0.05, fit_kwargs={"max_steps": 7})

        assert spy.call_args_list
        assert all(call.kwargs.get("max_steps") == 7 for call in spy.call_args_list)

    def test_upper_limit_bands_fit_kwargs(self, calc_factory):
        calc = calc_factory()
        record = []

        with mock.patch.object(
            _IdentityTestStat, "compute", autospec=True, side_effect=_IdentityTestStat.compute
        ) as spy:
            calc.upper_limit_bands(_RecordingSolver(record=record), level=0.05, fit_kwargs={"max_steps": 7})

        assert spy.call_args_list
        assert all(call.kwargs.get("max_steps") == 7 for call in spy.call_args_list)


# =============================================================================
# poi_key resolution — optional field with per-call override
# =============================================================================


class TestPoiKeyResolution:
    """poi_key is optional: mappings never need it, scalars resolve per-call
    then field, and the limit methods accept a per-call target POI."""

    def test_joint_calculator_without_poi_key(self):
        """A calculator with no poi_key runs joint tests from full mappings."""
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll_2poi,
            params=create_params_2poi(),
            observation=create_observation_2poi(10.0, 25.0),
            test_statistic=TMu(),
            distribution=TMuAsymptotic(dof=2),
            predict_fn=predict_fn_2poi,
        )

        result = calc.test({"mu_a": 1.0, "mu_b": 1.0})

        assert float(result.pnull) == pytest.approx(0.024326, rel=1e-3)

    def test_scalar_without_poi_key_raises(self, calc_factory):
        """A bare scalar cannot resolve without a POI key; the error says how."""
        calc = calc_factory(poi_key=None)

        with pytest.raises(ValueError, match="no POI key"):
            calc.test(1.0)

    def test_upper_limit_without_poi_key_raises(self, calc_factory):
        """The limit axis needs a POI key from the field or the call."""
        calc = calc_factory(poi_key=None)

        with pytest.raises(ValueError, match="no POI key"):
            calc.upper_limit(BisectionLimitSolver(bounds=(0.0, 5.0), tol=0.01))

    def test_upper_limit_per_call_poi_key_overrides_field(self):
        """The per-call poi_key picks the limit axis over the field."""
        calc = AsymptoticCalculator(
            nll_fn=_dummy_nll,
            params=sl.State.from_pytree({"mu": 0.0, "nu": 0.0}),
            observation=_DUMMY_OBS,
            poi_key="mu",
            test_statistic=_IdentityTestStat(),
            distribution=_ExponentialCLsDist(),
        )

        with mock.patch.object(
            _IdentityTestStat, "compute", autospec=True, side_effect=_IdentityTestStat.compute
        ) as spy:
            calc.upper_limit(BisectionLimitSolver(bounds=(0.0, 5.0), tol=0.5), level=0.05, poi_key="nu")

        tested_keys = {key for call in spy.call_args_list for key in call.args[4]}
        assert tested_keys == {"nu"}

    def test_unknown_poi_key_raises(self, calc_factory):
        """A resolved poi_key must name a model parameter."""
        calc = calc_factory(poi_key="typo")

        with pytest.raises(ValueError, match="typo"):
            calc.test(1.0)

    def test_toy_poi_alt_broadcasts_over_tested_keys(self):
        """A scalar poi_alt broadcasts over the tested point's POIs, so a
        keyless joint toy calculator still produces palt."""
        calc = ToyCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(10.0),
            test_statistic=QTilde(),
            toy_generator=ToyGenerator(predict_fn=predict_fn, ntoys=50),
            key=jax.random.key(3),
        )

        result = calc.test({"mu": 1.0})

        assert result.palt is not None


# =============================================================================
# confidence_region — criterion field over hypothesis points
# =============================================================================


class TestConfidenceRegion:
    """confidence_region evaluates criterion(test(point)) over hypothesis points."""

    def test_default_criterion_is_null_pvalue(self, calc_factory):
        """Default criterion is pnull: exp(-poi)*0.5 on the synthetic model."""
        calc = calc_factory()
        points = [{"mu": 0.0}, {"mu": 1.0}, {"mu": 2.0}]

        region = calc.confidence_region(points, level=0.05)

        assert region.values.shape == (3,)
        expected = [0.5, 0.5 * float(jnp.exp(-1.0)), 0.5 * float(jnp.exp(-2.0))]
        for got, want in zip(region.values, expected, strict=True):
            assert float(got) == pytest.approx(want, rel=1e-6)

    def test_inside_mask(self, calc_factory):
        """inside is values >= level: pnull(mu) = exp(-mu)/2 crosses 0.05 at mu=ln(10)."""
        calc = calc_factory()

        region = calc.confidence_region([{"mu": 1.0}, {"mu": 3.0}], level=0.05)

        assert bool(region.inside[0])  # exp(-1)/2 = 0.184 >= 0.05
        assert not bool(region.inside[1])  # exp(-3)/2 = 0.025 < 0.05

    def test_scalar_points_use_poi_key(self, calc_factory):
        """Bare scalars are valid points for the calculator's poi_key."""
        calc = calc_factory()

        region = calc.confidence_region([0.0, 1.0])

        assert float(region.values[0]) == pytest.approx(0.5, rel=1e-6)
        assert float(region.values[1]) == pytest.approx(0.5 * float(jnp.exp(-1.0)), rel=1e-6)

    def test_custom_criterion(self, calc_factory):
        """A user criterion (CLs here) replaces the default: exp(-poi)."""
        calc = calc_factory()

        region = calc.confidence_region([{"mu": 1.0}], criterion=calc.cls)

        assert float(region.values[0]) == pytest.approx(float(jnp.exp(-1.0)), rel=1e-6)

    def test_pytree_criterion_stacks_per_leaf(self, calc_factory):
        """A criterion returning a pytree yields one stacked array per leaf."""
        calc = calc_factory()

        region = calc.confidence_region(
            [{"mu": 0.0}, {"mu": 2.0}],
            criterion=lambda result: {"pnull": result.pnull, "palt": result.palt},
        )

        assert region.values["pnull"].shape == (2,)
        assert region.values["palt"].shape == (2,)
        assert float(region.values["pnull"][1]) == pytest.approx(0.5 * float(jnp.exp(-2.0)), rel=1e-6)
        assert float(region.values["palt"][0]) == pytest.approx(0.5, rel=1e-6)

    def test_fit_kwargs_forwarded(self, calc_factory):
        """fit_kwargs reach the test statistic in the scanned evaluations."""
        calc = calc_factory()

        with mock.patch.object(
            _IdentityTestStat, "compute", autospec=True, side_effect=_IdentityTestStat.compute
        ) as spy:
            calc.confidence_region([{"mu": 0.0}, {"mu": 1.0}], fit_kwargs={"max_steps": 7})

        assert spy.call_args_list
        assert all(call.kwargs.get("max_steps") == 7 for call in spy.call_args_list)

    def test_map_fn_override_matches_vmap(self, calc_factory):
        """A sequential lax.map produces the same field as the default vmap."""
        calc = calc_factory()
        points = [{"mu": 0.0}, {"mu": 1.0}, {"mu": 2.0}]

        default = calc.confidence_region(points)
        sequential = calc.confidence_region(points, map_fn=lambda fn: lambda xs: jax.lax.map(fn, xs))

        assert jnp.array_equal(default.values, sequential.values)

    def test_mismatched_point_keys_raise(self, calc_factory):
        """Points naming different POIs cannot form one scan."""
        calc = calc_factory()

        with pytest.raises(ValueError, match="same POI keys"):
            calc.confidence_region([{"mu": 0.0}, {"nu": 1.0}])

    def test_joint_two_poi_region_values(self):
        """2-POI TMu scan gives the chi-square_2 tail exp(-t/2) at each point.

        At {mu_a: 1, mu_b: 1} with n=(10, 25): t = 7.4320 (sum of the two
        channels), pnull = exp(-7.4320/2) = 0.024326. At the MLE point
        {mu_a: 0.5, mu_b: 2.0}: t = 0, pnull = 1.
        """
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll_2poi,
            params=create_params_2poi(),
            observation=create_observation_2poi(10.0, 25.0),
            poi_key="mu_a",
            test_statistic=TMu(),
            distribution=TMuAsymptotic(dof=2),
            predict_fn=predict_fn_2poi,
        )

        region = calc.confidence_region([{"mu_a": 1.0, "mu_b": 1.0}, {"mu_a": 0.5, "mu_b": 2.0}], level=0.05)

        assert float(region.values[0]) == pytest.approx(0.024326, rel=1e-3)
        assert float(region.values[1]) == pytest.approx(1.0, abs=1e-3)
        assert not bool(region.inside[0])
        assert bool(region.inside[1])


class TestToyCalculatorConfidenceRegion:
    """ToyCalculator.confidence_region regenerates toys with a fresh key per point."""

    def _make_calc(self, **overrides):
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)
        kwargs = {
            "nll_fn": poisson_nll,
            "params": params,
            "observation": observed,
            "poi_key": "mu",
            "test_statistic": QTilde(),
            "toy_generator": ToyGenerator(predict_fn=predict_fn, ntoys=100),
            "key": jax.random.key(11),
        }
        kwargs.update(overrides)
        return ToyCalculator(**kwargs)

    def test_reproducible_with_same_key(self):
        """The same calculator key gives an identical scan."""
        calc = self._make_calc()
        points = [{"mu": 0.5}, {"mu": 1.0}]

        r1 = calc.confidence_region(points, criterion=lambda r: r.pnull)
        r2 = calc.confidence_region(points, criterion=lambda r: r.pnull)

        assert jnp.array_equal(r1.values, r2.values)

    def test_distinct_keys_per_point(self):
        """Identical points in one scan get independent toy ensembles."""
        calc = self._make_calc()

        region = calc.confidence_region([{"mu": 1.0}, {"mu": 1.0}], criterion=lambda r: r.pnull)

        # Same hypothesis, different subkeys: the empirical p-values differ.
        assert float(region.values[0]) != float(region.values[1])

    def test_per_call_key_changes_ensembles(self):
        """A per-call key overrides the calculator's key field."""
        calc = self._make_calc()
        points = [{"mu": 1.0}]

        r_field = calc.confidence_region(points, criterion=lambda r: r.pnull)
        r_call = calc.confidence_region(points, criterion=lambda r: r.pnull, key=jax.random.key(99))

        assert float(r_field.values[0]) != float(r_call.values[0])

    def test_physics_ordering(self):
        """pnull is high at the well-fitting point and low far from it.

        n_obs=10 has mu_hat=0.5; testing mu=0.5 fits, mu=3.0 is strongly
        disfavored (expected 35 vs observed 10).
        """
        calc = self._make_calc()

        region = calc.confidence_region([{"mu": 0.5}, {"mu": 3.0}], criterion=lambda r: r.pnull, level=0.05)

        assert float(region.values[0]) > 0.3
        assert float(region.values[1]) < 0.05
        assert bool(region.inside[0])
        assert not bool(region.inside[1])


# =============================================================================
# calculator <-> solver seam — recording fakes (glue, no numerics)
# =============================================================================


class _RecordingSolver(LimitSolver):
    """Fake solver: records solve() arguments, evaluates the objective once
    at poi=1.7, and returns a sentinel value."""

    record: list = eqx.field(static=True)

    def solve(self, objective, level, *, key=None):
        value = objective(1.7, key)
        self.record.append({"level": level, "key": key, "value": value})
        return jnp.asarray(123.0)


class _RecordingStochasticSolver(_RecordingSolver, StochasticLimitSolver):
    """Stochastic-marked variant of the recording fake."""


class TestLimitSolverGlue:
    """Argument flow across the calculator/solver seam."""

    def test_solver_receives_level_and_result_passes_through(self, calc_factory):
        calc = calc_factory()
        record = []

        out = calc.upper_limit(_RecordingSolver(record=record), level=0.123)

        assert float(out) == 123.0
        assert record[0]["level"] == 0.123
        assert record[0]["key"] is None

    def test_objective_composes_criterion_and_test(self, calc_factory):
        """The objective handed to the solver is criterion∘test: with the
        identity test statistic and CLs(poi) = exp(-poi), evaluating at
        poi=1.7 must give e^-1.7 = 0.18268352 (hand value)."""
        calc = calc_factory()
        record = []

        calc.upper_limit(_RecordingSolver(record=record), level=0.05)

        assert float(record[0]["value"]) == pytest.approx(0.18268352, rel=1e-6)

    def test_bands_objective_returns_band_values(self, calc_factory):
        """upper_limit_bands hands the solver a BandValues-valued objective;
        the median band is CLs_med(1.7) = exp(-0.8 * 1.7) = e^-1.36
        = 0.256661 (hand value)."""
        calc = calc_factory()
        record = []

        calc.upper_limit_bands(_RecordingSolver(record=record), level=0.05)

        assert isinstance(record[0]["value"], BandValues)
        assert float(record[0]["value"].median) == pytest.approx(0.256661, rel=1e-5)

    def test_custom_criterion_flows_to_objective(self, calc_factory):
        """criterion=pnull: objective value at poi=1.7 is
        pnull(1.7) = exp(-1.7) * 0.5 = 0.09134176 (hand value)."""
        calc = calc_factory()
        record = []

        calc.upper_limit(_RecordingSolver(record=record), level=0.05, criterion=lambda result: result.pnull)

        assert float(record[0]["value"]) == pytest.approx(0.09134176, rel=1e-6)

    def test_pytree_criterion_flows_per_leaf(self, calc_factory):
        """A pytree-valued criterion reaches the solver leaf-for-leaf:
        {"cls": e^-1.7 = 0.18268352, "clsb": e^-1.7/2 = 0.09134176}."""
        calc = calc_factory()
        record = []

        calc.upper_limit(
            _RecordingSolver(record=record),
            level=0.05,
            criterion=lambda result: {"cls": calc.cls(result), "clsb": result.pnull},
        )

        value = record[0]["value"]
        assert float(value["cls"]) == pytest.approx(0.18268352, rel=1e-6)
        assert float(value["clsb"]) == pytest.approx(0.09134176, rel=1e-6)

    def test_custom_band_criterion_flows_to_objective(self, calc_factory):
        """criterion=expected pnull bands: median leaf at poi=1.7 is
        exp(-0.8 * 1.7) * 0.5 = 0.1283305 (hand value)."""
        calc = calc_factory()
        record = []

        calc.upper_limit_bands(
            _RecordingSolver(record=record),
            level=0.05,
            criterion=lambda result: calc.pvalue_bands(result).null_pvalue,
        )

        assert float(record[0]["value"].median) == pytest.approx(0.1283305, rel=1e-5)

    def test_field_solver_used_when_no_argument(self, calc_factory):
        record = []
        calc = calc_factory(limit_solver=_RecordingSolver(record=record))

        out = calc.upper_limit(level=0.05)

        assert float(out) == 123.0

    def test_per_call_solver_overrides_field(self, calc_factory):
        field_record, call_record = [], []
        calc = calc_factory(limit_solver=_RecordingSolver(record=field_record))

        calc.upper_limit(_RecordingSolver(record=call_record), level=0.05)

        assert call_record
        assert not field_record

    def test_no_solver_configured_raises(self, calc_factory):
        calc = calc_factory()

        with pytest.raises(ValueError, match="solver"):
            calc.upper_limit(level=0.05)


# =============================================================================
# Type-specific: ToyCalculator toy-regeneration machinery
# =============================================================================


class _ConstantDist(Distribution):
    """Fixed p-values, independent of the toys used to build it."""

    def null_pval(self, result):
        return jnp.array(0.25)

    def alt_pval(self, result):
        return jnp.array(0.5)


def _counting_toy_calc(gen, **extra):
    """ToyCalculator on the counting model with n_obs=10."""
    extra.setdefault("key", jax.random.key(99))
    return ToyCalculator(
        nll_fn=poisson_nll,
        params=create_params(mu_init=1.0),
        observation=create_observation(10.0),
        poi_key="mu",
        toy_generator=gen,
        **extra,
    )


def _synthetic_toy_calc(gen, **extra):
    """ToyCalculator on the dummy model with the exponential CLs distribution."""
    extra.setdefault("key", jax.random.key(99))
    extra.setdefault("distribution_factory", lambda toys: _ExponentialCLsDist())
    return ToyCalculator(
        nll_fn=_dummy_nll,
        params=_DUMMY_PARAMS,
        observation=_DUMMY_OBS,
        poi_key="mu",
        test_statistic=_IdentityTestStat(),
        toy_generator=gen,
        **extra,
    )


class TestToyCalculatorConstruction:
    """ToyCalculator is inherently stochastic: generator and key are required.

    Fixed distributions (asymptotic formulas or frozen external toy
    ensembles) belong on HypoTestCalculator; ToyCalculator exists for the
    regeneration machinery, so configurations without it are rejected at
    construction.
    """

    def test_missing_generator_raises(self):
        with pytest.raises(ValueError, match="toy_generator"):
            ToyCalculator(
                nll_fn=poisson_nll,
                params=create_params(mu_init=1.0),
                observation=create_observation(10.0),
                poi_key="mu",
                key=jax.random.key(0),
            )

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="key"):
            ToyCalculator(
                nll_fn=poisson_nll,
                params=create_params(mu_init=1.0),
                observation=create_observation(10.0),
                poi_key="mu",
                toy_generator=ToyGenerator(ntoys=4, predict_fn=predict_fn),
            )


class TestToyCalculatorTestWithKey:
    """test(poi, key=...) regenerates toys and builds the distribution per POI."""

    def test_degenerate_sampler_gives_exact_pvalues(self):
        """Every toy reproduces the observation, so q_toy == q_obs for all toys.

        SimpleEmpiricalDistribution tail counting is >=-inclusive, so
        pnull = palt = 1.0 exactly and CLs = 1.0.
        """
        gen = ToyGenerator(ntoys=50, sample_fn=lambda state, key: {"n": 10.0})
        calc = _counting_toy_calc(gen)

        result = calc.test(1.0, key=jax.random.key(0))

        assert float(result.pnull) == 1.0
        assert float(result.palt) == 1.0
        assert float(calc.cls(result)) == 1.0

    def test_poisson_toys_match_exact_tail_probabilities(self):
        """Empirical p-values approach exact Poisson tail probabilities.

        QTilde at poi_test=1 is monotone in n with the boundary at n_obs=10:
        q_toy >= q_obs exactly when n_toy <= 10.
        pnull -> P(n <= 10 | lambda = 1*10+5 = 15) = 0.11846
        palt  -> P(n <= 10 | lambda = 0*10+5 =  5) = 0.98630
        Tolerances are ~3 sigma of binomial MC noise at ntoys=1000.
        """
        gen = ToyGenerator(ntoys=1000, predict_fn=predict_fn)
        calc = _counting_toy_calc(gen)

        result = calc.test(1.0, key=jax.random.key(0))

        assert float(result.pnull) == pytest.approx(0.11846, abs=0.035)
        assert float(result.palt) == pytest.approx(0.98630, abs=0.02)

    def test_same_key_reproducible(self):
        """Identical keys produce identical toy ensembles and p-values."""
        captured = []

        def capturing_factory(toys):
            captured.append(toys)
            return SimpleEmpiricalDistribution.from_toys(toys)

        gen = ToyGenerator(ntoys=100, predict_fn=predict_fn)
        calc = _counting_toy_calc(gen, distribution_factory=capturing_factory)

        r1 = calc.test(1.0, key=jax.random.key(111))
        r2 = calc.test(1.0, key=jax.random.key(111))

        assert float(r1.pnull) == float(r2.pnull)
        assert jnp.array_equal(captured[0].q_null, captured[1].q_null)

    def test_different_keys_differ(self):
        """Different keys produce different toy ensembles."""
        captured = []

        def capturing_factory(toys):
            captured.append(toys)
            return SimpleEmpiricalDistribution.from_toys(toys)

        gen = ToyGenerator(ntoys=100, predict_fn=predict_fn)
        calc = _counting_toy_calc(gen, distribution_factory=capturing_factory)

        calc.test(1.0, key=jax.random.key(111))
        calc.test(1.0, key=jax.random.key(222))

        assert not jnp.array_equal(captured[0].q_null, captured[1].q_null)

    def test_field_key_used_when_no_argument(self):
        """test(poi) without a key argument draws with the field key: the
        ensembles are identical to an explicit test(poi, key=<field key>)
        call, and differ under a per-call override."""
        captured = []

        def capturing_factory(toys):
            captured.append(toys)
            return SimpleEmpiricalDistribution.from_toys(toys)

        field_key = jax.random.key(99)
        gen = ToyGenerator(ntoys=20, predict_fn=predict_fn)
        calc = _counting_toy_calc(gen, distribution_factory=capturing_factory, key=field_key)

        calc.test(1.0)
        calc.test(1.0, key=field_key)
        calc.test(1.0, key=jax.random.key(1))

        assert jnp.array_equal(captured[0].q_null, captured[1].q_null)
        assert not jnp.array_equal(captured[0].q_null, captured[2].q_null)

    def test_distribution_factory_swap(self):
        """Any Callable[[ToyResult], Distribution] can replace the default factory."""
        gen = ToyGenerator(ntoys=10, sample_fn=lambda state, key: {"n": 10.0})
        calc = _counting_toy_calc(gen, distribution_factory=lambda toys: _ConstantDist())

        result = calc.test(1.0, key=jax.random.key(0))

        assert float(result.pnull) == 0.25
        assert float(result.palt) == 0.5

    def test_poi_alt_threading(self):
        """poi_alt controls whether alternative-hypothesis toys are generated."""
        captured = []

        def capturing_factory(toys):
            captured.append(toys)
            return SimpleEmpiricalDistribution.from_toys(toys)

        gen = ToyGenerator(ntoys=20, predict_fn=predict_fn)

        calc_default = _counting_toy_calc(gen, distribution_factory=capturing_factory)
        calc_default.test(1.0, key=jax.random.key(0))
        assert captured[0].q_alt is not None
        assert captured[0].q_alt.shape == (20,)

        calc_no_alt = _counting_toy_calc(gen, distribution_factory=capturing_factory, poi_alt=None)
        with pytest.warns(UserWarning, match="without q_alt"):
            result = calc_no_alt.test(1.0, key=jax.random.key(0))
        assert captured[1].q_alt is None
        assert result.palt is None


class TestToyCalculatorKeyedLimits:
    """Keyed toy limits: type-based dispatch and key flow (glue), plus ONE
    end-to-end integration anchor for the otherwise-untested keyed pipeline."""

    def test_rootfind_raises_typeerror(self):
        """RootFind is not a StochasticLimitSolver: adaptive root finding
        assumes a deterministic criterion, and every ToyCalculator limit is
        stochastic (no key argument needed to trigger the guard)."""
        gen = ToyGenerator(ntoys=4, sample_fn=lambda state, key: {})
        calc = _synthetic_toy_calc(gen)

        with pytest.raises(TypeError, match="Stochastic"):
            calc.upper_limit(RootFindingLimitSolver(bounds=(0.01, 8.0)), level=0.05)

    def test_stochastic_solver_receives_key(self):
        """The per-call key reaches solver.solve unchanged."""
        gen = ToyGenerator(ntoys=4, sample_fn=lambda state, key: {})
        calc = _synthetic_toy_calc(gen)
        record = []
        key = jax.random.key(7)

        calc.upper_limit(_RecordingStochasticSolver(record=record), level=0.05, key=key)

        assert record[0]["key"] is not None
        assert jnp.array_equal(jax.random.key_data(record[0]["key"]), jax.random.key_data(key))

    def test_objective_threads_key_into_test(self):
        """The composed objective calls test(poi, key=...) per evaluation:
        the capturing factory sees one ToyResult per objective evaluation."""
        captured = []

        def capturing_factory(toys):
            captured.append(toys)
            return _ExponentialCLsDist()

        gen = ToyGenerator(ntoys=4, sample_fn=lambda state, key: {})
        calc = _synthetic_toy_calc(gen, distribution_factory=capturing_factory)

        calc.upper_limit(_RecordingStochasticSolver(record=[]), level=0.05, key=jax.random.key(1))

        assert len(captured) == 1  # recording solver evaluates the objective once

    def test_field_key_used_when_no_argument(self):
        """Without a key argument the field key drives the solver; a per-call
        key argument overrides it."""
        gen = ToyGenerator(ntoys=4, sample_fn=lambda state, key: {})
        field_key = jax.random.key(7)
        calc = _synthetic_toy_calc(gen, key=field_key)
        record = []
        override = jax.random.key(11)

        recording_solver = _RecordingStochasticSolver(record=record)
        calc.upper_limit(recording_solver, level=0.05)
        calc.upper_limit(recording_solver, level=0.05, key=override)

        assert jnp.array_equal(jax.random.key_data(record[0]["key"]), jax.random.key_data(field_key))
        assert jnp.array_equal(jax.random.key_data(record[1]["key"]), jax.random.key_data(override))

    def test_keyed_pipeline_integration_anchor(self):
        """End-to-end keyed path with an analytic anchor: the synthetic CLs
        is exp(-poi) regardless of the toys, so Bisection(tol=0) through the
        full generate -> factory -> p-value pipeline must return
        ln 20 = 2.9957322735539909 (bracket collapse; derivation in
        test_limit_solvers)."""
        gen = ToyGenerator(ntoys=4, sample_fn=lambda state, key: {})
        calc = _synthetic_toy_calc(gen)

        limit = calc.upper_limit(BisectionLimitSolver(bounds=(0.01, 8.0), tol=0.0), level=0.05, key=jax.random.key(1))

        assert float(limit) == pytest.approx(2.9957322735539909, rel=1e-3)

    def test_default_criterion_none_raises(self):
        """Default CLs criterion with palt=None (no alt toys) raises clearly."""
        gen = ToyGenerator(ntoys=4, sample_fn=lambda state, key: {})
        calc = ToyCalculator(
            nll_fn=_dummy_nll,
            params=_DUMMY_PARAMS,
            observation=_DUMMY_OBS,
            poi_key="mu",
            test_statistic=_IdentityTestStat(),
            toy_generator=gen,
            poi_alt=None,
            key=jax.random.key(3),
        )

        with pytest.warns(UserWarning, match="without q_alt"), pytest.raises(ValueError, match="criterion"):
            calc.upper_limit(BisectionLimitSolver(bounds=(0.01, 8.0), tol=0.0), level=0.05)


# =============================================================================
# Results carry the distribution that produced them
# =============================================================================


class _MarkerDist(Distribution):
    """Sentinel distribution whose p-values encode a recognizable marker."""

    marker: float

    def null_pval(self, result):
        return jnp.asarray(self.marker)

    def alt_pval(self, result):
        return jnp.asarray(self.marker)

    def pvalue_bands(self, result):
        values = [jnp.asarray(self.marker)] * 5
        return _make_expected_bands(values, values)


class TestResultCarriesDistribution:
    """pvalue_bands(result) must consult the distribution that produced the
    result's p-values, which the result carries (marker glue, no physics)."""

    def test_asymptotic_result_records_field_distribution(self):
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(10.0),
            poi_key="mu",
            distribution=_MarkerDist(marker=0.1111),
        )

        result = calc.test(1.0)

        assert isinstance(result.distribution, _MarkerDist)
        assert result.distribution.marker == 0.1111
        assert float(calc.pvalue_bands(result).null_pvalue.median) == pytest.approx(0.1111)

    def test_toy_result_records_factory_distribution(self):
        gen = ToyGenerator(ntoys=4, sample_fn=lambda state, key: {"n": 10.0})
        calc = ToyCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(10.0),
            poi_key="mu",
            toy_generator=gen,
            key=jax.random.key(0),
            distribution_factory=lambda toys: _MarkerDist(marker=0.2222),
        )

        result = calc.test(1.0)

        assert isinstance(result.distribution, _MarkerDist)
        assert result.distribution.marker == 0.2222
        assert float(result.pnull) == pytest.approx(0.2222)
        assert float(calc.pvalue_bands(result).null_pvalue.median) == pytest.approx(0.2222)

    def test_hand_built_result_without_distribution_raises(self):
        """pvalue_bands has no field to fall back on: a result that does not
        carry its distribution is rejected with a clear error."""
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(10.0),
            poi_key="mu",
        )
        ts = TSResult(value=jnp.array(1.0), test=jnp.array(1.0))
        result = HypoTestResult(q_obs=ts.value, pnull=None, palt=None, test_stat_result=ts)

        with pytest.raises(ValueError, match="distribution"):
            calc.pvalue_bands(result)
