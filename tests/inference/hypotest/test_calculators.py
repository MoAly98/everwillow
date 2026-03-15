"""Tests for hypothesis test calculators.

Tests the calculator orchestration with concrete expected values.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

import everwillow.statelib as sl
from everwillow.inference.hypotest import (
    AsymptoticCalculator,
    ExpectedBands,
    HypoTestCalculator,
    QMu,
    QTilde,
    QTildeAsymptotic,
    cl_s,
    significance,
)

# =============================================================================
# Test fixtures: Simple Poisson counting experiment
# =============================================================================

S = 10.0  # signal yield
B = 5.0  # background yield


def poisson_nll(params, observation):
    """Poisson NLL for a simple counting experiment."""
    mu = params["mu"]
    n_expected = mu * S + B
    n_observed = observation["n"]
    return n_expected - n_observed * jnp.log(n_expected)


def create_params(mu_init: float = 1.0) -> sl.State:
    """Create initial parameter state."""
    return sl.State.from_pytree({"mu": mu_init})


def create_observation(n: float) -> dict[str, float]:
    """Create observation dict."""
    return {"n": n}


def predict_fn(params_state: sl.State) -> dict[str, float]:
    """Prediction function for Asimov data."""
    mu = params_state.to_pytree()["mu"]
    return {"n": mu * S + B}


def expected_q(n_obs: float, mu_test: float) -> float:
    """Compute analytical q value."""
    n_exp_test = mu_test * S + B
    return 2.0 * (n_exp_test - n_obs - n_obs * math.log(n_exp_test / n_obs))


def normal_sf(x: float) -> float:
    """Standard normal survival function: 1 - CDF."""
    return 0.5 * (1 - math.erf(x / math.sqrt(2)))


# =============================================================================
# AsymptoticCalculator Tests
# =============================================================================


class TestAsymptoticCalculator:
    """Tests for AsymptoticCalculator."""

    def test_basic_result_structure(self):
        """Test that calculator returns proper HypoTestResult."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = AsymptoticCalculator(
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert hasattr(result, "q_obs")
        assert hasattr(result, "pnull")
        assert hasattr(result, "palt")
        assert hasattr(result, "cl_s")
        assert hasattr(result, "expected_bands")
        assert hasattr(result, "test_stat_result")

    def test_q_obs_at_mle(self):
        """At MLE, q=0.

        n_obs=15 for mu=1: mu_hat=1=mu_test, q=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = AsymptoticCalculator(
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert result.q_obs == pytest.approx(0.0, abs=1e-5)

    def test_q_asimov_with_asimov_observation(self):
        """Test q_asimov with explicit Asimov observation.

        Asimov at mu=1 (n=15), testing at mu=1: q_asimov=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)
        asimov = create_observation(15.0)

        calc = HypoTestCalculator(
            test_statistic=QTilde(), distribution=QTildeAsymptotic()
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            asimov_observation=asimov,
        )

        assert result.test_stat_result.q_asimov == pytest.approx(0.0, abs=1e-4)

    def test_q_asimov_with_predict_fn(self):
        """Test that predict_fn generates Asimov at mu_asimov.

        mu_asimov=0 by default.
        Asimov at mu=0: n_asimov = 5
        Testing at mu=1: q_asimov = 2*(15-5-5*ln(3)) ≈ 9.014
        """
        expected_q_asimov = expected_q(5.0, 1.0)  # ~9.014

        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = AsymptoticCalculator(
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert result.test_stat_result.q_asimov == pytest.approx(
            expected_q_asimov, rel=1e-3
        )

    def test_q_asimov_at_different_mu_test(self):
        """Test that Asimov is always at mu_asimov, regardless of mu_test.

        At mu_test=0: Asimov at mu=0 (n=5), testing at 0 → q_asimov=0
        At mu_test=2: Asimov at mu=0 (n=5), testing at 2 → q_asimov≈23.9
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = AsymptoticCalculator(
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )

        # Test at mu=0
        result_0 = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=0.0,
            predict_fn=predict_fn,
        )
        assert result_0.test_stat_result.q_asimov == pytest.approx(0.0, abs=1e-4)

        # Test at mu=2
        expected_q_asimov_2 = expected_q(5.0, 2.0)
        result_2 = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=2.0,
            predict_fn=predict_fn,
        )
        assert result_2.test_stat_result.q_asimov == pytest.approx(
            expected_q_asimov_2, rel=1e-3
        )

    def test_pvalues_computed(self):
        """Test that pnull and palt are finite after distribution call."""
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = AsymptoticCalculator(
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert jnp.isfinite(result.pnull)
        assert jnp.isfinite(result.palt)

    def test_custom_test_statistic(self):
        """Test calculator with QMu instead of QTilde."""
        params = create_params(mu_init=1.0)
        observed = create_observation(25.0)

        calc = AsymptoticCalculator(
            test_statistic=QMu(),
            distribution=QTildeAsymptotic(),
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        # QMu doesn't have boundary, so q > 0 for upward fluctuation
        assert float(result.q_obs) > 0.0


# =============================================================================
# HypoTestCalculator Tests (generic base)
# =============================================================================


class TestHypoTestCalculator:
    """Tests for the generic HypoTestCalculator base."""

    def test_default_test_statistic(self):
        """Test that default test statistic is QTilde."""
        calc = HypoTestCalculator()
        assert isinstance(calc.test_statistic, QTilde)

    def test_kwargs_passthrough(self):
        """Test that kwargs are forwarded to test statistic."""
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = HypoTestCalculator(
            test_statistic=QTilde(), distribution=QTildeAsymptotic()
        )
        # predict_fn passed as kwarg, forwarded to CowanTestStatistic
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert result.test_stat_result.q_asimov is not None

    def test_without_predict_fn(self):
        """Test that calculator works without predict_fn (no Asimov).

        Without Asimov, q_asimov is None and piecewise distributions
        (QTildeAsymptotic) return None for p-values that need it.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = HypoTestCalculator(
            test_statistic=QTilde(), distribution=QTildeAsymptotic()
        )
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            result = calc(
                poisson_nll,
                params,
                observed,
                ("mu",),
                poi_test=1.0,
            )

        assert result.test_stat_result.q_asimov is None
        assert result.pnull is None
        assert result.palt is None


# =============================================================================
# CLs Tests
# =============================================================================


class TestClS:
    """Tests for cl_s utility: CLs = pnull / palt (Cowan et al., ATLAS convention).

    Uses QMu asymptotic p-values (Cowan Eq. 57-59) as reference values.
    For QMu with μ tested, σ=1, Asimov at μ'=0:
        pnull = 1 - Φ(√q)        (p_μ: upper tail under signal)
        palt  = 1 - Φ(√q - μ/σ)  (CL_b: upper tail under background)
    """

    @pytest.mark.parametrize(
        ("pnull", "palt", "expected_cls"),
        [
            # QMu at median upper limit (μ=1.96σ, band N=0):
            # q = 1.96², pnull = 1-Φ(1.96) = 0.025, palt = 1-Φ(0) = 0.5
            (0.025, 0.5, 0.05),
            # QMu at +1σ band (μ=2.727σ, band N=1):
            # q = 1.727², pnull = 1-Φ(1.727) ≈ 0.0421, palt = Φ(1) ≈ 0.8413
            (0.0421, 0.8413, 0.05004),
            # QMu at -1σ band (μ=1.412σ, band N=-1):
            # q = 2.412², pnull = 1-Φ(2.412) ≈ 0.00793, palt = Φ(-1) ≈ 0.1587
            (0.00793, 0.1587, 0.04997),
        ],
        ids=["median-band", "plus-1sigma-band", "minus-1sigma-band"],
    )
    def test_qmu_cls_at_expected_upper_limit(self, pnull, palt, expected_cls):
        """CLs ≈ 0.05 at each band's expected upper limit (σ=1, α=0.05)."""
        result = float(cl_s(jnp.array(pnull), jnp.array(palt)))
        assert result == pytest.approx(expected_cls, rel=1e-3)

    def test_cls_inflated_when_no_sensitivity(self):
        """CLs method protects against false exclusion.

        When background also finds data unlikely (small palt), CLs
        is inflated even if pnull is small. This is the key property
        of the CLs method — it prevents excluding signals when
        the experiment has no sensitivity.
        """
        result = float(cl_s(jnp.array(0.001), jnp.array(0.002)))
        assert result == pytest.approx(0.5)


# =============================================================================
# ExpectedBands Tests
# =============================================================================


class TestExpectedBandsClsBands:
    """Tests for ExpectedBands.cls_bands() with known QMu values.

    Uses QMu p-values at the expected upper limit (σ=1, α=0.05).
    At the upper limit for each band, CLs = pnull/palt = 0.05.
    Different bands have different pnull and palt, verifying that
    cls_bands() correctly pairs the right elements of each tuple.
    """

    def test_cls_bands_qmu_at_expected_upper_limit(self):
        """CLs = 0.05 at each band's expected upper limit.

        QMu expected p-values at μ_up(N) = Φ⁻¹(1 - α·Φ(N)) + N:
            Band N | μ_up  | pnull = 1-Φ(μ_up-N) | palt = Φ(N)  | CLs
            -2     | 1.052 | 1-Φ(3.052)=0.001138 | Φ(-2)=0.02275| 0.05
            -1     | 1.412 | 1-Φ(2.412)=0.00793  | Φ(-1)=0.15866| 0.05
             0     | 1.960 | 1-Φ(1.960)=0.02500  | Φ(0) =0.50000| 0.05
            +1     | 2.727 | 1-Φ(1.727)=0.04213  | Φ(1) =0.84134| 0.05
            +2     | 3.656 | 1-Φ(1.656)=0.04883  | Φ(2) =0.97725| 0.05
        """
        bands = ExpectedBands(
            minus_2sigma=(jnp.array(0.001138), jnp.array(0.02275)),
            minus_1sigma=(jnp.array(0.00793), jnp.array(0.15866)),
            median=(jnp.array(0.02500), jnp.array(0.50000)),
            plus_1sigma=(jnp.array(0.04213), jnp.array(0.84134)),
            plus_2sigma=(jnp.array(0.04883), jnp.array(0.97725)),
        )

        cls_values = bands.cls_bands()

        # abs=1e-3 accounts for rounding in the hardcoded p-values above
        assert float(cls_values[0]) == pytest.approx(0.05, abs=1e-3)
        assert float(cls_values[1]) == pytest.approx(0.05, abs=1e-3)
        assert float(cls_values[2]) == pytest.approx(0.05, abs=1e-3)
        assert float(cls_values[3]) == pytest.approx(0.05, abs=1e-3)
        assert float(cls_values[4]) == pytest.approx(0.05, abs=1e-3)


class TestExpectedBandsSignificanceBands:
    """Tests for ExpectedBands significance band methods.

    Uses known QMu p-values for μ=2, σ=1, q_A=4:
        Z = Φ⁻¹(1-p) converts p-value to significance.
    """

    @pytest.fixture
    def qmu_bands(self) -> ExpectedBands:
        """ExpectedBands with known QMu p-values (μ=2, σ=1, q_A=4).

        Band   | pnull      | palt
        -2σ    | 3.167e-5   | 0.02275
        -1σ    | 0.00135    | 0.15866
        median | 0.02275    | 0.5
        +1σ    | 0.15866    | 0.84134
        +2σ    | 0.5        | 0.97725
        """
        return ExpectedBands(
            minus_2sigma=(jnp.array(3.167e-5), jnp.array(0.02275)),
            minus_1sigma=(jnp.array(0.00135), jnp.array(0.15866)),
            median=(jnp.array(0.02275), jnp.array(0.5)),
            plus_1sigma=(jnp.array(0.15866), jnp.array(0.84134)),
            plus_2sigma=(jnp.array(0.5), jnp.array(0.97725)),
        )

    def test_null_significance_bands(self, qmu_bands: ExpectedBands):
        """Z_null at each band: 4.0, 3.0, 2.0, 1.0, 0.0."""
        z_bands = qmu_bands.null_significance_bands()

        assert float(z_bands[0]) == pytest.approx(4.0, abs=0.01)
        assert float(z_bands[1]) == pytest.approx(3.0, abs=0.01)
        assert float(z_bands[2]) == pytest.approx(2.0, abs=0.01)
        assert float(z_bands[3]) == pytest.approx(1.0, abs=0.01)
        assert float(z_bands[4]) == pytest.approx(0.0, abs=0.01)

    def test_alt_significance_bands(self, qmu_bands: ExpectedBands):
        """Z_alt at each band: 2.0, 1.0, 0.0, -1.0, -2.0."""
        z_bands = qmu_bands.alt_significance_bands()

        assert float(z_bands[0]) == pytest.approx(2.0, abs=0.01)
        assert float(z_bands[1]) == pytest.approx(1.0, abs=0.01)
        assert float(z_bands[2]) == pytest.approx(0.0, abs=0.01)
        assert float(z_bands[3]) == pytest.approx(-1.0, abs=0.01)
        assert float(z_bands[4]) == pytest.approx(-2.0, abs=0.01)

    def test_significance_utility_known_values(self):
        """Test standalone significance() with known p-value → Z mappings."""
        assert float(significance(jnp.array(0.5))) == pytest.approx(0.0, abs=1e-6)
        assert float(significance(jnp.array(0.02275))) == pytest.approx(2.0, abs=0.01)
        assert float(significance(jnp.array(0.15866))) == pytest.approx(1.0, abs=0.01)
        assert float(significance(jnp.array(0.00135))) == pytest.approx(3.0, abs=0.01)
