"""Test statistics for hypothesis testing.

This module provides test statistic classes that compute likelihood ratios.
Each test statistic is an equinox Module that computes the test statistic value
and stores additional information in the extras dict.

The statistical interpretation (p-values) is handled by Distribution classes,
which are separate from the test statistics.

References:
    Cowan et al., "Asymptotic formulae for likelihood-based tests of new physics"
    Eur. Phys. J. C 71 (2011) 1554, arXiv:1007.1727
"""

from __future__ import annotations

import abc
import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

import everwillow._src.statelib as sl
from everwillow._src.inference.fitting import fit
from everwillow._src.inference.hypotest.results import TestStatResult
from everwillow._src.inference.hypotest.utils import constrained_fit, make_asimov, single_poi_key

PoiPoint: tp.TypeAlias = tp.Mapping[sl.K, ArrayLike]

__all__ = [
    "Q0",
    "CowanTestStatistic",
    "QMu",
    "QTilde",
    "TMu",
    "TestStatistic",
]


class TestStatistic(eqx.Module):
    """Abstract base class for test statistics.

    Test statistics compute likelihood ratios and store relevant data in the
    TestStatResult.extras dict. The statistical interpretation (p-values) is
    handled separately by Distribution classes.

    Subclasses must implement:
        - ``_compute``: Compute the core test statistic formula.

    """

    def compute(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_test: PoiPoint,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute the test statistic.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data passed to nll_fn.
            poi_test: The tested POI point, a mapping from POI key to value,
                e.g. ``{"mu": 1.0}`` or ``{"mu_ggF": 1.0, "mu_VBF": 0.5}``.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            TestStatResult with value, test, q_asimov, and extras.
        """
        q_obs, extras = self._compute(nll_fn, params, observation, poi_test, **fit_kwargs)

        point = {key: jnp.asarray(value) for key, value in poi_test.items()}
        return TestStatResult(value=q_obs, test=point, q_asimov=None, extras=extras)

    @abc.abstractmethod
    def _compute(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_test: PoiPoint,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute the core test statistic formula.

        Subclasses implement this method with their specific formula.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data passed to nll_fn.
            poi_test: The tested POI point, a mapping from POI key to value.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            Tuple of (q_value, extras_dict).
        """
        ...


class CowanTestStatistic(TestStatistic):
    r"""Cowan test statistics with Asimov-based variance estimation.

    There are five Cowan test statistics:

    1. TMu: used for two-sided confidence intervals
    2. TMuTilde: used for confidence intervals with a positive signal (Feldman-Cousins)
    3. Q0: used for discovery tests (rejecting the :math:`\mu = 0` hypothesis)
    4. QMu: used for exclusion of a non-zero signal hypothesis
    5. QMuTilde: used for exclusion of a non-zero signal hypothesis with a positive signal

    This subclass extends ``TestStatistic`` with methods to efficiently
    compute the variance of :math:`\hat{\mu}` using the Asimov dataset, as described in the Cowan paper.

    Asimov data can be provided in two ways:

    1. ``asimov_observation``: pre-computed Asimov dataset.
    2. ``predict_fn``: generate Asimov at ``poi_asimov`` (default depends on
       the test statistic; override via the ``poi_asimov`` kwarg).

    If neither is provided, ``q_asimov`` will be None. This can cause p-value computations for test statistics
    that require ``q_asimov`` to fail.

    Attributes:
        poi_asimov: Default POI point for Asimov dataset generation. A scalar
            sets every tested POI to that value.
    """

    poi_asimov: float | PoiPoint = 0.0

    def compute(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_test: PoiPoint,
        *,
        asimov_observation: PyTree | None = None,
        predict_fn: tp.Callable[[sl.State], PyTree] | None = None,
        poi_asimov: float | PoiPoint | None = None,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute the test statistic.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data passed to nll_fn.
            poi_test: The tested POI point, a mapping from POI key to value,
                e.g. ``{"mu": 1.0}``.
            asimov_observation: Pre-computed Asimov dataset.
            predict_fn: Function to generate expected observation from parameters.
            poi_asimov: POI point at which to generate the Asimov dataset. A scalar
                sets every tested POI to that value (0.0 for background-only, 1.0
                for nominal signal); a mapping sets each POI separately. Defaults
                to ``self.poi_asimov``.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            TestStatResult with value, test, q_asimov, and extras.
        """
        q_obs, extras = self._compute(nll_fn, params, observation, poi_test, **fit_kwargs)

        if poi_asimov is None:
            poi_asimov = self.poi_asimov

        asimov_obs = self._resolve_asimov(asimov_observation, predict_fn, params, poi_test, poi_asimov)

        q_asimov = None
        if asimov_obs is not None:
            q_asimov_val, asimov_extras = self._compute(nll_fn, params, asimov_obs, poi_test, **fit_kwargs)
            q_asimov = q_asimov_val
            extras["asimov_fit_constrained"] = asimov_extras.get("fit_constrained")
            extras["asimov_fit_free"] = asimov_extras.get("fit_free")

        point = {key: jnp.asarray(value) for key, value in poi_test.items()}
        return TestStatResult(value=q_obs, test=point, q_asimov=q_asimov, extras=extras)

    @staticmethod
    def _resolve_asimov(
        asimov_observation: PyTree | None,
        predict_fn: tp.Callable[[sl.State], PyTree] | None,
        params: sl.State,
        poi_test: PoiPoint,
        poi_asimov: float | PoiPoint,
    ) -> PyTree | None:
        """Resolve Asimov observation from explicit data or predict_fn.

        When ``predict_fn`` is used, the Asimov dataset is generated at the
        Asimov POI point (not at ``poi_test``). A scalar ``poi_asimov`` is
        broadcast to every tested POI; a mapping is used directly.
        """
        if asimov_observation is not None:
            return asimov_observation
        if predict_fn is not None:
            point = dict.fromkeys(poi_test, poi_asimov) if isinstance(poi_asimov, (int, float)) else poi_asimov
            return make_asimov(predict_fn, params, point)
        return None


class QTilde(CowanTestStatistic):
    r"""Modified profile likelihood ratio for upper limits (Eq. 16).

    The test statistic is:

    .. math::

        \tilde{q}_\mu = \begin{cases}
            -2 \ln \frac{L(\mu, \hat{\hat{\theta}}(\mu))}{L(0, \hat{\hat{\theta}}(0))}
                & \text{if } \hat{\mu} < 0 \\
            -2 \ln \frac{L(\mu, \hat{\hat{\theta}}(\mu))}{L(\hat{\mu}, \hat{\theta})}
                & \text{if } 0 \leq \hat{\mu} \leq \mu \\
            0 & \text{if } \hat{\mu} > \mu
        \end{cases}

    where :math:`\hat{\hat{\theta}}(\mu)` is the conditional MLE of the
    nuisance parameters given :math:`\mu`, and :math:`\hat{\mu}, \hat{\theta}`
    are the unconditional MLEs. The boundary at :math:`\hat{\mu} > \mu`
    protects against excluding signal when there is an upward fluctuation.

    This is the standard test statistic for CLs upper limit calculations.
    """

    def _compute(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_test: PoiPoint,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute q̃ for a single observation."""
        poi_key = single_poi_key(poi_test)
        poi_value = poi_test[poi_key]

        # Free fit (unconditional MLE)
        fit_free = fit(nll_fn, params, observation, **fit_kwargs)
        fitted_state: sl.State[Array] = fit_free.params
        mu_hat = fitted_state[poi_key]

        # Constrained fit at mu_test: L(μ, θ̂̂(μ))
        fit_constrained = constrained_fit(nll_fn, params, observation, poi_test, **fit_kwargs)

        # Constrained fit at μ=0: L(0, θ̂̂(0)) — denominator when μ̂ < 0
        # Both branches are always evaluated (JAX tracing); jnp.where selects.
        fit_zero = constrained_fit(nll_fn, params, observation, {poi_key: 0.0}, **fit_kwargs)

        # Eq. 16: denominator is L(0, θ̂̂(0)) when μ̂ < 0, else L(μ̂, θ̂)
        nll_denom = jnp.where(mu_hat < 0.0, fit_zero.nll, fit_free.nll)
        delta_nll = fit_constrained.nll - nll_denom
        q_raw = 2.0 * delta_nll

        # Boundary: q = 0 if μ̂ > μ (upward fluctuation)
        q = jnp.where(mu_hat <= poi_value, q_raw, 0.0)
        q = jnp.maximum(q, 0.0)

        extras = {
            "fit_constrained": fit_constrained,
            "fit_free": fit_free,
            "fit_zero": fit_zero,
            "mu_hat": mu_hat,
        }

        return q, extras


class QMu(CowanTestStatistic):
    r"""Profile likelihood ratio for upper limits (Eq. 14).

    The test statistic is:

    .. math::

        q_\mu = \begin{cases}
            -2 \ln \lambda(\mu) & \text{if } \hat{\mu} \leq \mu \\
            0 & \text{if } \hat{\mu} > \mu
        \end{cases}

    where :math:`\lambda(\mu) = L(\mu, \hat{\hat{\theta}}(\mu)) / L(\hat{\mu}, \hat{\theta})`
    is the profile likelihood ratio. The boundary at :math:`\hat{\mu} > \mu`
    protects against excluding signal when there is an upward fluctuation.
    """

    def _compute(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_test: PoiPoint,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute q_μ for a single observation."""
        poi_key = single_poi_key(poi_test)
        poi_value = poi_test[poi_key]

        fit_free = fit(nll_fn, params, observation, **fit_kwargs)
        fitted_state: sl.State[Array] = fit_free.params
        mu_hat = fitted_state[poi_key]

        fit_constrained = constrained_fit(nll_fn, params, observation, poi_test, **fit_kwargs)

        delta_nll = fit_constrained.nll - fit_free.nll
        q_raw = 2.0 * delta_nll

        # Boundary: q = 0 if mu_hat > poi_test (upward fluctuation)
        q = jnp.where(mu_hat <= poi_value, q_raw, 0.0)
        q = jnp.maximum(q, 0.0)

        extras = {
            "fit_constrained": fit_constrained,
            "fit_free": fit_free,
            "mu_hat": mu_hat,
        }

        return q, extras


class Q0(CowanTestStatistic):
    r"""Discovery test statistic for testing :math:`\mu = 0` (Eq. 12).

    The test statistic is:

    .. math::

        q_0 = \begin{cases}
            -2 \ln \lambda(0) & \text{if } \hat{\mu} \geq 0 \\
            0 & \text{if } \hat{\mu} < 0
        \end{cases}

    where :math:`\lambda(0) = L(0, \hat{\hat{\theta}}(0)) / L(\hat{\mu}, \hat{\theta})`
    is the profile likelihood ratio evaluated at :math:`\mu = 0`.
    The boundary at :math:`\hat{\mu} < 0` prevents "discovery" of negative signal.

    Attributes:
        poi_asimov: Default POI point for Asimov generation. Defaults to 1.0
            (signal hypothesis).
    """

    poi_asimov: float | PoiPoint = 1.0

    def compute(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_test: PoiPoint,
        *,
        asimov_observation: PyTree | None = None,
        predict_fn: tp.Callable[[sl.State], PyTree] | None = None,
        poi_asimov: float | PoiPoint | None = None,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute q_0 discovery test statistic.

        Note:
            The tested value is ignored; Q0 always tests μ=0 by design. The
            tested point still names the POI, so it must name exactly one.
        """
        poi_key = single_poi_key(poi_test)
        return super().compute(
            nll_fn,
            params,
            observation,
            {poi_key: 0.0},
            asimov_observation=asimov_observation,
            predict_fn=predict_fn,
            poi_asimov=poi_asimov,
            **fit_kwargs,
        )

    def _compute(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_test: PoiPoint,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute q_0 for a single observation."""
        # poi_test is always {poi_key: 0.0} due to the compute() override
        poi_key = single_poi_key(poi_test)

        fit_free = fit(nll_fn, params, observation, **fit_kwargs)
        fitted_state: sl.State[Array] = fit_free.params
        mu_hat = fitted_state[poi_key]

        fit_constrained = constrained_fit(nll_fn, params, observation, poi_test, **fit_kwargs)

        delta_nll = fit_constrained.nll - fit_free.nll
        q_raw = 2.0 * delta_nll

        q = jnp.where(mu_hat >= 0.0, q_raw, 0.0)
        q = jnp.maximum(q, 0.0)

        extras = {
            "fit_constrained": fit_constrained,
            "fit_free": fit_free,
            "mu_hat": mu_hat,
        }

        return q, extras


class TMu(CowanTestStatistic):
    r"""Profile likelihood ratio for two-sided confidence intervals (Eq. 8).

    The test statistic is:

    .. math::

        t_\mu = -2 \ln \lambda(\mu)

    where :math:`\lambda(\mu) = L(\mu, \hat{\hat{\theta}}(\mu)) / L(\hat{\mu}, \hat{\theta})`
    is the profile likelihood ratio. No boundary is applied; :math:`t_\mu` can
    take any non-negative value regardless of :math:`\hat{\mu}`.
    """

    def _compute(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_test: PoiPoint,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute t_μ for a single observation.

        Handles a joint point of any dimension: the constrained fit fixes every
        POI in ``poi_test``, and ``t`` follows a chi-square with as many degrees
        of freedom as POIs (Cowan Eq. 21).
        """
        fit_free = fit(nll_fn, params, observation, **fit_kwargs)
        fitted_state: sl.State[Array] = fit_free.params
        # The fitted POI estimate is the sub-mapping over the tested keys.
        mu_hat = {key: fitted_state[key] for key in poi_test}

        fit_constrained = constrained_fit(nll_fn, params, observation, poi_test, **fit_kwargs)

        delta_nll = fit_constrained.nll - fit_free.nll
        t = 2.0 * delta_nll

        extras = {
            "fit_constrained": fit_constrained,
            "fit_free": fit_free,
            "mu_hat": mu_hat,
        }

        return t, extras
