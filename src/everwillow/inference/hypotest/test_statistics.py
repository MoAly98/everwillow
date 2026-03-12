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
from jaxtyping import Array, PyTree

import everwillow as ew
import everwillow.statelib as sl
from everwillow.inference.hypotest._results import TestStatResult
from everwillow.inference.hypotest._utils import constrained_fit

__all__ = [
    "Q0",
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

    Asimov data can be provided in two ways:

    1. ``asimov_observation``: pre-computed Asimov dataset.
    2. ``predict_fn``: generate Asimov at ``mu_asimov`` (default depends on
       the test statistic; override via the ``mu_asimov`` kwarg).

    If neither is provided, ``q_asimov`` will be None.

    Subclasses must implement:
        - ``_compute_q``: Compute the core test statistic formula.

    Attributes:
        mu_asimov: Default POI value for Asimov dataset generation.
    """

    mu_asimov: float = 0.0

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        *,
        asimov_observation: PyTree | None = None,
        predict_fn: tp.Callable[[sl.State], PyTree] | None = None,
        mu_asimov: float | None = None,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute the test statistic.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data passed to nll_fn.
            poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
            poi_test: Test value for the POI.
            asimov_observation: Pre-computed Asimov dataset.
            predict_fn: Function to generate expected observation from parameters.
            mu_asimov: POI value at which to generate the Asimov dataset.
                Defaults to ``self.mu_asimov``.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            TestStatResult with value, test, q_asimov, and extras.
        """
        q_obs, extras = self._compute_q(
            nll_fn, params, observation, poi_key, poi_test, **fit_kwargs
        )

        if mu_asimov is None:
            mu_asimov = self.mu_asimov

        asimov_obs = self._resolve_asimov(
            asimov_observation, predict_fn, params, poi_key, mu_asimov
        )

        q_asimov = None
        if asimov_obs is not None:
            q_asimov_val, asimov_extras = self._compute_q(
                nll_fn, params, asimov_obs, poi_key, poi_test, **fit_kwargs
            )
            q_asimov = q_asimov_val
            extras["asimov_fit_constrained"] = asimov_extras.get("fit_constrained")
            extras["asimov_fit_free"] = asimov_extras.get("fit_free")

        return TestStatResult(
            value=q_obs, test=jnp.asarray(poi_test), q_asimov=q_asimov, extras=extras
        )

    @staticmethod
    def _resolve_asimov(
        asimov_observation: PyTree | None,
        predict_fn: tp.Callable[[sl.State], PyTree] | None,
        params: sl.State,
        poi_key: sl.K,
        mu_asimov: float,
    ) -> PyTree | None:
        """Resolve Asimov observation from explicit data or predict_fn.

        When ``predict_fn`` is used, the Asimov dataset is generated at
        ``mu_asimov`` (not at ``poi_test``).
        """
        if asimov_observation is not None:
            return asimov_observation
        if predict_fn is not None:
            asimov_params = sl.update(params, updates={poi_key: mu_asimov})
            return predict_fn(asimov_params)
        return None

    @abc.abstractmethod
    def _compute_q(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute the core test statistic formula.

        Subclasses implement this method with their specific formula.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data passed to nll_fn.
            poi_key: Canonical key for the parameter of interest.
            poi_test: Test value for the POI.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            Tuple of (q_value, extras_dict).
        """
        ...


class QTilde(TestStatistic):
    """Profile likelihood ratio with boundary handling for upper limits.

    The test statistic is:
        q̃_μ = -2 ln(L(μ)/L(μ̂)) if μ̂ ≤ μ
             = 0                 if μ̂ > μ

    The boundary at μ̂ > μ protects against excluding signal when there's
    an upward fluctuation. This is the standard test statistic for CLs
    upper limit calculations.
    """

    def _compute_q(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute q̃ for a single observation."""
        # Free fit (unconditional MLE)
        fit_free = ew.fit(nll_fn, params, observation, **fit_kwargs)
        fitted_state: sl.State[Array] = sl.State.from_pytree(fit_free.params)
        mu_hat = fitted_state[poi_key]

        # Constrained fit (POI fixed at test value)
        fixed: sl.State[float] = sl.State.from_pytree({poi_key: poi_test})
        fit_constrained = constrained_fit(
            nll_fn, params, observation, fixed, **fit_kwargs
        )

        # Profile likelihood ratio
        delta_nll = fit_constrained.nll - fit_free.nll
        q_raw = 2.0 * delta_nll

        # Boundary: q = 0 if mu_hat > poi_test (upward fluctuation)
        q = jnp.where(mu_hat <= poi_test, q_raw, 0.0)
        q = jnp.maximum(q, 0.0)

        extras = {
            "fit_constrained": fit_constrained,
            "fit_free": fit_free,
            "mu_hat": mu_hat,
            "poi_test": poi_test,
        }

        return q, extras


class QMu(TestStatistic):
    """Profile likelihood ratio without boundary handling.

    The test statistic is:
        q_μ = -2 ln(L(μ)/L(μ̂))

    No boundary is applied. Use for general hypothesis testing.
    """

    def _compute_q(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute q_μ for a single observation."""
        fit_free = ew.fit(nll_fn, params, observation, **fit_kwargs)
        fitted_state: sl.State[Array] = sl.State.from_pytree(fit_free.params)

        fixed: sl.State[float] = sl.State.from_pytree({poi_key: poi_test})
        fit_constrained = constrained_fit(
            nll_fn, params, observation, fixed, **fit_kwargs
        )

        delta_nll = fit_constrained.nll - fit_free.nll
        q = 2.0 * delta_nll

        extras = {
            "fit_constrained": fit_constrained,
            "fit_free": fit_free,
            "mu_hat": fitted_state[poi_key],
        }

        return q, extras


class Q0(TestStatistic):
    """Discovery test statistic for testing μ = 0.

    The test statistic is:
        q_0 = -2 ln(L(0)/L(μ̂)) if μ̂ ≥ 0
            = 0                  if μ̂ < 0

    The boundary at μ̂ < 0 prevents "discovery" of negative signal.

    Attributes:
        mu_asimov: Default POI value for Asimov generation. Defaults to 1.0
            (signal hypothesis).
    """

    mu_asimov: float = 1.0

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        *,
        asimov_observation: PyTree | None = None,
        predict_fn: tp.Callable[[sl.State], PyTree] | None = None,
        mu_asimov: float | None = None,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute q_0 discovery test statistic.

        Note:
            The ``poi_test`` argument is ignored; Q0 always tests μ=0 by design.
        """
        return super().__call__(
            nll_fn,
            params,
            observation,
            poi_key,
            0.0,
            asimov_observation=asimov_observation,
            predict_fn=predict_fn,
            mu_asimov=mu_asimov,
            **fit_kwargs,
        )

    def _compute_q(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute q_0 for a single observation."""
        # poi_test will always be 0.0 due to __call__ override
        fit_free = ew.fit(nll_fn, params, observation, **fit_kwargs)
        fitted_state: sl.State[Array] = sl.State.from_pytree(fit_free.params)
        mu_hat = fitted_state[poi_key]

        fixed: sl.State[float] = sl.State.from_pytree({poi_key: 0.0})
        fit_constrained = constrained_fit(
            nll_fn, params, observation, fixed, **fit_kwargs
        )

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


class TMu(TestStatistic):
    """Signed test statistic for two-sided confidence intervals.

    The test statistic is:
        t_μ = sign(μ̂ - μ) × q_μ

    where q_μ = -2 ln(L(μ)/L(μ̂)).
    """

    def _compute_q(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute t_μ for a single observation."""
        fit_free = ew.fit(nll_fn, params, observation, **fit_kwargs)
        fitted_state: sl.State[Array] = sl.State.from_pytree(fit_free.params)
        mu_hat = fitted_state[poi_key]

        fixed: sl.State[float] = sl.State.from_pytree({poi_key: poi_test})
        fit_constrained = constrained_fit(
            nll_fn, params, observation, fixed, **fit_kwargs
        )

        delta_nll = fit_constrained.nll - fit_free.nll
        q = 2.0 * delta_nll
        t = jnp.sign(mu_hat - poi_test) * q

        extras = {
            "fit_constrained": fit_constrained,
            "fit_free": fit_free,
            "mu_hat": mu_hat,
        }

        return t, extras
