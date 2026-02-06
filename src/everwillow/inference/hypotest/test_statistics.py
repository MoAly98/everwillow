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

    Subclasses must implement:
        - `__call__`: Compute the test statistic and populate extras

    The extras dict typically contains:
        - q_asimov: Test statistic value from Asimov dataset
        - mu_hat: MLE of the POI
        - fit_free: Free fit result
        - fit_constrained: Constrained fit result
    """

    @abc.abstractmethod
    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        *,
        asimov_nll_fn: tp.Callable[[PyTree], float] | None = None,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute the test statistic.

        Args:
            nll_fn: Negative log-likelihood function.
            params: Initial parameter state.
            poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
            poi_test: Test value for the POI.
            asimov_nll_fn: Optional NLL for Asimov dataset.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            TestStatResult with q value and extras.
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

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        *,
        asimov_nll_fn: tp.Callable[[PyTree], float] | None = None,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute q̃_μ test statistic."""
        q_obs, extras = self._compute_q(nll_fn, params, poi_key, poi_test, **fit_kwargs)

        # Compute q_asimov if asimov_nll_fn provided
        if asimov_nll_fn is not None:
            q_asimov, asimov_extras = self._compute_q(
                asimov_nll_fn, params, poi_key, poi_test, **fit_kwargs
            )
            extras["q_asimov"] = q_asimov
            extras["asimov_fit_constrained"] = asimov_extras.get("fit_constrained")
            extras["asimov_fit_free"] = asimov_extras.get("fit_free")
        else:
            extras["q_asimov"] = q_obs

        return TestStatResult(q=q_obs, extras=extras)

    def _compute_q(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute q̃ for a single NLL function."""
        # Free fit (unconditional MLE)
        fit_free = ew.fit(nll_fn, params, **fit_kwargs)
        fitted_state: sl.State[Array] = sl.State.from_pytree(fit_free.params)
        mu_hat = fitted_state[poi_key]

        # Constrained fit (POI fixed at test value)
        fixed: sl.State[float] = sl.State.from_pytree({poi_key: poi_test})
        fit_constrained = constrained_fit(nll_fn, params, fixed, **fit_kwargs)

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

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        *,
        asimov_nll_fn: tp.Callable[[PyTree], float] | None = None,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute q_μ test statistic."""
        q_obs, extras = self._compute_q(nll_fn, params, poi_key, poi_test, **fit_kwargs)

        if asimov_nll_fn is not None:
            q_asimov, _ = self._compute_q(
                asimov_nll_fn, params, poi_key, poi_test, **fit_kwargs
            )
            extras["q_asimov"] = q_asimov
        else:
            extras["q_asimov"] = q_obs

        return TestStatResult(q=q_obs, extras=extras)

    def _compute_q(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute q_μ for a single NLL function."""
        fit_free = ew.fit(nll_fn, params, **fit_kwargs)
        fitted_state: sl.State[Array] = sl.State.from_pytree(fit_free.params)

        fixed: sl.State[float] = sl.State.from_pytree({poi_key: poi_test})
        fit_constrained = constrained_fit(nll_fn, params, fixed, **fit_kwargs)

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
    """

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        *,
        asimov_nll_fn: tp.Callable[[PyTree], float] | None = None,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute q_0 discovery test statistic.

        Note:
            The ``poi_test`` argument is ignored; Q0 always tests μ=0 by design.
        """
        _ = poi_test  # Unused; Q0 always tests μ=0
        q_obs, extras = self._compute_q(nll_fn, params, poi_key, **fit_kwargs)

        if asimov_nll_fn is not None:
            q_asimov, _ = self._compute_q(asimov_nll_fn, params, poi_key, **fit_kwargs)
            extras["q_asimov"] = q_asimov
        else:
            extras["q_asimov"] = q_obs

        return TestStatResult(q=q_obs, extras=extras)

    def _compute_q(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute q_0 for a single NLL function."""
        fit_free = ew.fit(nll_fn, params, **fit_kwargs)
        fitted_state: sl.State[Array] = sl.State.from_pytree(fit_free.params)
        mu_hat = fitted_state[poi_key]

        fixed: sl.State[float] = sl.State.from_pytree({poi_key: 0.0})
        fit_constrained = constrained_fit(nll_fn, params, fixed, **fit_kwargs)

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

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        *,
        asimov_nll_fn: tp.Callable[[PyTree], float] | None = None,
        **fit_kwargs: tp.Any,
    ) -> TestStatResult:
        """Compute t_μ signed test statistic."""
        t_obs, extras = self._compute_t(nll_fn, params, poi_key, poi_test, **fit_kwargs)

        if asimov_nll_fn is not None:
            t_asimov, _ = self._compute_t(
                asimov_nll_fn, params, poi_key, poi_test, **fit_kwargs
            )
            extras["q_asimov"] = t_asimov
        else:
            extras["q_asimov"] = t_obs

        return TestStatResult(q=t_obs, extras=extras)

    def _compute_t(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        **fit_kwargs: tp.Any,
    ) -> tuple[Array, dict[str, tp.Any]]:
        """Compute t_μ for a single NLL function."""
        fit_free = ew.fit(nll_fn, params, **fit_kwargs)
        fitted_state: sl.State[Array] = sl.State.from_pytree(fit_free.params)
        mu_hat = fitted_state[poi_key]

        fixed: sl.State[float] = sl.State.from_pytree({poi_key: poi_test})
        fit_constrained = constrained_fit(nll_fn, params, fixed, **fit_kwargs)

        delta_nll = fit_constrained.nll - fit_free.nll
        q = 2.0 * delta_nll
        t = jnp.sign(mu_hat - poi_test) * q

        extras = {
            "fit_constrained": fit_constrained,
            "fit_free": fit_free,
            "mu_hat": mu_hat,
        }

        return t, extras
