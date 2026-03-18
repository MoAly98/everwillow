"""Hypothesis test calculators.

This module provides calculators that orchestrate hypothesis testing by
computing the test statistic, then delegating p-value computation to
Distribution objects.

- ``HypoTestCalculator``: Generic base — forwards all kwargs to the
  test statistic.
- ``AsymptoticCalculator``: Extends the base with explicit Asimov args
  (``predict_fn``, ``mu_asimov``) for Cowan et al. asymptotic workflows.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
from jaxtyping import PyTree

import everwillow.statelib as sl
from everwillow.inference.hypotest._utils import cl_s
from everwillow.inference.hypotest.distributions import Distribution, QTildeAsymptotic
from everwillow.inference.hypotest.results import HypoTestResult
from everwillow.inference.hypotest.test_statistics import QTilde, TestStatistic

__all__ = ["AsymptoticCalculator", "HypoTestCalculator"]


class HypoTestCalculator(eqx.Module):
    """Generic hypothesis test calculator.

    Orchestrates hypothesis testing by:
    1. Computing the test statistic on observed data
    2. Delegating p-value computation to a Distribution object

    All keyword arguments are forwarded to the test statistic's
    ``__call__``. This includes both test-statistic-specific arguments
    (e.g. ``predict_fn``, ``mu_asimov`` for ``CowanTestStatistic``)
    and fit options (e.g. ``solver``, ``max_steps``).

    Attributes:
        test_statistic: Test statistic to use. Defaults to QTilde.
        distribution: Distribution for p-value computation.
            Defaults to QTildeAsymptotic.
    """

    test_statistic: TestStatistic = eqx.field(default_factory=QTilde)
    distribution: Distribution = eqx.field(default_factory=QTildeAsymptotic)

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        **kwargs: tp.Any,
    ) -> HypoTestResult:
        """Run hypothesis test.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data passed to nll_fn.
            poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
            poi_test: Test value for the POI.
            **kwargs: Forwarded to the test statistic. Includes both
                test-statistic-specific args (e.g. ``predict_fn``,
                ``mu_asimov`` for Cowan test statistics) and fit options.

        Returns:
            HypoTestResult with observed p-values.
        """
        ts_result = self.test_statistic(
            nll_fn, params, observation, poi_key, poi_test, **kwargs
        )

        pnull = self.distribution.null_pval(ts_result)
        palt = self.distribution.alt_pval(ts_result)

        cl_s_value = None
        if pnull is not None and palt is not None:
            cl_s_value = cl_s(pnull, palt)

        expected_bands = self.distribution.expected_pvalues(ts_result)

        return HypoTestResult(
            q_obs=ts_result.value,
            pnull=pnull,
            palt=palt,
            cl_s=cl_s_value,
            test_stat_result=ts_result,
            expected_bands=expected_bands,
        )


class AsymptoticCalculator(HypoTestCalculator):
    """Calculator for Cowan et al. asymptotic hypothesis tests.

    Extends ``HypoTestCalculator`` with explicit ``predict_fn`` and
    ``mu_asimov`` keyword arguments for Asimov dataset generation.

    Example:
        >>> calc = AsymptoticCalculator(
        ...     test_statistic=QTilde(),
        ...     distribution=QTildeAsymptotic(),
        ... )
        >>> result = calc(
        ...     nll_fn, params, observed, ("mu",), poi_test=1.0,
        ...     predict_fn=my_predict_fn,
        ... )
    """

    def __call__(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        *,
        predict_fn: tp.Callable[[sl.State], PyTree] | None = None,
        mu_asimov: float = 0.0,
        **kwargs: tp.Any,
    ) -> HypoTestResult:
        """Run asymptotic hypothesis test.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data passed to nll_fn.
            poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
            poi_test: Test value for the POI.
            predict_fn: Function to generate expected observation from parameters.
                Used to create the Asimov dataset at ``mu_asimov``.
            mu_asimov: POI value for Asimov dataset generation.
                Defaults to 0.0 (background-only, for exclusion tests).
                Use 1.0 for discovery tests.
            **kwargs: Additional arguments forwarded to the test statistic
                (e.g. fit options).

        Returns:
            HypoTestResult with observed p-values.
        """
        return super().__call__(
            nll_fn,
            params,
            observation,
            poi_key,
            poi_test,
            predict_fn=predict_fn,
            mu_asimov=mu_asimov,
            **kwargs,
        )
