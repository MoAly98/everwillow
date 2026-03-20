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
from jaxtyping import Array, PyTree

import everwillow.statelib as sl
from everwillow.inference.hypotest._utils import cl_s
from everwillow.inference.hypotest.distributions import Distribution, QTildeAsymptotic
from everwillow.inference.hypotest.results import ExpectedBands, HypoTestResult
from everwillow.inference.hypotest.test_statistics import QTilde, TestStatistic

__all__ = ["AsymptoticCalculator", "HypoTestCalculator"]


class HypoTestCalculator(eqx.Module):
    """Generic hypothesis test calculator.

    Orchestrates hypothesis testing by:
    1. Computing the test statistic on observed data
    2. Delegating p-value computation to a Distribution object

    The calculator stores all model-specific arguments at construction time,
    so ``test(poi_test)`` only takes the varying parameter. Additional
    keyword arguments to ``test()`` are forwarded to the test statistic.

    Attributes:
        nll_fn: Negative log-likelihood function taking (params, observation).
        params: Initial parameter state.
        observation: Observed data passed to nll_fn.
        poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
        test_statistic: Test statistic to use. Defaults to QTilde.
        distribution: Distribution for p-value computation.
            Defaults to QTildeAsymptotic.
    """

    nll_fn: tp.Callable[[PyTree, PyTree], float]
    params: sl.State
    observation: PyTree
    poi_key: sl.K
    test_statistic: TestStatistic = eqx.field(default_factory=QTilde)
    distribution: Distribution = eqx.field(default_factory=QTildeAsymptotic)

    def test(
        self,
        poi_test: float,
        **kwargs: tp.Any,
    ) -> HypoTestResult:
        """Run hypothesis test.

        Args:
            poi_test: Test value for the POI.
            **kwargs: Forwarded to the test statistic. Includes both
                test-statistic-specific args (e.g. ``predict_fn``,
                ``mu_asimov`` for Cowan test statistics) and fit options.

        Returns:
            HypoTestResult with observed p-values.
        """
        ts_result = self.test_statistic(
            self.nll_fn,
            self.params,
            self.observation,
            self.poi_key,
            poi_test,
            **kwargs,
        )

        pnull = self.distribution.null_pval(ts_result)
        palt = self.distribution.alt_pval(ts_result)

        return HypoTestResult(
            q_obs=ts_result.value,
            pnull=pnull,
            palt=palt,
            test_stat_result=ts_result,
        )

    def cls(self, result: HypoTestResult) -> Array | None:
        """Compute CLs = pnull / palt from a hypothesis test result.

        Args:
            result: HypoTestResult from ``test()``.

        Returns:
            CLs value, or None if either p-value is None.
        """
        if result.pnull is None or result.palt is None:
            return None
        return cl_s(result.pnull, result.palt)

    def expected(self, result: HypoTestResult) -> ExpectedBands | None:
        """Compute expected p-values at standard sigma bands.

        Delegates to the distribution's ``expected_pvalues`` method.

        Args:
            result: HypoTestResult from ``test()``.

        Returns:
            ExpectedBands with p-values at each sigma level.

        Raises:
            NotImplementedError: If the distribution does not support
                expected p-value computation.
        """
        return self.distribution.expected_pvalues(result.test_stat_result)


class AsymptoticCalculator(HypoTestCalculator):
    """Calculator for Cowan et al. asymptotic hypothesis tests.

    Extends ``HypoTestCalculator`` with ``predict_fn`` and ``mu_asimov``
    fields for Asimov dataset generation. These are injected into the
    test statistic call automatically by ``test()``.

    Example:
        >>> calc = AsymptoticCalculator(
        ...     nll_fn=nll_fn, params=params, observation=observed,
        ...     poi_key=("mu",), predict_fn=my_predict_fn,
        ... )
        >>> result = calc.test(poi_test=1.0)

    Attributes:
        predict_fn: Function to generate expected observation from parameters.
            Used to create the Asimov dataset at ``mu_asimov``.
        mu_asimov: POI value for Asimov dataset generation.
            Defaults to 0.0 (background-only, for exclusion tests).
            Use 1.0 for discovery tests.
    """

    predict_fn: tp.Callable[[sl.State], PyTree] | None = None
    mu_asimov: float = 0.0

    def test(
        self,
        poi_test: float,
        **kwargs: tp.Any,
    ) -> HypoTestResult:
        """Run asymptotic hypothesis test.

        Injects ``predict_fn`` and ``mu_asimov`` from init fields,
        unless overridden in kwargs.

        Args:
            poi_test: Test value for the POI.
            **kwargs: Additional arguments forwarded to the test statistic
                (e.g. fit options). Can override ``predict_fn`` and
                ``mu_asimov`` for one-off use.

        Returns:
            HypoTestResult with observed p-values.
        """
        kwargs.setdefault("predict_fn", self.predict_fn)
        kwargs.setdefault("mu_asimov", self.mu_asimov)
        return super().test(poi_test, **kwargs)
