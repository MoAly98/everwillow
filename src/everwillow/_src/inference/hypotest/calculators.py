"""Hypothesis test calculators.

This module provides calculators that orchestrate hypothesis testing by
computing the test statistic, then delegating p-value computation to
Distribution objects.

- ``HypoTestCalculator``: Generic base — forwards all kwargs to the
  test statistic, and provides upper-limit methods driven by a
  user-suppliable criterion (CLs by default).
- ``AsymptoticCalculator``: Extends the base with Asimov dataset config
  (``predict_fn``/``mu_asimov`` or ``asimov_observation``) for Cowan et al.
  asymptotic workflows.
- ``ToyCalculator``: Extends the base with toy-based p-values — a composed
  ToyGenerator regenerates toy ensembles at each tested POI and a
  distribution factory turns them into p-values.
"""

from __future__ import annotations

import abc
import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

import everwillow._src.statelib as sl
from everwillow._src.inference.hypotest.distributions import (
    Distribution,
    QTildeAsymptotic,
    SimpleEmpiricalDistribution,
)
from everwillow._src.inference.hypotest.limit_solvers import LimitSolver, StochasticLimitSolver
from everwillow._src.inference.hypotest.results import (
    BandValues,
    ExpectedBands,
    HypoTestResult,
    RegionResult,
    ToyResult,
)
from everwillow._src.inference.hypotest.test_statistics import PoiPoint, QTilde, TestStatistic
from everwillow._src.inference.hypotest.toys import ToyGenerator
from everwillow._src.inference.hypotest.utils import cl_s

__all__ = ["AsymptoticCalculator", "HypoTestCalculator", "ToyCalculator"]


def _require_criterion_value(value):
    """Reject criteria that produce None with an actionable error."""
    if value is None:
        msg = (
            "the limit criterion returned None. The default CLs criterion needs "
            "both pnull and palt (e.g. alternative-hypothesis toys or an Asimov "
            "dataset); provide the missing ingredient or pass a custom criterion."
        )
        raise ValueError(msg)
    return value


class HypoTestCalculator(eqx.Module):
    """Abstract core of the hypothesis test calculators.

    Holds everything shared by the concrete calculators: the statistical
    model, the test statistic, the limit solver, and all machinery downstream
    of ``test()`` (CLs, p-value bands, upper limits). Subclasses implement
    ``test()``, which is the single point where they differ: where the
    p-values come from. ``AsymptoticCalculator`` computes them from a fixed
    distribution; ``ToyCalculator`` regenerates toy ensembles per call.

    Attributes:
        nll_fn: Negative log-likelihood function taking (params, observation).
        params: Initial parameter state.
        observation: Observed data passed to nll_fn.
        poi_key: Canonical key of the default parameter of interest, e.g.
            "mu". Optional: a full point mapping never needs it. It is
            consulted only where a bare scalar must be resolved into a point,
            i.e. ``test(1.0)`` and the limit methods (whose solvers walk one
            POI axis); the limit methods also take a per-call ``poi_key``
            override, which wins over the field.
        test_statistic: Test statistic to use. Defaults to QTilde.
        limit_solver: Solver for computing upper limits. Defaults to None.
            Limits cannot be computed if limit_solver is None and no solver
            is passed per call.
    """

    nll_fn: tp.Callable[[PyTree, PyTree], float]
    params: sl.State
    observation: PyTree
    poi_key: sl.K | None = None
    test_statistic: TestStatistic = eqx.field(default_factory=QTilde)
    limit_solver: LimitSolver | None = None

    def _resolve_poi_key(self, poi_key: sl.K | None = None) -> sl.K:
        """Resolve the target POI: a per-call key overrides the field.

        Raises:
            ValueError: If neither is set, or the resolved key does not name
                a model parameter.
        """
        resolved = poi_key if poi_key is not None else self.poi_key
        if resolved is None:
            msg = (
                "no POI key: set the poi_key field or pass poi_key per call; "
                "joint tests pass a full point mapping instead"
            )
            raise ValueError(msg)
        if resolved not in self.params:
            msg = f"poi_key {resolved!r} does not name a model parameter; known keys: {tuple(self.params)}"
            raise ValueError(msg)
        return resolved

    def _as_point(self, poi_test: float | PoiPoint) -> PoiPoint:
        """Normalise a tested POI into a point mapping.

        A mapping is already a point and is used as-is (e.g. for joint
        multi-POI tests); anything else names the value for this calculator's
        ``poi_key``. The non-mapping branch must cover JAX arrays and tracers,
        since the limit solvers evaluate ``test()`` at traced scalar POIs.
        """
        return poi_test if isinstance(poi_test, tp.Mapping) else {self._resolve_poi_key(): poi_test}

    @abc.abstractmethod
    def test(
        self,
        poi_test: float | PoiPoint,
        **kwargs: tp.Any,
    ) -> HypoTestResult:
        """Run a hypothesis test at ``poi_test`` and return a HypoTestResult.

        Implementations must record the distribution that produced the
        p-values on the result (``distribution=...``); everything downstream
        (``pvalue_bands``, the limit methods) relies on it.

        Args:
            poi_test: The tested POI, either a scalar value for this
                calculator's ``poi_key`` or a full point mapping for a joint
                (multi-POI) test.
            **kwargs: Forwarded to the test statistic computation and the
                fits underneath.

        Returns:
            HypoTestResult with observed p-values.
        """

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

    def pvalue_bands(self, result: HypoTestResult) -> ExpectedBands | None:
        """Compute expected p-values at standard sigma bands.

        Delegates to the ``pvalue_bands`` method of the distribution the
        result carries, so bands always come from the same distribution
        that produced the result's p-values.

        Args:
            result: HypoTestResult from ``test()``.

        Returns:
            ExpectedBands with p-values at each sigma level.

        Raises:
            ValueError: If the result does not carry a distribution
                (e.g. it was built by hand rather than by ``test()``).
            NotImplementedError: If the distribution does not support
                expected p-value computation.
        """
        if result.distribution is None:
            msg = "this result carries no distribution; produce results with test()"
            raise ValueError(msg)
        return result.distribution.pvalue_bands(result.test_stat_result)

    def _band_criterion(
        self,
        criterion: tp.Callable[[HypoTestResult], BandValues] | None,
    ) -> tp.Callable[[HypoTestResult], BandValues | None]:
        """Resolve the expected-band criterion, defaulting to per-band CLs."""
        if criterion is not None:
            return criterion

        def default_criterion(result: HypoTestResult) -> BandValues | None:
            bands = self.pvalue_bands(result)
            return None if bands is None else bands.cl_s

        return default_criterion

    def _resolve_solver(self, solver: LimitSolver | None) -> LimitSolver:
        """Resolve the limit solver: a per-call solver overrides the field."""
        solver = solver if solver is not None else self.limit_solver
        if solver is None:
            msg = "no limit solver: set the limit_solver field or pass one per call"
            raise ValueError(msg)
        return solver

    def _solve_limit(
        self,
        solver: LimitSolver | None,
        level: float,
        criterion: tp.Callable[[HypoTestResult], PyTree],
        fit_kwargs: dict[str, tp.Any] | None,
        poi_key: sl.K | None = None,
    ) -> PyTree:
        """Compose criterion(test(poi)) into a solver objective and solve it."""
        solver = self._resolve_solver(solver)
        key_for_limit = self._resolve_poi_key(poi_key)

        def objective(poi: float, key: PRNGKeyArray | None) -> PyTree:
            del key  # the base calculator has no randomness to route into test()
            return _require_criterion_value(criterion(self.test({key_for_limit: poi}, **(fit_kwargs or {}))))

        return solver.solve(objective, level)

    def upper_limit(
        self,
        solver: LimitSolver | None = None,
        *,
        level: float = 0.05,
        criterion: tp.Callable[[HypoTestResult], PyTree] | None = None,
        poi_key: sl.K | None = None,
        fit_kwargs: dict[str, tp.Any] | None = None,
    ) -> Array:
        """Find the upper limit: the POI value where the criterion crosses ``level``.

        The solver searches the curve criterion(test(poi)). The default
        criterion is CLs, so by default this is the CLs upper limit at the
        1 - level confidence level.

        A single expected (blind) limit needs no band machinery: build the
        calculator with the Asimov dataset as its ``observation`` and call
        this method.

        Args:
            solver: Limit solver for this call. Defaults to the
                ``limit_solver`` field.
            level: Target criterion value. Defaults to 0.05 (a 95% CL limit).
            criterion: Maps a HypoTestResult to the quantity the limit is
                defined on. Defaults to CLs. For example
                ``lambda result: result.pnull`` sets a CLs+b limit instead.
                A criterion returning several quantities (any pytree) gives
                one limit per leaf.
            poi_key: The POI the limit is set on, for this call. Defaults to
                the ``poi_key`` field. In a multi-POI model the other POIs
                are profiled in every fit unless pinned via ``fit_kwargs``.
            fit_kwargs: Fit options forwarded to every ``test()`` evaluation,
                e.g. ``fixed`` parameters or ``bounds`` transforms. Passed as
                a dict because names like ``bounds`` and ``solver`` mean
                other things at this level.

        Returns:
            The POI value where the criterion equals ``level``.

        Raises:
            ValueError: If no solver or POI key is configured, or the
                criterion returns None (the default criterion needs palt,
                e.g. from an Asimov dataset or alternative-hypothesis toys).
        """
        crit = criterion if criterion is not None else self.cls
        return self._solve_limit(solver, level, crit, fit_kwargs, poi_key)

    def upper_limit_bands(
        self,
        solver: LimitSolver | None = None,
        *,
        level: float = 0.05,
        criterion: tp.Callable[[HypoTestResult], BandValues] | None = None,
        poi_key: sl.K | None = None,
        fit_kwargs: dict[str, tp.Any] | None = None,
    ) -> BandValues:
        """Find the expected upper limits at the standard sigma bands (Brazil bands).

        The solver searches every band of the band criterion at once. The
        default criterion is the per-band expected CLs from
        ``pvalue_bands``. With a ``GridScanLimitSolver`` all five band
        limits come from a single grid pass.

        Args:
            solver: Limit solver for this call. Defaults to the
                ``limit_solver`` field.
            level: Target criterion value. Defaults to 0.05 (95% CL limits).
            criterion: Maps a HypoTestResult to per-band values (a
                BandValues). Defaults to per-band expected CLs. For example
                ``lambda result: calc.pvalue_bands(result).null_pvalue``
                uses the expected pnull bands instead.
            poi_key: The POI the limits are set on, for this call. Defaults
                to the ``poi_key`` field.
            fit_kwargs: Fit options forwarded to every ``test()`` evaluation.

        Returns:
            BandValues with one limit per sigma band.

        Raises:
            ValueError: If no solver or POI key is configured, or the
                criterion returns None (the distribution cannot compute
                expected bands, e.g. without an Asimov test statistic).
        """
        return self._solve_limit(solver, level, self._band_criterion(criterion), fit_kwargs, poi_key)

    def _region_criterion(
        self,
        criterion: tp.Callable[[HypoTestResult], PyTree] | None,
    ) -> tp.Callable[[HypoTestResult], PyTree]:
        """Resolve the region criterion, defaulting to the null p-value.

        The null p-value is the joint-region criterion: with ``TMu`` and
        ``TMuAsymptotic(dof=k)`` its ``level`` sublevel set is the standard
        chi-square region. CLs is not the default because it is a
        one-sided single-POI construction.
        """
        if criterion is not None:
            return criterion

        def default_criterion(result: HypoTestResult) -> Array | None:
            return result.pnull

        return default_criterion

    def _stack_points(
        self,
        points: tp.Iterable[float | PoiPoint],
    ) -> tuple[tuple[PoiPoint, ...], dict[sl.K, Array]]:
        """Normalise points and stack their values leaf-wise for mapping.

        Every point must name the same POI keys so the scan can be expressed
        as one mapped evaluation over stacked value arrays.
        """
        normalized = tuple(self._as_point(p) for p in points)
        if not normalized:
            msg = "confidence_region needs at least one hypothesis point"
            raise ValueError(msg)
        keys = tuple(normalized[0])
        if any(tuple(p) != keys for p in normalized):
            msg = "all points in a scan must name the same POI keys"
            raise ValueError(msg)
        stacked = {key: jnp.stack([jnp.asarray(p[key]) for p in normalized]) for key in keys}
        return normalized, stacked

    def confidence_region(
        self,
        points: tp.Iterable[float | PoiPoint],
        *,
        level: float = 0.05,
        criterion: tp.Callable[[HypoTestResult], PyTree] | None = None,
        fit_kwargs: dict[str, tp.Any] | None = None,
        map_fn: tp.Callable = jax.vmap,
    ) -> RegionResult:
        """Evaluate the region criterion over hypothesis points.

        A point belongs to the joint confidence region when the criterion is
        at least ``level`` (not excluded at the 1 - level confidence level).
        The default criterion is the null p-value, so with ``TMu`` and
        ``TMuAsymptotic(dof=k)`` this is the standard chi-square region with
        one degree of freedom per POI. Contour the returned values at
        ``level`` downstream for a 2-D region plot.

        The scanned grid must respect the physical boundaries of the model
        (e.g. expected yields stay positive), otherwise those points evaluate
        to NaN.

        Args:
            points: Hypothesis points to scan. Each is a mapping from POI key
                to value, or a bare scalar for this calculator's ``poi_key``.
                All points must name the same POI keys.
            level: Criterion value defining region membership. Defaults to
                0.05 (a 95% CL region).
            criterion: Maps a HypoTestResult to the quantity the region is
                defined on. Defaults to the null p-value. A criterion
                returning several quantities (any pytree) gives one stacked
                array per leaf.
            fit_kwargs: Fit options forwarded to every ``test()`` evaluation,
                e.g. ``fixed`` parameters pinning POIs that are not scanned.
            map_fn: Maps the per-point evaluation over the stacked points.
                Defaults to ``jax.vmap``. Replace with e.g.
                ``lambda fn: partial(jax.lax.map, fn, batch_size=32)`` to scan
                in chunks when memory is tight, or a Python loop for
                step-through debugging.

        Returns:
            RegionResult with the criterion values per point and the
            membership mask at ``level``.

        Raises:
            ValueError: If no points are given, the points name different POI
                keys, or the criterion returns None.
        """
        crit = self._region_criterion(criterion)
        normalized, stacked = self._stack_points(points)

        def eval_point(point: PoiPoint) -> PyTree:
            return _require_criterion_value(crit(self.test(point, **(fit_kwargs or {}))))

        values = map_fn(eval_point)(stacked)
        return RegionResult(points=normalized, values=values, level=level)


class AsymptoticCalculator(HypoTestCalculator):
    """Calculator with p-values from a fixed distribution.

    Named for its default (the Cowan et al. asymptotic formulas), but the
    ``distribution`` field accepts any fixed Distribution, including a
    frozen empirical ensemble built once from externally generated toys
    (``SimpleEmpiricalDistribution.from_toys``). For distributions that must
    be regenerated per tested POI, use ``ToyCalculator``.

    The asymptotic formulas need an Asimov dataset for the alternative
    p-value and the expected bands. It can be provided in two ways:

    1. **Pre-computed**: pass ``asimov_observation`` directly. This is
       useful when the Asimov dataset is expensive to generate or when
       the model prediction function is not available (e.g. combined
       models with multiple observation channels).
    2. **On-the-fly**: pass ``predict_fn`` and ``poi_asimov``. The Asimov
       dataset is generated at each ``test()`` call by setting the POI
       to ``poi_asimov`` and calling ``predict_fn``.

    When both are provided, ``asimov_observation`` takes precedence and
    ``predict_fn`` / ``poi_asimov`` are ignored.

    Example:
        >>> calc = AsymptoticCalculator(
        ...     nll_fn=nll_fn, params=params, observation=observed,
        ...     poi_key="mu", predict_fn=my_predict_fn,
        ... )
        >>> result = calc.test(poi_test=1.0)

    Attributes:
        distribution: Distribution for p-value computation.
            Defaults to QTildeAsymptotic.
        predict_fn: Function mapping parameter state to expected observation.
            Used to create the Asimov dataset at ``poi_asimov``.
        poi_asimov: POI point for Asimov dataset generation. A scalar sets
            every POI to that value; defaults to 0.0 (background-only, for
            exclusion tests). Use 1.0 for discovery tests.
        asimov_observation: Pre-computed Asimov dataset. When provided,
            this is used directly instead of generating one via
            ``predict_fn`` / ``poi_asimov``.
    """

    distribution: Distribution = eqx.field(default_factory=QTildeAsymptotic)
    predict_fn: tp.Callable[[sl.State], PyTree] | None = None
    poi_asimov: float | PoiPoint = 0.0
    asimov_observation: PyTree | None = None

    def test(
        self,
        poi_test: float | PoiPoint,
        **kwargs: tp.Any,
    ) -> HypoTestResult:
        """Run a hypothesis test against the fixed distribution.

        Injects ``predict_fn``, ``poi_asimov``, and ``asimov_observation``
        from init fields, unless overridden in kwargs.

        Args:
            poi_test: The tested POI, either a scalar value for ``poi_key`` or
                a full point mapping for a joint test.
            **kwargs: Additional arguments forwarded to the test statistic
                (e.g. fit options). Can override ``predict_fn``,
                ``poi_asimov``, or ``asimov_observation`` for one-off use.

        Returns:
            HypoTestResult with observed p-values.
        """
        kwargs.setdefault("predict_fn", self.predict_fn)
        kwargs.setdefault("poi_asimov", self.poi_asimov)
        kwargs.setdefault("asimov_observation", self.asimov_observation)

        ts_result = self.test_statistic.compute(
            self.nll_fn,
            self.params,
            self.observation,
            self._as_point(poi_test),
            **kwargs,
        )

        return HypoTestResult(
            q_obs=ts_result.value,
            pnull=self.distribution.null_pval(ts_result),
            palt=self.distribution.alt_pval(ts_result),
            test_stat_result=ts_result,
            distribution=self.distribution,
        )


class ToyCalculator(HypoTestCalculator):
    """Calculator with toy-based (Monte Carlo) p-values.

    Every test regenerates the toy ensembles at the tested POI::

        test(poi)             # toys thrown with the calculator's key:
        test(poi, key=key)    # ... or with a per-call key override
                              #   toys = toy_generator.generate(..., poi, key)
                              #   dist = distribution_factory(toys)
                              #   p-values from dist

    Regenerating per tested POI matters for limits: the distribution of the
    test statistic depends on the hypothesis being tested, so a limit scan
    against a single fixed toy ensemble would use the wrong distribution away
    from the POI it was generated at. The limit methods here thread a fresh
    key into every solver step. For a fixed distribution (asymptotic
    formulas, or a frozen ensemble built once from external toys) use
    ``AsymptoticCalculator`` instead.

    The calculator is a pure function of its inputs: the same key gives
    identical ensembles on every call, and one key for a whole analysis is
    statistically sound (reuse correlates outputs, it does not bias them).
    Pass a per-call key for an independent replica, e.g. to estimate the
    Monte Carlo spread of a limit.

    Each pipeline stage is swappable: the sampling scheme lives on the
    ``toy_generator``, and ``distribution_factory`` chooses how raw toy
    test statistics become p-values (any EmpiricalDistribution subclass,
    e.g. one with smoothed or tail-fitted p-values).

    Example:
        >>> gen = ToyGenerator(predict_fn=my_predict_fn, ntoys=2000)
        >>> calc = ToyCalculator(
        ...     nll_fn=nll_fn, params=params, observation=observed,
        ...     poi_key="mu", test_statistic=QTilde(), toy_generator=gen,
        ...     key=jax.random.key(42),
        ... )
        >>> result = calc.test(1.0)
        >>> limit = calc.upper_limit(BisectionLimitSolver(bounds=(0.0, 5.0), tol=0.01))

    Attributes:
        toy_generator: Sampling engine drawing the toy ensembles. Required.
        distribution_factory: Turns a ToyResult into a Distribution.
            Defaults to ``SimpleEmpiricalDistribution.from_toys``
            (plain tail counting).
        poi_alt: POI point of the alternative hypothesis used for the
            second toy ensemble (needed for palt and hence CLs). A scalar sets
            every tested POI to that value, like ``poi_asimov``; defaults to
            0.0 (background-only, for exclusion tests). Set to None to
            generate null-hypothesis toys only.
        key: PRNG key for toy generation. Required; every stochastic method
            uses it unless a per-call key overrides it.
    """

    toy_generator: ToyGenerator | None = None
    distribution_factory: tp.Callable[[ToyResult], Distribution] = eqx.field(
        default=SimpleEmpiricalDistribution.from_toys, static=True
    )
    poi_alt: float | PoiPoint | None = 0.0
    key: PRNGKeyArray | None = None

    def __check_init__(self):
        # Dataclass field ordering forces defaults here, so required-ness is
        # enforced after construction instead of by the signature.
        if self.toy_generator is None or self.key is None:
            msg = (
                "ToyCalculator requires a toy_generator and a key; "
                "for a fixed distribution use HypoTestCalculator subclasses "
                "such as AsymptoticCalculator"
            )
            raise ValueError(msg)

    def test(
        self,
        poi_test: float | PoiPoint,
        *,
        key: PRNGKeyArray | None = None,
        **kwargs: tp.Any,
    ) -> HypoTestResult:
        """Run a hypothesis test from toys regenerated at ``poi_test``.

        Args:
            poi_test: The tested POI, either a scalar value for ``poi_key`` or
                a full point mapping for a joint test.
            key: PRNG key for toy generation. Defaults to the calculator's
                ``key`` field.
            **kwargs: Forwarded to the test statistic computation and to the
                fits performed for each toy.

        Returns:
            HypoTestResult with observed p-values.
        """
        key = key if key is not None else self.key
        # __check_init__ guarantees the generator (and key) are set
        generator = tp.cast(ToyGenerator, self.toy_generator)

        point = self._as_point(poi_test)
        # A scalar poi_alt broadcasts over the tested POIs, like poi_asimov.
        poi_alt = self.poi_alt
        if poi_alt is not None and not isinstance(poi_alt, tp.Mapping):
            poi_alt = dict.fromkeys(point, poi_alt)

        toys = generator.generate(
            self.nll_fn,
            self.params,
            self.observation,
            point,
            test_statistic=self.test_statistic,
            poi_alt=poi_alt,
            key=key,
            **kwargs,
        )
        distribution = self.distribution_factory(toys)

        ts_result = self.test_statistic.compute(
            self.nll_fn,
            self.params,
            self.observation,
            point,
            **kwargs,
        )

        return HypoTestResult(
            q_obs=ts_result.value,
            pnull=distribution.null_pval(ts_result),
            palt=distribution.alt_pval(ts_result),
            test_stat_result=ts_result,
            distribution=distribution,
        )

    def _solve_limit(
        self,
        solver: LimitSolver | None,
        level: float,
        criterion: tp.Callable[[HypoTestResult], PyTree],
        fit_kwargs: dict[str, tp.Any] | None,
        poi_key: sl.K | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> PyTree:
        """Stochastic variant: thread a fresh key into every test() evaluation."""
        key = key if key is not None else self.key
        key_for_limit = self._resolve_poi_key(poi_key)

        solver = self._resolve_solver(solver)
        if not isinstance(solver, StochasticLimitSolver):
            msg = (
                f"{type(solver).__name__} is not a StochasticLimitSolver: it may reuse or "
                "interpolate through evaluations, which toy noise breaks. Use "
                "GridScanLimitSolver or BisectionLimitSolver for toy-based limits."
            )
            raise TypeError(msg)

        def objective(poi: float, eval_key: PRNGKeyArray | None) -> PyTree:
            return _require_criterion_value(
                criterion(self.test({key_for_limit: poi}, key=eval_key, **(fit_kwargs or {})))
            )

        return solver.solve(objective, level, key=key)

    def upper_limit(
        self,
        solver: LimitSolver | None = None,
        *,
        level: float = 0.05,
        criterion: tp.Callable[[HypoTestResult], PyTree] | None = None,
        poi_key: sl.K | None = None,
        key: PRNGKeyArray | None = None,
        fit_kwargs: dict[str, tp.Any] | None = None,
    ) -> Array:
        """Find the upper limit from toys regenerated at every solver step.

        The solver must be a `StochasticLimitSolver`, and every solver
        evaluation regenerates the toy ensembles at the tested POI. For a
        limit against a fixed distribution use ``AsymptoticCalculator``.

        Args:
            solver: Limit solver for this call. Defaults to the
                ``limit_solver`` field.
            level: Target criterion value. Defaults to 0.05 (a 95% CL limit).
            criterion: Maps a HypoTestResult to the quantity the limit is
                defined on. Defaults to CLs.
            poi_key: The POI the limit is set on, for this call. Defaults to
                the ``poi_key`` field.
            key: PRNG key driving the solver and the per-evaluation toys.
                Defaults to the calculator's ``key`` field.
            fit_kwargs: Fit options forwarded to every ``test()`` evaluation.

        Returns:
            The POI value where the criterion equals ``level``.

        Raises:
            TypeError: If the solver is not a `StochasticLimitSolver`.
            ValueError: If no solver or POI key is configured, or the
                criterion returns None (the default criterion needs palt,
                e.g. from alternative-hypothesis toys via ``poi_alt``).
        """
        crit = criterion if criterion is not None else self.cls
        return self._solve_limit(solver, level, crit, fit_kwargs, poi_key, key=key)

    def upper_limit_bands(
        self,
        solver: LimitSolver | None = None,
        *,
        level: float = 0.05,
        criterion: tp.Callable[[HypoTestResult], BandValues] | None = None,
        poi_key: sl.K | None = None,
        key: PRNGKeyArray | None = None,
        fit_kwargs: dict[str, tp.Any] | None = None,
    ) -> BandValues:
        """Find the expected (Brazil-band) upper limits from regenerated toys.

        The solver must be a `StochasticLimitSolver`; with a
        ``GridScanLimitSolver`` all bands come from a single keyed grid pass.
        For bands against a fixed distribution use ``AsymptoticCalculator``.

        Args:
            solver: Limit solver for this call. Defaults to the
                ``limit_solver`` field.
            level: Target criterion value. Defaults to 0.05 (95% CL limits).
            criterion: Maps a HypoTestResult to per-band values (a
                BandValues). Defaults to per-band expected CLs.
            poi_key: The POI the limits are set on, for this call. Defaults
                to the ``poi_key`` field.
            key: PRNG key driving the solver and the per-evaluation toys.
            fit_kwargs: Fit options forwarded to every ``test()`` evaluation.

        Returns:
            BandValues with one limit per sigma band.

        Raises:
            TypeError: If a key is given with a solver that is not a
                `StochasticLimitSolver`.
            ValueError: If no solver or POI key is configured, or the
                criterion returns None.
        """
        return self._solve_limit(solver, level, self._band_criterion(criterion), fit_kwargs, poi_key, key=key)

    def confidence_region(
        self,
        points: tp.Iterable[float | PoiPoint],
        *,
        level: float = 0.05,
        criterion: tp.Callable[[HypoTestResult], PyTree] | None = None,
        key: PRNGKeyArray | None = None,
        fit_kwargs: dict[str, tp.Any] | None = None,
        map_fn: tp.Callable = jax.vmap,
    ) -> RegionResult:
        """Region scan with toy ensembles regenerated at every point.

        Each point receives its own subkey (split from ``key``), so the
        ensembles are independent per hypothesis and the whole scan is
        reproducible from the calculator's key.

        Args:
            points: Hypothesis points to scan, as in the base method.
            level: Criterion value defining region membership.
            criterion: Maps a HypoTestResult to the region quantity. Defaults
                to the null p-value.
            key: PRNG key driving the per-point toy ensembles. Defaults to the
                calculator's ``key`` field.
            fit_kwargs: Fit options forwarded to every ``test()`` evaluation.
            map_fn: Maps the per-point evaluation over the stacked points and
                subkeys. Defaults to ``jax.vmap``; see the base method for
                memory-friendly alternatives.

        Returns:
            RegionResult with the criterion values per point and the
            membership mask at ``level``.
        """
        key = key if key is not None else self.key
        crit = self._region_criterion(criterion)
        normalized, stacked = self._stack_points(points)
        subkeys = jax.random.split(key, len(normalized))

        def eval_point(point_and_key: tuple[PoiPoint, PRNGKeyArray]) -> PyTree:
            point, eval_key = point_and_key
            return _require_criterion_value(crit(self.test(point, key=eval_key, **(fit_kwargs or {}))))

        values = map_fn(eval_point)((stacked, subkeys))
        return RegionResult(points=normalized, values=values, level=level)
