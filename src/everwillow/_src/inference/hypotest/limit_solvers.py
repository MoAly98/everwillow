"""
This module contains implementations of solver methods to find the upper limits on a parameter of interest
(POI) in a hypothesis test. The module contains an ABC definition for limit solvers,
as well as a concrete implementation of a few useful and commonly used ones both for
asymptotic and toy-based calculators.

The solvers defined here should be instantiated then passed to a Calculator instance.
"""

from __future__ import annotations

import abc
import typing as tp

import equinox as eqx
import jax
import optimistix as optx
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree


class LimitSolver(eqx.Module):
    """
    Abstract base class for limit solvers.

    Subclasses should implement the `solve` method to find the upper limit of POIs
    in a hypothesis test.
    """

    @abc.abstractmethod
    def solve(
        self,
        objective: tp.Callable[[float, PRNGKeyArray | None], PyTree],
        level: float,
        *,
        key: PRNGKeyArray | None = None,
    ) -> PyTree:
        """Find the POI value at which each leaf of the objective crosses ``level``.

        The objective is called as ``objective(poi, key)`` and may return any
        pytree of scalars — a single CLs value, a BandValues of expected-band
        values, or a custom container of criteria. The returned pytree has the
        same structure, holding the level-crossing POI per leaf.

        Args:
            objective: Criterion curve(s) to solve, ``objective(poi, key) -> PyTree``.
            level: Target value of the criterion (e.g. 0.05 for a 95% CL limit).
            key: None for deterministic criteria; a PRNG key when each
                evaluation draws fresh randomness (e.g. toy resampling).

        Returns:
            PyTree matching the objective's structure, with the crossing POI
            per leaf.
        """


class StochasticLimitSolver(LimitSolver):
    """
    An abstract subclass of `LimitSolver` that remains valid for limit solving in toy-based hypothesis tests.
    This means the algorithm's result stays statistically meaningful when each objective evaluation
    carries independent random noise (fresh PRNG key per evaluation).

    A criterion is deterministic if evaluating it twice at the same POI gives the
    same value. Criteria computed from asymptotic formulas behave this way.
    Toy-based criteria do not: every evaluation throws a new set of toys, so the
    returned CLs or p-value fluctuates with the statistical precision of the toy
    ensemble. Solvers under this base are safe for toys because each evaluation is
    used once, as a comparison against `level`, and never enters a fit or an
    interpolation. Solvers outside this base may reuse or interpolate through
    earlier evaluations, which is only valid for deterministic criteria.
    """


class RootFindingLimitSolver(LimitSolver):
    """
    A concrete subclass of `LimitSolver` that finds the upper limit by adaptive root
    finding of objective(poi) = level via optimistix. The objective must be
    deterministic: evaluating it at the same POI must always return the same value.
    Asymptotic criteria satisfy this. Toy-based criteria do not, so use a
    `StochasticLimitSolver` for those.

    The default optimistix algorithm is bisection, the same search geometry as
    `BisectionLimitSolver`. The difference is the contract. This solver may reuse
    and interpolate through previous evaluations, and it converges on tolerances in
    POI space. That is only valid when every evaluation is exact and repeatable.

    Attributes:
        bounds: A tuple of lower and upper bounds for the search.
        solver: The optimistix root finder to use, configured at construction
            (e.g. ``optx.Bisection(rtol=1e-6, atol=1e-8)`` or ``optx.Newton(...)``).
            Defaults to bisection with rtol=1e-4, atol=1e-6.
        maxiter: The maximum number of solver steps to perform.
    """

    bounds: tuple[float, float]
    solver: optx.AbstractRootFinder = eqx.field(default_factory=lambda: optx.Bisection(rtol=1e-4, atol=1e-6))
    maxiter: int = 100

    def solve(
        self,
        objective: tp.Callable[[float, PRNGKeyArray | None], PyTree],
        level: float,
        *,
        key: PRNGKeyArray | None = None,
    ) -> PyTree:
        # One probe evaluation to learn the leaf structure of the criterion
        leaves, treedef = jax.tree.flatten(objective(self.bounds[0], key))
        initial_guess = jnp.asarray((self.bounds[0] + self.bounds[1]) / 2.0)

        def leaf_limit(leaf_index: int) -> Array:
            def distance_to_level(poi, _args):
                value = jax.tree.flatten(objective(poi, key))[0][leaf_index]
                return value - level

            solution = optx.root_find(
                distance_to_level,
                self.solver,
                initial_guess,
                options={"lower": self.bounds[0], "upper": self.bounds[1]},
                max_steps=self.maxiter,
                throw=False,
            )
            return eqx.error_if(
                solution.value,
                solution.result != optx.RESULTS.successful,
                "Root finding did not converge within maxiter steps",
            )

        return jax.tree.unflatten(treedef, [leaf_limit(i) for i in range(len(leaves))])


class GridScanLimitSolver(StochasticLimitSolver):
    """
    A concrete subclass of `StochasticLimitSolver` that performs a grid scan over specified POI values
    to find the upper limit. It works by evaluating the objective function on a grid of POI values,
    then interpolating to find where it crosses the target level.

    This is useful when:
    - The objective function is expensive and you want to reuse evaluations
    - You need to visualize the objective curve
    - Root-finding fails due to non-monotonicity

    Note:
        The accuracy depends on the density of scan points near the crossing.
        There is an inherent assumption that the objective function is monotonically
        decreasing with larger POI values. This follows PyHF's convention.

    Attributes:
        scan: The array of POI values to scan for finding the upper limit
    """

    scan: Array

    def solve(
        self,
        objective: tp.Callable[[float, PRNGKeyArray | None], PyTree],
        level: float,
        *,
        key: PRNGKeyArray | None = None,
    ) -> PyTree:
        # 1. evaluate the objective function on the scan points
        if key is None:
            objective_values = jax.vmap(lambda poi: objective(poi, None))(self.scan)
        else:
            keys = jax.random.split(key, len(self.scan))
            objective_values = jax.vmap(objective)(self.scan, keys)

        # 2. Interpolate between grid points to find the crossing, per criterion curve.
        # The vmap above evaluated every criterion at every scan point in one pass,
        # so each leaf of `objective_values` holds one criterion sampled on the whole
        # grid, shape (len(scan),). A scalar CLs objective gives one such curve; a
        # BandValues objective gives five, one per band. Each curve gets its own
        # independent crossing, which is why the interpolation maps over leaves.

        def leaf_crossing(curve: Array) -> Array:
            """Find where one criterion curve crosses `level`.

            jnp.interp requires ascending x values, and the curves fall as the
            POI grows, so curve and scan are reversed together. The pairs stay
            matched; this relies on the monotonically decreasing criterion
            assumption from the class docstring."""
            cross = jnp.interp(level, curve[::-1], self.scan[::-1])
            at_lower = jnp.isclose(cross, self.scan[0])
            at_upper = jnp.isclose(cross, self.scan[-1])
            return eqx.error_if(
                cross,
                at_lower | at_upper,
                "GridScanLimitSolver: no crossing within the scan range. "
                "The limit sits at the scan boundary, so the scan range may be too narrow.",
            )

        return jax.tree.map(leaf_crossing, objective_values)


class BisectionLimitSolver(StochasticLimitSolver):
    """
    A concrete subclass of `StochasticLimitSolver` that performs an explicitly
    stepped bisection search, throwing fresh toys at every step.

    The full algorithm::

        lo, hi = bounds
        for i in 1..maxiter:
            mid   = (lo + hi) / 2
            value = objective(mid, fold_in(key, i))   # fresh toys, used once
            if |value - level| < tol:
                stop                                  # within toy statistical precision
            if value > level:
                lo = mid                              # crossing lies above mid
            else:
                hi = mid                              # crossing lies below mid
        return (lo + hi) / 2

    Each evaluation is consumed on the spot as a single comparison against
    `level`, then discarded. Nothing is fitted or interpolated from stored
    values, so a toy fluctuation can only affect one branching decision, never
    the shape of the search. `RootFindingLimitSolver` steps the same way by
    default, but optimistix does not support throwing fresh toys per evaluated
    point, so this custom bisection exists to provide exactly that.

    Attributes:
        bounds: A tuple of lower and upper bounds for the search.
        tol: Early-exit threshold: stop as soon as ``abs(objective(mid) - level) < tol``.
            Set to the Monte Carlo noise of the criterion for toy-based limits.
            tol=0.0 disables the early exit and runs all maxiter halvings.
        maxiter: The maximum number of iterations to perform.
    """

    bounds: tuple[float, float]
    tol: float = 1e-2
    maxiter: int = 100

    def solve(
        self,
        objective: tp.Callable[[float, PRNGKeyArray | None], PyTree],
        level: float,
        *,
        key: PRNGKeyArray | None = None,
    ) -> PyTree:
        # One probe evaluation to learn the leaf structure of the criterion
        leaves, treedef = jax.tree.flatten(objective(self.bounds[0], key))

        def leaf_limit(leaf_index):
            # Generate a key to use for toy draw in this iteration
            if key is None:

                def iteration_key(itr):
                    return None  # deterministic: None flows through
            else:

                def iteration_key(itr):
                    return jax.random.fold_in(jax.random.fold_in(key, leaf_index), itr)  # fresh toys per step

            def leaf_objective_value(poi, iteration):
                return jax.tree.leaves(objective(poi, iteration_key(iteration)))[leaf_index]

            def continue_while(state: tuple[Array, Array, Array, Array]) -> Array:
                """
                Condition for the bisection while loop: continue while the iteration
                count is less than `maxiter` and the crossing of objective with
                `level` has not been found.
                """
                iteration, _lo, _hi, converged = state
                return (iteration < self.maxiter) & (~converged)

            def bisect(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
                """
                The algorithm to run in the bisection while loop.
                """
                (
                    iteration,
                    lo,
                    hi,
                    _converged,
                ) = state
                # Get the middle of current search bracket
                mid = (lo + hi) / 2

                # Evaluate the objective at the middle of the search bracket
                mid_value = leaf_objective_value(mid, iteration)

                # Check if the middle value crosses the level
                converged = jnp.abs(mid_value - level) < self.tol

                # Update the search bracket based on the middle value
                # narrow it: if mid_value is below level, lo moves to mid; if above, hi moves to mid
                # This assumes the objective is montonically decreasing with increasing POIs
                new_lo = jnp.where(mid_value > level, mid, lo)
                new_hi = jnp.where(mid_value <= level, mid, hi)

                return iteration + 1, new_lo, new_hi, converged

            initial_lo = self.bounds[0]
            initial_hi = self.bounds[1]

            initial_state = (0, initial_lo, initial_hi, jnp.asarray(False))
            result = jax.lax.while_loop(
                continue_while,
                bisect,
                initial_state,
            )
            (
                _iteration,
                lo,
                hi,
                _converged,
            ) = result

            final_mid = (lo + hi) / 2.0
            at_lower = jnp.isclose(final_mid, initial_lo)
            at_upper = jnp.isclose(final_mid, initial_hi)

            return eqx.error_if(
                final_mid,
                at_lower | at_upper,
                "BisectionLimitSolver: root not found within bounds. "
                "The limit is at the search boundary, suggesting the bounds are too narrow.",
            )

        return jax.tree.unflatten(treedef, [leaf_limit(i) for i in range(len(leaves))])
