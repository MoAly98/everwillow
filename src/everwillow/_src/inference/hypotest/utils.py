"""Utilities for hypothesis testing."""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

import everwillow._src.statelib as sl
from everwillow._src.inference.fitting import FitResult, fit

__all__ = [
    "cl_s",
    "constrained_fit",
    "make_asimov",
    "ncx2_sf",
    "sigma_from_asimov",
    "significance",
    "single_poi_key",
]


def ncx2_sf(x: Array, dof: int, nc: Array, *, n_terms: int = 64) -> Array:
    r"""Survival function of the non-central chi-square distribution.

    Evaluated as a Poisson-weighted mixture of central chi-square tails:

    .. math::

        P(X \geq x) = \sum_{j\geq 0} \frac{e^{-nc/2}(nc/2)^j}{j!}\,
                      P(\chi^2_{dof + 2j} \geq x)

    truncated at ``n_terms`` terms, which is fully vectorised and jittable. The
    series reduces to the central chi-square survival at ``nc=0``. The default
    term count is accurate to better than 1e-10 for non-centralities up to ~50,
    which covers asymptotic multi-POI p-values; raise it for larger ``nc``.

    Args:
        x: Point at which to evaluate the survival function.
        dof: Degrees of freedom (the number of POIs).
        nc: Non-centrality parameter.
        n_terms: Number of series terms.

    Returns:
        Survival probability :math:`P(X \geq x)`.
    """
    x = jnp.maximum(x, 0.0)
    half = nc / 2.0
    j = jnp.arange(n_terms)
    # Poisson(j; nc/2) weights in log space; xlogy(0, 0) = 0 keeps nc=0 well-defined.
    log_weights = -half + jax.scipy.special.xlogy(j, half) - jax.scipy.special.gammaln(j + 1.0)
    tails = jax.scipy.stats.chi2.sf(x, dof + 2.0 * j)
    return jnp.sum(jnp.exp(log_weights) * tails)


def single_poi_key(point: tp.Mapping[sl.K, ArrayLike]) -> sl.K:
    """Return the sole POI key, rejecting points that name more than one.

    Used where a scalar POI is required: the one-sided test statistics (which
    order on a scalar :math:`\\hat\\mu`) and the sigma-based expected bands.
    """
    keys = tuple(point)
    if len(keys) != 1:
        msg = f"expected a single POI, but the point names {len(keys)}: {keys}"
        raise ValueError(msg)
    return keys[0]


def make_asimov(
    predict_fn: tp.Callable[[sl.State], PyTree],
    params: sl.State,
    poi_asimov: tp.Mapping[sl.K, ArrayLike],
) -> PyTree:
    """Generate an Asimov dataset at a given POI point.

    Sets each POI in ``poi_asimov`` on the parameter state and calls
    ``predict_fn`` to produce the expected observation.

    Args:
        predict_fn: Function mapping parameter state to expected observation.
        params: Parameter state (used as template).
        poi_asimov: POI point at which to generate the Asimov dataset, a mapping
            from POI key to value, e.g. ``{"mu": 0.0}``.

    Returns:
        Expected observation (Asimov dataset).
    """
    asimov_params = sl.update(params, updates=poi_asimov)
    return predict_fn(asimov_params)


def sigma_from_asimov(mu: Array, q_asimov: Array, mu_asimov: float = 0.0) -> Array:
    r"""Extract :math:`\sigma` (uncertainty on :math:`\hat{\mu}`) from an Asimov test statistic.

    Uses the relation :math:`t_{\mu,A} \approx (\mu - \mu')^2/\sigma^2` to solve for :math:`\sigma`.

    Args:
        mu: POI value being tested.
        q_asimov: Test statistic evaluated on Asimov data.
        mu_asimov: POI value used to generate the Asimov dataset.
            Defaults to 0.0 (background-only, for exclusion tests).

    Returns:
        Estimated :math:`\sigma = |\mu - \mu_\text{asimov}| / \sqrt{q_\text{asimov}}`.
    """
    return jnp.abs(mu - mu_asimov) / jnp.sqrt(jnp.maximum(q_asimov, 1e-10))


def significance(p: Array) -> Array:
    r"""Convert p-value to significance: :math:`Z = \Phi^{-1}(1 - p)`.

    Args:
        p: p-value (scalar or array).

    Returns:
        Significance Z.
    """
    return -jax.scipy.stats.norm.ppf(p)


def cl_s(pnull: Array, palt: Array) -> Array:
    r"""Compute :math:`\text{CL}_s = p_\text{null} / p_\text{alt}`.

    :math:`\text{CL}_s = P(q \geq q_\text{obs} \mid \text{signal+background})
    / P(q \geq q_\text{obs} \mid \text{background})`

    The CLs method protects against excluding signal
    hypotheses when there is no sensitivity: if palt is small
    (background also finds data unlikely), CLs stays large.

    Args:
        pnull: p-value under null hypothesis (:math:`\mu' = \mu`, signal+background).
        palt: p-value under alternative hypothesis (:math:`\mu' = 0`, background-only).

    Returns:
        CLs value. Protected against division by zero.
    """
    return pnull / jnp.maximum(palt, 1e-10)


def constrained_fit(
    nll_fn: tp.Callable[[PyTree, PyTree], float],
    params: sl.State,
    observation: PyTree,
    poi_fixed: sl.State | tp.Mapping[sl.K, ArrayLike],
    **fit_kwargs: tp.Any,
) -> FitResult:
    """Perform constrained fit, merging POI constraint with user-fixed params.

    Merges ``poi_fixed`` (the POI constraint from the test statistic) with any
    user-specified ``fixed`` params in ``fit_kwargs``. When all parameters end
    up fixed, the NLL is evaluated directly without running the optimizer.

    Args:
        nll_fn: Negative log-likelihood function taking (params, observation).
        params: Initial parameter state.
        observation: Observed data passed to nll_fn.
        poi_fixed: The POI point to constrain, either a mapping from POI key to
            value (e.g. ``{"mu": 1.0}``) or an already-built State.
        **fit_kwargs: Additional arguments passed to fit(). If ``fixed`` is
            present, it is merged with ``poi_fixed`` (``poi_fixed`` wins on
            overlapping keys).

    Returns:
        FitResult with fitted parameters and NLL value.
    """
    if not isinstance(poi_fixed, sl.State):
        poi_fixed = sl.State.from_pytree(poi_fixed)

    user_fixed = fit_kwargs.pop("fixed", None)
    merged_fixed = sl.merge(user_fixed, poi_fixed) if user_fixed else poi_fixed

    # Check if fixing these params leaves any free parameters
    free_keys = set(params.mapping.keys()) - set(merged_fixed.mapping.keys())

    if len(free_keys) == 0:
        # All parameters are fixed - just evaluate NLL
        updated_params = sl.update(params, updates=merged_fixed)
        nll_value = jnp.asarray(nll_fn(updated_params.to_pytree(), observation))
        return FitResult(
            params=updated_params,
            nll=nll_value,
            success=jnp.asarray(True),
            solver_result=None,
        )

    return fit(nll_fn, params, observation, fixed=merged_fixed, **fit_kwargs)
