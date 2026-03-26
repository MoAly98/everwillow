"""Utility helpers shared by the example implementations."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import jax.numpy as jnp
import pyhs3
from jax.scipy.special import gammaln
from pyhs3.typing.aliases import TensorVar
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify

import everwillow as ew

warnings.filterwarnings("ignore", category=UserWarning, module="pytensor")


def jaxify_distribution(
    model: pyhs3.Model,
    distribution_name: str,
) -> tuple[list[TensorVar], Callable[..., Sequence[float]]]:
    """Convert a PyTensor distribution graph into a JAX-callable function."""

    distribution = model.distributions[distribution_name]
    inputs = [var for var in graph_inputs([distribution]) if var.name is not None]

    function_graph = FunctionGraph(inputs=inputs, outputs=[distribution], clone=True)
    mode.JAX.optimizer.rewrite(function_graph)

    return inputs, jax_funcify(function_graph)


def log_normal_modifier(theta: float, up: float, down: float) -> jnp.ndarray:
    """Compute the log-normal scaling corresponding to the nuisance parameter."""

    theta_array = jnp.asarray(theta)
    scale_up = jnp.exp(theta_array * jnp.log(up))
    scale_down = jnp.exp(theta_array * jnp.log(1.0 / down))
    return jnp.where(theta_array >= 0.0, scale_up, scale_down)


def shape_interpolate(nominal: float, up: float, theta: float) -> jnp.ndarray:
    """Linearly interpolate between nominal and up templates."""

    nominal_array = jnp.asarray(nominal)
    up_array = jnp.asarray(up)
    theta_array = jnp.asarray(theta)
    return nominal_array + theta_array * (up_array - nominal_array)


def poisson_logpdf(observed: float, mean: float) -> jnp.ndarray:
    """Return log(P(X=observed | mean)) for a Poisson random variable."""

    observed_array = jnp.asarray(observed)
    mean_array = jnp.asarray(mean)
    return jnp.where(
        mean_array > 0.0,
        observed_array * jnp.log(jnp.maximum(mean_array, 1e-12)) - mean_array - gammaln(observed_array + 1.0),
        -jnp.inf,
    )


def plot_history(history: ew.HistoryCallback, ax=None, **kwargs):
    """Plot NLL convergence history.

    Args:
        ax: Optional matplotlib axes. If None, creates a new figure.
        **kwargs: Additional arguments passed to ax.plot().

    Returns:
        The matplotlib axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    plot_kwargs = {"marker": ".", "markersize": 3, "linewidth": 1}
    plot_kwargs.update(kwargs)

    ax.plot(history.steps, history.nlls, **plot_kwargs)
    ax.set_xlabel("Step")
    ax.set_ylabel("NLL")
    ax.set_title("Optimization Convergence")
    ax.grid(True, alpha=0.3)

    return ax
