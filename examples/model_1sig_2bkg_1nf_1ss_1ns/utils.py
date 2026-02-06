"""Utility helpers shared by the example implementations."""

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
        observed_array * jnp.log(jnp.maximum(mean_array, 1e-12))
        - mean_array
        - gammaln(observed_array + 1.0),
        -jnp.inf,
    )


class HistoryCallback:
    """Callback that records optimization history for analysis and plotting.

    This callback accumulates step indices and NLL values during interactive
    fitting, which can then be plotted to visualize convergence.

    Examples:
        >>> from utils import HistoryCallback
        >>> import everwillow as ew
        >>>
        >>> history = HistoryCallback()
        >>> result = ew.ifit(nll_fn, params, callback=history)
        >>>
        >>> # Plot convergence
        >>> history.plot()
        >>>
        >>> # Access raw data
        >>> print(history.steps)  # [0, 1, 2, ...]
        >>> print(history.nlls)   # [123.4, 45.6, ...]
    """

    def __init__(self) -> None:
        self._steps: list[int] = []
        self._nlls: list[float] = []

    def __call__(self, step: int, free_state, state) -> None:
        """Record a step during optimization.

        Args:
            step: Current iteration index.
            free_state: Current free parameter values (State).
            state: Solver state with NLL accessible via state.f_info.f.
        """
        del free_state  # unused, but part of Callback signature
        self._steps.append(step)
        self._nlls.append(float(state.f_info.f))

    @property
    def steps(self) -> list[int]:
        """List of step indices."""
        return self._steps

    @property
    def nlls(self) -> list[float]:
        """List of NLL values at each step."""
        return self._nlls

    def clear(self) -> None:
        """Clear recorded history."""
        self._steps.clear()
        self._nlls.clear()

    def plot(self, ax=None, **kwargs):
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

        ax.plot(self._steps, self._nlls, **plot_kwargs)
        ax.set_xlabel("Step")
        ax.set_ylabel("NLL")
        ax.set_title("Optimization Convergence")
        ax.grid(True, alpha=0.3)

        return ax
