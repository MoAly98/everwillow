from __future__ import annotations

import typing as tp

import jax

import everwillow.statelib as sl

if tp.TYPE_CHECKING:
    from everwillow.inference.fitting import SolverState


# Callback signature for interactive fitting: (iteration_idx, y, state) -> None
# Matches the signature pattern used by optimistix' solver.step and solver.terminate
class Callback(tp.Protocol):
    def __call__(self, iteration: int, y: sl.State, state: SolverState) -> None: ...


class HistoryCallback(Callback):
    """Callback that records optimization history for analysis and plotting.

    This callback accumulates step indices and NLL values during interactive
    fitting, which can then be plotted to visualize convergence.

    Examples:
        >>> import everwillow as ew
        >>>
        >>> history = ew.inference.HistoryCallback()
        >>> result = ew.ifit(nll_fn, params, callbacks=[history])
        >>>
        >>> # Access history
        >>> print(history.steps)  # [0, 1, 2, ...]
        >>> print(history.nlls)   # [123.4, 45.6, ...]
    """

    def __init__(self) -> None:
        self._steps: list[int] = []
        self._nlls: list[jax.Array] = []

    def __call__(self, iteration: int, y: sl.State, state: SolverState) -> None:
        """Record a step during optimization.

        Args:
            iteration: Current iteration index.
            y: Current free parameter values (State). Unused.
            state: Solver state with NLL accessible via state.f_info.f.
        """
        del y  # unused, but part of Callback signature
        self._steps.append(iteration)
        self._nlls.append(state.f_info.f)

    @property
    def steps(self) -> list[int]:
        """List of step indices."""
        return self._steps

    @property
    def nlls(self) -> list[jax.Array]:
        """List of NLL values at each step."""
        return self._nlls
