"""Compare ifit convergence across pyhs3, pyhf, and evermore models."""

from __future__ import annotations

from functools import partial

import evermore as evm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from evermore_model import build_components, loss
from flax import nnx
from pyhf_model import build_pyhf
from pyhf_model import nll_fn as pyhf_nll_fn
from pyhs3_model import build_pyhs3
from rich.console import Console

import everwillow as ew
import everwillow.statelib as sl
from everwillow import HistoryCallback

jax.config.update("jax_enable_x64", True)


def fit_pyhs3_ifit(max_steps: int = 150) -> tuple[HistoryCallback, float]:
    """Run ifit on pyhs3 model and return history."""
    inputs, jaxified, fixed_values, initial = build_pyhs3()

    def nll(params):
        merged = {**fixed_values, **params}
        ordered = [merged[var.name] for var in inputs]
        probability = jaxified(*ordered)[0]
        return -jnp.log(jnp.asarray(probability))

    history = HistoryCallback()
    init_state = sl.State.from_pytree(initial)
    result = ew.ifit(nll, init_state, max_steps=max_steps, callbacks=[history])
    return history, float(result.nll)


def fit_pyhf_ifit(max_steps: int = 150) -> tuple[HistoryCallback, float]:
    """Run ifit on pyhf model and return history."""
    model, data_vector, init, _slices = build_pyhf()
    nll = pyhf_nll_fn(model, data_vector)

    history = HistoryCallback()
    init_state = sl.State.from_pytree(init)
    result = ew.ifit(nll, init_state, max_steps=max_steps, callbacks=[history])
    return history, float(result.nll)


def fit_evermore_ifit(max_steps: int = 150) -> tuple[HistoryCallback, float]:
    """Run ifit on evermore model and return history."""
    components = build_components()
    params, hists, observation = components

    graphdef, dynamic, static = nnx.split(params, evm.filter.is_parameter, ...)
    args = (graphdef, static, hists, observation)

    history = HistoryCallback()
    init_state = sl.State.from_pytree(dynamic)
    result = ew.ifit(partial(loss, args=args), init_state, max_steps=max_steps, callbacks=[history])
    return history, float(result.nll)


def plot_convergence_comparison(
    histories: dict[str, HistoryCallback],
    final_nlls: dict[str, float],
    output_path: str = "ifit_convergence.png",
) -> None:
    """Plot overlaid convergence histories for all models."""
    _, ax = plt.subplots(figsize=(10, 6))

    colors = {"pyhs3": "#1f77b4", "pyhf": "#ff7f0e", "evermore": "#2ca02c"}

    for name, history in histories.items():
        final_nll = final_nlls[name]
        ax.plot(
            history.steps,
            history.nlls,
            label=f"{name} (final: {final_nll:.4f})",
            color=colors.get(name),
            marker=".",
            markersize=4,
            linewidth=1.5,
            alpha=0.8,
        )

    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Negative Log-Likelihood", fontsize=12)
    ax.set_title("ifit Convergence Comparison", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add a zoomed inset for final convergence if there are enough steps
    min_steps = min(len(h.steps) for h in histories.values())
    if min_steps > 10:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax_inset = inset_axes(ax, width="40%", height="40%", loc="center right")
        zoom_start = max(0, min_steps - 15)

        for name, history in histories.items():
            ax_inset.plot(
                history.steps[zoom_start:],
                history.nlls[zoom_start:],
                color=colors.get(name),
                marker=".",
                markersize=3,
                linewidth=1,
            )

        ax_inset.set_title("Final steps", fontsize=9)
        ax_inset.grid(True, alpha=0.3)
        ax_inset.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved convergence plot to {output_path}")


def main() -> None:
    console = Console()
    console.rule("ifit Convergence Comparison")

    max_steps = 100

    console.print("\n[bold]Running ifit on pyhs3 model...[/bold]")
    pyhs3_history, pyhs3_nll = fit_pyhs3_ifit(max_steps=max_steps)

    console.print("\n[bold]Running ifit on pyhf model...[/bold]")
    pyhf_history, pyhf_nll = fit_pyhf_ifit(max_steps=max_steps)

    console.print("\n[bold]Running ifit on evermore model...[/bold]")
    evermore_history, evermore_nll = fit_evermore_ifit(max_steps=max_steps)

    console.print("\n")
    console.rule("Results Summary")

    console.print(f"  pyhs3:    {len(pyhs3_history.steps):3d} steps, NLL = {pyhs3_nll:.6f}")
    console.print(f"  pyhf:     {len(pyhf_history.steps):3d} steps, NLL = {pyhf_nll:.6f}")
    console.print(f"  evermore: {len(evermore_history.steps):3d} steps, NLL = {evermore_nll:.6f}")

    histories = {
        "pyhs3": pyhs3_history,
        "pyhf": pyhf_history,
        "evermore": evermore_history,
    }
    final_nlls = {
        "pyhs3": pyhs3_nll,
        "pyhf": pyhf_nll,
        "evermore": evermore_nll,
    }

    console.print("\n")
    plot_convergence_comparison(histories, final_nlls)


if __name__ == "__main__":
    main()
