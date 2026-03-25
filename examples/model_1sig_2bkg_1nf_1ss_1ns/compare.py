"""Compare fits across libraries for the 1sig_2bkg_1nf_1ss_1ns example."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import matplotlib.pyplot as plt
import numpy as np

# Evermore model
from evermore_model import build_components, summarise_evermore_fit
from evermore_model import fit_with_everwillow as fit_evermore_with_everwillow
from evermore_model import fit_with_iminuit as fit_evermore_with_iminuit
from evermore_model import fit_with_optimistix as fit_evermore_with_optimistix
from evermore_model import fit_with_scipy as fit_evermore_with_scipy

# PyHF model
from pyhf_model import build_pyhf, summarise_pyhf
from pyhf_model import fit_with_everwillow as fit_pyhf_with_everwillow
from pyhf_model import fit_with_optimistix as fit_pyhf_with_optimistix
from pyhf_model import fit_with_pyhf_native as fit_pyhf_with_scipy
from pyhf_model import fit_with_pyhf_native_minuit as fit_pythf_with_iminuit

# PyHS3 model
from pyhs3_model import build_pyhs3, summarise_pyhs3_fit
from pyhs3_model import fit_with_everwillow as fit_pyhs3_with_everwillow
from pyhs3_model import fit_with_iminuit as fit_pyhs3_with_iminuit
from pyhs3_model import fit_with_optimistix as fit_pyhs3_with_optimistix
from pyhs3_model import fit_with_scipy as fit_pyhs3_with_scipy
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table

warnings.filterwarnings("ignore", category=UserWarning, module="pytensor")
jax.config.update("jax_enable_x64", True)


N_STEPS = 10_000


def _format(value: float, sign: bool = False, unit: None = None) -> str:
    s = f"{value:+.6f}" if sign else f"{value:.6f}"
    if unit:
        s += rf" \[{unit}]"
    return s


def _format_dict(d: dict[str, float]) -> str:
    return "\n".join(f"{k}={_format(v, sign=True)}" for k, v in sorted(d.items()))


def time_and_run(func: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    return end - start, result


class Benchmark(NamedTuple):
    name: str
    params: dict[str, float]
    expected_yields: dict[str, float]
    nll: float
    runtime_cold: float
    runtime_hot: float

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        table = Table(title=self.name, show_header=True)
        table.add_column("Bestfit params", style="bold magenta")
        table.add_column("Postfit yields", style="bold magenta")
        table.add_column("Postfit NLL", style="bold cyan")
        table.add_column("Runtime (cold)", style="bold green")
        table.add_column("Runtime (hot)", style="bold green")

        table.add_row(
            _format_dict(self.params),
            _format_dict(self.expected_yields),
            _format(self.nll, sign=True, unit="a.u."),
            _format(self.runtime_cold, sign=False, unit="s"),
            _format(self.runtime_hot, sign=False, unit="s"),
        )
        yield table


def benchmark_pyhs3_everwillow():
    inputs, jaxified, fixed_values, initial = build_pyhs3()

    @jax.jit
    def fun():
        return fit_pyhs3_with_everwillow(inputs, jaxified, fixed_values, initial, max_steps=N_STEPS)

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    expected_yields = summarise_pyhs3_fit(params)

    # Hot run
    runtime_hot, _ = time_and_run(fun)

    return Benchmark(
        name="pyhs3 + everwillow",
        params=params,
        expected_yields=expected_yields,
        nll=nll_value,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhs3_everwillow_ifit():
    inputs, jaxified, fixed_values, initial = build_pyhs3()

    # ifit can't be jitted due to Python side effects in the loop
    def fun():
        return fit_pyhs3_with_everwillow(inputs, jaxified, fixed_values, initial, max_steps=N_STEPS, interactive=True)

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    expected_yields = summarise_pyhs3_fit(params)

    # Hot run
    runtime_hot, _ = time_and_run(fun)

    return Benchmark(
        name="pyhs3 + everwillow.ifit",
        params=params,
        expected_yields=expected_yields,
        nll=nll_value,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhs3_optimistix():
    inputs, jaxified, fixed_values, initial = build_pyhs3()

    @jax.jit
    def fun():
        return fit_pyhs3_with_optimistix(inputs, jaxified, fixed_values, initial, max_steps=N_STEPS)

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    expected_yields = summarise_pyhs3_fit(params)

    # Hot run
    runtime_hot, _ = time_and_run(fun)

    return Benchmark(
        name="pyhs3 + optimistix",
        params=params,
        expected_yields=expected_yields,
        nll=nll_value,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhs3_iminuit():
    inputs, jaxified, fixed_values, initial = build_pyhs3()

    # can't be jitted due to iminuit
    # @jax.jit
    def fun():
        return fit_pyhs3_with_iminuit(inputs, jaxified, fixed_values, initial, max_steps=N_STEPS)

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    expected_yields = summarise_pyhs3_fit(params)

    # Hot run
    runtime_hot, _ = time_and_run(fun)

    return Benchmark(
        name="pyhs3 + iminuit",
        params=params,
        expected_yields=expected_yields,
        nll=nll_value,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhs3_scipy():
    inputs, jaxified, fixed_values, initial = build_pyhs3()

    # @jax.jit
    def fun():
        return fit_pyhs3_with_scipy(inputs, jaxified, fixed_values, initial, max_steps=N_STEPS)

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    expected_yields = summarise_pyhs3_fit(params)

    # Hot run
    runtime_hot, _ = time_and_run(fun)

    return Benchmark(
        name="pyhs3 + scipy.minimizer",
        params=params,
        expected_yields=expected_yields,
        nll=nll_value,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_evermore_native():
    components = build_components()
    fun = jax.jit(lambda: fit_evermore_with_optimistix(components, max_steps=N_STEPS))
    # Cold run
    runtime_cold, (params, nll) = time_and_run(fun)
    expected_yields = summarise_evermore_fit(params)
    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="evermore + optimistix",
        params=params,
        expected_yields=expected_yields,
        nll=nll,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_evermore_everwillow():
    components = build_components()

    @jax.jit
    def fun():
        return fit_evermore_with_everwillow(components, max_steps=N_STEPS)

    # Cold run
    runtime_cold, (params, nll) = time_and_run(fun)
    expected_yields = summarise_evermore_fit(params)
    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="evermore + everwillow",
        params=params,
        expected_yields=expected_yields,
        nll=nll,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_evermore_everwillow_ifit():
    components = build_components()

    # ifit can't be jitted due to Python side effects in the loop
    def fun():
        return fit_evermore_with_everwillow(components, max_steps=N_STEPS, interactive=True)

    # Cold run
    runtime_cold, (params, nll) = time_and_run(fun)
    expected_yields = summarise_evermore_fit(params)
    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="evermore + everwillow.ifit",
        params=params,
        expected_yields=expected_yields,
        nll=nll,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_evermore_iminuit():
    components = build_components()

    # @jax.jit
    def fun():
        return fit_evermore_with_iminuit(components, max_steps=N_STEPS)

    # Cold run
    runtime_cold, (params, nll) = time_and_run(fun)
    expected_yields = summarise_evermore_fit(params)
    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="evermore + iminuit",
        params=params,
        expected_yields=expected_yields,
        nll=nll,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_evermore_scipy():
    components = build_components()

    # can't be jitted due to scipy
    def fun():
        return fit_evermore_with_scipy(components, max_steps=N_STEPS)

    # Cold run
    runtime_cold, (params, nll) = time_and_run(fun)
    expected_yields = summarise_evermore_fit(params)
    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="evermore + scipy.minimizer",
        params=params,
        expected_yields=expected_yields,
        nll=nll,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhf_native():
    pyhf_model, pyhf_data, pyhf_init, pyhf_slices = build_pyhf()

    # can't be jitted due to pyhf internals
    def fun():
        return fit_pyhf_with_scipy(pyhf_model, pyhf_data, pyhf_init, pyhf_slices, maxiter=N_STEPS)

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    # expected yields
    expected_yields = summarise_pyhf(params)

    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="pyhf + scipy.minimizer",
        params=params,
        nll=nll_value,
        expected_yields=expected_yields,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhf_native_minuit():
    pyhf_model, pyhf_data, pyhf_init, pyhf_slices = build_pyhf()

    # can't be jitted due to pyhf internals
    def fun():
        return fit_pythf_with_iminuit(pyhf_model, pyhf_data, pyhf_init, pyhf_slices, maxiter=N_STEPS)

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    # expected yields
    expected_yields = summarise_pyhf(params)

    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="pyhf + iminuit",
        params=params,
        nll=nll_value,
        expected_yields=expected_yields,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhf_optimistix():
    pyhf_model, pyhf_data, pyhf_init, pyhf_slices = build_pyhf()

    @jax.jit
    def fun():
        return fit_pyhf_with_optimistix(pyhf_model, pyhf_data, pyhf_init, pyhf_slices, max_steps=N_STEPS)

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    expected_yields = summarise_pyhf(params)
    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="pyhf + optimistix",
        params=params,
        nll=nll_value,
        expected_yields=expected_yields,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhf_everwillow():
    pyhf_model, pyhf_data, pyhf_init, pyhf_slices = build_pyhf()

    @jax.jit
    def fun():
        return fit_pyhf_with_everwillow(pyhf_model, pyhf_data, pyhf_init, pyhf_slices, max_steps=N_STEPS)

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    expected_yields = summarise_pyhf(params)
    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="pyhf + everwillow",
        params=params,
        nll=nll_value,
        expected_yields=expected_yields,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhf_everwillow_ifit():
    pyhf_model, pyhf_data, pyhf_init, pyhf_slices = build_pyhf()

    # ifit can't be jitted due to Python side effects in the loop
    def fun():
        return fit_pyhf_with_everwillow(
            pyhf_model,
            pyhf_data,
            pyhf_init,
            pyhf_slices,
            max_steps=N_STEPS,
            interactive=True,
        )

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    expected_yields = summarise_pyhf(params)
    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="pyhf + everwillow.ifit",
        params=params,
        nll=nll_value,
        expected_yields=expected_yields,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def plot_benchmark_grid(benchmarks: list[Benchmark], output_path: str = "benchmark_comparison.png") -> None:
    """Create a 2D grid visualization of benchmark results.

    Args:
        benchmarks: List of Benchmark results
        output_path: Path to save the output plot
    """
    # Parse benchmark names into (model, optimizer) tuples
    parsed = []
    for b in benchmarks:
        parts = b.name.split(" + ")
        if len(parts) == 2:
            model, optimizer = parts
            parsed.append((model.strip(), optimizer.strip(), b))

    # Get unique models and optimizers
    models = sorted({p[0] for p in parsed})
    # Fixed order for optimizers: everwillow, optimistix, iminuit, scipy
    optimizer_order = [
        "everwillow",
        "everwillow.ifit",
        "optimistix",
        "iminuit",
        "scipy.minimizer",
    ]
    all_optimizers = {p[1] for p in parsed}
    optimizers = [o for o in optimizer_order if o in all_optimizers]

    # Create model->idx and optimizer->idx mappings
    model_idx = {m: i for i, m in enumerate(models)}
    optimizer_idx = {o: i for i, o in enumerate(optimizers)}

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute log-scale sizes
    all_times = [b.runtime_cold for _, _, b in parsed] + [b.runtime_hot for _, _, b in parsed]
    min_time, max_time = min(all_times), max(all_times)

    def time_to_size(t):
        """Convert runtime to bubble size using log scale."""
        log_t = np.log10(t)
        log_min = np.log10(min_time)
        log_max = np.log10(max_time)
        # Scale to reasonable bubble sizes (100 to 2000)
        normalized = (log_t - log_min) / (log_max - log_min)
        return 100 + normalized * 1900

    # Plot bubbles
    for model, optimizer, b in parsed:
        x = model_idx[model]
        y = optimizer_idx[optimizer]

        # Cold bubble (larger, pastel blue, behind)
        cold_size = time_to_size(b.runtime_cold)

        # Hot bubble (smaller, pastel red, in front)
        hot_size = time_to_size(b.runtime_hot)

        # Calculate bubble radii in data coordinates
        # scatter 's' parameter is in points^2, need to convert to data units
        fig_height_inches = fig.get_figheight()
        fig_width_inches = fig.get_figwidth()
        y_range = len(optimizers) + 0.4 + 0.6  # total y-axis range
        x_range = len(models) + 0.4 + 0.4  # total x-axis range

        # Convert points to data coordinates (accounting for aspect ratio)
        points_to_data_y = y_range / (72 * fig_height_inches)
        points_to_data_x = x_range / (72 * fig_width_inches)

        # Use x-direction for horizontal separation
        cold_radius_x = np.sqrt(cold_size) * points_to_data_x / 2
        hot_radius_x = np.sqrt(hot_size) * points_to_data_x / 2
        cold_radius_y = np.sqrt(cold_size) * points_to_data_y / 2
        hot_radius_y = np.sqrt(hot_size) * points_to_data_y / 2

        # Calculate dynamic offset to ensure separation
        bubble_gap = 0.03  # minimum gap between bubbles
        required_offset = (cold_radius_x + hot_radius_x) / 2 + bubble_gap

        # Draw bubbles with dynamic offset
        ax.scatter(
            x - required_offset,
            y,
            s=cold_size,
            c="#a8d8ea",
            alpha=0.6,
            edgecolors="#5da9c4",
            linewidth=1.5,
            zorder=1,
        )
        ax.scatter(
            x + required_offset,
            y,
            s=hot_size,
            c="#ffb3ba",
            alpha=0.6,
            edgecolors="#ff6b7a",
            linewidth=1.5,
            zorder=2,
        )

        # Add time annotations in milliseconds
        cold_ms = b.runtime_cold * 1000
        hot_ms = b.runtime_hot * 1000

        # Format: use appropriate precision based on magnitude
        def format_ms(ms):
            if ms < 0.01:
                return f"{ms:.4f}"
            if ms < 1:
                return f"{ms:.3f}"
            if ms < 10:
                return f"{ms:.2f}"
            return f"{ms:.1f}"

        # Position annotations slightly above the bubble top
        annotation_gap = 0.02  # small gap above bubble

        # Annotate cold time (above cold bubble)
        ax.text(
            x - required_offset,
            y + cold_radius_y + annotation_gap,
            f"{format_ms(cold_ms)}ms",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2d6a7a",
            weight="bold",
        )

        # Annotate hot time (above hot bubble)
        ax.text(
            x + required_offset,
            y + hot_radius_y + annotation_gap,
            f"{format_ms(hot_ms)}ms",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#c93545",
            weight="bold",
        )

    # Set axis properties
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=12, weight="bold")
    ax.set_yticks(range(len(optimizers)))
    ax.set_yticklabels(optimizers, fontsize=11)

    # Extend axis limits to prevent bubble overlap with borders
    ax.set_xlim(-0.4, len(models) - 1 + 0.4)
    ax.set_ylim(-0.6, len(optimizers) - 1 + 2.0)

    ax.set_xlabel("Model Framework", fontsize=13, weight="bold")
    ax.set_ylabel("Optimizer", fontsize=13, weight="bold")
    ax.set_title("Benchmark: Model vs Optimizer Runtime", fontsize=15, weight="bold", pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Create legend in upper right corner
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#a8d8ea", edgecolor="#5da9c4", alpha=0.6, label="Cold run"),
        Patch(facecolor="#ffb3ba", edgecolor="#ff6b7a", alpha=0.6, label="Hot run"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=12,
        framealpha=0.95,
        edgecolor="gray",
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved benchmark plot to {output_path}")


def plot_parameter_deviations(
    benchmarks: list[Benchmark],
    reference_name: str = "evermore + optimistix",
    output_path: str = "parameter_deviations.png",
) -> None:
    """Create a parameter deviation plot showing differences from reference method.

    Args:
        benchmarks: List of Benchmark results
        reference_name: Name of the benchmark to use as reference (truth)
        output_path: Path to save the output plot
    """
    # Find reference benchmark
    reference = None
    for b in benchmarks:
        if b.name == reference_name:
            reference = b
            break

    if reference is None:
        print(f"Warning: Reference method '{reference_name}' not found")
        return

    # Get all parameter names from reference
    param_names = sorted(reference.params.keys())

    # Parse benchmark names into (model, optimizer) and organize
    parsed = []
    for b in benchmarks:
        if b.name == reference_name:
            continue
        parts = b.name.split(" + ")
        if len(parts) == 2:
            model, optimizer = parts[0].strip(), parts[1].strip()
            parsed.append((model, optimizer, b))

    # Fixed order for optimizers: everwillow, everwillow.ifit, optimistix, iminuit, scipy
    optimizer_order = [
        "everwillow",
        "everwillow.ifit",
        "optimistix",
        "iminuit",
        "scipy.minimizer",
    ]

    # Sort by model first, then by optimizer order
    def sort_key(item):
        model, optimizer, _ = item
        opt_idx = optimizer_order.index(optimizer) if optimizer in optimizer_order else 999
        return (model, opt_idx)

    parsed.sort(key=sort_key)

    # Get unique optimizers in specified order
    all_optimizers = {p[1] for p in parsed}
    optimizers = [o for o in optimizer_order if o in all_optimizers]

    # Base colors for models (one per model framework)
    model_colors = {
        "evermore": "#1f77b4",  # blue
        "pyhf": "#ff7f0e",  # orange
        "pyhs3": "#2ca02c",  # green
    }

    # Hatching patterns for optimizers
    optimizer_hatches = {
        0: "",  # solid (everwillow)
        1: "...",  # dots (everwillow.ifit)
        2: "///",  # diagonal lines (optimistix)
        3: "\\\\\\",  # opposite diagonal (iminuit)
        4: "xxx",  # crosshatch (scipy.minimizer)
    }

    # Setup figure
    _fig, ax = plt.subplots(figsize=(10, 6))

    # Plot deviations
    legend_items = []
    for model, optimizer, b in parsed:
        # Get base color for model
        base_color = model_colors.get(model, "#888888")

        # Get hatch pattern for optimizer
        optimizer_idx = optimizers.index(optimizer)
        hatch = optimizer_hatches.get(optimizer_idx, "")

        # Calculate absolute deviations
        deviations = []
        for param in param_names:
            ref_val = reference.params[param]
            fit_val = b.params.get(param, ref_val)
            abs_diff = fit_val - ref_val
            deviations.append(abs_diff)

        # Plot as horizontal scatter with transparency
        y_positions = np.arange(len(param_names))
        ax.scatter(
            deviations,
            y_positions,
            s=150,
            facecolors=base_color,
            label=b.name,
            zorder=3,
            edgecolors="black",
            linewidth=1.5,
            hatch=hatch,
            alpha=0.7,
        )

        legend_items.append((b.name, base_color, hatch))

    # Add reference line at zero
    ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Reference: {reference_name}",
        zorder=1,
    )

    # Set axis properties with reference values in brackets (scientific notation, 2 sig figs)
    ax.set_yticks(np.arange(len(param_names)))
    y_labels = []
    for param in param_names:
        val = reference.params[param]
        # Format with 2 significant figures in scientific notation
        y_labels.append(f"{param} [{val:.2g}]")
    ax.set_yticklabels(y_labels, fontsize=10)

    # Extend y-axis to make room for legend at top
    ax.set_ylim(-0.5, len(param_names) - 1 + 3.0)

    ax.set_xlabel("Absolute Parameter Deviation from Reference", fontsize=13, weight="bold")
    ax.set_ylabel("Parameter", fontsize=13, weight="bold")
    ax.set_title("Parameter Deviations Across Methods", fontsize=15, weight="bold", pad=20)

    # Format x-axis with scientific notation in LaTeX
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(11)
    ax.xaxis.get_offset_text().set_weight("bold")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="x")
    ax.set_axisbelow(True)

    # Create custom legend with ordered items
    from matplotlib.patches import Patch

    legend_handles = []

    # Add reference method and line
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Reference: evermore + optimistix",
        )
    )

    # Add method items
    for name, color, hatch in legend_items:
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                hatch=hatch,
                label=name,
                linewidth=1.5,
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=9,
        framealpha=0.98,
        edgecolor="gray",
        fancybox=True,
        shadow=True,
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved parameter deviation plot to {output_path}")


def main() -> None:
    console = Console()
    jax.config.update("jax_enable_x64", True)

    console.print("[bold]Benchmarking fits for 1sig_2bkg_1nf_1ss_1ns model (64-bit precision)[/]")

    benchmarks = []

    console.print("Running [i]pyhs3 + optimistix[/i]...")
    pyhs3_optimistix_benchmark = benchmark_pyhs3_optimistix()
    console.print(pyhs3_optimistix_benchmark)
    benchmarks.append(pyhs3_optimistix_benchmark)

    console.print("Running [i]pyhs3 + iminuit[/i]...")
    pyhs3_iminuit_benchmark = benchmark_pyhs3_iminuit()
    console.print(pyhs3_iminuit_benchmark)
    benchmarks.append(pyhs3_iminuit_benchmark)

    console.print("Running [i]pyhs3 + everwillow[/i]...")
    pyhs3_everwillow_benchmark = benchmark_pyhs3_everwillow()
    console.print(pyhs3_everwillow_benchmark)
    benchmarks.append(pyhs3_everwillow_benchmark)

    console.print("Running [i]pyhs3 + everwillow.ifit[/i]...")
    pyhs3_everwillow_ifit_benchmark = benchmark_pyhs3_everwillow_ifit()
    console.print(pyhs3_everwillow_ifit_benchmark)
    benchmarks.append(pyhs3_everwillow_ifit_benchmark)

    console.print("Running [i]pyhs3 + scipy.minimizer[/i]...")
    pyhs3_scipy_benchmark = benchmark_pyhs3_scipy()
    console.print(pyhs3_scipy_benchmark)
    benchmarks.append(pyhs3_scipy_benchmark)

    console.print("Running [i]evermore + optimistix[/i]...")
    evermore_native_benchmark = benchmark_evermore_native()
    console.print(evermore_native_benchmark)
    benchmarks.append(evermore_native_benchmark)

    console.print("Running [i]evermore + iminuit[/i]...")
    evermore_iminuit_benchmark = benchmark_evermore_iminuit()
    console.print(evermore_iminuit_benchmark)
    benchmarks.append(evermore_iminuit_benchmark)

    console.print("Running [i]evermore + everwillow[/i]...")
    evermore_ew_benchmark = benchmark_evermore_everwillow()
    console.print(evermore_ew_benchmark)
    benchmarks.append(evermore_ew_benchmark)

    console.print("Running [i]evermore + everwillow.ifit[/i]...")
    evermore_ew_ifit_benchmark = benchmark_evermore_everwillow_ifit()
    console.print(evermore_ew_ifit_benchmark)
    benchmarks.append(evermore_ew_ifit_benchmark)

    console.print("Running [i]evermore + scipy.minimizer[/i]...")
    evermore_scipy_benchmark = benchmark_evermore_scipy()
    console.print(evermore_scipy_benchmark)
    benchmarks.append(evermore_scipy_benchmark)

    console.print("Running [i]pyhf + optimistix[/i]...")
    pyhf_optimistix_benchmark = benchmark_pyhf_optimistix()
    console.print(pyhf_optimistix_benchmark)
    benchmarks.append(pyhf_optimistix_benchmark)

    console.print("Running [i]pyhf + scipy.minimizer[/i]...")
    pyhf_native_benchmark = benchmark_pyhf_native()
    console.print(pyhf_native_benchmark)
    benchmarks.append(pyhf_native_benchmark)

    console.print("Running [i]pyhf + iminuit[/i]...")
    pyhf_native_minuit_benchmark = benchmark_pyhf_native_minuit()
    console.print(pyhf_native_minuit_benchmark)
    benchmarks.append(pyhf_native_minuit_benchmark)

    console.print("Running [i]pyhf + everwillow[/i]...")
    pyhf_ew_benchmark = benchmark_pyhf_everwillow()
    console.print(pyhf_ew_benchmark)
    benchmarks.append(pyhf_ew_benchmark)

    console.print("Running [i]pyhf + everwillow.ifit[/i]...")
    pyhf_ew_ifit_benchmark = benchmark_pyhf_everwillow_ifit()
    console.print(pyhf_ew_ifit_benchmark)
    benchmarks.append(pyhf_ew_ifit_benchmark)

    # Generate visualizations
    console.print("\n[bold]Generating benchmark visualizations...[/]")
    plot_benchmark_grid(benchmarks)
    plot_parameter_deviations(benchmarks, reference_name="evermore + optimistix")


if __name__ == "__main__":
    # python examples/model_1sig_2bkg_1nf_1ss_1ns/compare.py
    main()
