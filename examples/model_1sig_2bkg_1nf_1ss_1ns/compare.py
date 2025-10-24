"""Compare fits across libraries for the 1sig_2bkg_1nf_1ss_1ns example."""

import time
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
from evermore_model import build_components, fit_with_optimistix, summarise_evermore_fit
from evermore_model import fit_with_everwillow as fit_evermore_with_everwillow
from pyhf_model import build_pyhf, fit_with_pyhf_native, summarise_pyhf
from pyhf_model import fit_with_everwillow as fit_pyhf_with_everwillow
from pyhs3_model import build_pyhs3, summarise_pyhs3_fit
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table

import everwillow as ew

jax.config.update("jax_enable_x64", True)


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

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
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


N_STEPS = 1000


def benchmark_pyhs3():
    nll, initial = build_pyhs3()

    def fun():
        return ew.fit(jax.jit(nll), initial, max_steps=N_STEPS)

    # Cold run
    runtime_cold, fit = time_and_run(fun)
    params = dict(fit.params)
    nll_value = float(fit.nll)

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


def benchmark_evermore_native():
    components = build_components()
    fun = jax.jit(lambda: fit_with_optimistix(components, max_steps=N_STEPS))
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
    # enable once https://github.com/MoAly98/everwillow/pull/27 is merged
    # fun = jax.jit(fit_evermore_with_everwillow)
    components = build_components()

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


def benchmark_pyhf_native():
    pyhf_model, pyhf_data, pyhf_init, pyhf_slices = build_pyhf()
    # can't be jitted due to pyhf internals

    def fun():
        return fit_with_pyhf_native(
            pyhf_model, pyhf_data, pyhf_init, pyhf_slices, maxiter=N_STEPS
        )

    # Cold run
    runtime_cold, (params, nll_value) = time_and_run(fun)
    # expected yields
    expected_yields = summarise_pyhf(params)

    # Hot run
    runtime_hot, _ = time_and_run(fun)
    return Benchmark(
        name="pyhf + native",
        params=params,
        nll=nll_value,
        expected_yields=expected_yields,
        runtime_cold=runtime_cold,
        runtime_hot=runtime_hot,
    )


def benchmark_pyhf_everwillow():
    pyhf_model, pyhf_data, pyhf_init, pyhf_slices = build_pyhf()

    # enable once https://github.com/MoAly98/everwillow/pull/27 is merged
    # fun = jax.jit(lambda: fit_pyhf_with_everwillow(pyhf_model, pyhf_data, pyhf_init))
    def fun():
        return fit_pyhf_with_everwillow(
            pyhf_model, pyhf_data, pyhf_init, pyhf_slices, max_steps=N_STEPS
        )

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


def main() -> None:
    console = Console()
    jax.config.update("jax_enable_x64", True)

    console.print(
        "[bold]Benchmarking fits for 1sig_2bkg_1nf_1ss_1ns model (64-bit precision)[/]"
    )

    console.print("Running [i]pyhs3 + everwillow[/i]...")
    pyhs3_benchmark = benchmark_pyhs3()
    console.print(pyhs3_benchmark)

    console.print("Running [i]evermore + optimistix[/i]...")
    evermore_native_benchmark = benchmark_evermore_native()
    console.print(evermore_native_benchmark)

    console.print("Running [i]evermore + everwillow[/i]...")
    evermore_ew_benchmark = benchmark_evermore_everwillow()
    console.print(evermore_ew_benchmark)

    console.print("Running [i]pyhf + native[/i]...")
    pyhf_native_benchmark = benchmark_pyhf_native()
    console.print(pyhf_native_benchmark)

    console.print("Running [i]pyhf + everwillow[/i]...")
    pyhf_ew_benchmark = benchmark_pyhf_everwillow()
    console.print(pyhf_ew_benchmark)


if __name__ == "__main__":
    # python examples/model_1sig_2bkg_1nf_1ss_1ns/compare.py
    main()
