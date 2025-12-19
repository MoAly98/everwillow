"""Compare fits across libraries for the 1sig_2bkg_1nf_1ss_1ns example."""

from __future__ import annotations

import jax
import everwillow as ew
from rich.console import Console
from rich.table import Table

try:  # Package import for -m execution
    from .evermore_model import build_evermore_setup, fit_with_everwillow as fit_evermore_with_everwillow, fit_with_optimistix, summarise_evermore_fit
    from .model_config import expected_components
    from .pyhf_model import (
        build_pyhf_setup,
        fit_pyhf_with_everwillow,
        fit_with_pyhf,
        summarise_pyhf_fit,
    )
    from .pyhs3_model import build_pyhs3_setup, summarise_pyhs3_fit
except ImportError:  # Script execution fallback
    from evermore_model import build_evermore_setup, fit_with_everwillow as fit_evermore_with_everwillow, fit_with_optimistix, summarise_evermore_fit
    from model_config import expected_components
    from pyhf_model import (
        build_pyhf_setup,
        fit_pyhf_with_everwillow,
        fit_with_pyhf,
        summarise_pyhf_fit,
    )
    from pyhs3_model import build_pyhs3_setup, summarise_pyhs3_fit


def _format(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    console = Console()
    jax.config.update("jax_enable_x64", True)

    console.rule("Building models")

    pyhs3_setup = build_pyhs3_setup()
    evermore_setup = build_evermore_setup()
    pyhf_setup = build_pyhf_setup()

    console.rule("Running fits")

    pyhs3_fit = ew.fit(
        pyhs3_setup.negative_log_likelihood,
        pyhs3_setup.initial_params,
        max_steps=150,
    )
    pyhs3_params = dict(pyhs3_fit.params)
    pyhs3_nll = float(pyhs3_fit.nll)

    evermore_native = fit_with_optimistix(evermore_setup)
    evermore_ew = fit_evermore_with_everwillow(evermore_setup)

    pyhf_ew = fit_pyhf_with_everwillow(pyhf_setup)
    pyhf_native = fit_with_pyhf(pyhf_setup)

    console.rule("Fit parameter comparison")

    parameter_table = Table(show_header=True, header_style="bold")
    parameter_table.add_column("Parameter")
    parameter_table.add_column("pyhs3 + everwillow", justify="right")
    parameter_table.add_column("evermore + everwillow", justify="right")
    parameter_table.add_column("evermore + optimistix", justify="right")
    parameter_table.add_column("pyhf + everwillow", justify="right")
    parameter_table.add_column("pyhf + native", justify="right")

    parameters = ["mu", "norm1", "norm2", "shape1"]

    for name in parameters:
        parameter_table.add_row(
            name,
            _format(pyhs3_params[name]),
            _format(evermore_ew.params[name]),
            _format(evermore_native.params[name]),
            _format(pyhf_ew.params[name]),
            _format(pyhf_native.params[name]),
        )

    parameter_table.add_row(
        "NLL",
        _format(pyhs3_nll),
        _format(evermore_ew.nll),
        _format(evermore_native.nll),
        _format(pyhf_ew.nll),
        _format(pyhf_native.nll),
    )

    console.print(parameter_table)

    console.rule("Expected yields at best fit")

    yields_table = Table(show_header=True, header_style="bold")
    yields_table.add_column("Component")
    yields_table.add_column("pyhs3 + everwillow", justify="right")
    yields_table.add_column("evermore + everwillow", justify="right")
    yields_table.add_column("evermore + optimistix", justify="right")
    yields_table.add_column("pyhf + everwillow", justify="right")
    yields_table.add_column("pyhf + native", justify="right")

    pyhs3_yields = summarise_pyhs3_fit(pyhs3_params)
    evermore_yields = summarise_evermore_fit(evermore_ew.params)
    evermore_native_yields = summarise_evermore_fit(evermore_native.params)
    pyhf_everwillow_yields = summarise_pyhf_fit(pyhf_ew.params)
    pyhf_native_yields = summarise_pyhf_fit(pyhf_native.params)

    for component in ["signal", "bkg1", "bkg2", "total"]:
        yields_table.add_row(
            component,
            _format(pyhs3_yields[component]),
            _format(evermore_yields[component]),
            _format(evermore_native_yields[component]),
            _format(pyhf_everwillow_yields[component]),
            _format(pyhf_native_yields[component]),
        )

    console.print(yields_table)


if __name__ == "__main__":
    main()
