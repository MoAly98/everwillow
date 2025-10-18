"""Compare fits across libraries for the 1sig_2bkg_1nf_1ss_1ns example."""

import jax
from rich.console import Console
from rich.table import Table

import everwillow as ew

try:  # Package import for -m execution
    from .evermore_model import fit_with_everwillow as fit_evermore_with_everwillow
    from .evermore_model import fit_with_optimistix, summarise_evermore_fit
    from .pyhf_model import (
        build_pyhf,
        fit_with_pyhf_native,
        summarise_pyhf,
        vector_to_dict,
    )
    from .pyhf_model import (
        fit_with_everwillow as fit_pyhf_with_everwillow,
    )
    from .pyhf_model import (
        nll_fn as pyhf_nll,
    )
    from .pyhs3_model import build_pyhs3, summarise_pyhs3_fit
except ImportError:  # Script execution fallback
    from evermore_model import fit_with_everwillow as fit_evermore_with_everwillow
    from evermore_model import fit_with_optimistix, summarise_evermore_fit
    from pyhf_model import (
        build_pyhf,
        fit_with_pyhf_native,
        summarise_pyhf,
        vector_to_dict,
    )
    from pyhf_model import (
        fit_with_everwillow as fit_pyhf_with_everwillow,
    )
    from pyhf_model import (
        nll_fn as pyhf_nll,
    )
    from pyhs3_model import build_pyhs3, summarise_pyhs3_fit


def _format(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    console = Console()
    jax.config.update("jax_enable_x64", True)

    console.rule("Building models")

    pyhs3_nll, pyhs3_init = build_pyhs3()
    pyhf_model, pyhf_data, pyhf_init, pyhf_slices = build_pyhf()

    console.rule("Running fits")

    pyhs3_fit = ew.fit(pyhs3_nll, pyhs3_init, max_steps=150)
    pyhs3_params = dict(pyhs3_fit.params)
    pyhs3_nll = float(pyhs3_fit.nll)

    evermore_native_params, evermore_native_nll = fit_with_optimistix()
    evermore_ew_params, evermore_ew_nll = fit_evermore_with_everwillow()

    pyhf_ew_vector, pyhf_ew_nll = fit_pyhf_with_everwillow(
        pyhf_model, pyhf_data, pyhf_init
    )
    pyhf_nll_fn = pyhf_nll(pyhf_model, pyhf_data)
    pyhf_native_vector = fit_with_pyhf_native(pyhf_model, pyhf_data, pyhf_init)
    pyhf_native_nll = float(pyhf_nll_fn(pyhf_native_vector))

    pyhf_ew_params = vector_to_dict(pyhf_ew_vector, pyhf_slices)
    pyhf_native_params = vector_to_dict(pyhf_native_vector, pyhf_slices)

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
            _format(evermore_ew_params[name]),
            _format(evermore_native_params[name]),
            _format(pyhf_ew_params[name]),
            _format(pyhf_native_params[name]),
        )

    parameter_table.add_row(
        "NLL",
        _format(pyhs3_nll),
        _format(evermore_ew_nll),
        _format(evermore_native_nll),
        _format(pyhf_ew_nll),
        _format(pyhf_native_nll),
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
    evermore_yields = summarise_evermore_fit(evermore_ew_params)
    evermore_native_yields = summarise_evermore_fit(evermore_native_params)
    pyhf_everwillow_yields = summarise_pyhf(pyhf_ew_vector, pyhf_slices)
    pyhf_native_yields = summarise_pyhf(pyhf_native_vector, pyhf_slices)

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
