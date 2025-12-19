"""Compare fits across libraries for the 1sig_2bkg_1nf_1ss_1ns example."""

import jax
from rich.console import Console
from rich.table import Table

try:  # Package import for -m execution
    from .evermore_model import fit_with_everwillow as fit_evermore_with_everwillow
    from .evermore_model import fit_with_optimistix, summarise_evermore_fit
except ImportError:  # Script execution fallback
    from evermore_model import fit_with_everwillow as fit_evermore_with_everwillow
    from evermore_model import summarise_evermore_fit


def _format(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    console = Console()
    jax.config.update("jax_enable_x64", True)

    evermore_ew_params, evermore_ew_nll = fit_evermore_with_everwillow()
    evermore_iew_params, evermore_iew_nll = fit_evermore_with_everwillow(
        interactive=True
    )

    console.rule("Fit parameter comparison")

    parameter_table = Table(show_header=True, header_style="bold")
    parameter_table.add_column("Parameter")
    parameter_table.add_column("evermore + everwillow", justify="right")
    parameter_table.add_column("evermore + everwillow (interactive)", justify="right")

    parameters = ["mu", "norm1", "norm2", "shape1"]

    for name in parameters:
        parameter_table.add_row(
            name,
            _format(evermore_ew_params[name]),
            _format(evermore_iew_params[name]),
        )

    parameter_table.add_row(
        "NLL",
        _format(evermore_ew_nll),
        _format(evermore_iew_nll),
    )

    console.print(parameter_table)

    console.rule("Expected yields at best fit")

    yields_table = Table(show_header=True, header_style="bold")
    yields_table.add_column("Component")
    yields_table.add_column("evermore + everwillow", justify="right")
    yields_table.add_column("evermore + everwillow (interactive)", justify="right")

    evermore_yields = summarise_evermore_fit(evermore_ew_params)
    evermore_interactive_yields = summarise_evermore_fit(evermore_iew_params)

    for component in ["signal", "bkg1", "bkg2", "total"]:
        yields_table.add_row(
            component,
            _format(evermore_yields[component]),
            _format(evermore_interactive_yields[component]),
        )

    console.print(yields_table)


if __name__ == "__main__":
    main()
