"""pyhf implementation of the 1sig_2bkg_1nf_1ss_1ns example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
import pyhf

pyhf.set_backend("jax")
tensorlib = pyhf.tensorlib

from .model_config import (
    DEFAULT_DATA,
    ModelData,
    default_initial_params,
    expected_components,
)


@dataclass(frozen=True)
class PyhfSetup:
    workspace: pyhf.Workspace
    model: pyhf.pdf.Model
    data: list[float]


@dataclass(frozen=True)
class PyhfFitResult:
    params: dict[str, float]
    nll: float


def _build_workspace(data: ModelData) -> pyhf.Workspace:
    down_bkg1 = data.bkg1_shape_down
    down_bkg2 = data.bkg2_shape_down

    spec = {
        "channels": [
            {
                "name": "singlebin",
                "samples": [
                    {
                        "name": "signal",
                        "data": [data.signal_nominal],
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None},
                        ],
                    },
                    {
                        "name": "bkg1",
                        "data": [data.bkg1_nominal],
                        "modifiers": [
                            {
                                "name": "norm1",
                                "type": "normsys",
                                "data": {"hi": data.norm1_up, "lo": data.norm1_down},
                            },
                            {
                                "name": "shape1",
                                "type": "histosys",
                                "data": {
                                    "hi_data": [data.bkg1_shape_up],
                                    "lo_data": [down_bkg1],
                                },
                            },
                        ],
                    },
                    {
                        "name": "bkg2",
                        "data": [data.bkg2_nominal],
                        "modifiers": [
                            {
                                "name": "norm2",
                                "type": "normsys",
                                "data": {"hi": data.norm2_up, "lo": data.norm2_down},
                            },
                            {
                                "name": "shape1",
                                "type": "histosys",
                                "data": {
                                    "hi_data": [data.bkg2_shape_up],
                                    "lo_data": [down_bkg2],
                                },
                            },
                        ],
                    },
                ],
            },
        ],
        "observations": [
            {"name": "singlebin", "data": [data.observed]},
        ],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [
                        {"name": "mu", "inits": [1.0], "bounds": [[0.0, 5.0]]},
                        {"name": "norm1", "inits": [0.0], "bounds": [[-5.0, 5.0]]},
                        {"name": "norm2", "inits": [0.0], "bounds": [[-5.0, 5.0]]},
                        {"name": "shape1", "inits": [0.0], "bounds": [[-5.0, 5.0]]},
                    ],
                },
            },
        ],
        "version": "1.0.0",
    }

    return pyhf.Workspace(spec)


def build_pyhf_setup(data: ModelData = DEFAULT_DATA) -> PyhfSetup:
    workspace = _build_workspace(data)
    model = workspace.model(measurement_name="Measurement")
    model_data = workspace.data(model)
    return PyhfSetup(workspace=workspace, model=model, data=model_data)


def _dict_to_array(model: pyhf.pdf.Model, params: Mapping[str, float]) -> jnp.ndarray:
    array = jnp.asarray(model.config.suggested_init(), dtype=jnp.float64)
    for name, value in params.items():
        index = model.config.par_slice(name)
        array = array.at[index].set(value)
    return array


def _array_to_dict(model: pyhf.pdf.Model, array) -> dict[str, float]:
    tensor = jnp.asarray(array)
    return {
        name: float(tensor[model.config.par_slice(name)][0])
        for name in model.config.par_names
    }


def pyhf_negative_log_likelihood(setup: PyhfSetup) -> Callable[[Mapping[str, float]], jnp.ndarray]:
    """Return an NLL callable compatible with everwillow."""

    def nll(params: Mapping[str, float]) -> jnp.ndarray:
        param_array = _dict_to_array(setup.model, params)
        log_probs = setup.model.logpdf(param_array, setup.data)
        return -tensorlib.sum(log_probs)

    return nll


def fit_with_pyhf(
    setup: PyhfSetup,
    maxiter: int | None = None,
) -> PyhfFitResult:
    """Fit the model with pyhf's MLE optimiser."""

    initial = _dict_to_array(setup.model, default_initial_params())

    fit_kwargs = {}
    if maxiter is not None:
        fit_kwargs["optimizer"] = pyhf.optimize.scipy_minimize(
            options={"maxiter": maxiter}
        )

    fit_array = pyhf.infer.mle.fit(
        setup.data,
        setup.model,
        init_pars=tensorlib.tolist(initial),
        par_bounds=tensorlib.tolist(setup.model.config.suggested_bounds()),
        **fit_kwargs,
    )

    params = _array_to_dict(setup.model, fit_array)
    nll = -float(tensorlib.sum(setup.model.logpdf(fit_array, setup.data)))

    return PyhfFitResult(params=params, nll=nll)


def fit_pyhf_with_everwillow(
    setup: PyhfSetup,
    max_steps: int = 150,
) -> PyhfFitResult:
    """Fit the pyhf model using everwillow's optimiser."""

    import everwillow as ew

    nll_fn = pyhf_negative_log_likelihood(setup)
    result = ew.fit(
        nll_fn,
        dict(default_initial_params()),
        max_steps=max_steps,
    )

    return PyhfFitResult(
        params={key: float(value) for key, value in result.params.items()},
        nll=float(result.nll),
    )


def summarise_pyhf_fit(
    params: Mapping[str, float],
    data: ModelData = DEFAULT_DATA,
) -> dict[str, float]:
    """Mirror pyhs3 and evermore summaries for consistency."""

    return expected_components(params, data=data)
