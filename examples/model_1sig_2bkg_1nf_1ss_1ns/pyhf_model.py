"""Compact pyhf example helpers."""

import jax.numpy as jnp
import pyhf

from .model_config import DEFAULT_DATA, ModelData, expected_components

pyhf.set_backend("jax")


def _workspace(data: ModelData) -> pyhf.Workspace:
    return pyhf.Workspace(
        {
            "channels": [
                {
                    "name": "singlebin",
                    "samples": [
                        {
                            "name": "signal",
                            "data": [data.signal_nominal],
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None}
                            ],
                        },
                        {
                            "name": "bkg1",
                            "data": [data.bkg1_nominal],
                            "modifiers": [
                                {
                                    "name": "norm1",
                                    "type": "normsys",
                                    "data": {
                                        "hi": data.norm1_up,
                                        "lo": data.norm1_down,
                                    },
                                },
                                {
                                    "name": "shape1",
                                    "type": "histosys",
                                    "data": {
                                        "hi_data": [data.bkg1_shape_up],
                                        "lo_data": [data.bkg1_shape_down],
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
                                    "data": {
                                        "hi": data.norm2_up,
                                        "lo": data.norm2_down,
                                    },
                                },
                                {
                                    "name": "shape1",
                                    "type": "histosys",
                                    "data": {
                                        "hi_data": [data.bkg2_shape_up],
                                        "lo_data": [data.bkg2_shape_down],
                                    },
                                },
                            ],
                        },
                    ],
                }
            ],
            "observations": [{"name": "singlebin", "data": [data.observed]}],
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
                }
            ],
            "version": "1.0.0",
        }
    )


def build_pyhf(
    data: ModelData = DEFAULT_DATA,
) -> tuple[pyhf.pdf.Model, jnp.ndarray, jnp.ndarray, dict[str, slice]]:
    ws = _workspace(data)
    model = ws.model(measurement_name="Measurement")
    dataset = jnp.asarray(ws.data(model), dtype=jnp.float64)
    init = jnp.asarray(model.config.suggested_init(), dtype=jnp.float64)
    slices = {name: model.config.par_slice(name) for name in model.config.par_names}
    return model, dataset, init, slices


def vector_to_dict(theta: jnp.ndarray, slices: dict[str, slice]) -> dict[str, float]:
    return {name: float(theta[slice_][0]) for name, slice_ in slices.items()}


def dict_to_vector(
    params: dict[str, float], theta: jnp.ndarray, slices: dict[str, slice]
):
    vector = theta.copy()
    for name, value in params.items():
        vector = vector.at[slices[name]].set(value)
    return vector


def nll_fn(model: pyhf.pdf.Model, data_vector: jnp.ndarray):
    def nll(theta: jnp.ndarray) -> jnp.ndarray:
        return -jnp.sum(model.logpdf(theta, data_vector))

    return nll


def fit_with_pyhf_native(
    model: pyhf.pdf.Model,
    data_vector: jnp.ndarray,
    init: jnp.ndarray,
    *,
    maxiter: int | None = None,
) -> jnp.ndarray:
    kwargs = {}
    if maxiter is not None:
        kwargs["optimizer"] = pyhf.optimize.scipy_minimize(options={"maxiter": maxiter})
    return jnp.asarray(
        pyhf.infer.mle.fit(
            data_vector,
            model,
            init_pars=init.tolist(),
            par_bounds=model.config.suggested_bounds(),
            **kwargs,
        ),
        dtype=jnp.float64,
    )


def fit_with_everwillow(
    model: pyhf.pdf.Model,
    data_vector: jnp.ndarray,
    init: jnp.ndarray,
    *,
    max_steps: int = 150,
) -> tuple[jnp.ndarray, float]:
    import everwillow as ew

    nll = nll_fn(model, data_vector)
    result = ew.fit(nll, init, max_steps=max_steps)
    return jnp.asarray(result.params, dtype=jnp.float64), float(result.nll)


def summarise_pyhf(
    theta: jnp.ndarray, slices: dict[str, slice], data: ModelData = DEFAULT_DATA
):
    return expected_components(vector_to_dict(theta, slices), data=data)
