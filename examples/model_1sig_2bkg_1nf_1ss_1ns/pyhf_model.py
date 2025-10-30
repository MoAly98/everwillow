"""Compact pyhf example helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pyhf
from model_config import DEFAULT_DATA, ModelData, expected_components

import everwillow as ew

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision
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
    return {name: theta[slice_][0] for name, slice_ in slices.items()}


def dict_to_vector(
    params: dict[str, float], theta: jnp.ndarray, slices: dict[str, slice]
):
    vector = theta.copy()
    for name, value in params.items():
        vector = vector.at[slices[name]].set(value)
    return vector


def nll_fn(model: pyhf.pdf.Model, data_vector: jnp.ndarray):
    @jax.jit
    def nll(theta: jnp.ndarray) -> jnp.ndarray:
        return -jnp.sum(model.logpdf(theta, data_vector))

    return nll


def fit_with_pyhf_native(
    model: pyhf.pdf.Model,
    data_vector: jnp.ndarray,
    init: jnp.ndarray,
    slices: dict[str, slice],
    *,
    maxiter: int | None = None,
) -> jnp.ndarray:
    kwargs = {}
    if maxiter is not None:
        kwargs["maxiter"] = maxiter
    kwargs["tolerance"] = 1e-8
    params = jnp.asarray(
        pyhf.infer.mle.fit(
            data_vector,
            model,
            init_pars=init.tolist(),
            par_bounds=model.config.suggested_bounds(),
            **kwargs,
        ),
        dtype=jnp.float64,
    )
    nll = nll_fn(model, data_vector)
    return vector_to_dict(params, slices), nll(params)


def fit_with_pyhf_native_minuit(
    model: pyhf.pdf.Model,
    data_vector: jnp.ndarray,
    init: jnp.ndarray,
    slices: dict[str, slice],
    *,
    maxiter: int | None = None,
) -> np.ndarray:
    pyhf.set_backend(
        "jax",
        pyhf.optimize.minuit_optimizer(tolerance=1e-8, verbose=0, strategy=2),
    )
    kwargs = {}
    if maxiter is not None:
        kwargs["maxiter"] = maxiter
    params = np.asarray(
        pyhf.infer.mle.fit(
            data_vector,
            model,
            init_pars=init.tolist(),
            par_bounds=model.config.suggested_bounds(),
            **kwargs,
        ),
        dtype=np.float64,
    )
    nll = nll_fn(model, data_vector)
    return vector_to_dict(params, slices), nll(params)


def fit_with_everwillow(
    model: pyhf.pdf.Model,
    data_vector: jnp.ndarray,
    init: jnp.ndarray,
    slices: dict[str, slice],
    *,
    max_steps: int = 150,
) -> tuple[dict[str, float], float]:
    nll = nll_fn(model, data_vector)
    result = ew.fit(nll, init, max_steps=max_steps)
    params = jnp.asarray(result.params, dtype=jnp.float64)
    return vector_to_dict(params, slices), result.nll


def fit_with_optimistix(
    model: pyhf.pdf.Model,
    data_vector: jnp.ndarray,
    init: jnp.ndarray,
    slices: dict[str, slice],
    *,
    max_steps: int = 10_000,
) -> tuple[dict[str, float], float]:
    nll = nll_fn(model, data_vector)

    # Wrapper that adapts nll(theta) to optimistix's nll(theta, args) signature
    def nll_optx(theta: jnp.ndarray, args: tuple) -> jnp.ndarray:
        return nll(theta)

    solver = optx.BFGS(rtol=1e-5, atol=1e-7)
    result = optx.minimise(
        nll_optx,
        solver,
        init,
        args=(),
        has_aux=False,
        max_steps=max_steps,
    )

    params = jnp.asarray(result.value, dtype=jnp.float64)
    nll_value = result.state.f_info.f
    return vector_to_dict(params, slices), nll_value


def summarise_pyhf(params: dict[str, float], data: ModelData = DEFAULT_DATA):
    return expected_components(params, data=data)
