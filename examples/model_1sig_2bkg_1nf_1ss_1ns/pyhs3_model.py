"""Minimal helpers for building the pyhs3 model used in the comparison."""

from __future__ import annotations

from collections.abc import Mapping

import iminuit
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pyhs3
from model_config import (
    DEFAULT_DATA,
    ModelData,
    default_initial_params,
    expected_components,
    gaussian_constraint_width,
)
from pyhs3.data import PointData
from pyhs3.distributions import GaussianDist, PoissonDist, ProductDist
from pyhs3.functions import GenericFunction
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from scipy.optimize import minimize
from utils import jaxify_distribution

import everwillow as ew
import everwillow.statelib as sl

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision


def nll_fn(inputs, jaxified, fixed_values):
    """Create NLL function from pyhs3 model components."""

    @jax.jit
    def nll(params: Mapping[str, float]) -> jnp.ndarray:
        merged = {**fixed_values, **params}
        ordered = [merged[var.name] for var in inputs]
        probability = jaxified(*ordered)[0]
        return -jnp.log(jnp.asarray(probability))

    return nll


def build_pyhs3(
    data: ModelData = DEFAULT_DATA,
) -> tuple[list, callable, dict[str, float], dict[str, float]]:
    """Return (inputs, jaxified, fixed_values, initial-parameter dict)."""

    workspace = pyhs3.Workspace(
        metadata=Metadata(hs3_version="0.2"),
        distributions=_build_distributions(),
        functions=_build_functions(),
        parameter_points=[
            ParameterSet(
                name="default_values",
                parameters=[ParameterPoint(name=k, value=v) for k, v in default_initial_params().items()],
            )
        ],
        data=_build_data_points(data),
    )

    model = workspace.model()
    inputs, jaxified = jaxify_distribution(model, "model")

    initial = {point.name: float(point.value) for point in workspace.parameter_points[0].parameters}
    fixed_values = {point.name: float(point.value) for point in workspace.data}

    return inputs, jaxified, fixed_values, initial


def fit_with_everwillow(
    inputs,
    jaxified,
    fixed_values,
    initial: dict[str, float],
    *,
    max_steps: int = 150,
    interactive: bool = False,
) -> tuple[dict[str, float], float]:
    """Fit using everwillow optimizer."""
    nll = nll_fn(inputs, jaxified, fixed_values)
    init_state = sl.State.from_pytree(initial)

    if not interactive:
        result = ew.fit(nll, init_state, max_steps=max_steps)
    else:
        result = ew.ifit(nll, init_state, max_steps=max_steps)

    params = dict(result.params)
    return params, result.nll


def fit_with_optimistix(
    inputs,
    jaxified,
    fixed_values,
    initial: dict[str, float],
    *,
    max_steps: int = 10_000,
) -> tuple[dict[str, float], float]:
    """Fit using optimistix BFGS optimizer."""
    nll = nll_fn(inputs, jaxified, fixed_values)

    # Convert dict params to array for optimistix
    param_names = sorted(initial.keys())
    init_array = jnp.array([initial[name] for name in param_names])

    # Wrapper that converts array -> dict -> NLL
    def nll_array(params_array: jnp.ndarray, args: tuple) -> jnp.ndarray:
        params_dict = {name: params_array[i] for i, name in enumerate(param_names)}
        return nll(params_dict)

    solver = optx.BFGS(rtol=1e-5, atol=1e-7)
    result = optx.minimise(
        nll_array,
        solver,
        init_array,
        args=(),
        has_aux=False,
        max_steps=max_steps,
    )

    # Convert result back to dict
    best_params_array = result.value
    best_params = {name: best_params_array[i] for i, name in enumerate(param_names)}
    nll_value = result.state.f_info.f

    return best_params, nll_value


def fit_with_iminuit(
    inputs,
    jaxified,
    fixed_values,
    initial: dict[str, float],
    *,
    max_steps: int = 10_000,
) -> tuple[dict[str, float], float]:
    """Fit using iminuit optimizer."""
    nll = nll_fn(inputs, jaxified, fixed_values)

    # Convert dict params to array for iminuit
    param_names = sorted(initial.keys())
    init_array = np.array([initial[name] for name in param_names])

    # Wrapper that converts array -> dict -> NLL
    def nll_array(params_array):
        params_dict = {name: float(params_array[i]) for i, name in enumerate(param_names)}
        return nll(params_dict)

    # Gradient function using JAX
    def grad_nll_array(params_array):
        # Convert to jax array
        jax_array = jnp.array(params_array)

        # Create dict and compute gradient
        def nll_for_grad(arr):
            params_dict = {name: arr[i] for i, name in enumerate(param_names)}
            return nll(params_dict)

        grad_jax = jax.grad(nll_for_grad)(jax_array)
        return np.array(grad_jax)

    # Setup Minuit
    minuit = iminuit.Minuit(
        nll_array,
        init_array,
        grad=grad_nll_array,
    )
    minuit.errordef = iminuit.Minuit.LIKELIHOOD
    minuit.strategy = 2
    minuit.tol = 1e-8

    # Minimize
    minuit.migrad(ncall=max_steps, use_simplex=False)

    # Convert result back to dict
    best_params = {name: minuit.values[i] for i, name in enumerate(param_names)}
    return best_params, minuit.fval


def fit_with_scipy(
    inputs,
    jaxified,
    fixed_values,
    initial: dict[str, float],
    *,
    max_steps: int = 10_000,
) -> tuple[dict[str, float], float]:
    """Fit using scipy.optimize.minimize with SLSQP."""
    nll = nll_fn(inputs, jaxified, fixed_values)

    # Convert dict params to array for scipy
    param_names = sorted(initial.keys())
    init_array = np.array([initial[name] for name in param_names])

    # Wrapper that converts array -> dict -> NLL
    def nll_array(params_array):
        params_dict = {name: float(params_array[i]) for i, name in enumerate(param_names)}
        return nll(params_dict)

    # Gradient function using JAX
    def grad_nll_array(params_array):
        # Convert to jax array
        jax_array = jnp.array(params_array)

        # Create dict and compute gradient
        def nll_for_grad(arr):
            params_dict = {name: arr[i] for i, name in enumerate(param_names)}
            return nll(params_dict)

        grad_jax = jax.grad(nll_for_grad)(jax_array)
        return np.array(grad_jax)

    # Minimize using scipy
    result = minimize(
        nll_array,
        init_array,
        method="SLSQP",
        jac=grad_nll_array,
        options={"maxiter": max_steps, "ftol": 1e-8},
    )

    # Convert result back to dict
    best_params = {name: result.x[i] for i, name in enumerate(param_names)}
    return best_params, result.fun


def summarise_pyhs3_fit(
    params: Mapping[str, float],
    data: ModelData = DEFAULT_DATA,
) -> dict[str, float]:
    return expected_components(params, data=data)


def _build_distributions():
    return [
        PoissonDist(name="main_poisson", x="n_obs", mean="n_expected"),
        GaussianDist(
            name="norm1_constraint",
            x="a_norm1",
            mean="norm1",
            sigma=gaussian_constraint_width(),
        ),
        GaussianDist(
            name="norm2_constraint",
            x="a_norm2",
            mean="norm2",
            sigma=gaussian_constraint_width(),
        ),
        GaussianDist(
            name="shape1_constraint",
            x="a_shape1",
            mean="shape1",
            sigma=gaussian_constraint_width(),
        ),
        ProductDist(
            type="product_dist",
            name="model",
            factors=[
                "main_poisson",
                "norm1_constraint",
                "norm2_constraint",
                "shape1_constraint",
            ],
        ),
    ]


def _build_functions():
    return [
        GenericFunction(
            type="generic_function",
            name="signal_expected",
            expression="mu * signal_nominal",
        ),
        GenericFunction(
            type="generic_function",
            name="bkg1_lnN_factor",
            expression="exp(norm1 * (log(1.1)**((norm1 + abs(norm1))/(2*abs(norm1) + 1e-10)) * log(1.0/0.9)**((abs(norm1) - norm1)/(2*abs(norm1) + 1e-10))))",
        ),
        GenericFunction(
            type="generic_function",
            name="bkg1_shape_interp",
            expression="bkg1_nominal + shape1 * (bkg1_shape_up - bkg1_nominal)",
        ),
        GenericFunction(
            type="generic_function",
            name="bkg1_expected",
            expression="bkg1_lnN_factor * bkg1_shape_interp",
        ),
        GenericFunction(
            type="generic_function",
            name="bkg2_lnN_factor",
            expression="exp(norm2 * (log(1.05)**((norm2 + abs(norm2))/(2*abs(norm2) + 1e-10)) * log(1.0/0.95)**((abs(norm2) - norm2)/(2*abs(norm2) + 1e-10))))",
        ),
        GenericFunction(
            type="generic_function",
            name="bkg2_shape_interp",
            expression="bkg2_nominal + shape1 * (bkg2_shape_up - bkg2_nominal)",
        ),
        GenericFunction(
            type="generic_function",
            name="bkg2_expected",
            expression="bkg2_lnN_factor * bkg2_shape_interp",
        ),
        GenericFunction(
            type="generic_function",
            name="n_expected",
            expression="signal_expected + bkg1_expected + bkg2_expected",
        ),
    ]


def _build_data_points(data: ModelData):
    return [
        PointData(name="n_obs", value=data.observed),
        PointData(name="a_norm1", value=0.0),
        PointData(name="a_norm2", value=0.0),
        PointData(name="a_shape1", value=0.0),
        PointData(name="signal_nominal", value=data.signal_nominal),
        PointData(name="bkg1_nominal", value=data.bkg1_nominal),
        PointData(name="bkg1_shape_up", value=data.bkg1_shape_up),
        PointData(name="bkg2_nominal", value=data.bkg2_nominal),
        PointData(name="bkg2_shape_up", value=data.bkg2_shape_up),
    ]
