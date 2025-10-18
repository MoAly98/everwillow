"""Minimal helpers for building the pyhs3 model used in the comparison."""

from collections.abc import Mapping

import jax.numpy as jnp
import pyhs3
from pyhs3.data import PointData
from pyhs3.distributions import GaussianDist, PoissonDist, ProductDist
from pyhs3.functions import GenericFunction
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterSet

from .model_config import (
    DEFAULT_DATA,
    ModelData,
    default_initial_params,
    expected_components,
    gaussian_constraint_width,
)
from .utils import jaxify_distribution


def build_pyhs3(
    data: ModelData = DEFAULT_DATA,
) -> tuple[callable, dict[str, float]]:
    """Return (negative log-likelihood, initial-parameter dict)."""

    workspace = pyhs3.Workspace(
        metadata=Metadata(hs3_version="0.2"),
        distributions=_build_distributions(),
        functions=_build_functions(),
        parameter_points=[
            ParameterSet(
                name="default_values",
                parameters=[
                    ParameterPoint(name=k, value=v)
                    for k, v in default_initial_params().items()
                ],
            )
        ],
        data=_build_data_points(data),
    )

    model = workspace.model()
    inputs, jaxified = jaxify_distribution(model, "model")

    initial = {
        point.name: float(point.value)
        for point in workspace.parameter_points[0].parameters
    }
    fixed_values = {point.name: float(point.value) for point in workspace.data}

    def nll(params: Mapping[str, float]) -> jnp.ndarray:
        merged = {**fixed_values, **params}
        ordered = [merged[var.name] for var in inputs]
        probability = jaxified(*ordered)[0]
        return -jnp.log(jnp.asarray(probability))

    return nll, initial


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
