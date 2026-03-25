"""Shared configuration for the 1sig_2bkg_1nf_1ss_1ns example."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
from utils import log_normal_modifier, shape_interpolate


@dataclass(frozen=True)
class ModelData:
    """Numerical inputs describing the toy analysis."""

    observed: float = 37.0
    signal_nominal: float = 3.0
    bkg1_nominal: float = 10.0
    bkg1_shape_up: float = 12.0
    bkg1_shape_down: float = 8.0
    bkg2_nominal: float = 20.0
    bkg2_shape_up: float = 23.0
    bkg2_shape_down: float = 19.0
    norm1_up: float = 1.1
    norm1_down: float = 0.9
    norm2_up: float = 1.05
    norm2_down: float = 0.95
    constraint_width: float = 1.0


DEFAULT_DATA = ModelData()


def default_initial_params() -> dict[str, float]:
    """Return the nominal starting point for all optimisations."""

    return {
        "mu": 1.0,
        "norm1": 0.0,
        "norm2": 0.0,
        "shape1": 0.0,
    }


def expected_components(
    params: Mapping[str, float],
    data: ModelData = DEFAULT_DATA,
    as_arrays: bool = False,
) -> dict[str, float | jnp.ndarray]:
    """Compute expected event yields for each physics component.

    The expectation mirrors the expressions implemented in the pyhs3 model.
    """

    mu = params["mu"]
    norm1 = params["norm1"]
    norm2 = params["norm2"]
    shape1 = params["shape1"]

    signal_expected = jnp.asarray(mu) * data.signal_nominal

    bkg1_interp = shape_interpolate(data.bkg1_nominal, data.bkg1_shape_up, shape1)
    bkg1_modifier = log_normal_modifier(norm1, data.norm1_up, data.norm1_down)
    bkg1_expected = bkg1_modifier * bkg1_interp

    bkg2_interp = shape_interpolate(data.bkg2_nominal, data.bkg2_shape_up, shape1)
    bkg2_modifier = log_normal_modifier(norm2, data.norm2_up, data.norm2_down)
    bkg2_expected = bkg2_modifier * bkg2_interp

    total_expected = signal_expected + bkg1_expected + bkg2_expected

    if as_arrays:
        return {
            "signal": signal_expected,
            "bkg1": bkg1_expected,
            "bkg2": bkg2_expected,
            "total": total_expected,
        }

    return {
        "signal": float(signal_expected),
        "bkg1": float(bkg1_expected),
        "bkg2": float(bkg2_expected),
        "total": float(total_expected),
    }


def free_parameter_names() -> tuple[str, ...]:
    """Names of the floating parameters used in all implementations."""

    return ("mu", "norm1", "norm2", "shape1")


def gaussian_constraint_width() -> float:
    """Width of the Gaussian constraints (shared across modifiers)."""

    return DEFAULT_DATA.constraint_width
