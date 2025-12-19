"""evermore reference implementation for the 1sig_2bkg_1nf_1ss_1ns example.

This module mirrors the structure of the upstream evermore example by
embedding the same model and loss definitions, then wrapping them with
helpers that make it easy to run fits from the everwillow examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

import evermore as evm

from .model_config import DEFAULT_DATA, ModelData, default_initial_params, expected_components

jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# Evermore reference model (copied from evermore/examples/model.py)
# -----------------------------------------------------------------------------


class Params(eqx.Module):
    mu: evm.Parameter
    norm1: evm.NormalParameter
    norm2: evm.NormalParameter
    shape1: evm.NormalParameter


def build_hists(data: ModelData) -> dict[str, dict[str, jnp.ndarray]]:
    """Create the histogram dictionary used by the evermore example."""

    return {
        "nominal": {
            "signal": jnp.array([data.signal_nominal]),
            "bkg1": jnp.array([data.bkg1_nominal]),
            "bkg2": jnp.array([data.bkg2_nominal]),
        },
        "shape_up": {
            "bkg1": jnp.array([data.bkg1_shape_up]),
            "bkg2": jnp.array([data.bkg2_shape_up]),
        },
        "shape_down": {
            "bkg1": jnp.array([data.bkg1_shape_down]),
            "bkg2": jnp.array([data.bkg2_shape_down]),
        },
    }


def build_params() -> Params:
    """Instantiate the evermore parameters with the same defaults as upstream."""

    return Params(
        mu=evm.Parameter(name="mu"),
        norm1=evm.NormalParameter(name="norm1"),
        norm2=evm.NormalParameter(name="norm2"),
        shape1=evm.NormalParameter(name="shape1"),
    )


def model(
    params: Params,
    hists: dict[str, dict[str, jnp.ndarray]],
) -> dict[str, jnp.ndarray]:
    """Evermore expectation model (identical to the reference example)."""

    expectations: dict[str, jnp.ndarray] = {}

    # signal process
    sig_mod = params.mu.scale()
    expectations["signal"] = sig_mod(hists["nominal"]["signal"])

    # bkg1 process
    bkg1_lnN = params.norm1.scale_log(up=jnp.array([1.1]), down=jnp.array([0.9]))
    bkg1_shape = params.shape1.morphing(
        up_template=hists["shape_up"]["bkg1"],
        down_template=hists["shape_down"]["bkg1"],
    )
    expectations["bkg1"] = (bkg1_lnN @ bkg1_shape)(hists["nominal"]["bkg1"])

    # bkg2 process
    bkg2_lnN = params.norm2.scale_log(up=jnp.array([1.05]), down=jnp.array([0.95]))
    bkg2_shape = params.shape1.morphing(
        up_template=hists["shape_up"]["bkg2"],
        down_template=hists["shape_down"]["bkg2"],
    )
    expectations["bkg2"] = (bkg2_lnN @ bkg2_shape)(hists["nominal"]["bkg2"])

    return expectations


@eqx.filter_jit
def loss(
    dynamic: Params,
    static: Params,
    hists: dict[str, dict[str, jnp.ndarray]],
    observation: jnp.ndarray,
) -> jnp.ndarray:
    """Evermore loss function copied from the reference example."""

    params = evm.tree.combine(dynamic, static)
    expectations = model(params, hists)
    constraints = evm.loss.get_log_probs(params)
    loss_val = (
        evm.pdf.PoissonContinuous(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)


# -----------------------------------------------------------------------------
# Helper utilities for the everwillow example
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvermoreSetup:
    params: Params
    hists: dict[str, dict[str, jnp.ndarray]]
    observation: jnp.ndarray
    data: ModelData = DEFAULT_DATA


def build_evermore_setup(data: ModelData = DEFAULT_DATA) -> EvermoreSetup:
    """Create Params, histograms, and observation matching the reference example."""

    return EvermoreSetup(
        params=build_params(),
        hists=build_hists(data),
        observation=jnp.array([data.observed]),
        data=data,
    )


def _params_to_dict(params: Params) -> dict[str, float]:
    return {
        "mu": float(params.mu.value),
        "norm1": float(params.norm1.value),
        "norm2": float(params.norm2.value),
        "shape1": float(params.shape1.value),
    }


def initial_parameter_dict(setup: EvermoreSetup) -> dict[str, float]:
    """Return numeric starting values for optimisation."""

    defaults = default_initial_params()
    # Replace the evermore defaults with the explicit initial values we use elsewhere.
    return {
        "mu": defaults["mu"],
        "norm1": defaults["norm1"],
        "norm2": defaults["norm2"],
        "shape1": defaults["shape1"],
    }


@dataclass(frozen=True)
class EvermoreFitResult:
    params: dict[str, float]
    nll: float


def fit_with_optimistix(
    setup: EvermoreSetup,
    max_steps: int = 10_000,
) -> EvermoreFitResult:
    """Run the evermore native optimistix fit (as in nll_fit_optimistix.py)."""

    dynamic, static = evm.tree.partition(setup.params)

    def optx_loss(dynamic_params, args):
        static_params, hists, observation = args
        return loss(dynamic_params, static_params, hists, observation)

    solver = optx.BFGS(rtol=1e-5, atol=1e-7)
    result = optx.minimise(
        optx_loss,
        solver,
        dynamic,
        has_aux=False,
        args=(static, setup.hists, setup.observation),
        max_steps=max_steps,
    )

    bestfit_params = evm.tree.combine(result.value, static)
    nll_value = float(
        loss(result.value, static, setup.hists, setup.observation)
    )

    return EvermoreFitResult(params=_params_to_dict(bestfit_params), nll=nll_value)


def fit_with_everwillow(
    setup: EvermoreSetup,
    max_steps: int = 150,
) -> EvermoreFitResult:
    """Fit the evermore loss with everwillow (mirroring the example wrapper)."""

    import everwillow as ew

    # Partition once so we can reconstruct the Params tree inside the wrapper.
    dynamic, static = evm.tree.partition(setup.params)

    def nll_fn(param_dict: Mapping[str, float]) -> jnp.ndarray:
        updated = evm.tree.update_values(
            dynamic,
            values=Params(
                mu=param_dict["mu"],
                norm1=param_dict["norm1"],
                norm2=param_dict["norm2"],
                shape1=param_dict["shape1"],
            ),
        )
        return loss(updated, static, setup.hists, setup.observation)

    result = ew.fit(
        nll_fn,
        initial_parameter_dict(setup),
        max_steps=max_steps,
    )

    params = {name: float(value) for name, value in result.params.items()}
    nll_value = float(result.nll)

    return EvermoreFitResult(params=params, nll=nll_value)


def summarise_evermore_fit(
    params: Mapping[str, float],
    data: ModelData = DEFAULT_DATA,
) -> dict[str, float]:
    """Return expected component yields for comparison tables."""

    return expected_components(params, data=data)
