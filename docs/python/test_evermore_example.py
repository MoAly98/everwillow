"""Standalone evermore counting experiment example."""

from __future__ import annotations

import typing as tp
from functools import partial

import evermore as evm
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PyTree

import everwillow as ew

jax.config.update("jax_enable_x64", True)


# type defs
Hist1D: tp.TypeAlias = Float[Array, " nbins"]
Args: tp.TypeAlias = tuple[
    nnx.GraphDef,  # graphdef
    nnx.State,  # state
    PyTree[Hist1D],  # hists
]


class Model(nnx.Module):
    def __init__(
        self,
        mu: evm.Parameter,
        norm1: evm.NormalParameter,
        norm2: evm.NormalParameter,
        shape: evm.NormalParameter,
    ):
        self.mu = mu
        self.norm1 = norm1
        self.norm2 = norm2
        self.shape = shape

    def __call__(self, hists: PyTree[Hist1D]) -> PyTree[Hist1D]:
        expectations = {}

        # signal process
        sig_mod = self.mu.scale()
        expectations["signal"] = sig_mod(hists["nominal"]["signal"])

        # bkg1 process
        bkg1_lnN = self.norm1.scale_log_asymmetric(
            up=jnp.array([1.1]), down=jnp.array([0.9])
        )
        bkg1_shape = self.shape.morphing(
            up_template=hists["shape_up"]["bkg1"],
            down_template=hists["shape_down"]["bkg1"],
        )
        # combine modifiers
        bkg1_mod = bkg1_lnN @ bkg1_shape
        expectations["bkg1"] = bkg1_mod(hists["nominal"]["bkg1"])

        # bkg2 process
        bkg2_lnN = self.norm2.scale_log_asymmetric(
            up=jnp.array([1.05]), down=jnp.array([0.95])
        )
        bkg2_shape = self.shape.morphing(
            up_template=hists["shape_up"]["bkg2"],
            down_template=hists["shape_down"]["bkg2"],
        )
        # combine modifiers
        bkg2_mod = bkg2_lnN @ bkg2_shape
        expectations["bkg2"] = bkg2_mod(hists["nominal"]["bkg2"])

        # return the modified expectations
        return expectations


hists = {
    "nominal": {
        "signal": jnp.array([3.0]),
        "bkg1": jnp.array([10.0]),
        "bkg2": jnp.array([20.0]),
    },
    "shape_up": {
        "bkg1": jnp.array([12.0]),
        "bkg2": jnp.array([23.0]),
    },
    "shape_down": {
        "bkg1": jnp.array([8.0]),
        "bkg2": jnp.array([19.0]),
    },
}


model = Model(
    mu=evm.Parameter(name="mu"),
    norm1=evm.NormalParameter(name="norm1"),
    norm2=evm.NormalParameter(name="norm2"),
    shape=evm.NormalParameter(name="shape"),
)

observation = jnp.array([37.0])
expectations = model(hists)


@nnx.jit
def loss(dynamic: nnx.State, observation: Hist1D, args: Args) -> Float[Array, ""]:
    # unpack
    (graphdef, static, hists) = args
    # reconstruct model
    model = nnx.merge(graphdef, dynamic, static)
    # calculate expectation
    expectations = model(hists)
    # calculate constraints
    constraints = evm.loss.get_log_probs(model)
    loss_val = (
        evm.pdf.PoissonContinuous(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    # sum all up
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)


graphdef, dynamic, static = nnx.split(model, evm.filter.is_dynamic_parameter, ...)

# Perform the fit
result = ew.fit(
    partial(loss, args=(graphdef, static, hists)),
    dynamic,
    observation,
)

print(result.params)
# {
#   'mu': Array(2.33333346, dtype=float64),
#   'norm1': Array(3.0642646e-08, dtype=float64),
#   'norm2': Array(2.33507612e-08, dtype=float64),
#   'shape': Array(-1.66275153e-08, dtype=float64),
# }
