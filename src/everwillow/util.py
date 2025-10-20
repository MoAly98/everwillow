from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike


__all__ = ["float_array"]


def float_array(x: ArrayLike) -> ArrayLike:
    """Convert input to a float array."""
    return jnp.asarray(x, dtype=jnp.result_type(float))