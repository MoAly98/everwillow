"""
Abstract base and implementations of parameter space transforms.
"""

import abc
import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "AbstractParameterTransformation",
    "MinuitTransform",
    "OneSidedLogTransform",
    "SigmoidTransform",
    "SoftPlusTransform",
]


def _logit(x: ArrayLike) -> ArrayLike:
    """Compute ``log(x / (1 - x))`` in a numerically stable way."""
    return jnp.log(x) - jnp.log1p(-x)


def _sigmoid(x: ArrayLike) -> ArrayLike:
    """Compute ``1 / (1 + exp(-x))`` without overflow."""
    return jnp.where(
        x >= 0,
        1.0 / (1.0 + jnp.exp(-x)),
        jnp.exp(x) / (1.0 + jnp.exp(x)),
    )


class AbstractParameterTransformation(eqx.Module):
    """
    Abstract base for parameter transformations.

    Subclasses implement ``unwrap`` (bounded → unconstrained) and ``wrap``
    (unconstrained → bounded) using JAX-compatible array ops.
    """

    @abc.abstractmethod
    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Transform a value from its constrained space to the real line."""

    @abc.abstractmethod
    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Transform a value from the real line back to its constrained space."""


class MinuitTransform(AbstractParameterTransformation):
    """
    Minuit-style transform for parameters with finite lower and upper bounds.

    ``unwrap`` converts a bounded value into an unconstrained internal representation,
    while ``wrap`` inverts the mapping. Both bounds must be finite and ``lower < upper``.
    Example:

        >>> transform = MinuitTransform(lower=0.0, upper=1.0)
        >>> jnp.isclose(transform.wrap(transform.unwrap(0.3)), 0.3)
        Array(True, dtype=bool)

    Reference: https://root.cern.ch/download/minuit.pdf (Sec. 1.2.1).
    """

    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)

    def __name__(self) -> str:
        return "MinuitTransform"

    # check for finite boundaries
    def __post_init__(self):
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            message = f"{self.__name__} requires finite lower/upper bounds."
            raise ValueError(message)
        if self.lower >= self.upper:
            message = (
                f"{self.__name__} requires lower bound to be strictly less than "
                "upper bound."
            )
            raise ValueError(message)

    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Convert a bounded value into an unconstrained representation."""
        value = jnp.asarray(value)
        value = eqx.error_if(value, ~jnp.isfinite(value), "Value must be finite.")
        error_msg = (
            f"Value passed to {self.__name__} is exactly at or outside the boundaries "
            f"[{self.lower}, {self.upper}]."
        )
        value = eqx.error_if(value, value <= self.lower, error_msg)
        value = eqx.error_if(value, value >= self.upper, error_msg)
        # this formula turns user-provided "external" parameter values into "internal" values
        return jnp.arcsin(2.0 * (value - self.lower) / (self.upper - self.lower) - 1.0)

    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Convert an unconstrained value back into the bounded interval."""
        return self.lower + (self.upper - self.lower) / 2 * (jnp.sin(value) + 1)


class SigmoidTransform(AbstractParameterTransformation):
    """
    Logit/sigmoid pair for parameters with finite lower and upper bounds.

    ``unwrap`` applies the logit of the affine-scaled value, ``wrap`` applies the sigmoid.
    Bounds must be finite with ``lower < upper``.

    Example:

        >>> transform = SigmoidTransform(lower=-2.0, upper=3.0)
        >>> value = -1.1
        >>> jnp.isclose(transform.wrap(transform.unwrap(value)), value)
        Array(True, dtype=bool)
    """

    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)

    def __name__(self) -> str:
        return "SigmoidTransform"

    # check for finite boundaries
    def __post_init__(self):
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            message = f"{self.__name__} requires finite lower/upper bounds."
            raise ValueError(message)
        if self.lower >= self.upper:
            message = (
                f"{self.__name__} requires lower bound to be strictly less than "
                "upper bound."
            )
            raise ValueError(message)

    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Convert a bounded value into an unconstrained representation."""
        value = jnp.asarray(value)
        value = eqx.error_if(value, ~jnp.isfinite(value), "Value must be finite.")
        error_msg = (
            f"Value passed to {self.__name__} is exactly at or outside the boundaries "
            f"[{self.lower}, {self.upper}]."
        )
        value = eqx.error_if(value, value <= self.lower, error_msg)
        value = eqx.error_if(value, value >= self.upper, error_msg)
        # this formula turns user-provided "external" parameter values into "internal" values
        return _logit((value - self.lower) / (self.upper - self.lower))

    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Convert an unconstrained value back into the bounded interval."""
        return self.lower + (self.upper - self.lower) * _sigmoid(value)


class OneSidedLogTransform(AbstractParameterTransformation):
    """
    Log transform for parameters with exactly one finite bound.

    Direction ``"lower"`` enforces ``value > bound`` (via ``log(value - bound)``),
    while direction ``"upper"`` enforces ``value < bound`` (via ``log(bound - value)``).

    Example:

        >>> transform = OneSidedLogTransform(bound=0.0, direction="lower")
        >>> jnp.isclose(transform.wrap(transform.unwrap(2.0)), 2.0)
        Array(True, dtype=bool)
    """

    bound: float = eqx.field(static=True)
    direction: str = eqx.field(static=True)  # 'lower' or 'upper'

    def __name__(self) -> str:
        return "OneSidedLogTransform"

    # check for finite lower boundary
    def __post_init__(self):
        if self.direction not in ("lower", "upper"):
            message = f"Unsupported direction {self.direction!r} for {self.__name__}."
            raise ValueError(message)
        if not math.isfinite(self.bound):
            message = f"{self.__name__} requires a finite bound."
            raise ValueError(message)

    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Convert a single-sided bounded value into an unconstrained representation."""
        value = jnp.asarray(value)
        value = eqx.error_if(value, ~jnp.isfinite(value), "Value must be finite.")
        if self.direction == "lower":
            error_msg = (
                f"Value passed to {self.__name__} must be greater than lower bound "
                f"{self.bound}."
            )
            value = eqx.error_if(value, value <= self.bound, error_msg)
            return jnp.log(value - self.bound)

        error_msg = f"Value passed to {self.__name__} must be less than upper bound {self.bound}."
        value = eqx.error_if(value, value >= self.bound, error_msg)
        return jnp.log(self.bound - value)

    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Convert an unconstrained value back into the one-sided bounded space."""
        value = jnp.asarray(value)
        if self.direction == "lower":
            return self.bound + jnp.exp(value)
        return self.bound - jnp.exp(value)


class SoftPlusTransform(AbstractParameterTransformation):
    """
    Applies the softplus transformation to parameters, projecting them from real space (R) to positive space (R+).
    This transformation is useful for enforcing the positivity of parameters and does not require lower or upper boundaries.

    ``unwrap`` computes the inverse softplus (with validation), while ``wrap`` applies
    ``jax.nn.softplus``.

    Example:

        >>> transform = SoftPlusTransform()
        >>> jnp.isclose(transform.wrap(transform.unwrap(0.8)), 0.8)
        Array(True, dtype=bool)
    """

    def __name__(self) -> str:
        return "SoftPlusTransform"

    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Apply the inverse softplus, validating positivity and finiteness."""
        value = jnp.asarray(value)
        value = eqx.error_if(value, ~jnp.isfinite(value), "Value must be finite.")
        value = eqx.error_if(
            value, value < 0, "Expected positive inputs to inv_softplus."
        )
        return jnp.log(-jnp.expm1(-value)) + value

    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Apply the softplus function."""
        return jax.nn.softplus(value)
