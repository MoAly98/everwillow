"""
Abstract base and implementations of parameter space transforms.
"""

import abc
import dataclasses
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import checkify
from jaxtyping import ArrayLike

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


def _safe_assert(fn: tp.Callable, *args, **kwargs) -> None:
    error, _ = fn(*args, **kwargs)
    checkify.check_error(error)
    return


@checkify.checkify
def _check_in_bounds(value: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> None:
    """Check if a value is bounded between lower and upper.

    Args:
        value: The value to check.
        lower: The lower bound.
        upper: The upper bound.

    Raises:
        ValueError: If any element of value is outside the bounds.
    """
    checkify.check(
        jnp.all((value > lower) & (value < upper)),
        "value needs to be bounded between {lower} and {upper}, got {value}",
        value=jnp.asarray(value),
        lower=jnp.asarray(lower),
        upper=jnp.asarray(upper),
    )


@checkify.checkify
def _check_left_gt_right(left: ArrayLike, right: ArrayLike) -> None:
    """Check if the left value is greater than the right value.

    Args:
        left: The left value.
        right: The right value.

    Raises:
        ValueError: If left is not greater than right.
    """
    checkify.check(
        left > right,
        "left value needs to be greater than right value, got right={right}, left={left}",
        right=jnp.asarray(right),
        left=jnp.asarray(left),
    )


@checkify.checkify
def _check_is_finite(value: ArrayLike) -> None:
    """Check if a value is finite.

    Args:
        value: The value to check.

    Raises:
        ValueError: If any element of value is not finite.
    """
    checkify.check(
        jnp.isfinite(value),
        "value needs to be finite, got {value}",
        value=jnp.asarray(value),
    )


@checkify.checkify
def _check_is_non_negative(value: ArrayLike) -> None:
    checkify.check(
        jnp.all(value >= 0),
        "value needs to be non-negative, got {value}",
        value=jnp.asarray(value),
    )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=[],
)
@dataclasses.dataclass
class AbstractParameterTransformation(abc.ABC):
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


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["lower", "upper"],
    meta_fields=[],
)
@dataclasses.dataclass
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

    lower: ArrayLike
    upper: ArrayLike

    def __post_init__(self):
        # checks
        _safe_assert(_check_is_finite, self.lower)
        _safe_assert(_check_is_finite, self.upper)
        _safe_assert(_check_left_gt_right, self.upper, self.lower)

    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Convert a bounded value into an unconstrained representation."""
        value = jnp.asarray(value)
        _safe_assert(_check_is_finite, value)
        _safe_assert(_check_in_bounds, value, lower=self.lower, upper=self.upper)
        # this formula turns user-provided "external" parameter values into "internal" values
        return jnp.arcsin(2.0 * (value - self.lower) / (self.upper - self.lower) - 1.0)

    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Convert an unconstrained value back into the bounded interval."""
        return self.lower + (self.upper - self.lower) / 2 * (jnp.sin(value) + 1)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["lower", "upper"],
    meta_fields=[],
)
@dataclasses.dataclass
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

    lower: ArrayLike
    upper: ArrayLike

    def __post_init__(self):
        # checks
        _safe_assert(_check_is_finite, self.lower)
        _safe_assert(_check_is_finite, self.upper)
        _safe_assert(_check_left_gt_right, self.upper, self.lower)

    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Convert a bounded value into an unconstrained representation."""
        value = jnp.asarray(value)
        _safe_assert(_check_is_finite, value)
        _safe_assert(_check_in_bounds, value, lower=self.lower, upper=self.upper)
        # this formula turns user-provided "external" parameter values into "internal" values
        return _logit((value - self.lower) / (self.upper - self.lower))

    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Convert an unconstrained value back into the bounded interval."""
        return self.lower + (self.upper - self.lower) * _sigmoid(value)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["bound"],
    meta_fields=["direction"],
)
@dataclasses.dataclass
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

    bound: ArrayLike
    direction: str  # 'lower' or 'upper'

    def __post_init__(self):
        # checks
        if self.direction not in ("lower", "upper"):
            message = (
                f"Unsupported direction {self.direction!r} for {type(self).__name__}."
            )
            raise ValueError(message)
        _safe_assert(_check_is_finite, self.bound)

    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Convert a single-sided bounded value into an unconstrained representation."""
        value = jnp.asarray(value)
        _safe_assert(_check_is_finite, value)
        if self.direction == "lower":
            # checks
            _safe_assert(_check_left_gt_right, value, self.bound)
            return jnp.log(value - self.bound)
        # direction == "upper"
        _safe_assert(_check_left_gt_right, self.bound, value)
        return jnp.log(self.bound - value)

    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Convert an unconstrained value back into the one-sided bounded space."""
        value = jnp.asarray(value)
        if self.direction == "lower":
            return self.bound + jnp.exp(value)
        return self.bound - jnp.exp(value)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=[],
)
@dataclasses.dataclass
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

    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Apply the inverse softplus, validating positivity and finiteness."""
        value = jnp.asarray(value)
        _safe_assert(_check_is_finite, value)
        _safe_assert(_check_is_non_negative, value)
        return jnp.log(-jnp.expm1(-value)) + value

    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Apply the softplus function."""
        return jax.nn.softplus(value)
