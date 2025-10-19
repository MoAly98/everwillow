'''
Abstract base and implementations of parameter space transforms.
'''
import abc
import math

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
import jax

__all__ = [
    "AbstractParameterTransformation",
    "MinuitTransform",
    "SigmoidTransform",
    "OneSidedLogTransform",
    "SoftPlusTransform",
]

def _logit(x: ArrayLike) -> ArrayLike:
    """Compute logit function: log(x / (1 - x))"""
    return jnp.log(x) - jnp.log1p(-x)


def _sigmoid(x: ArrayLike) -> ArrayLike:
    """Compute sigmoid function: 1 / (1 + exp(-x)).

    Numerically stable implementation that avoids overflow.
    """
    return jnp.where(
        x >= 0,
        1.0 / (1.0 + jnp.exp(-x)),
        jnp.exp(x) / (1.0 + jnp.exp(x)),
    )

class AbstractParameterTransformation(eqx.Module):
    """
    Abstract base class for parameter transformations.

    This class defines the interface for parameter transformations, which are used to map parameters
    between different spaces (e.g., from constrained to unconstrained space). Subclasses must implement
    the `unwrap` and `wrap` methods to define the specific transformation logic.
    """

    @abc.abstractmethod
    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """
        Transform a parameter from its meaningful (e.g. bounded) space to the real unconstrained space.

        Args:
            parameter (AbstractParameter): The parameter to be transformed.

        Returns:
            AbstractParameter: The transformed parameter.
        """

    @abc.abstractmethod
    def wrap(self, value: ArrayLike) -> ArrayLike:
        """
        Transform a parameter from the real unconstrained space back to its meaningful (e.g. bounded) space. (Inverse of `unwrap`)

        Args:
            parameter (AbstractParameter): The parameter to be transformed.

        Returns:
            AbstractParameter: The parameter transformed back to its original space.
        """

class MinuitTransform(AbstractParameterTransformation):
    """
    Transform parameters based on Minuit's conventions. This transformation is used to map parameters with finite
    lower and upper boundaries to an unconstrained space. Both lower and upper boundaries
    are required and must be finite.

    Use `unwrap` to transform parameters into the unconstrained space and `wrap` to transform them back into the bounded space.

    Reference:
    https://root.cern.ch/download/minuit.pdf (Sec. 1.2.1 The transformation for parameters with limits.)

    Example:

        .. code-block:: python
        # TODO:: add example usage
    """

    lower: float = eqx.static_field()
    upper: float = eqx.static_field()

    def __name__(self) -> str:
        return "MinuitTransform"

    # check for finite boundaries
    def __post_init__(self):
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError(f"{self.__name__} requires finite lower/upper bounds.")
        if self.lower >= self.upper:
            raise ValueError(f"{self.__name__} requires lower bound to be strictly less than upper bound.")


    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Unwrap the external (bounded) value to the internal (unconstrained) value."""
        value = jnp.asarray(value)
        value = eqx.error_if(value, ~jnp.isfinite(value), "Value must be finite.")
        error_msg = (
            f"Value passed to {self.__name__} is exactly at or outside the boundaries "
            f"[{self.lower}, {self.upper}]."
        )
        value = eqx.error_if(value, value <= self.lower, error_msg)
        value = eqx.error_if(value, value >= self.upper, error_msg)
        # this formula turns user-provided "external" parameter values into "internal" values
        return jnp.arcsin(
            2.0 * (value - self.lower) / (self.upper - self.lower) - 1.0
        )


    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Wrap the internal (unconstrained) value to the external (bounded) value."""
        return self.lower + (self.upper - self.lower) / 2 * (
            jnp.sin(value) + 1
        )

class SigmoidTransform(AbstractParameterTransformation):
    """
    Transform parameters based on the sigmoid function. This transformation is used to map parameters with finite
    lower and upper boundaries to an unconstrained space. Both lower and upper boundaries
    are required and must be finite.

    Use `unwrap` to transform parameters into the unconstrained space and `wrap` to transform them back into the bounded space.

    Example:

        .. code-block:: python
        # TODO:: add example usage
    """

    lower: float = eqx.static_field()
    upper: float = eqx.static_field()

    def __name__(self) -> str:
        return "SigmoidTransform"

    # check for finite boundaries
    def __post_init__(self):
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError(f"{self.__name__} requires finite lower/upper bounds.")
        if self.lower >= self.upper:
            raise ValueError(f"{self.__name__} requires lower bound to be strictly less than upper bound.")


    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Unwrap the external (bounded) value to the internal (unconstrained) value."""
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
        """Wrap the internal (unconstrained) value to the external (bounded) value."""
        return self.lower + (self.upper - self.lower) * _sigmoid(value)

class OneSidedLogTransform(AbstractParameterTransformation):
    """
    Log transform for parameters with exactly one finite bound.

    Use `unwrap` to transform parameters into the unconstrained space and `wrap` to transform them back into the bounded space.

    Example:

        .. code-block:: python
    """

    bound: float = eqx.static_field()
    direction: str = eqx.static_field()  # 'lower' or 'upper'



    def __name__(self) -> str:
        return "OneSidedLogTransform"

    # check for finite lower boundary
    def __post_init__(self):
        if self.direction not in ("lower", "upper"):
            raise ValueError(f"Unsupported direction {self.direction!r} for {self.__name__}.")
        if not math.isfinite(self.bound):
            raise ValueError(f"{self.__name__} requires a finite bound.")


    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """Unwrap the external (bounded) value to the internal (unconstrained) value."""
        value = jnp.asarray(value)
        value = eqx.error_if(value, ~jnp.isfinite(value), "Value must be finite.")
        if self.direction == "lower":
            error_msg = (
                f"Value passed to {self.__name__} must be greater than lower bound "
                f"{self.bound}."
            )
            value = eqx.error_if(value, value <= self.bound, error_msg)
            return jnp.log(value - self.bound)

        error_msg = (
            f"Value passed to {self.__name__} must be less than upper bound {self.bound}."
        )
        value = eqx.error_if(value, value >= self.bound, error_msg)
        return jnp.log(self.bound - value)

    def wrap(self, value: ArrayLike) -> ArrayLike:
        """Wrap the internal (unconstrained) value to the external (bounded) value."""
        value = jnp.asarray(value)
        if self.direction == "lower":
            return self.bound + jnp.exp(value)
        return self.bound - jnp.exp(value)


class SoftPlusTransform(AbstractParameterTransformation):
    """
    Applies the softplus transformation to parameters, projecting them from real space (R) to positive space (R+).
    This transformation is useful for enforcing the positivity of parameters and does not require lower or upper boundaries.

    Use `unwrap` to transform parameters into the unconstrained real space and `wrap` to transform them back into the positive real space.

    Reference:
    https://github.com/danielward27/paramax/blob/main/paramax/utils.py

    Example:

    .. code-block:: python
    TODO:: add example
    """

    def __name__(self) -> str:
        return "SoftPlusTransform"

    def unwrap(self, value: ArrayLike) -> ArrayLike:
        """The inverse of the softplus function, checking for positive inputs."""
        value = jnp.asarray(value)
        value = eqx.error_if(value, ~jnp.isfinite(value), "Value must be finite.")
        value = eqx.error_if(value, value < 0, "Expected positive inputs to inv_softplus.")
        value_t = jnp.log(-jnp.expm1(-value)) + value
        return value_t

    def wrap(self, value: ArrayLike) -> ArrayLike:
        return jax.nn.softplus(value)
