from __future__ import annotations

import dataclasses
import typing as tp

from .state import FlatState, PyTree, split_state

P = tp.ParamSpec("P")


@dataclasses.dataclass(frozen=True)
class Model:
    """Callable wrapper around a log-density function.

    Attributes:
        logpdf: Callable that evaluates the log-density for a pytree-shaped
            parameter container.

    Examples:
        >>> scale = 0.5
        >>> model = Model(logpdf=lambda tree: -(tree["a"] ** 2 + tree["b"] ** 2) * scale)
        >>> params = FlatState.from_pytree({"a": 1.0, "b": 2.0})
        >>> model(params)
        -2.5
    """

    logpdf: tp.Callable[[PyTree], float]

    def __call__(
        self,
        parameters: PyTree | FlatState,
    ) -> float:
        """Evaluate the wrapped log-density function.

        Args:
            parameters: Either a ``FlatState`` produced by
                ``FlatState.from_pytree`` or a raw pytree compatible with
                ``logpdf``.

        Returns:
            Log-density computed by ``logpdf``.
        """

        if isinstance(parameters, FlatState):
            pytree = parameters.to_pytree()
        else:
            pytree = parameters
        return self.logpdf(pytree)


@dataclasses.dataclass(frozen=True)
class CombinedModel:
    """Sum multiple model evaluations over a shared merged state.

    Attributes:
        models: Ordered collection of ``Model`` instances that each consume one
            segment of a merged ``FlatState``.
    """

    models: tuple[Model, ...]

    @classmethod
    def combine(
        cls,
        *models: Model,
    ) -> "CombinedModel":
        """Create a ``CombinedModel`` from individual model components.

        Args:
            *models: Ordered sequence of models to combine.

        Returns:
            CombinedModel configured to sum the provided models.
        """
        return cls(models=tuple(models))

    def __call__(
        self,
        parameters: FlatState,
    ) -> float:
        """Evaluate all models on the provided merged state.

        Args:
            parameters: Merged ``FlatState`` containing one internal state per
                model in the construction order.

        Returns:
            Sum of the log-density values returned by each model.
        """

        states = self._normalize_states(parameters)
        return sum(
            model(state) for model, state in zip(self.models, states, strict=True)
        )

    def _normalize_states(
        self,
        parameters: FlatState,
    ) -> tuple[FlatState, ...]:
        """Split and validate the merged state before evaluation.

        Args:
            parameters: Merged ``FlatState`` expected to contain one segment per
                model.

        Returns:
            Tuple of per-model ``FlatState`` instances.

        Raises:
            TypeError: If ``parameters`` is not a ``FlatState``.
            ValueError: If the number of segments does not match the number of models.
        """
        if not isinstance(parameters, FlatState):
            raise TypeError("parameters must be a FlatState")
        states = split_state(parameters)
        if len(states) != len(self.models):
            raise ValueError(
                f"Expected {len(self.models)} states, received {len(states)}"
            )
        return states
