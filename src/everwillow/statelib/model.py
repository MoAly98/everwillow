"""Model abstractions built on :class:`~everwillow.statelib.state.State`."""

from __future__ import annotations

__all__ = ["CombinedModel", "Model"]

import dataclasses
import typing as tp

from jaxtyping import PyTree

from everwillow.statelib.state import FrozenChainMap, MergeMetadata, State, V, split


@dataclasses.dataclass(frozen=True)
class Model:
    """Callable wrapper around a log-density function.

    Examples:
        >>> scale = 0.5
        >>> model = Model(logpdf=lambda tree: -(tree["a"] ** 2 + tree["b"] ** 2) * scale)
        >>> params = State.from_pytree({"a": 1.0, "b": 2.0})
        >>> model(params)
        -2.5
    """

    logpdf: tp.Callable[[PyTree], float]  #: Evaluates the log-density for the pytree.

    def __call__(self, parameters: State) -> float:
        """Evaluate the wrapped log-density function.

        Args:
            parameters: a ``State`` produced by ``State.from_pytree``

        Returns:
            Log-density computed by ``logpdf``.
        """

        if not isinstance(parameters, State):
            msg = "parameters must be a State"  # type: ignore[unreachable]
            raise TypeError(msg)
        pytree = parameters.to_pytree()
        return self.logpdf(pytree)


@dataclasses.dataclass(frozen=True)
class CombinedModel:
    """Sum multiple model evaluations over a shared merged state."""

    models: tuple[Model, ...]  #: Ordered ``Model`` instances consumed per segment.

    @classmethod
    def combine(
        cls,
        *models: Model,
    ) -> CombinedModel:
        """Create a ``CombinedModel`` from individual model components.

        Args:
            *models: Ordered sequence of models to combine.

        Returns:
            CombinedModel configured to sum the provided models.
        """
        return cls(models=tuple(models))

    def __call__(
        self,
        parameters: FrozenChainMap[V],
        merge_metadata: MergeMetadata,
    ) -> float:
        """Evaluate all models on the provided merged state.

        Args:
            parameters: Merged ``State`` containing one internal state per
                model in the construction order.

        Returns:
            Sum of the log-density values returned by each model.
        """
        if not isinstance(merge_metadata, MergeMetadata):
            message = "merge_metadata must be a MergeMetadata instance"  # type: ignore[unreachable]
            raise TypeError(message)

        states = split(parameters, metadata=merge_metadata)
        if len(states) != len(self.models):
            msg = f"Expected {len(self.models)} states, received {len(states)}"
            raise ValueError(msg)

        total = 0.0
        for model, state in zip(self.models, states, strict=True):
            total += model(state)
        return total
