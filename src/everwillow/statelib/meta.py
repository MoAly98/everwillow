from __future__ import annotations

import dataclasses
import typing as tp
from functools import partial

import jax.tree_util as jtu
from jaxtyping import PyTreeDef

if tp.TYPE_CHECKING:
    from everwillow.statelib.state import FrozenChainMap, K, State, V


__all__ = ["MergeMeta", "TreeDefMeta"]


@partial(
    jtu.register_dataclass,
    data_fields=[],
    meta_fields=["treedef", "keys"],
)
@dataclasses.dataclass(frozen=True, slots=True)
class TreeDefMeta:
    """Metadata retaining the treedef of a pytree."""

    treedef: PyTreeDef | None
    keys: tp.Sequence[K]

    def to_pytree(
        self,
        mapping: tp.Mapping[K, V],
    ) -> State[V]:
        """Reconstruct a ``State`` from flattened values.

        Args:
            values: Flattened values corresponding to the original pytree.

        Returns:
            Reconstructed ``State`` object.
        """
        # can't reconstruct pytree with treedef=None
        if self.treedef is None:
            msg = "Cannot reconstruct pytree with 'treedef=None'"
            raise ValueError(msg)

        # can't convert to pytree if keys don't match
        if set(self.keys) != set(mapping.keys()):
            missing = set(self.keys) - set(mapping.keys())
            extra = set(mapping.keys()) - set(self.keys)
            msg = f"Mapping keys do not match treedef keys. Missing: {missing}, Extra: {extra}"
            raise KeyError(msg)

        # this order of keys is important here to preserve original pytree order
        return jtu.tree_unflatten(
            treedef=self.treedef, leaves=(mapping[k] for k in self.keys)
        )


@partial(
    jtu.register_dataclass,
    data_fields=[],
    meta_fields=["treedefmetas"],
)
@dataclasses.dataclass(frozen=True, slots=True)
class MergeMeta:
    """Metadata retained when multiple :class:`State` objects are merged.

    The metadata stores the treedef for each original state alongside the group
    of keys that belong to that state so the merged mapping can be split later.
    """

    treedefmetas: tuple[TreeDefMeta, ...]

    def split(
        self,
        chain_map: FrozenChainMap[V],
    ) -> tuple[State[V], ...]:
        """Partition a merged FrozenChainMap back into individual states.

        Args:
            chain_map: FrozenChainMap produced by :func:`merge`.

        Returns:
            Tuple of :class:`State` objects restoring the original order.

        Examples:
            >>> state_a = State.from_pytree({"a": 1})
            >>> merged, mergemeta = merge(state_a)
            >>> mergemeta.split(merged)[0].to_pytree()
            {'a': 1}
        """
        from everwillow.statelib.state import State

        return tuple(
            State(mapping, treedefmeta=treedefmeta)
            for mapping, treedefmeta in zip(
                chain_map.maps, self.treedefmetas, strict=True
            )
        )
