from __future__ import annotations

import dataclasses
import typing as tp
from functools import partial

import jax.tree_util as jtu
from jaxtyping import PyTree, PyTreeDef

if tp.TYPE_CHECKING:
    from everwillow._src.statelib.state import K, V


__all__ = ["TreeDefMeta"]


@partial(
    jtu.register_dataclass,
    data_fields=[],
    meta_fields=["treedef", "keys", "merged"],
)
@dataclasses.dataclass(frozen=True, slots=True)
class TreeDefMeta:
    """Metadata retaining the treedef of a pytree."""

    treedef: PyTreeDef
    keys: tp.Sequence[K]
    merged: bool = False

    def to_pytree(
        self,
        mapping: tp.Mapping[K, V],
    ) -> PyTree[V]:
        """Reconstruct a ``PyTree`` from a (State) mapping.

        Args:
            mapping: A mapping that holds the values matching the pytree definition.

        Returns:
            Reconstructed ``PyTree`` object.
        """
        # can't convert to pytree if keys don't match
        self_keys = set(self.keys)
        mapping_keys = set(mapping.keys())
        if self_keys != mapping_keys:
            missing = self_keys - mapping_keys
            extra = mapping_keys - self_keys
            msg = f"Mapping keys do not match treedef keys. Missing: {missing}, Extra: {extra}"
            raise KeyError(msg)

        del self_keys, mapping_keys  # unused after check

        # this order of keys is important here to preserve original pytree order
        return jtu.tree_unflatten(
            treedef=self.treedef, leaves=(mapping[k] for k in self.keys)
        )
