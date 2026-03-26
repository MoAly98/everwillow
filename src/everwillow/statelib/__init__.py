"""Immutable state management for JAX pytrees."""

from __future__ import annotations

from everwillow._src.statelib import State as State
from everwillow._src.statelib import Transform as Transform
from everwillow._src.statelib import apply_transformations as apply_transformations
from everwillow._src.statelib import canonicalize_key as canonicalize_key
from everwillow._src.statelib import combine_partitions as combine_partitions
from everwillow._src.statelib import merge as merge
from everwillow._src.statelib import partition as partition
from everwillow._src.statelib import split as split
from everwillow._src.statelib import update as update

__all__ = [
    "State",
    "Transform",
    "apply_transformations",
    "canonicalize_key",
    "combine_partitions",
    "merge",
    "partition",
    "split",
    "update",
]
