"""
Compatibility layer exposing the everwillow statelib API under the historic
``statelib`` import path required by the tests.
"""

from __future__ import annotations

from everwillow.statelib import (CombinedModel, FlatState, Model, Transform,
                                 apply_transformations, map_state,
                                 merge_states, split_state, update_state)

__all__ = [
    "CombinedModel",
    "FlatState",
    "Model",
    "Transform",
    "apply_transformations",
    "map_state",
    "merge_states",
    "split_state",
    "update_state",
]
