"""Immutable mapping helpers for working with JAX pytrees.

The utilities in this module provide read-only dictionaries that keep track of
the canonical keys JAX emits when flattening nested structures.
"""

from __future__ import annotations

import typing as tp
from types import EllipsisType, MappingProxyType

import jax.tree_util as jtu
import treescope
import treescope.repr_lib
from jaxtyping import ArrayLike, PyTree

try:
    from IPython import get_ipython

    _in_ipython = get_ipython() is not None
except ImportError:
    _in_ipython = False

from everwillow._src.statelib.meta import TreeDefMeta

__all__ = [
    "K",
    "State",
    "V",
    "canonicalize_key",
    "combine_partitions",
    "merge",
    "partition",
    "split",
    "update",
]


K: tp.TypeAlias = str | tuple[str | int, ...]
V = tp.TypeVar("V", bound=ArrayLike | None)


def _flatten_iterables(x: tp.Any) -> tp.Iterator[tp.Any]:
    """Flatten any iterable except strings/bytes."""
    if isinstance(x, tp.Iterable) and not isinstance(x, (str, bytes)):
        for y in x:
            yield from _flatten_iterables(y)
    else:
        yield x


@tp.overload
def canonicalize_key(path: tuple[tp.Any, ...]) -> str: ...


@tp.overload
def canonicalize_key(path: tuple[tp.Any, ...], *, sep: str) -> str: ...


@tp.overload
def canonicalize_key(
    path: tuple[tp.Any, ...], *, sep: None
) -> tuple[str | int, ...]: ...


def canonicalize_key(path: tuple[tp.Any, ...], *, sep: str | None = ".") -> K:
    """Convert a JAX key path to plain Python entries.

    Args:
        path: Key path emitted by :func:`jax.tree_util.tree_flatten_with_path`.
        sep: Separator used to join the entries into a string. Defaults to
            ``"."``. When ``None`` the key is returned as a tuple.

    Returns:
        Canonical key representation that can be used to index a :class:`State`.

    Raises:
        TypeError: If ``path`` contains an unsupported key type.
        ValueError: If a key segment contains the separator string.

    Examples:
        >>> import jax.tree_util as jtu
        >>> canonicalize_key((jtu.DictKey("a"), jtu.SequenceKey(0)))
        'a.0'

        Use ``sep=None`` for tuple keys:

        >>> canonicalize_key((jtu.DictKey("a"), jtu.SequenceKey(0)), sep=None)
        ('a', 0)
    """
    result: list[tp.Any] = []
    for entry in path:
        if isinstance(entry, jtu.DictKey):
            result.extend(_flatten_iterables(entry.key))
        elif isinstance(entry, jtu.GetAttrKey):
            result.append(entry.name)
        elif isinstance(entry, jtu.SequenceKey):
            result.append(entry.idx)
        elif isinstance(entry, jtu.FlattenedIndexKey):
            result.extend(_flatten_iterables(entry.key))
        else:
            msg = f"Unrecognised key path entry: {entry}"
            raise TypeError(msg)
    if sep is not None:
        for entry in result:
            s = str(entry)
            if sep in s:
                msg = (
                    f"Key segment {s!r} contains the separator {sep!r}. "
                    f"Use sep=None for tuple keys, or choose a different separator."
                )
                raise ValueError(msg)
        return sep.join(map(str, result))

    return tuple(result)


class BaseMapping(tp.Mapping[K, V], tp.Generic[V]):
    """Read-only mapping facade used by the state containers.

    This class wraps an immutable mapping and exposes the standard mapping
    protocol while providing helpers that are convenient for tests.

    Examples:
        Create a :class:`State` and access its mapping interface:

        >>> state = State.from_pytree({"a": 1.0})
        >>> isinstance(state, BaseMapping)
        True
        >>> state["a"]
        1.0
    """

    __slots__ = ("_mapping",)

    _mapping: tp.Mapping[K, V]

    def __getitem__(self, key: K) -> V:
        return self._mapping[key]

    def __iter__(self) -> tp.Iterator[K]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def to_dict(self) -> dict[K, V]:
        return dict(self._mapping)

    @property
    def mapping(self) -> tp.Mapping[K, V]:
        """Read-only view of the underlying mapping.

        Returns:
            :class:`types.MappingProxyType` exposing the raw key/value pairs.

        Examples:
            >>> state = State.from_pytree({"a": 1.0})
            >>> "a" in state.mapping
            True
        """

        return self._mapping


@jtu.register_pytree_with_keys_class
class State(BaseMapping[V]):
    """Container that stores flattened pytrees keyed by canonical keys.

    The state keeps track of the pytree definition so it can be converted back
    to the original nested structure.

    Examples:
        >>> state = State.from_pytree({"a": {"b": 2.0}})
        >>> state["a.b"]
        2.0
        >>> state.to_pytree()
        {'a': {'b': 2.0}}
    """

    __slots__ = ("_treedefmeta",)

    def __init__(
        self,
        mapping: tp.Mapping[K, V],
        *,
        treedefmeta: TreeDefMeta,
    ) -> None:
        """Initialise a state from an existing mapping.

        Args:
            mapping: Mapping whose keys are canonical tuples and values are
                pytree leaves.
            treedefmeta: TreeDefMeta instance containing the pytree definition
                and (ordered) keys for reconstruction.
        """
        # Ensure the mapping is immutable
        self._mapping = MappingProxyType(mapping)

        if not isinstance(treedefmeta, TreeDefMeta):
            msg = "'treedefmeta' must be a TreeDefMeta instance. Use State.from_pytree() instead."  # type: ignore[unreachable]
            raise TypeError(msg)
        self._treedefmeta = treedefmeta

    @classmethod
    def from_pytree(
        cls,
        pytree: PyTree[V],
        *,
        is_leaf: tp.Callable[[V], bool] | None = None,
        sep: str | None = ".",
    ) -> State[V]:
        """Build a :class:`State` instance from an arbitrary pytree.

        Args:
            pytree: Nested structure supported by :mod:`jax.tree_util`.
            is_leaf: Optional callable passed to :func:`jax.tree_util.tree_flatten_with_path`
                to customize which nodes are treated as leaves.
            sep: Separator used to join key entries when constructing
                public keys. Defaults to ``"."``. When ``None``, keys are
                returned as tuples.

        Returns:
            New :class:`State` representing ``pytree``.

        Examples:
            >>> State.from_pytree({"a": [1, 2]}).mapping
            mappingproxy({'a.0': 1, 'a.1': 2})
        """

        if isinstance(pytree, State):
            msg = f"{pytree=} is already a State instance"
            raise TypeError(msg)

        # flatten the pytree with paths to build canonical keys
        path_leaves, treedef = jtu.tree_flatten_with_path(pytree, is_leaf=is_leaf)
        data, keys = {}, []
        for path, leaf in path_leaves:
            key = canonicalize_key(path, sep=sep)
            data[key] = leaf
            keys.append(key)

        treedefmeta = TreeDefMeta(treedef=treedef, keys=tuple(keys), merged=False)
        return cls(data, treedefmeta=treedefmeta)

    def to_pytree(self) -> PyTree[V]:
        """Reconstruct the stored pytree using the cached tree definition.

        Returns:
            Pytree with the same structure used to create the state.

        Examples:
            >>> state = State.from_pytree({"x": 1})
            >>> state.to_pytree()
            {'x': 1}
        """

        return self.treedefmeta.to_pytree(self.mapping)

    @property
    def treedefmeta(self) -> TreeDefMeta:
        return self._treedefmeta

    @property
    def is_merged(self) -> bool:
        """Whether this state was produced by :func:`merge`.

        Returns:
            ``True`` if the internal treedef is a compound tuple of its children.
        """
        tdm = self._treedefmeta
        return tdm.merged

    @property
    def notnone(self) -> dict[K, V]:
        """Return a filtered view excluding keys with None values.

        This is useful after :func:`partition` to see only the active entries.

        Returns:
            Dictionary containing only non-None entries.

        Examples:
            >>> state = State.from_pytree({"a": 1, "b": 2})
            >>> left, _ = partition(state, predicate=lambda k, _: k == "a")
            >>> left.notnone
            {'a': 1}
        """
        return {k: v for k, v in self._mapping.items() if v is not None}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()!r})"

    def __treescope_repr__(self, path, subtree_renderer):
        return treescope.repr_lib.render_dictionary_wrapper(
            object_type=type(self),
            wrapped_dict=self.to_dict(),
            path=path,
            subtree_renderer=subtree_renderer,
        )

    def show(self) -> None:
        """Pretty-print this State with rich array visualization."""
        if _in_ipython:
            treescope.display(self, ignore_exceptions=True)
        else:
            print(treescope.render_to_text(self))

    # jax.tree_util.register_pytree_with_keys_class methods
    def tree_flatten_with_keys(self):
        # .to_dict() because jax.tree_util already knows how to flatten dicts
        children_with_keys = ((jtu.GetAttrKey("_mapping"), self.to_dict()),)
        aux_data = (self._treedefmeta,)
        return children_with_keys, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[TreeDefMeta, ...],
        children: tuple[tp.Mapping[K, V], ...],
    ) -> State[V]:
        (treedefmeta,) = aux_data
        (mapping,) = children
        return cls(mapping, treedefmeta=treedefmeta)


def merge(*states: State[V]) -> State[V]:
    """Combine several States into one.

    When states share overlapping keys, the last value wins.

    Args:
        *states: Sequence of :class:`State` instances to merge (at least one).

    Returns:
        New :class:`State` containing all key/value pairs from the inputs.

    Raises:
        ValueError: If no states are provided.
    """
    if len(states) < 1:
        msg = "merge requires at least one state"
        raise ValueError(msg)

    all_keys: list[K] = []
    all_vals: list[V] = []
    child_treedefs: list[jtu.PyTreeDef] = []
    for s in states:
        all_keys.extend(s.treedefmeta.keys)
        all_vals.extend(s[k] for k in s.treedefmeta.keys)
        child_treedefs.append(s.treedefmeta.treedef)

    compound_treedef = jtu.treedef_tuple(child_treedefs)
    mapping = dict(zip(all_keys, all_vals, strict=False))
    return State(
        mapping, treedefmeta=TreeDefMeta(compound_treedef, tuple(all_keys), merged=True)
    )


def split(state: State[V]) -> tuple[State[V], ...]:
    """Split a merged State back into original States.

    For overlapping keys, all returned segments receive the merged value.

    Args:
        state: :class:`State` instance created by :func:`merge`.

    Returns:
        Tuple of :class:`State` instances corresponding to the original inputs
        used to create ``state``.

    Raises:
        ValueError: If ``state`` was not produced by :func:`merge`.
    """

    if not state.is_merged:
        msg = "split requires a state produced by merge"
        raise ValueError(msg)

    child_treedefs = jtu.treedef_children(state.treedefmeta.treedef)
    offset, states = 0, []
    for child_td in child_treedefs:
        n = child_td.num_leaves
        child_keys = state.treedefmeta.keys[offset : offset + n]
        child_map = {k: state[k] for k in child_keys}
        states.append(State(child_map, treedefmeta=TreeDefMeta(child_td, child_keys)))
        offset += n
    return tuple(states)


def partition(
    state: State[V],
    *,
    predicate: tp.Callable[[K, V], bool],
) -> tuple[State[V], State[V]]:
    """Split a state into two partitions based on a predicate.

    Args:
        state: :class:`State` instance to partition.
        predicate: Callable returning ``True`` for items that should go into
            the first partition.

    Returns:
        Tuple ``(left, right)`` containing two :class:`State` partitioned from
        the original state. Elements not satisfying the predicate are set to ``None``
        in ``left`` and vice versa for ``right``.

    Examples:
        >>> state = State.from_pytree({"a": 1, "b": 2})
        >>> left, right = partition(state, predicate=lambda key, _: key == "a")
        >>> right.notnone
        {'b': 2}
    """

    left_data: dict[K, V] = {}
    right_data: dict[K, V] = {}
    for key, value in state.items():
        if predicate(key, value):
            left_data[key] = value
            right_data[key] = tp.cast(V, None)
        else:
            left_data[key] = tp.cast(V, None)
            right_data[key] = value
    return (
        State(left_data, treedefmeta=state.treedefmeta),
        State(right_data, treedefmeta=state.treedefmeta),
    )


def combine_partitions(
    left: State[V],
    right: State[V],
) -> State[V]:
    """Merge two partitions that originated from the same state.

    Args:
        left: First partition returned by :func:`partition`.
        right: Second partition returned by :func:`partition`.

    Returns:
        :class:`State` containing the union of both partitions.

    Raises:
        ValueError: If the partitions do not share the same treedefmeta.

    Examples:
        >>> state = State.from_pytree({"a": 1, "b": 2})
        >>> left, right = partition(state, predicate=lambda key, _: key == "a")
        >>> combine_partitions(left, right)["b"]
        2
    """
    if left.treedefmeta != right.treedefmeta:
        msg = "partitions must originate from the same original state"
        raise ValueError(msg)
    return jtu.tree_map(
        lambda x1, x2: x1 if x1 is not None else x2,
        left,
        right,
        is_leaf=lambda x: x is None,
    )


def update(
    state: State[V],
    *,
    updates: tp.Mapping[K, V | EllipsisType],
) -> State[V]:
    """Return a new state with specific entries replaced.

    Args:
        state: Original :class:`State` to copy.
        updates: Mapping of existing keys to replacement values. Entries whose
            value is ``Ellipsis`` are ignored, which makes it easy to reuse the
            same dictionary across multiple updates.

    Returns:
        New :class:`State` with the replacements applied.

    Raises:
        KeyError: If ``updates`` includes a key that is not present in ``state``.

    Examples:
        >>> base = State.from_pytree({"a": 1, "b": 2})
        >>> update(base, updates={"b": 99}).to_dict()
        {'a': 1, 'b': 99}
    """
    if not isinstance(state, State):
        msg = "Can only update State types"  # type: ignore[unreachable]
        raise TypeError(msg)

    if missing := set(updates.keys()) - set(state.keys()):
        msg = f"cannot update missing keys: {missing}"
        raise KeyError(msg)

    data = dict(state.mapping)
    for key, value in updates.items():
        if value is ...:
            continue
        data[key] = value
    return State(data, treedefmeta=state.treedefmeta)
