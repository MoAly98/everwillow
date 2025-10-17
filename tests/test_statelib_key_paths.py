from __future__ import annotations

import jax.tree_util as jtu
import pytest

import everwillow.statelib.key_paths as kps


def test_canonical_keys_from_jax_key_paths() -> None:
    """Canonical JAX key objects convert to tuple form."""
    assert kps.canonical_key((jtu.DictKey("a"), jtu.SequenceKey(0))) == ("a", 0)
    assert kps.canonical_key((jtu.GetAttrKey("b"), jtu.FlattenedIndexKey(2))) == (
        "b",
        2,
    )


def test_canonical_keys_from_unrecognised_entry_raises() -> None:
    """Unexpected key objects surface a ValueError."""
    with pytest.raises(ValueError, match="Unrecognised key path entry"):
        kps.canonical_key((42,))


def test__ensure_public_key() -> None:
    """Normalise tuple keys without altering their value."""
    key = ("a", "b", 0)
    public_key = kps.ensure_public_key(key)
    assert public_key == key


def test__ensure_public_key_non_tuple_raises() -> None:
    """Reject non-tuple keys when normalising."""
    with pytest.raises(KeyError, match="FlatState keys must be tuples"):
        kps.ensure_public_key("not-a-tuple")  # type: ignore[arg-type]


def test___make_key_entry_no_template() -> None:
    """Factory chooses key types from values when no template is provided."""
    key = ("a", "b", 0)
    for _index, value in enumerate(key):
        entry = kps._make_key_entry(value, None)
        if isinstance(value, int):
            assert isinstance(entry, jtu.SequenceKey)
            assert entry.idx == value
        else:
            assert isinstance(entry, jtu.DictKey)
            assert entry.key == value


def test___make_key_entry_with_template() -> None:
    """Factory mirrors template types when provided."""
    templates = (
        jtu.DictKey("a"),
        jtu.GetAttrKey("b"),
        jtu.FlattenedIndexKey(0),
        jtu.SequenceKey(0),
    )
    values = ("x", "y", 1, 2)
    expected_types = (
        jtu.DictKey,
        jtu.GetAttrKey,
        jtu.FlattenedIndexKey,
        jtu.SequenceKey,
    )

    for value, template, expected_type in zip(
        values, templates, expected_types, strict=False
    ):
        entry = kps._make_key_entry(value, template)
        assert isinstance(entry, expected_type)
        if isinstance(entry, jtu.DictKey):
            assert entry.key == value
        elif isinstance(entry, jtu.GetAttrKey):
            assert entry.name == value
        elif isinstance(entry, jtu.SequenceKey):
            assert entry.idx == value
        elif isinstance(entry, jtu.FlattenedIndexKey):
            assert entry.key == value


def test_derive_key_path_from_canonical_key() -> None:
    """Derive key path entries by pairing values with templates."""
    canonical_key = ("a", "b", 0)
    template_path = (jtu.DictKey("a"), jtu.GetAttrKey("b"), jtu.SequenceKey(0))

    derived_path = tuple(
        kps._make_key_entry(value, template)
        for value, template in zip(canonical_key, template_path, strict=False)
    )

    assert derived_path == template_path


def test_derive_key_path_from_canonical_key_without_template() -> None:
    """Fallback derivation selects default key classes when template is missing."""
    canonical_key = ("a", "b", 0)
    derived_path = kps.derive_key_path(canonical_key, template=None)

    expected_path = (
        jtu.DictKey("a"),
        jtu.DictKey("b"),
        jtu.SequenceKey(0),
    )

    assert derived_path == expected_path


def test_derive_key_path_from_public_key_with_template() -> None:
    """Public keys plus templates rebuild original key objects."""
    key = ("a", "b", 0)
    public_key = kps.ensure_public_key(key)
    template_path = (jtu.DictKey("x"), jtu.GetAttrKey("y"), jtu.SequenceKey(1))
    derived_path = kps.derive_key_path(public_key, template=template_path)
    expected_path = (
        jtu.DictKey("a"),
        jtu.GetAttrKey("b"),
        jtu.SequenceKey(0),
    )
    assert derived_path == expected_path


def test_derive_key_path_from_public_key_without_template() -> None:
    """Public keys without templates fall back to default key types."""
    key = ("a", "b", 0)
    public_key = kps.ensure_public_key(key)
    derived_path = kps.derive_key_path(public_key, template=None)
    expected_path = (
        jtu.DictKey("a"),
        jtu.DictKey("b"),
        jtu.SequenceKey(0),
    )
    assert derived_path == expected_path
