"""Tests for :mod:`everwillow.statelib.key_paths`."""

from __future__ import annotations

import jax.tree_util as jtu
import pytest

import everwillow.statelib.key_paths as kps


class TestCanonicalKey:
    """Ensure canonical path conversion behaves as expected."""

    def test_converts_known_jax_keys(self) -> None:
        """A sequence of JAX key objects becomes a plain tuple."""

        assert kps.canonical_key((jtu.DictKey("a"), jtu.SequenceKey(0))) == ("a", 0)
        assert kps.canonical_key((jtu.GetAttrKey("b"), jtu.FlattenedIndexKey(2))) == (
            "b",
            2,
        )

    def test_rejects_unknown_entries(self) -> None:
        """Unsupported key entries raise a ``ValueError``."""

        with pytest.raises(ValueError, match="Unrecognised key path entry"):
            kps.canonical_key((42,))


class TestEnsurePublicKey:
    """Normalisation rules for external key paths."""

    def test_accepts_tuple_input(self) -> None:
        """Tuple paths are returned unchanged."""

        key = ("a", "b", 0)
        assert kps.ensure_public_key(key) == key

    def test_rejects_non_tuple(self) -> None:
        """Non-tuple inputs raise a ``KeyError``."""

        with pytest.raises(KeyError, match="FlatState keys must be tuples"):
            kps.ensure_public_key("not-a-tuple")  # type: ignore[arg-type]


class TestMakeKeyEntry:
    """Creation of specific JAX key objects."""

    def test_without_template(self) -> None:
        """Value type drives the chosen key class when no template is given."""

        key = ("a", "b", 0)
        for _index, value in enumerate(key):
            entry = kps._make_key_entry(value, None)
            if isinstance(value, int):
                assert isinstance(entry, jtu.SequenceKey)
                assert entry.idx == value
            else:
                assert isinstance(entry, jtu.DictKey)
                assert entry.key == value

    def test_with_template(self) -> None:
        """Templates preserve the original JAX key types."""

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


class TestDeriveKeyPath:
    """Reconstruction of JAX key-path tuples."""

    def test_from_canonical_key_with_template(self) -> None:
        """Pairing values with a template yields the same path."""

        canonical_key = ("a", "b", 0)
        template_path = (jtu.DictKey("a"), jtu.GetAttrKey("b"), jtu.SequenceKey(0))

        derived_path = tuple(
            kps._make_key_entry(value, template)
            for value, template in zip(canonical_key, template_path, strict=False)
        )

        assert derived_path == template_path

    def test_from_canonical_key_without_template(self) -> None:
        """Default key classes are inferred when the template is missing."""

        canonical_key = ("a", "b", 0)
        derived_path = kps.derive_key_path(canonical_key, template=None)

        expected_path = (
            jtu.DictKey("a"),
            jtu.DictKey("b"),
            jtu.SequenceKey(0),
        )

        assert derived_path == expected_path

    def test_from_public_key_with_template(self) -> None:
        """Public keys combined with a template reconstruct original path objects."""

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

    def test_from_public_key_without_template(self) -> None:
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
