"""Tests for everwillow.parameters.transforms."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import everwillow._src.parameters.transforms as transforms  # noqa: PLC2701

jax.config.update("jax_enable_x64", True)

ATOL = 1e-8


class TestMinuitTransform:
    """Minuit arcsin/sin transform behaviour."""

    def test_unwrap_expected_value(self):
        """unwrap matches the Minuit arcsin formula."""
        transform = transforms.MinuitTransform(lower=0.0, upper=1.0)
        value = 0.25
        expected = math.asin(
            2.0 * (value - transform.lower) / (transform.upper - transform.lower) - 1.0
        )
        result = transform.unwrap(value)
        assert jnp.isclose(result, expected, atol=ATOL)

    def test_wrap_expected_value(self):
        """wrap matches the Minuit sine formula."""
        transform = transforms.MinuitTransform(lower=-2.0, upper=2.0)
        internal = 0.75
        expected = transform.lower + (transform.upper - transform.lower) / 2.0 * (
            math.sin(internal) + 1.0
        )
        result = transform.wrap(internal)
        assert jnp.isclose(result, expected, atol=ATOL)

    def test_round_trip(self):
        """wrap(unwrap(x)) returns the original value."""
        transform = transforms.MinuitTransform(lower=0.0, upper=1.0)
        original = 0.42
        recovered = transform.wrap(transform.unwrap(original))
        assert jnp.isclose(recovered, original, atol=ATOL)

    @pytest.mark.parametrize("boundary", ["lower", "upper"])
    def test_raises_on_boundary(self, boundary):
        """unwrap rejects values at either boundary."""
        transform = transforms.MinuitTransform(lower=0.0, upper=1.0)
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="MinuitTransform"
        ):
            transform.unwrap(getattr(transform, boundary))

    def test_init_requires_finite_bounds(self):
        """constructor enforces finite and ordered bounds."""
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="lower bound must be finite"
        ):
            transforms.MinuitTransform(lower=jnp.inf, upper=1.0)
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="upper bound must be finite"
        ):
            transforms.MinuitTransform(lower=0.0, upper=jnp.inf)
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="requires lower bound"
        ):
            transforms.MinuitTransform(lower=1.0, upper=0.5)


class TestSigmoidTransform:
    """Logit/Sigmoid transform behaviour."""

    def test_unwrap_expected_value(self):
        """unwrap equals logit of the affine-scaled value."""
        transform = transforms.SigmoidTransform(lower=2.0, upper=5.0)
        value = 2.5
        scaled = (value - transform.lower) / (transform.upper - transform.lower)
        expected = jnp.log(scaled) - jnp.log1p(-scaled)
        result = transform.unwrap(value)
        assert jnp.isclose(result, expected, atol=ATOL)

    def test_wrap_expected_value(self):
        """wrap equals sigmoid of the affine-scaled value."""
        transform = transforms.SigmoidTransform(lower=-1.0, upper=4.0)
        internal = -0.3
        expected = transform.lower + (
            transform.upper - transform.lower
        ) * jax.nn.sigmoid(internal)
        result = transform.wrap(internal)
        assert jnp.isclose(result, expected, atol=ATOL)

    def test_round_trip(self):
        """wrap(unwrap(x)) returns the original value."""
        transform = transforms.SigmoidTransform(lower=0.0, upper=2.0)
        original = 0.7
        recovered = transform.wrap(transform.unwrap(original))
        assert jnp.isclose(recovered, original, atol=ATOL)

    @pytest.mark.parametrize("boundary", ["lower", "upper"])
    def test_raises_on_boundary(self, boundary):
        """unwrap rejects values at either boundary."""
        transform = transforms.SigmoidTransform(lower=-1.0, upper=1.0)
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="SigmoidTransform"
        ):
            transform.unwrap(getattr(transform, boundary))

    def test_init_requires_valid_bounds(self):
        """constructor enforces finite and ordered bounds."""
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="lower bound must be finite"
        ):
            transforms.SigmoidTransform(lower=jnp.inf, upper=1.0)
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="upper bound must be finite"
        ):
            transforms.SigmoidTransform(lower=-1.0, upper=jnp.inf)
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="requires lower bound"
        ):
            transforms.SigmoidTransform(lower=1.0, upper=1.0)


class TestOneSidedLogTransform:
    """Single-sided log transform behaviour."""

    @pytest.mark.parametrize(
        ("direction", "bound", "value"),
        [
            ("lower", 0.0, 2.0),
            ("upper", 5.0, 3.0),
        ],
    )
    def test_unwrap_expected_value(self, direction, bound, value):
        """unwrap equals the expected log expression."""
        transform = transforms.OneSidedLogTransform(bound=bound, direction=direction)
        result = transform.unwrap(value)
        if direction == "lower":
            expected = jnp.log(value - bound)
        else:
            expected = jnp.log(bound - value)
        assert jnp.isclose(result, expected, atol=ATOL)

    @pytest.mark.parametrize(
        ("direction", "bound", "internal"),
        [
            ("lower", -3.0, 1.5),
            ("upper", 5.0, -0.1),
        ],
    )
    def test_wrap_expected_value(self, direction, bound, internal):
        """wrap equals the expected exp-based expression."""
        transform = transforms.OneSidedLogTransform(bound=bound, direction=direction)
        result = transform.wrap(internal)
        if direction == "lower":
            expected = bound + jnp.exp(internal)
        else:
            expected = bound - jnp.exp(internal)
        assert jnp.isclose(result, expected, atol=ATOL)

    @pytest.mark.parametrize(
        ("direction", "bound", "value"),
        [
            ("lower", -1.0, 1.3),
            ("upper", 2.0, 1.5),
        ],
    )
    def test_round_trip(self, direction, bound, value):
        """wrap(unwrap(x)) returns the original value."""
        transform = transforms.OneSidedLogTransform(bound=bound, direction=direction)
        recovered = transform.wrap(transform.unwrap(value))
        assert jnp.isclose(recovered, value, atol=ATOL)

    def test_raises_on_invalid_direction(self):
        """constructor rejects unsupported directions."""
        with pytest.raises(ValueError, match="unsupported direction"):
            transforms.OneSidedLogTransform(bound=0.0, direction="sideways")

    def test_raises_on_infinite_bound(self):
        """constructor rejects non-finite bounds."""
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="bound must be finite"
        ):
            transforms.OneSidedLogTransform(bound=jnp.inf, direction="lower")

    @pytest.mark.parametrize(
        ("direction", "bound", "match"),
        [
            ("lower", 0.0, "greater than lower bound"),
            ("upper", 1.0, "less than upper bound"),
        ],
    )
    def test_raises_on_bound_violation(self, direction, bound, match):
        """unwrap raises if value is outside the permitted side."""
        transform = transforms.OneSidedLogTransform(bound=bound, direction=direction)
        with pytest.raises((eqx.EquinoxRuntimeError, ValueError), match=match):
            transform.unwrap(transform.bound)


class TestSoftPlusTransform:
    """SoftPlus-based positivity transform behaviour."""

    def test_unwrap_expected_value(self):
        """unwrap matches the analytic inverse softplus."""
        transform = transforms.SoftPlusTransform()
        value = 1.2
        expected = jnp.log(-jnp.expm1(-value)) + value
        result = transform.unwrap(value)
        assert jnp.isclose(result, expected, atol=ATOL)

    def test_wrap_expected_value(self):
        """wrap equals jax.nn.softplus."""
        transform = transforms.SoftPlusTransform()
        internal = -0.7
        expected = jax.nn.softplus(internal)
        result = transform.wrap(internal)
        assert jnp.isclose(result, expected, atol=ATOL)

    def test_round_trip(self):
        """wrap(unwrap(x)) returns the original value."""
        transform = transforms.SoftPlusTransform()
        original = 0.8
        recovered = transform.wrap(transform.unwrap(original))
        assert jnp.isclose(recovered, original, atol=ATOL)

    def test_raises_on_negative_input(self):
        """unwrap enforces non-negative inputs."""
        transform = transforms.SoftPlusTransform()
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="expected positive inputs"
        ):
            transform.unwrap(-0.1)

    def test_raises_on_zero_input(self):
        """unwrap enforces non-negative inputs."""
        transform = transforms.SoftPlusTransform()
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="expected positive inputs"
        ):
            transform.unwrap(0.0)


class TestInternalHelpers:
    """Validate behaviour of module-private helpers."""

    def test_logit_and_sigmoid_round_trip(self):
        """_logit and _sigmoid are mutual inverses."""
        raw = jnp.asarray(0.3)
        transformed = transforms._logit(raw)
        restored = transforms._sigmoid(transformed)
        assert jnp.isclose(restored, raw, atol=ATOL)

    def test_sigmoid_limits(self):
        """_sigmoid saturates correctly for large magnitudes."""
        assert jnp.isclose(transforms._sigmoid(50.0), 1.0, atol=1e-6)
        assert jnp.isclose(transforms._sigmoid(-50.0), 0.0, atol=1e-6)
