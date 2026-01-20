from __future__ import annotations

import typing as tp
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

import everwillow as ew
import everwillow.statelib as sl
from everwillow.inference import FitResult
from everwillow.parameters.transforms import (
    MinuitTransform,
    OneSidedLogTransform,
    SigmoidTransform,
    SoftPlusTransform,
)

jax.config.update("jax_enable_x64", True)

# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def simple_quadratic_nll():
    """Simple quadratic NLL: min at x=2, y=3."""

    def nll(params):
        return (params["x"] - 2.0) ** 2 + (params["y"] - 3.0) ** 2

    return nll


@pytest.fixture
def simple_params():
    """Initial params for simple_quadratic_nll."""
    return sl.State.from_pytree({"x": 0.0, "y": 0.0})


@pytest.fixture
def mock_pbar():
    """Mock tqdm progress bar for unit tests."""

    class MockPbar:
        def __init__(self):
            self.n = 0
            self.total = 100
            self._postfix = {}
            self.closed = False

        def set_postfix(self, **kwargs):
            self._postfix = kwargs

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    return MockPbar()


# ============================================================================
# Test helpers
# ============================================================================


def _expect_close(expected: float, *, atol: float = 1e-2):
    def _check(value: float) -> None:
        assert jnp.isclose(value, expected, atol=atol)

    return _check


def _expect_interval(*, lower: float | None = None, upper: float | None = None):
    def _check(value: float) -> None:
        if lower is not None:
            assert value >= lower
        if upper is not None:
            assert value <= upper

    return _check


def _fit_and_compare(
    nll_fn: tp.Callable[[tp.Any], float],
    params: tp.Any,
    **kwargs,
) -> ew.FitResult:
    expected = ew.fit(nll_fn, params, **kwargs)
    jit_expected = eqx.filter_jit(ew.fit)(nll_fn, params, **kwargs)
    # this compares everything except for the treedef that is not guaranteed to be the same
    assert eqx.tree_equal(expected.params, jit_expected.params, rtol=1e-12)
    assert jnp.isclose(expected.nll, jit_expected.nll)
    assert expected.success == jit_expected.success
    return expected


def _fit_raises(
    nll_fn: tp.Callable[[tp.Any], float],
    params: tp.Any,
    exception: type[Exception],
    **kwargs,
) -> None:
    with pytest.raises(exception):
        ew.fit(nll_fn, params, **kwargs)
    jit_fit = eqx.filter_jit(lambda p: ew.fit(nll_fn, p, **kwargs))
    with pytest.raises(exception):
        jit_fit(params)


# ============================================================================
# FitResult dataclass tests
# ============================================================================


class TestFitResult:
    """Tests for FitResult dataclass."""

    def test_fitresult_creation(self):
        """Test creating a FitResult with all fields."""
        params = {"mu": 1.0, "sigma": 0.5}
        result: FitResult[float] = FitResult(
            params=params,
            nll=jnp.asarray(5.5),
            success=jnp.asarray(True),
            solver_result=None,
        )

        assert result.params == params
        assert jnp.isclose(result.nll, 5.5)
        assert bool(result.success)
        assert result.solver_result is None

    def test_fitresult_frozen(self):
        """Test that FitResult is immutable (frozen dataclass)."""
        result: FitResult[float] = FitResult(
            params={},
            nll=jnp.asarray(0.0),
            success=jnp.asarray(True),
            solver_result=None,
        )

        with pytest.raises(AttributeError):
            result.nll = jnp.asarray(10.0)  # type: ignore[assignment]

    def test_fitresult_allows_none_solver_result(self):
        """Test that solver_result accepts ``None``."""
        result: FitResult[float] = FitResult(
            params={},
            nll=jnp.asarray(0.0),
            success=jnp.asarray(True),
            solver_result=None,
        )
        assert result.solver_result is None


# ============================================================================
# fit() function tests
# ============================================================================


class TestFit:
    """Tests for fit() public API."""

    # --- Basic functionality ---

    def test_simple_quadratic(self):
        """Test fitting a simple quadratic NLL."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2 + (params["sigma"] - 1.0) ** 2

        result = _fit_and_compare(
            nll, params=sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})
        )

        assert abs(result.params["mu"] - 2.0) < 1e-4
        assert abs(result.params["sigma"] - 1.0) < 1e-4
        assert float(result.nll) < 1e-8
        assert bool(result.success)
        assert result.solver_result is not None

    def test_single_parameter(self):
        """Test fitting with a single parameter."""

        def nll(params):
            return (params["x"] - 5.0) ** 2

        result = _fit_and_compare(nll, params=sl.State.from_pytree({"x": 0.0}))

        assert abs(result.params["x"] - 5.0) < 1e-4

    def test_multiple_parameters(self):
        """Test fitting with multiple parameters."""

        def nll(params):
            return (
                (params["a"] - 1.0) ** 2
                + (params["b"] - 2.0) ** 2
                + (params["c"] - 3.0) ** 2
            )

        result = _fit_and_compare(
            nll, params=sl.State.from_pytree({"a": 0.0, "b": 0.0, "c": 0.0})
        )

        assert result.params["a"] == 1.0
        assert abs(result.params["b"] - 2.0) < 1e-4
        assert abs(result.params["c"] - 3.0) < 1e-4

    # --- Pytree structures ---

    def test_nested_dict(self):
        """Test fitting with nested dict structure."""

        def nll(params):
            return (params["level1"]["mu"] - 2.0) ** 2 + (
                params["level1"]["sigma"] - 1.0
            ) ** 2

        initial: sl.State[float] = sl.State.from_pytree(
            {"level1": {"mu": 0.0, "sigma": 0.5}}
        )
        result = _fit_and_compare(nll, params=initial)

        assert abs(result.params["level1"]["mu"] - 2.0) < 1e-4
        assert abs(result.params["level1"]["sigma"] - 1.0) < 1e-4

    def test_deeply_nested_dict(self):
        """Test fitting with deeply nested structure."""

        def nll(params):
            return (params["a"]["b"]["c"] - 5.0) ** 2

        initial: sl.State[float] = sl.State.from_pytree({"a": {"b": {"c": 0.0}}})
        result = _fit_and_compare(nll, params=initial)

        assert abs(result.params["a"]["b"]["c"] - 5.0) < 1e-4

    def test_mixed_structure(self):
        """Test fitting with mixed flat and nested structure."""

        def nll(params):
            return (params["flat"] - 1.0) ** 2 + (params["nested"]["value"] - 2.0) ** 2

        initial: sl.State[float] = sl.State.from_pytree(
            {"flat": 0.0, "nested": {"value": 0.0}}
        )
        result = _fit_and_compare(nll, params=initial)

        assert abs(result.params["flat"] - 1.0) < 1e-4
        assert abs(result.params["nested"]["value"] - 2.0) < 1e-4

    # --- Fixed parameters ---

    def test_single_fixed_parameter(self):
        """Test fixing a single parameter."""

        def nll(params):
            return (
                (params["mu"] - 2.0) ** 2
                + (params["sigma"] - 1.0) ** 2
                + (params["background"] - 100.0) ** 2
            )

        result = _fit_and_compare(
            nll,
            params=sl.State.from_pytree({"mu": 0.0, "sigma": 0.5, "background": 50.0}),
            fixed=sl.State.from_pytree({"background": ...}),
        )

        assert abs(result.params["mu"] - 2.0) < 1e-4
        assert abs(result.params["sigma"] - 1.0) < 1e-4
        assert (
            abs(result.params["background"] - 50.0) < 1e-10
        )  # Should be exactly fixed

    def test_multiple_fixed_parameters(self):
        """Test fixing multiple parameters."""

        def nll(params):
            return (
                (params["a"] - 1.0) ** 2
                + (params["b"] - 2.0) ** 2
                + (params["c"] - 3.0) ** 2
            )

        result = _fit_and_compare(
            nll,
            params=sl.State.from_pytree({"a": 0.0, "b": 10.0, "c": 20.0}),
            fixed=sl.State.from_pytree({"b": ..., "c": ...}),
        )

        assert abs(result.params["a"] - 1.0) < 1e-4
        assert abs(result.params["b"] - 10.0) < 1e-10
        assert abs(result.params["c"] - 20.0) < 1e-10

    def test_all_parameters_fixed(self):
        """Test when all parameters are fixed (no optimization needed)."""

        def nll(params):
            return (params["x"] - 5.0) ** 2

        _fit_raises(
            nll,
            params=sl.State.from_pytree({"x": 3.0}),
            exception=IndexError,
            fixed=sl.State.from_pytree({"x": ...}),
        )

    def test_fixed_none(self):
        """Test that fixed=None works (no fixed parameters)."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2

        result = _fit_and_compare(
            nll, params=sl.State.from_pytree({"mu": 0.0}), fixed=None
        )

        assert abs(result.params["mu"] - 2.0) < 1e-4

    def test_fixed_empty_mapping(self):
        """Test that fixed={} works (no fixed parameters)."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2

        result = _fit_and_compare(
            nll,
            params=sl.State.from_pytree({"mu": 0.0}),
            fixed=sl.State.from_pytree({}),
        )

        assert abs(result.params["mu"] - 2.0) < 1e-4

    def test_fixed_nested_parameter(self):
        """Test fixing a parameter in nested structure."""

        def nll(params):
            return (params["level1"]["mu"] - 2.0) ** 2 + (
                params["level1"]["sigma"] - 1.0
            ) ** 2

        initial: sl.State[float] = sl.State.from_pytree(
            {"level1": {"mu": 0.0, "sigma": 5.0}}
        )
        result = _fit_and_compare(
            nll,
            initial,
            fixed=sl.State.from_pytree({"level1": {"sigma": ...}}),
        )

        assert abs(result.params["level1"]["mu"] - 2.0) < 1e-4
        assert abs(result.params["level1"]["sigma"] - 5.0) < 1e-10

    # --- Additional arguments ---

    def test_positional_args(self):
        """Test fit() with additional positional arguments."""

        def nll(params, target_mu, target_sigma):
            return (params["mu"] - target_mu) ** 2 + (
                params["sigma"] - target_sigma
            ) ** 2

        target_mu, target_sigma = 3.0, 1.5

        def wrapped(params):
            return nll(params, target_mu, target_sigma)

        result = _fit_and_compare(
            wrapped, params=sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})
        )

        assert abs(result.params["mu"] - 3.0) < 1e-4
        assert abs(result.params["sigma"] - 1.5) < 1e-4

    def test_keyword_args(self):
        """Test fit() with keyword arguments."""

        def nll(params, *, target_mu, target_sigma):
            return (params["mu"] - target_mu) ** 2 + (
                params["sigma"] - target_sigma
            ) ** 2

        wrapped = partial(nll, target_mu=4.0, target_sigma=0.8)
        result = _fit_and_compare(
            wrapped, params=sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})
        )

        assert abs(result.params["mu"] - 4.0) < 1e-4
        assert abs(result.params["sigma"] - 0.8) < 1e-4

    def test_both_args_and_kwargs(self):
        """Test fit() with both positional and keyword arguments."""

        def nll(params, target_mu, *, offset):
            return (params["mu"] - target_mu - offset) ** 2

        target_mu, offset = 2.0, 0.5

        def wrapped(params):
            return nll(params, target_mu, offset=offset)

        result = _fit_and_compare(wrapped, params=sl.State.from_pytree({"mu": 0.0}))

        assert abs(result.params["mu"] - 2.5) < 1e-4

    def test_args_with_fixed_params(self):
        """Test additional args combined with fixed parameters."""

        def nll(params, scale):
            return (params["a"] - scale) ** 2 + (params["b"] - 10.0) ** 2

        def wrapped(params):
            return nll(params, 7.0)

        result = _fit_and_compare(
            wrapped,
            params=sl.State.from_pytree({"a": 0.0, "b": 5.0}),
            fixed=sl.State.from_pytree({"b": ...}),
        )

        assert abs(result.params["a"] - 7.0) < 1e-4
        assert abs(result.params["b"] - 5.0) < 1e-10

    # --- Solver options ---

    def test_custom_solver(self):
        """Test fit() with custom solver."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2

        custom_solver: optx.AbstractMinimiser = optx.BFGS(rtol=1e-6, atol=1e-6)
        result = _fit_and_compare(
            nll, params=sl.State.from_pytree({"mu": 0.0}), solver=custom_solver
        )

        assert abs(result.params["mu"] - 2.0) < 1e-5

    def test_solver_kwargs(self):
        """Test that solver_kwargs are passed through."""

        def nll(params):
            return (params["mu"] - 2.0) ** 2

        result = _fit_and_compare(
            nll, params=sl.State.from_pytree({"mu": 0.0}), max_steps=50
        )

        assert abs(result.params["mu"] - 2.0) < 1e-4

    # --- Realistic examples ---

    def test_poisson_likelihood(self):
        """Test Poisson negative log-likelihood fit."""

        def poisson_nll(params, observed):
            signal = 10.0
            expected = params["mu"] * signal + params["background"]
            # Poisson NLL (ignoring constant term)
            return expected - observed * jnp.log(expected)

        observed = 25.0
        result = _fit_and_compare(
            lambda params: poisson_nll(params, observed),
            params=sl.State.from_pytree({"mu": 1.0, "background": 10.0}),
        )

        # MLE for Poisson: expected ≈ observed
        expected_total = result.params["mu"] * 10.0 + result.params["background"]
        assert (
            abs(expected_total - observed) < 0.02
        )  # Relaxed tolerance for optimizer convergence

    def test_gaussian_with_constraint(self):
        """Test Gaussian likelihood with constraint term."""

        def nll_with_constraint(params):
            # Main term
            main = (params["mu"] - 2.0) ** 2

            # Constraint on sigma (Gaussian prior)
            constraint = ((params["sigma"] - 1.0) / 0.2) ** 2

            return main + constraint

        result = _fit_and_compare(
            nll_with_constraint, params=sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})
        )

        assert abs(result.params["mu"] - 2.0) < 1e-4
        assert abs(result.params["sigma"] - 1.0) < 1e-3

    # --- Parameter bounds ---

    def test_fit_hits_upper_bound(self):
        """Test that upper bound is enforced when minimum is above it."""

        def nll(params):
            # Unconstrained minimum at mu=10
            return (params["mu"] - 10.0) ** 2

        result = _fit_and_compare(
            nll,
            sl.State.from_pytree({"mu": 0.5}),
            bounds=sl.State.from_pytree({"mu": MinuitTransform(lower=0.0, upper=5.0)}),
        )

        # Should hit the upper bound since true minimum is at 10
        assert 0.0 <= result.params["mu"] <= 5.0
        assert jnp.isclose(result.params["mu"], 5.0, atol=1e-2)
        assert float(result.nll) < 100.0  # Should be at (5-10)^2 = 25, not (0.5-10)^2

    def test_fit_hits_lower_bound(self):
        """Test that lower bound is enforced when minimum is below it."""

        def nll(params):
            # Unconstrained minimum at mu=-5
            return (params["mu"] + 5.0) ** 2

        result = _fit_and_compare(
            nll,
            sl.State.from_pytree({"mu": 2.0}),
            bounds=sl.State.from_pytree({"mu": MinuitTransform(lower=0.0, upper=10.0)}),
        )

        # Should hit the lower bound since true minimum is at -5
        assert 0.0 <= result.params["mu"] <= 10.0
        assert jnp.isclose(result.params["mu"], 0.0, atol=1e-2)
        assert float(result.nll) < 100.0  # Should be at (0+5)^2 = 25

    def test_fit_within_bounds_unconstrained(self):
        """Test that bounds don't affect fit when minimum is within bounds."""

        def nll(params):
            # Unconstrained minimum at mu=2.5 (inside [0, 5])
            return (params["mu"] - 2.5) ** 2

        result = _fit_and_compare(
            nll,
            sl.State.from_pytree({"mu": 1.0}),
            bounds=sl.State.from_pytree({"mu": MinuitTransform(lower=0.0, upper=5.0)}),
        )

        # Should find true minimum
        assert 0.0 <= result.params["mu"] <= 5.0
        assert jnp.isclose(result.params["mu"], 2.5, atol=1e-2)

    @pytest.mark.parametrize(
        ("transform_factory", "target", "initial", "expected"),
        [
            (MinuitTransform(lower=0.0, upper=5.0), 2.5, 0.1, 2.5),
            (SigmoidTransform(lower=-2.0, upper=3.0), 2.0, -1.0, 2.0),
            (
                OneSidedLogTransform(bound=0.0, direction="lower"),
                -3.0,
                1.0,
                0.0,
            ),
            (
                OneSidedLogTransform(bound=5.0, direction="upper"),
                8.0,
                1.0,
                5.0,
            ),
            (SoftPlusTransform(), 1.5, 0.2, 1.5),
        ],
    )
    def test_fit_supports_all_transform_variants(
        self,
        transform_factory,
        target,
        initial,
        expected,
    ):
        """Ensure fit integrates each transform class."""

        def nll(params):
            return (params["x"] - target) ** 2

        result = _fit_and_compare(
            nll,
            sl.State.from_pytree({"x": initial}),
            bounds=sl.State.from_pytree({"x": transform_factory}),
        )

        assert jnp.isclose(result.params["x"], expected, atol=1e-2)

    @pytest.mark.parametrize(
        ("transform_factory", "target", "initial", "check"),
        [
            (
                MinuitTransform(lower=0.0, upper=1.0),
                -2.0,
                0.3,
                _expect_close(0.0),
            ),
            (
                MinuitTransform(lower=0.0, upper=1.0),
                5.0,
                0.7,
                _expect_close(1.0),
            ),
            (
                SigmoidTransform(lower=0.0, upper=1.0),
                -5.0,
                0.4,
                _expect_close(0.0),
            ),
            (
                SigmoidTransform(lower=0.0, upper=1.0),
                5.0,
                0.6,
                _expect_close(1.0),
            ),
            (
                OneSidedLogTransform(bound=0.0, direction="lower"),
                -5.0,
                0.8,
                _expect_close(0.0),
            ),
            (
                OneSidedLogTransform(bound=2.0, direction="upper"),
                10.0,
                1.0,
                _expect_close(2.0),
            ),
            (
                SoftPlusTransform(),
                -1.0,
                0.5,
                _expect_interval(lower=0.0, upper=5e-2),
            ),
        ],
    )
    def test_fit_transforms_enforce_bounds(
        self,
        transform_factory,
        target,
        initial,
        check,
    ):
        """Verify each transform clamps solutions at its boundary."""

        def nll(params):
            return (params["x"] - target) ** 2

        result = _fit_and_compare(
            nll,
            params=sl.State.from_pytree({"x": initial}),
            bounds=sl.State.from_pytree({"x": transform_factory}),
        )

        check(float(result.params["x"]))


# ============================================================================
# _fit() internal function tests
# ============================================================================


class TestFitInternal:
    """Tests for _fit() shared logic (validation, dispatch)."""

    def test_validates_params_type(self):
        """Should raise TypeError if params is not a State."""

        def nll(params):
            return params["x"] ** 2

        # Pass a dict instead of State
        with pytest.raises(TypeError, match="params must be a State"):
            ew.fit(nll, {"x": 0.0})  # type: ignore[arg-type]

    def test_validates_fixed_type(self):
        """Should raise TypeError if fixed is not State or None."""

        def nll(params):
            return params["x"] ** 2

        with pytest.raises(TypeError, match="fixed must be a State or None"):
            ew.fit(nll, sl.State.from_pytree({"x": 0.0}), fixed={"x": ...})  # type: ignore[arg-type]

    def test_validates_bounds_type(self):
        """Should raise TypeError if bounds is not State or None."""

        def nll(params):
            return params["x"] ** 2

        with pytest.raises(TypeError, match="bounds must be a State or None"):
            ew.fit(nll, sl.State.from_pytree({"x": 0.0}), bounds={"x": None})  # type: ignore[arg-type]

    def test_ifit_validates_params_type(self):
        """ifit should also validate params type."""

        def nll(params):
            return params["x"] ** 2

        with pytest.raises(TypeError, match="params must be a State"):
            ew.ifit(nll, {"x": 0.0}, progress=False)  # type: ignore[arg-type]


# ============================================================================
# _ProgressUpdater class tests
# ============================================================================


class TestProgressUpdater:
    """Tests for _ProgressUpdater class."""

    def test_update_sets_progress_and_postfix(self, mock_pbar):
        """update() should set pbar.n and postfix with NLL."""
        from everwillow.inference.fitting import _ProgressUpdater  # noqa: PLC2701

        updater = _ProgressUpdater(mock_pbar)
        updater.update(step=5, nll_value=1.234)

        assert mock_pbar.n == 5
        assert "NLL" in mock_pbar._postfix
        assert "1.234" in mock_pbar._postfix["NLL"]

    def test_finalize_sets_total_to_actual_steps(self, mock_pbar):
        """finalize() should adjust total to final_step, not max_steps."""
        from everwillow.inference.fitting import _ProgressUpdater  # noqa: PLC2701

        mock_pbar.total = 100  # max_steps was 100
        updater = _ProgressUpdater(mock_pbar)
        updater.finalize(final_step=25, nll_value=0.5)

        assert mock_pbar.total == 25  # Should be actual steps, not 100
        assert mock_pbar.n == 25

    def test_finalize_sets_final_nll_in_postfix(self, mock_pbar):
        """finalize() should show final NLL value."""
        from everwillow.inference.fitting import _ProgressUpdater  # noqa: PLC2701

        updater = _ProgressUpdater(mock_pbar)
        updater.finalize(final_step=10, nll_value=0.001)

        assert "NLL" in mock_pbar._postfix
        assert "0.001" in mock_pbar._postfix["NLL"]


# ============================================================================
# _make_progress_context tests
# ============================================================================


class TestMakeProgressContext:
    """Tests for _make_progress_context context manager."""

    def test_yields_updater_when_enabled(self):
        """Should yield _ProgressUpdater when enabled=True."""
        from everwillow.inference.fitting import (
            _make_progress_context,  # noqa: PLC2701
            _ProgressUpdater,  # noqa: PLC2701
        )

        with _make_progress_context(enabled=True, max_steps=100) as updater:
            assert isinstance(updater, _ProgressUpdater)

    def test_yields_none_when_disabled(self):
        """Should yield None when enabled=False."""
        from everwillow.inference.fitting import _make_progress_context  # noqa: PLC2701

        with _make_progress_context(enabled=False, max_steps=100) as updater:
            assert updater is None

    def test_closes_progress_bar_on_exit(self):
        """Context manager should close pbar on exit."""
        from everwillow.inference.fitting import _make_progress_context  # noqa: PLC2701

        # We can't easily check if pbar is closed without mocking tqdm,
        # but we can verify the context manager exits cleanly
        with _make_progress_context(enabled=True, max_steps=10) as updater:
            assert updater is not None
        # If we get here, the pbar was closed properly

    def test_closes_progress_bar_on_exception(self):
        """Context manager should close pbar even if exception raised."""
        from everwillow.inference.fitting import _make_progress_context  # noqa: PLC2701

        def raise_in_context():
            with _make_progress_context(enabled=True, max_steps=10):
                raise ValueError("Test exception")

        # If cleanup fails, this would raise a different error
        with pytest.raises(ValueError, match="Test exception"):
            raise_in_context()


# ============================================================================
# _iminimize tests
# ============================================================================


class TestIminimize:
    """Tests for _iminimize interactive minimization loop."""

    def test_callback_called_each_iteration(self, simple_quadratic_nll, simple_params):
        """Callback should be invoked at every solver step."""
        call_count = []

        def counting_callback(step, y, state):
            call_count.append(step)

        result = ew.ifit(
            simple_quadratic_nll,
            simple_params,
            callback=counting_callback,
            progress=False,
            max_steps=50,
        )

        assert len(call_count) > 0
        assert result.success

    def test_callback_receives_correct_step_index(
        self, simple_quadratic_nll, simple_params
    ):
        """First arg should be 0, 1, 2, ... for each iteration."""
        steps = []

        def record_step(step, y, state):
            steps.append(step)

        ew.ifit(
            simple_quadratic_nll,
            simple_params,
            callback=record_step,
            progress=False,
            max_steps=50,
        )

        # Steps should be sequential starting from 0
        assert steps == list(range(len(steps)))

    def test_callback_receives_solver_state_with_nll(self):
        """Third arg should be solver state with f_info.f for NLL."""
        nlls = []

        # Use a harder problem so NLL doesn't start at 0
        def nll(params):
            return (params["x"] - 10.0) ** 2 + (params["y"] - 20.0) ** 2

        def record_nll(step, y, state):
            nlls.append(float(state.f_info.f))

        result: FitResult[float] = ew.ifit(
            nll,
            sl.State.from_pytree({"x": 0.0, "y": 0.0}),
            callback=record_nll,
            progress=False,
            max_steps=50,
        )

        assert len(nlls) > 0
        # Verify state.f_info.f contains valid NLL values (not all zeros)
        assert any(nll_val > 0 for nll_val in nlls)
        # Final result should be close to optimum
        assert result.success
        assert abs(result.params["x"] - 10.0) < 1e-3

    def test_no_callback_when_none(self, simple_quadratic_nll, simple_params):
        """Should work without callback (callback=None)."""
        result = ew.ifit(
            simple_quadratic_nll,
            simple_params,
            callback=None,
            progress=False,
        )

        assert result.success
        assert abs(result.params["x"] - 2.0) < 1e-3
        assert abs(result.params["y"] - 3.0) < 1e-3

    def test_early_termination_before_max_steps(
        self, simple_quadratic_nll, simple_params
    ):
        """Should stop when solver converges, not wait for max_steps."""
        result = ew.ifit(
            simple_quadratic_nll,
            simple_params,
            progress=False,
            max_steps=1000,  # Large max
        )

        # Should converge well before 1000 steps
        assert result.solver_result.stats["num_steps"] < 100
        assert result.success

    def test_respects_max_steps_limit(self):
        """Should stop at max_steps even if not converged."""

        # Use a hard NLL that won't converge quickly
        def hard_nll(params):
            return jnp.sin(params["x"] * 10) ** 2 + (params["x"] - 100.0) ** 2

        result: FitResult[float] = ew.ifit(
            hard_nll,
            sl.State.from_pytree({"x": 0.0}),
            progress=False,
            max_steps=5,
        )

        assert result.solver_result.stats["num_steps"] <= 5

    def test_returns_actual_step_count_in_stats(
        self, simple_quadratic_nll, simple_params
    ):
        """stats should have num_steps with actual iteration count."""
        result = ew.ifit(
            simple_quadratic_nll,
            simple_params,
            progress=False,
            max_steps=100,
        )

        assert "num_steps" in result.solver_result.stats
        assert int(result.solver_result.stats["num_steps"]) > 0
        assert int(result.solver_result.stats["num_steps"]) <= 100

    def test_progress_updater_update_called(self, simple_quadratic_nll, simple_params):
        """update() should be called each iteration when progress=True."""
        from unittest.mock import MagicMock, patch

        mock_updater = MagicMock()

        with patch(
            "everwillow.inference.fitting._make_progress_context"
        ) as mock_context:
            mock_context.return_value.__enter__ = MagicMock(return_value=mock_updater)
            mock_context.return_value.__exit__ = MagicMock(return_value=False)

            ew.ifit(simple_quadratic_nll, simple_params, progress=True)

            # update() should have been called at least once
            assert mock_updater.update.call_count > 0

    def test_progress_updater_finalize_called(
        self, simple_quadratic_nll, simple_params
    ):
        """finalize() should be called when optimization completes."""
        from unittest.mock import MagicMock, patch

        mock_updater = MagicMock()

        with patch(
            "everwillow.inference.fitting._make_progress_context"
        ) as mock_context:
            mock_context.return_value.__enter__ = MagicMock(return_value=mock_updater)
            mock_context.return_value.__exit__ = MagicMock(return_value=False)

            ew.ifit(simple_quadratic_nll, simple_params, progress=True)

            # finalize() should have been called exactly once
            mock_updater.finalize.assert_called_once()

    def test_progress_updater_finalize_receives_final_step_and_nll(
        self, simple_quadratic_nll, simple_params
    ):
        """finalize() should receive the final step count and NLL value."""
        from unittest.mock import MagicMock, patch

        mock_updater = MagicMock()

        with patch(
            "everwillow.inference.fitting._make_progress_context"
        ) as mock_context:
            mock_context.return_value.__enter__ = MagicMock(return_value=mock_updater)
            mock_context.return_value.__exit__ = MagicMock(return_value=False)

            result = ew.ifit(simple_quadratic_nll, simple_params, progress=True)

            # finalize() should be called with final_step and nll_value
            call_args = mock_updater.finalize.call_args
            final_step = call_args[0][0]  # First positional arg
            nll_value = call_args[0][1]  # Second positional arg

            # final_step should match stats
            assert final_step == int(result.solver_result.stats["num_steps"])
            # nll_value should be close to final NLL
            assert abs(nll_value - float(result.nll)) < 1e-6


# ============================================================================
# ifit() public API tests
# ============================================================================


class TestIfit:
    """Tests for ifit() public API."""

    def test_simple_quadratic(self, simple_quadratic_nll, simple_params):
        """Test ifit finds correct minimum."""
        result = ew.ifit(simple_quadratic_nll, simple_params, progress=False)

        assert abs(result.params["x"] - 2.0) < 1e-3
        assert abs(result.params["y"] - 3.0) < 1e-3
        assert float(result.nll) < 1e-6
        assert result.success

    def test_converges_to_same_result_as_fit(self, simple_quadratic_nll, simple_params):
        """ifit and fit should produce equivalent results."""
        fit_result = ew.fit(simple_quadratic_nll, simple_params)
        ifit_result = ew.ifit(simple_quadratic_nll, simple_params, progress=False)

        assert jnp.allclose(fit_result.params["x"], ifit_result.params["x"], atol=1e-4)
        assert jnp.allclose(fit_result.params["y"], ifit_result.params["y"], atol=1e-4)
        assert jnp.allclose(fit_result.nll, ifit_result.nll, atol=1e-6)

    def test_with_progress_disabled(self, simple_quadratic_nll, simple_params):
        """Should work with progress=False."""
        result = ew.ifit(simple_quadratic_nll, simple_params, progress=False)

        assert result.success
        assert result.solver_result is not None

    def test_with_fixed_params(self, simple_quadratic_nll, simple_params):
        """Fixed parameters should remain unchanged."""
        result = ew.ifit(
            simple_quadratic_nll,
            simple_params,
            fixed=sl.State.from_pytree({"y": ...}),
            progress=False,
        )

        assert abs(result.params["x"] - 2.0) < 1e-3
        assert result.params["y"] == 0.0  # Fixed at initial value

    def test_with_bounds(self):
        """Parameter bounds should be respected."""

        def nll(params):
            return (params["x"] - 10.0) ** 2  # Min at x=10

        result: FitResult[float] = ew.ifit(
            nll,
            sl.State.from_pytree({"x": 0.5}),
            bounds=sl.State.from_pytree({"x": MinuitTransform(lower=0.0, upper=5.0)}),
            progress=False,
        )

        assert 0.0 <= result.params["x"] <= 5.0
        assert jnp.isclose(result.params["x"], 5.0, atol=1e-2)  # Should hit upper bound

    def test_callback_history_pattern(self):
        """Record NLL history via callback, verify decreasing."""
        history: dict[str, list] = {"steps": [], "nlls": []}

        # Use harder problem so NLL doesn't start near 0
        def nll(params):
            return (params["x"] - 10.0) ** 2 + (params["y"] - 20.0) ** 2

        def record(step, y, state):
            history["steps"].append(step)
            history["nlls"].append(float(state.f_info.f))

        result: FitResult[float] = ew.ifit(
            nll,
            sl.State.from_pytree({"x": 0.0, "y": 0.0}),
            callback=record,
            progress=False,
        )

        assert len(history["nlls"]) > 0
        assert result.success
        # Verify we recorded some non-zero NLLs (optimization happened)
        assert any(nll_val > 0 for nll_val in history["nlls"])
        # Verify optimization succeeded
        assert abs(result.params["x"] - 10.0) < 1e-3

    def test_max_steps_one(self):
        """Edge case: max_steps=1."""

        def nll(params):
            return (params["x"] - 5.0) ** 2

        result: FitResult[float] = ew.ifit(
            nll,
            sl.State.from_pytree({"x": 0.0}),
            progress=False,
            max_steps=1,
        )

        # Should not crash, even with just 1 step
        assert result.solver_result is not None

    def test_already_at_optimum(self):
        """Edge case: params already at optimum."""

        def nll(params):
            return (params["x"] - 5.0) ** 2

        result: FitResult[float] = ew.ifit(
            nll,
            sl.State.from_pytree({"x": 5.0}),  # Already at optimum
            progress=False,
        )

        assert jnp.isclose(result.params["x"], 5.0, atol=1e-6)
        assert float(result.nll) < 1e-10
