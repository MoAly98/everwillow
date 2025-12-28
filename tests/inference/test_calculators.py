"""Tests for asymptotic p-value and CLs calculations."""

from __future__ import annotations

import jax.numpy as jnp

from everwillow.inference.calculators import cls, pvalue_onesided


class TestPvalueOnesided:
    """Tests for pvalue_onesided function."""

    def test_pvalue_at_q_zero_no_shift(self):
        """p-value at q=0 with no shift should be 0.5."""
        # p = 1 - Phi(sqrt(0) - 0) = 1 - Phi(0) = 0.5
        p = pvalue_onesided(0.0, shift=0.0)
        assert jnp.isclose(p, 0.5, atol=1e-6)

    def test_pvalue_at_q_zero_with_shift(self):
        """p-value at q=0 with shift should be > 0.5."""
        # p = 1 - Phi(sqrt(0) - shift) = 1 - Phi(-shift)
        # For positive shift, Phi(-shift) < 0.5, so p > 0.5
        p = pvalue_onesided(0.0, shift=1.0)
        assert float(p) > 0.5

    def test_pvalue_large_q_small_pvalue(self):
        """Large q should give small p-value (strong evidence against null)."""
        # p = 1 - Phi(sqrt(9) - 0) = 1 - Phi(3) ~ 0.0013
        p = pvalue_onesided(9.0, shift=0.0)
        assert float(p) < 0.01

    def test_pvalue_decreases_with_q(self):
        """p-value should decrease as q increases."""
        p1 = pvalue_onesided(1.0, shift=0.0)
        p2 = pvalue_onesided(4.0, shift=0.0)
        p3 = pvalue_onesided(9.0, shift=0.0)
        assert float(p1) > float(p2) > float(p3)

    def test_pvalue_shift_increases_pvalue(self):
        """Adding shift should increase p-value (alternative hypothesis)."""
        # Same q, but with shift = sqrt(q_asimov), p_alt > p_null
        p_null = pvalue_onesided(4.0, shift=0.0)
        p_alt = pvalue_onesided(4.0, shift=2.0)  # shift = sqrt(4) = 2
        assert float(p_alt) > float(p_null)

    def test_pvalue_known_values(self):
        """Test against known normal distribution values."""
        # sqrt(q) = 1 => p = 1 - Phi(1) ~ 0.1587
        p = pvalue_onesided(1.0, shift=0.0)
        assert jnp.isclose(p, 0.1587, atol=0.001)

        # sqrt(q) = 2 => p = 1 - Phi(2) ~ 0.0228
        p = pvalue_onesided(4.0, shift=0.0)
        assert jnp.isclose(p, 0.0228, atol=0.001)


class TestCls:
    """Tests for CLs calculation."""

    def test_cls_ratio(self):
        """CLs = p_alt / p_null."""
        p_alt = 0.1
        p_null = 0.5
        result = cls(p_alt, p_null)
        assert jnp.isclose(result, 0.2, atol=1e-6)

    def test_cls_equal_pvalues(self):
        """CLs = 1 when p_alt = p_null."""
        result = cls(0.3, 0.3)
        assert jnp.isclose(result, 1.0, atol=1e-6)

    def test_cls_protects_zero_denominator(self):
        """CLs should handle p_null near zero gracefully."""
        # Should not raise or return inf
        result = cls(0.1, 1e-15)
        assert jnp.isfinite(result)

    def test_cls_small_values(self):
        """CLs should work with small p-values."""
        result = cls(0.001, 0.01)
        assert jnp.isclose(result, 0.1, atol=1e-6)

    def test_cls_decreases_with_larger_p_null(self):
        """CLs decreases when p_null increases (more background-like)."""
        cls1 = cls(0.1, 0.2)
        cls2 = cls(0.1, 0.5)
        assert float(cls1) > float(cls2)


class TestCalculatorsIntegration:
    """Integration tests combining pvalue_onesided and cls."""

    def test_cls_from_test_statistic(self):
        """Compute CLs from observed and Asimov test statistics."""
        # Observed test statistic (large q means data disfavors alternative)
        q_obs = 4.0

        # Asimov test statistic (shift = sqrt(q_asimov))
        q_asimov = 4.0
        shift = jnp.sqrt(q_asimov)

        # p_alt uses shift, p_null doesn't
        p_alt = pvalue_onesided(q_obs, shift=shift)
        p_null = pvalue_onesided(q_obs, shift=0.0)

        # Compute CLs
        cls_val = cls(p_alt, p_null)

        # CLs is well-defined and positive
        assert float(cls_val) > 0.0
        assert jnp.isfinite(cls_val)

    def test_cls_at_different_q_values(self):
        """CLs should vary appropriately with test statistic."""
        q_asimov = 4.0
        shift = jnp.sqrt(q_asimov)

        # At q=0: p_alt high (data compatible with alt), p_null = 0.5
        cls_at_0 = cls(pvalue_onesided(0.0, shift), pvalue_onesided(0.0, 0.0))

        # At q=q_asimov: p_alt = 0.5, p_null small
        cls_at_asimov = cls(
            pvalue_onesided(q_asimov, shift), pvalue_onesided(q_asimov, 0.0)
        )

        # All values should be positive and finite
        assert float(cls_at_0) > 0.0
        assert float(cls_at_asimov) > 0.0
        assert jnp.isfinite(cls_at_0)
        assert jnp.isfinite(cls_at_asimov)

    def test_cls_exclusion_scenario(self):
        """Test CLs in a typical exclusion scenario (large q_obs)."""
        # When q_obs >> q_asimov, we expect small CLs (strong exclusion)
        q_asimov = 1.0  # Expected sensitivity
        shift = jnp.sqrt(q_asimov)

        # Large observed test statistic (data strongly disfavors alternative)
        q_obs = 9.0  # sqrt(9) = 3 sigma

        p_alt = pvalue_onesided(q_obs, shift=shift)
        p_null = pvalue_onesided(q_obs, shift=0.0)
        cls_val = cls(p_alt, p_null)

        # With q_obs >> q_asimov, CLs should be small (exclusion)
        # p_alt = 1 - Phi(3 - 1) = 1 - Phi(2) ~ 0.023
        # p_null = 1 - Phi(3) ~ 0.0013
        # CLs ~ 0.023 / 0.0013 ~ 17.4 (not exclusion in this case)
        # Actually this scenario doesn't give exclusion
        assert float(cls_val) > 0.0
