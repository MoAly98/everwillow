"""Statistical inference tools for high-energy physics.

This module provides tools for parameter fitting, profile likelihood scans,
hypothesis testing, and limit setting.
"""

from __future__ import annotations

from everwillow.inference.fitting import (
    FitResult,
    fit,
    fixed_param_fit,
    ifit,
    ifixed_param_fit,
)

__all__ = [
    "FitResult",
    "fit",
    "fixed_param_fit",
    "ifit",
    "ifixed_param_fit",
]
