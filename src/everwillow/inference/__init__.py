"""Statistical inference tools for high-energy physics.

This module provides tools for parameter fitting, profile likelihood scans,
hypothesis testing, and limit setting.
"""

from __future__ import annotations

from everwillow.inference.fitting import FitResult, fit, ifit
from everwillow.inference.uncertainty import (
    correlation_matrix,
    covariance_matrix,
    hessian_matrix,
    uncertainties,
)

__all__ = [
    "FitResult",
    "correlation_matrix",
    "covariance_matrix",
    "fit",
    "hessian_matrix",
    "ifit",
    "uncertainties",
]
