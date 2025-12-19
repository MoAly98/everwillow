"""Statistical inference tools for high-energy physics.

This module provides tools for parameter fitting, profile likelihood scans,
hypothesis testing, and limit setting.
"""

from __future__ import annotations

from everwillow.inference.fitting import Callback, FitResult, fit, ifit

__all__ = [
    "Callback",
    "FitResult",
    "fit",
    "ifit",
]
