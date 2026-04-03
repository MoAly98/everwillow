"""
everwillow: statistical inference steps for high-energy physics analyses with JAX.
"""

from __future__ import annotations

__version__ = "0.1.2"

# Core API
from everwillow._src.inference.fitting import FitResult, fit, ifit, prepare

__all__ = [
    "FitResult",
    "fit",
    "ifit",
    "prepare",
]
