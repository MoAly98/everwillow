"""Core fitting functionality for statistical inference."""

from __future__ import annotations

from everwillow._src.inference.fitting import FitResult as FitResult
from everwillow._src.inference.fitting import fit as fit
from everwillow._src.inference.fitting import ifit as ifit

__all__ = ["FitResult", "fit", "ifit"]
