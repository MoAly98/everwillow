"""
everwillow: statistical inference steps for high-energy physics analyses with JAX.
"""

from __future__ import annotations

import datetime

__name__ = "everwillow"
__author__ = "everwillow developers"
__copyright__ = f"Copyright {datetime.datetime.now().year}, everwillow developers"
__credits__ = ["Mohamed Aly", "Peter Fackeldey", "Massimiliano Galli"]
__contact__ = "https://github.com/MoAly98/everwillow"
__version__ = "0.0.1"

# Core API
from everwillow._src.inference.fitting import FitResult, fit, ifit

__all__ = [
    "FitResult",
    "fit",
    "ifit",
]
