"""Callback protocols and implementations for interactive fitting."""

from __future__ import annotations

from everwillow._src.inference.callback import Callback as Callback
from everwillow._src.inference.callback import HistoryCallback as HistoryCallback

__all__ = ["Callback", "HistoryCallback"]
