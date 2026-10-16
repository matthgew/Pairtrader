"""Cointegration pairs-trading lab: scan, backtest, optimise, paper-trade."""
from .backtest import run_backtest
from .strategy import Params, compute_signals

__all__ = ["Params", "compute_signals", "run_backtest"]
