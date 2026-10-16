"""Z-score mean-reversion signals on a rolling cointegration spread.

Position convention (for the pair y/x):
  +1 = long the spread  = BUY y, SELL beta units of x   (spread is unusually LOW)
  -1 = short the spread = SELL y, BUY beta units of x   (spread is unusually HIGH)
   0 = flat

Everything at bar t uses only data up to and including bar t. The backtester
applies the position from bar t to the return from t to t+1, so there is no
look-ahead.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Params:
    hedge_window: int = 120   # bars used to estimate the rolling hedge ratio
    z_window: int = 60        # bars used for the spread's rolling mean / std
    entry_z: float = 2.0      # open a trade when |z| exceeds this
    exit_z: float = 0.5       # close when |z| falls back inside this
    stop_z: float = 4.0       # stop out if |z| blows past this (relationship may have broken)
    max_hold: int = 0         # force exit after this many bars (0 = no limit)

    def to_dict(self) -> dict:
        return asdict(self)


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta of y on x (cov/var), using only past data."""
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


def next_position(prev: float, z: float, held: int, p: Params) -> tuple[float, int]:
    """One step of the trading rule. Shared by the backtest and the live loop.

    Returns (new_position, bars_held).
    """
    if np.isnan(z):
        return 0.0, 0
    if prev == 0:
        if p.entry_z < z < p.stop_z:
            return -1.0, 0     # spread too high -> sell y, buy x
        if -p.stop_z < z < -p.entry_z:
            return 1.0, 0      # spread too low  -> buy y, sell x
        return 0.0, 0
    held += 1
    reverted = (prev > 0 and z > -p.exit_z) or (prev < 0 and z < p.exit_z)
    stopped = abs(z) > p.stop_z
    timed_out = bool(p.max_hold) and held >= p.max_hold
    if reverted or stopped or timed_out:
        return 0.0, 0
    return prev, held


def compute_signals(y: pd.Series, x: pd.Series, p: Params) -> pd.DataFrame:
    beta = rolling_hedge_ratio(y, x, p.hedge_window)
    spread = y - beta * x
    mu = spread.rolling(p.z_window).mean()
    sd = spread.rolling(p.z_window).std()
    z = (spread - mu) / sd

    pos = np.zeros(len(z))
    held = 0
    for t, zt in enumerate(z.values):
        prev = pos[t - 1] if t else 0.0
        pos[t], held = next_position(prev, zt, held, p)

    return pd.DataFrame(
        {"y": y, "x": x, "beta": beta, "spread": spread, "z": z, "position": pos},
        index=y.index,
    )
