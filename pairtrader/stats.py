"""Cointegration statistics: Engle-Granger test, hedge ratio, half-life, pair scanning."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS of y on x: returns (beta, intercept) so spread = y - beta*x - intercept."""
    model = sm.OLS(y.values, sm.add_constant(x.values)).fit()
    return float(model.params[1]), float(model.params[0])


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life (in bars) from an AR(1) fit of the spread's changes."""
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    model = sm.OLS(delta.values, sm.add_constant(lag.values)).fit()
    lam = model.params[1]
    return float(-np.log(2) / lam) if lam < 0 else float("inf")


def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """Test y and x for cointegration and describe the spread."""
    _, pvalue, _ = coint(y, x)
    beta, alpha = hedge_ratio(y, x)
    spread = y - beta * x - alpha
    return {
        "pvalue": float(pvalue),
        "hedge_ratio": beta,
        "intercept": alpha,
        "half_life": half_life(spread),
        "spread_std": float(spread.std()),
        "correlation": float(y.corr(x)),
    }


def scan_pairs(prices: pd.DataFrame, max_pvalue: float = 0.05) -> pd.DataFrame:
    """Test every pair of columns; return the cointegrated ones sorted by p-value.

    Note: testing many pairs creates false positives (multiple comparisons).
    Always confirm a pair out-of-sample before trusting it.
    """
    rows = []
    for a, b in combinations(prices.columns, 2):
        res = engle_granger(prices[a], prices[b])
        rows.append({"y": a, "x": b, **res})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df.pvalue <= max_pvalue].sort_values("pvalue").reset_index(drop=True)
