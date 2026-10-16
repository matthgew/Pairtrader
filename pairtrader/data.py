"""Price data: real prices via yfinance (cached to disk) and synthetic cointegrated pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data_cache")


def load_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices, one column per ticker."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = f"{'_'.join(sorted(tickers))}_{start}_{end or 'now'}_{interval}.csv".replace(":", "-")
    path = CACHE_DIR / key
    if use_cache and path.exists() and end is not None:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf  # imported lazily so tests run offline

    raw = yf.download(tickers, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {tickers}")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices[tickers].dropna(how="any")
    prices.to_csv(path)
    return prices


def synthetic_pair(
    n: int = 1500,
    hedge_ratio: float = 1.5,
    half_life: float = 15.0,
    spread_vol: float = 1.0,
    drift_vol: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate two prices that are cointegrated by construction.

    x is a random walk; y = hedge_ratio * x + OU spread. Useful for sanity-checking
    the strategy: if it can't make money here, something is broken.
    """
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, drift_vol, n))
    theta = np.log(2) / half_life
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] - theta * s[t - 1] + rng.normal(0, spread_vol)
    y = 20 + hedge_ratio * x + s
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Y": y, "X": x}, index=idx)
