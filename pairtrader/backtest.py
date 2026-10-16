"""Vectorised backtest of a pair position series, with transaction costs."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import Params, compute_signals


@dataclass
class Result:
    params: Params
    frame: pd.DataFrame          # signals + weights + returns + equity
    metrics: dict = field(default_factory=dict)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def _bars_per_year(index: pd.Index) -> float:
    if len(index) < 3:
        return 252.0
    secs = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float))
    if secs >= 20 * 3600:          # daily bars: count only trading days actually present
        span_years = (index[-1] - index[0]).days / 365.25
        return len(index) / span_years if span_years > 0 else 252.0
    return 365.25 * 24 * 3600 / secs


def metrics(returns: pd.Series, position: pd.Series, bars_per_year: float) -> dict:
    r = returns.fillna(0.0)
    equity = (1 + r).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    years = len(r) / bars_per_year if bars_per_year else 0
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else 0.0
    vol = float(r.std() * np.sqrt(bars_per_year))
    sharpe = float(r.mean() / r.std() * np.sqrt(bars_per_year)) if r.std() > 0 else 0.0
    dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    entries = int(((position != 0) & (position.shift(1).fillna(0) == 0)).sum())
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "calmar": cagr / abs(dd) if dd < 0 else 0.0,
        "n_trades": entries,
        "time_in_market": float((position != 0).mean()),
    }


def _trade_log(f: pd.DataFrame) -> pd.DataFrame:
    rows, start, side = [], None, 0.0
    pos = f["position"].values
    for i, p in enumerate(pos):
        prev = pos[i - 1] if i else 0.0
        if prev == 0 and p != 0:
            start, side = i, p
        elif prev != 0 and p != prev:
            seg = f["strategy_ret"].iloc[start + 1: i + 1]
            rows.append({"entry": f.index[start], "exit": f.index[i], "side": "long_spread" if side > 0 else "short_spread",
                         "bars": i - start, "return": float((1 + seg).prod() - 1),
                         "entry_z": float(f["z"].iloc[start]), "exit_z": float(f["z"].iloc[i])})
            if p != 0:
                start, side = i, p
    return pd.DataFrame(rows)


def run_backtest(
    y: pd.Series,
    x: pd.Series,
    params: Params | None = None,
    cost_bps: float = 5.0,
    leverage: float = 1.0,
) -> Result:
    """Backtest the pair. cost_bps is charged per unit of gross weight traded (each leg).

    Weights are dollar-based: gross exposure (|w_y| + |w_x|) equals `leverage`
    whenever a trade is on. The hedge ratio is re-estimated daily, so holding
    a trade incurs small rebalancing costs too.
    """
    p = params or Params()
    f = compute_signals(y, x, p)
    gross = f["y"] + f["beta"].abs() * f["x"]
    f["w_y"] = (f["position"] * f["y"] / gross * leverage).fillna(0.0)
    f["w_x"] = (-f["position"] * f["beta"] * f["x"] / gross * leverage).fillna(0.0)

    ry, rx = f["y"].pct_change(), f["x"].pct_change()
    gross_ret = f["w_y"].shift(1) * ry + f["w_x"].shift(1) * rx
    turnover = f["w_y"].diff().abs() + f["w_x"].diff().abs()
    f["cost"] = turnover.fillna(0.0) * cost_bps / 1e4
    f["strategy_ret"] = gross_ret.fillna(0.0) - f["cost"]
    f["equity"] = (1 + f["strategy_ret"]).cumprod()

    m = metrics(f["strategy_ret"], f["position"], _bars_per_year(f.index))
    m["total_cost"] = float(f["cost"].sum())
    return Result(params=p, frame=f, metrics=m, trades=_trade_log(f))
