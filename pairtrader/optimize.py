"""Parameter search with an honest out-of-sample check, plus an experiment log.

The trap with "maximise the return" is overfitting: with enough knobs, any
backtest can be made to look great on the data it was tuned on. So:
  1. Tune on the TRAIN slice only.
  2. Rank by a risk-adjusted score (Sharpe by default), not raw return.
  3. Report the TEST slice result for the chosen params - that's the number
     that matters. A big train/test gap means you overfit.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import pandas as pd

from .backtest import _bars_per_year, metrics, run_backtest
from .strategy import Params

DEFAULT_GRID = {
    "hedge_window": [60, 120, 250],
    "z_window": [20, 40, 60],
    "entry_z": [1.5, 2.0, 2.5],
    "exit_z": [0.0, 0.5],
    "stop_z": [3.5, 5.0],
}

EXPERIMENT_LOG = Path("results/experiments.jsonl")


def split(prices: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(prices) * train_frac)
    return prices.iloc[:cut], prices.iloc[cut:]


def grid_search(
    y: pd.Series,
    x: pd.Series,
    grid: dict | None = None,
    score: str = "sharpe",
    cost_bps: float = 5.0,
    min_trades: int = 5,
) -> pd.DataFrame:
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = Params(**dict(zip(keys, combo)))
        if p.exit_z >= p.entry_z or p.stop_z <= p.entry_z:
            continue
        m = run_backtest(y, x, p, cost_bps=cost_bps).metrics
        rows.append({**p.to_dict(), **m})
    df = pd.DataFrame(rows)
    df = df[df.n_trades >= min_trades]   # a 2-trade "winner" is luck, not skill
    return df.sort_values(score, ascending=False).reset_index(drop=True)


def optimize_pair(
    prices: pd.DataFrame,
    y_col: str,
    x_col: str,
    grid: dict | None = None,
    train_frac: float = 0.7,
    score: str = "sharpe",
    cost_bps: float = 5.0,
    note: str = "",
) -> dict:
    """Tune on train, evaluate on test, and append the experiment to the log."""
    train, test = split(prices, train_frac)
    board = grid_search(train[y_col], train[x_col], grid, score, cost_bps)
    if board.empty:
        raise ValueError("No parameter set produced enough trades on the train slice.")
    best = Params(**{k: type(v)(board.iloc[0][k]) for k, v in Params().to_dict().items()})

    # Warm up the rolling windows on the tail of train so test starts trading at once.
    warm = max(best.hedge_window, best.z_window)
    test_with_warmup = pd.concat([train.iloc[-warm:], test])
    oos = run_backtest(test_with_warmup[y_col], test_with_warmup[x_col], best, cost_bps=cost_bps)
    oos_ret = oos.frame["strategy_ret"].iloc[warm:]
    test_metrics = metrics(oos_ret, oos.frame["position"].iloc[warm:], _bars_per_year(test.index))

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pair": f"{y_col}/{x_col}",
        "train": [str(train.index[0].date()), str(train.index[-1].date())],
        "test": [str(test.index[0].date()), str(test.index[-1].date())],
        "score": score,
        "cost_bps": cost_bps,
        "best_params": best.to_dict(),
        "train_metrics": {k: float(board.iloc[0][k]) for k in ("total_return", "sharpe", "max_drawdown", "n_trades")},
        "test_metrics": {k: test_metrics[k] for k in ("total_return", "sharpe", "max_drawdown", "n_trades")},
        "n_combos_tried": int(len(board)),
        "note": note,
    }
    EXPERIMENT_LOG.parent.mkdir(exist_ok=True)
    with EXPERIMENT_LOG.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return {"record": record, "leaderboard": board}
