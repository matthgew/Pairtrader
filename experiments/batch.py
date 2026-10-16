"""Batch experiment: many candidate pairs, cointegration judged on TRAIN only.

Usage:  python experiments/batch.py [--start 2015-01-01] [--end 2026-10-01] [--note "..."]
Writes results/batch_<date>.csv and appends each pair to results/experiments.jsonl.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from pairtrader.data import load_prices  # noqa: E402
from pairtrader.optimize import optimize_pair, split, walk_forward  # noqa: E402
from pairtrader.stats import engle_granger  # noqa: E402

# (y, x, cost_bps, why it might be linked)
CANDIDATES = [
    ("KO", "PEP", 5, "soft drinks"),
    ("XOM", "CVX", 5, "oil majors"),
    ("V", "MA", 5, "card networks"),
    ("HD", "LOW", 5, "home improvement"),
    ("UNP", "CSX", 5, "railroads"),
    ("BHP", "RIO", 5, "miners"),
    ("BP", "SHEL", 5, "euro oil majors"),
    ("GOOGL", "GOOG", 3, "share classes"),
    ("GDX", "GDXJ", 5, "gold miner ETFs"),
    ("XLE", "XOP", 5, "energy ETFs"),
    ("EWA", "EWC", 5, "commodity-country ETFs"),
    ("USO", "BNO", 5, "WTI vs Brent"),
    ("TLT", "IEF", 3, "treasury duration"),
    ("SPY", "RSP", 3, "cap vs equal weight"),
    ("XLU", "VPU", 3, "utilities ETFs"),
    ("ETH-USD", "BTC-USD", 10, "crypto majors (daily)"),
    ("AAPL", "MSFT", 5, "control: big tech, weak economic link"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-10-01")
    ap.add_argument("--note", default="batch")
    ap.add_argument("--walk-forward", action="store_true", help="also run walk-forward re-tuning")
    ap.add_argument("--coint-filter", action="store_true", help="add rolling cointegration filter to the grid")
    args = ap.parse_args()

    grid = None
    if args.coint_filter:
        from pairtrader.optimize import DEFAULT_GRID
        grid = {**DEFAULT_GRID, "coint_window": [250], "coint_pvalue": [0.10]}

    rows = []
    for y, x, cost, why in CANDIDATES:
        try:
            prices = load_prices([y, x], start=args.start, end=args.end)
            train, _ = split(prices)
            eg = engle_granger(train[y], train[x])        # TRAIN only - no peeking
            out = optimize_pair(prices, y, x, grid=grid, cost_bps=cost,
                                note=f"{args.note}: {why}; train coint p={eg['pvalue']:.3f}")
            r = out["record"]
            row = {"pair": f"{y}/{x}", "why": why, "cost_bps": cost,
                   "train_coint_p": round(eg["pvalue"], 4), "half_life": round(eg["half_life"], 1),
                   "train_sharpe": round(r["train_metrics"]["sharpe"], 2),
                   "test_sharpe": round(r["test_metrics"]["sharpe"], 2),
                   "test_return": round(r["test_metrics"]["total_return"], 4),
                   "test_maxdd": round(r["test_metrics"]["max_drawdown"], 4),
                   "test_trades": r["test_metrics"]["n_trades"]}
            if args.walk_forward:
                wf = walk_forward(prices, y, x, grid=grid, cost_bps=cost)
                row.update(wf_sharpe=round(wf["sharpe"], 2), wf_return=round(wf["total_return"], 4),
                           wf_maxdd=round(wf["max_drawdown"], 4), wf_trades=wf["n_trades"])
            rows.append(row)
            print(row, flush=True)
        except Exception as e:
            print(f"[skip] {y}/{x}: {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    out_path = Path("results") / f"batch_{time.strftime('%Y%m%d_%H%M')}_{args.note.replace(' ', '_')}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{df.to_string(index=False)}\n\nSaved {out_path}")


if __name__ == "__main__":
    main()
