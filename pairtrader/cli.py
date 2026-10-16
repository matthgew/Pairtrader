"""Command line: python -m pairtrader <scan|backtest|optimize|paper> ..."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .backtest import run_backtest
from .data import load_prices, synthetic_pair
from .live import LiveConfig, run_forever
from .optimize import optimize_pair
from .stats import engle_granger, scan_pairs
from .strategy import Params


def _prices(args) -> pd.DataFrame:
    if args.synthetic:
        return synthetic_pair(seed=args.seed)
    return load_prices(args.tickers, start=args.start, end=args.end, interval=args.interval)


def _params(args) -> Params:
    return Params(hedge_window=args.hedge_window, z_window=args.z_window, entry_z=args.entry_z,
                  exit_z=args.exit_z, stop_z=args.stop_z, max_hold=args.max_hold)


def _fmt(m: dict) -> str:
    pct = {"total_return", "cagr", "ann_vol", "max_drawdown", "time_in_market", "total_cost"}
    return "\n".join(f"  {k:<15} {v:>10.2%}" if k in pct else f"  {k:<15} {v:>10.2f}" for k, v in m.items())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pairtrader", description="Cointegration pairs trading lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data_args(p):
        p.add_argument("tickers", nargs="*", default=["KO", "PEP"], help="yfinance symbols, e.g. KO PEP or BTC-USD ETH-USD")
        p.add_argument("--start", default="2018-01-01")
        p.add_argument("--end", default=None)
        p.add_argument("--interval", default="1d")
        p.add_argument("--synthetic", action="store_true", help="use a generated cointegrated pair (offline)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--cost-bps", type=float, default=5.0, help="cost per leg traded, basis points")

    def strat_args(p):
        d = Params()
        for k, v in d.to_dict().items():
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)

    p_scan = sub.add_parser("scan", help="find cointegrated pairs among tickers")
    data_args(p_scan)
    p_scan.add_argument("--max-pvalue", type=float, default=0.05)

    p_bt = sub.add_parser("backtest", help="backtest one pair (first two tickers: Y X)")
    data_args(p_bt)
    strat_args(p_bt)
    p_bt.add_argument("--save", help="write the per-bar frame to this CSV")

    p_opt = sub.add_parser("optimize", help="grid search on train, report out-of-sample test")
    data_args(p_opt)
    p_opt.add_argument("--train-frac", type=float, default=0.7)
    p_opt.add_argument("--score", default="sharpe", choices=["sharpe", "total_return", "calmar", "cagr"])
    p_opt.add_argument("--grid", help='JSON dict overriding the grid, e.g. \'{"entry_z":[1.5,2,3]}\'')
    p_opt.add_argument("--note", default="", help="free-text note saved to the experiment log")

    p_live = sub.add_parser("paper", help="run the paper-trading loop forever")
    data_args(p_live)
    strat_args(p_live)
    p_live.add_argument("--poll-seconds", type=int, default=3600)
    p_live.add_argument("--history-period", default="60d")
    p_live.add_argument("--allocation", type=float, default=0.5)
    p_live.add_argument("--max-drawdown", type=float, default=0.15)
    p_live.add_argument("--broker", choices=["paper", "alpaca"], default="paper")

    args = ap.parse_args(argv)

    if args.cmd == "scan":
        prices = _prices(args)
        found = scan_pairs(prices, args.max_pvalue)
        print("No cointegrated pairs found." if found.empty else found.to_string(index=False))

    elif args.cmd == "backtest":
        prices = _prices(args)
        yc, xc = prices.columns[:2]
        print(f"Pair {yc}/{xc}  bars={len(prices)}  {prices.index[0].date()} -> {prices.index[-1].date()}")
        print("Full-sample cointegration (descriptive only):", json.dumps({k: round(v, 4) for k, v in engle_granger(prices[yc], prices[xc]).items()}))
        res = run_backtest(prices[yc], prices[xc], _params(args), cost_bps=args.cost_bps)
        print(_fmt(res.metrics))
        if not res.trades.empty:
            print(f"\nLast trades:\n{res.trades.tail(10).to_string(index=False)}")
        if args.save:
            res.frame.to_csv(args.save)

    elif args.cmd == "optimize":
        prices = _prices(args)
        yc, xc = prices.columns[:2]
        grid = json.loads(args.grid) if args.grid else None
        if grid:
            from .optimize import DEFAULT_GRID
            grid = {**DEFAULT_GRID, **grid}
        out = optimize_pair(prices, yc, xc, grid=grid, train_frac=args.train_frac,
                            score=args.score, cost_bps=args.cost_bps, note=args.note)
        print("Top 10 on TRAIN:")
        cols = list(Params().to_dict())[:-1] + ["sharpe", "total_return", "max_drawdown", "n_trades"]
        print(out["leaderboard"][cols].head(10).to_string(index=False))
        r = out["record"]
        print(f"\nBest params: {r['best_params']}")
        print(f"TRAIN: {r['train_metrics']}")
        print(f"TEST (out-of-sample, the honest number): {r['test_metrics']}")
        print("Logged to results/experiments.jsonl")

    elif args.cmd == "paper":
        yc, xc = args.tickers[:2]
        cfg = LiveConfig(y=yc, x=xc, params=_params(args), interval=args.interval,
                         history_period=args.history_period, poll_seconds=args.poll_seconds,
                         allocation=args.allocation, max_drawdown=args.max_drawdown,
                         broker=args.broker, cost_bps=args.cost_bps)
        run_forever(cfg)


if __name__ == "__main__":
    main()
