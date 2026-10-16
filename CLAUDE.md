# Working in this repo (instructions for Claude)

This is a research lab for cointegration pairs trading. The owner wants you to run
experiments to find strategies that actually make money. Your job is to find edges that
are **real**, not ones that only exist in the backtest.

## Experiment loop

1. Read `results/experiments.jsonl` first so you don't repeat what's been tried.
2. Form a hypothesis (new pair, new universe, new rule, different bar size) and write it in `--note`.
3. Run `python -m pairtrader optimize ...` or edit the code and run it.
4. Judge it by the **test** metrics only. Train metrics are just the tuning set.
5. Summarise for the owner: what you tried, train vs test, and whether it's worth paper trading.

## Rules that keep results honest

* Never tune on, peek at, or pick parameters using the test slice. If you re-run
  the optimiser many times on the same test slice, it slowly becomes a train slice.
  Say so when that's happening, and hold out a fresher period when you can.
* Prefer Sharpe/Calmar over raw return; raw return rewards leverage and luck.
* Require a decent trade count (>= ~20 out of sample) before claiming anything.
* Keep realistic costs (>= 5 bps per leg for liquid US stocks, >= 10 bps for crypto).
* Scanning many pairs produces false positives; a pair that's only cointegrated in
  one window is a coincidence.
* Any new strategy logic must go through `next_position` (or a function like it) so the
  backtest and the live loop stay identical, and must pass `tests/test_core.py::test_no_lookahead`.
* Run `pytest -q` after code changes.

## Hard limits

* Do not switch any broker to real money, set `paper=False`, or add a live broker.
  Only the owner does that.
* Do not commit API keys; they live in environment variables or `.env` (gitignored).

## Ideas worth trying

* Kalman-filter hedge ratio instead of rolling OLS
* Johansen test / baskets of 3+ assets
* Re-test cointegration on a rolling window and only trade while it holds
* Half-life-based z_window and max_hold
* Volatility-scaled position sizing
* Walk-forward optimisation (re-tune every N months, chain the test periods)
* Sector ETF pairs, share classes (GOOG/GOOGL), country ETFs, crypto majors
