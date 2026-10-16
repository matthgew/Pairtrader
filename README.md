# pairtrader

A small, honest lab for **cointegration pairs trading**, the classic stat-arb idea:
find two assets whose prices are tied together over the long run, and when the gap
between them stretches unusually far, bet on it closing.

* Gap too **wide** (spread high, z > entry): **sell** the one that ran up, **buy** the one that lagged.
* Gap too **narrow** (spread low, z < -entry): the reverse.
* Close the trade when the gap returns toward normal, or bail out at a stop if it keeps widening.

You trade the *spread*, not either asset alone. "Sell when it goes up, buy when it goes
down" on a single market is a different strategy and loses badly when the market trends.

Built so Claude (or you) can run experiments, tune parameters and paper-trade it around
the clock, with guardrails against fooling yourself.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows (use .venv/bin/pip on mac/linux)
pytest -q
```

> Windows tip: if `pip` fails with "No such file or directory ... long path", put the
> venv in a short path (e.g. `C:\venvs\pairtrader`) or enable Windows long paths.

## Commands

```bash
# Find cointegrated pairs in a basket
python -m pairtrader scan XOM CVX COP EWA EWC GLD GDX --start 2021-01-01

# Backtest one pair (Y X) with chosen params
python -m pairtrader backtest KO PEP --entry-z 2 --exit-z 0.5 --z-window 60

# Tune on the first 70%, then score the untouched last 30%
python -m pairtrader optimize KO PEP --score sharpe --note "baseline"

# Hourly crypto (trades 24/7)
python -m pairtrader optimize BTC-USD ETH-USD --interval 1h --start 2025-01-01

# Offline sanity check on a pair that is cointegrated by construction
python -m pairtrader backtest --synthetic

# Paper trade forever (simulated money, state survives restarts)
python -m pairtrader paper BTC-USD ETH-USD --interval 1h --poll-seconds 3600
```

Every `optimize` run is appended to `results/experiments.jsonl` with its train **and**
test metrics, so you can see over time which ideas held up.

## How it works

| File | What it does |
|---|---|
| `pairtrader/stats.py` | Engle-Granger cointegration test, OLS hedge ratio, half-life, pair scanner |
| `pairtrader/strategy.py` | Rolling hedge ratio -> spread -> z-score -> entry/exit/stop rule (`next_position`) |
| `pairtrader/backtest.py` | Dollar-weighted backtest with per-leg costs, no look-ahead, metrics and trade log |
| `pairtrader/optimize.py` | Grid search on train, out-of-sample test, experiment log |
| `pairtrader/live.py` | Paper-trading loop that reuses the exact same rule, with kill switches |
| `pairtrader/data.py` | yfinance loader (cached) and synthetic cointegrated pair generator |

## What the first runs showed

On data through 2026-09-30, with 5 bps cost per leg:

| Pair | Train (tuned) | Test (out-of-sample) |
|---|---|---|
| KO / PEP, daily 2018–2026 | +9.3%, Sharpe 0.35 | **−8.2%**, Sharpe −0.35 |
| BTC / ETH, hourly 2025–2026 | +23.1%, Sharpe 1.67 | **−4.7%**, Sharpe −1.30 |

This is normal. Textbook pairs have been arbitraged for decades, and a grid search
always finds *something* that worked in the past. The test column is the one to trust.
Making real money requires finding a pair and a rule that hold up out of sample,
across several periods, after realistic costs.

## Running it 24/7

* **Crypto** trades around the clock; stocks only during market hours (the loop still
  runs, it just sees no new bars overnight).
* The loop has to run on a machine that stays on: your PC with sleep disabled, a cheap
  VPS, or a Raspberry Pi. Start it with `python -m pairtrader paper ...` inside
  `tmux`/`screen`, or as a Windows Task Scheduler / systemd service.
* `--broker alpaca` sends orders to a free **Alpaca paper account** instead of the local
  simulator (`pip install alpaca-py`, set `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`).
* Kill switches: create a file named `STOP` in the working directory, or let the
  `--max-drawdown` limit (default 15% from peak) flatten everything and exit.
* Logs: `results/live/<Y>_<X>_log.csv`.

## Going live (real money)

Not implemented on purpose. Before considering it: paper-trade for months, compare live
fills to the backtest, and size it at money you can afford to lose. Switching the Alpaca
client to `paper=False` is a deliberate one-line change you should make yourself, with
your own judgement. Nothing here is financial advice.
