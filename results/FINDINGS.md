# Findings

Running summary of what the experiments say. The raw numbers are in `experiments.jsonl` and `batch_*.csv`.

## 2026-10-16: batch 1 (baseline) and batch 2 (cointegration filter)

**Setup:** 17 candidate pairs, daily bars from 2015-01-01 to 2026-09-30, realistic costs
(3–10 bps per leg). Walk-forward testing: tune on 3 years, trade the next 6 months, roll
forward, then chain the out-of-sample blocks (about 2018–2026). Rerun with:

```bash
python experiments/batch.py --walk-forward --note "batch1 baseline"
python experiments/batch.py --walk-forward --coint-filter --note "batch2 coint filter"
```

**Batch 1, plain z-score rule:** mean walk-forward Sharpe was **−0.23**, and only 4 of 17
pairs made money. The best was GOOGL/GOOG at Sharpe 0.49, but that's only +2.9% over 8
years. Many pairs that scored well on the tuning data lost on unseen data: XOM/CVX went
from train Sharpe 0.52 to walk-forward −0.49, and BP/SHEL from 0.53 to −0.51.

**Batch 2, only open trades while the trailing 250-day window is cointegrated (p ≤ 0.10):**
mean walk-forward Sharpe was **−0.07**, and 5 of 15 pairs made money (2 pairs were skipped
for too few trades). Drawdowns shrank a lot; for example EWA/EWC went from −18% to −3.5%.

| Pair | Baseline WF Sharpe | Filtered WF Sharpe | Filtered trades |
|---|---|---|---|
| EWA/EWC | −0.14 | **0.46** | 12 |
| V/MA | −0.41 | **0.40** | 11 |
| SPY/RSP | −0.60 | 0.29 | 3 |
| GOOGL/GOOG | 0.49 | 0.15 | 23 |

**What this means**

* No pair shows a tradeable edge yet. The filter's improvement comes mostly from *trading
  less*: it sits out the periods when the relationship breaks down. That's valuable as
  risk control, but 11–12 trades over 8 years is far too few to tell skill from luck.
* EWA/EWC and V/MA with the filter are the only candidates worth a closer look. Next
  steps for them: hourly bars (more trades), a test on 2010–2015 data, and paper trading.
* Plain z-score pairs trading on well-known pairs is not profitable after costs in this
  period. That matches what's widely reported since the early 2000s.

## Ideas queue (next)

1. A bigger universe with train-only selection: scan ~100 tickers within each sector,
   keep the pairs cointegrated on train, and test them as a portfolio. Many small edges
   could add up to something.
2. Hourly bars for EWA/EWC, V/MA and GOOGL/GOOG (more trades, so stronger statistics).
3. A Kalman-filter hedge ratio.
4. Volatility-scaled sizing, and running several pairs at once.
