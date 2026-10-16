import numpy as np
import pandas as pd
import pytest

from pairtrader import live
from pairtrader.backtest import run_backtest
from pairtrader.data import synthetic_pair
from pairtrader.optimize import optimize_pair
from pairtrader.stats import engle_granger, scan_pairs
from pairtrader.strategy import Params, compute_signals, next_position


@pytest.fixture
def pair():
    return synthetic_pair(n=1200, seed=1)


def test_synthetic_pair_is_cointegrated(pair):
    res = engle_granger(pair.Y, pair.X)
    assert res["pvalue"] < 0.01
    assert res["hedge_ratio"] == pytest.approx(1.5, abs=0.1)
    assert 5 < res["half_life"] < 40


def test_random_walks_are_not_cointegrated():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": 100 + rng.normal(size=1000).cumsum(), "B": 100 + rng.normal(size=1000).cumsum()})
    assert engle_granger(df.A, df.B)["pvalue"] > 0.05
    assert scan_pairs(df).empty


def test_rule_buys_cheap_spread_and_sells_rich_spread():
    p = Params(entry_z=2, exit_z=0.5, stop_z=4)
    assert next_position(0, -2.5, 0, p)[0] == 1.0     # spread low -> buy y / sell x
    assert next_position(0, 2.5, 0, p)[0] == -1.0     # spread high -> sell y / buy x
    assert next_position(1, -1.0, 3, p)[0] == 1.0     # still stretched -> hold
    assert next_position(1, -0.2, 3, p)[0] == 0.0     # reverted -> exit
    assert next_position(1, -4.5, 3, p)[0] == 0.0     # blew through stop -> exit
    assert next_position(0, 4.5, 0, p)[0] == 0.0      # don't enter beyond the stop


def test_no_lookahead(pair):
    """Signals at bar t must not change when future bars are appended."""
    p = Params()
    full = compute_signals(pair.Y, pair.X, p)
    part = compute_signals(pair.Y.iloc[:800], pair.X.iloc[:800], p)
    pd.testing.assert_series_equal(full.position.iloc[:800], part.position)


def test_strategy_profits_on_cointegrated_pair(pair):
    res = run_backtest(pair.Y, pair.X, Params(), cost_bps=2)
    assert res.metrics["n_trades"] > 5
    assert res.metrics["sharpe"] > 0.5


def test_costs_reduce_returns(pair):
    cheap = run_backtest(pair.Y, pair.X, cost_bps=0).metrics["total_return"]
    pricey = run_backtest(pair.Y, pair.X, cost_bps=50).metrics["total_return"]
    assert pricey < cheap


def test_optimize_logs_out_of_sample(pair, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = optimize_pair(pair, "Y", "X", grid={"hedge_window": [60, 120], "z_window": [30],
                                               "entry_z": [1.5, 2.0], "exit_z": [0.5], "stop_z": [4.0]})
    rec = out["record"]
    assert "test_metrics" in rec and rec["test"][0] > rec["train"][1]
    assert (tmp_path / "results" / "experiments.jsonl").exists()


def test_live_step_with_paper_broker(pair, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(live, "STATE_DIR", tmp_path / "live")
    cfg = live.LiveConfig(y="Y", x="X", params=Params(hedge_window=60, z_window=30, entry_z=1.0))
    broker = live.PaperBroker("test")
    state = {}
    for t in range(200, 400):                 # replay history bar by bar
        state = live.step(cfg, broker, state, prices=pair.iloc[:t])
    trades_happened = (tmp_path / "live" / "test_broker.json").exists()
    assert trades_happened
    eq = broker.equity({"Y": pair.Y.iloc[399], "X": pair.X.iloc[399]})
    assert 50_000 < eq < 150_000


def test_stop_file_flattens(pair, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(live, "STATE_DIR", tmp_path / "live")
    cfg = live.LiveConfig(y="Y", x="X", params=Params(hedge_window=60, z_window=30, entry_z=0.1))
    broker = live.PaperBroker("stop")
    state = live.step(cfg, broker, {}, prices=pair.iloc[:300])
    (tmp_path / "STOP").touch()
    state = live.step(cfg, broker, state, prices=pair.iloc[:301])
    assert state["halted"] and not broker.get_positions()
