"""Always-on paper-trading loop.

Every `poll_seconds` it pulls fresh prices, recomputes the spread z-score,
applies the same `next_position` rule the backtest uses, and rebalances a
broker to the target holdings.

SAFETY: only paper brokers are implemented. Real-money trading is deliberately
not wired up - see README "Going live" before you even think about it.

Kill switches:
  * create a file named STOP in the working directory -> flatten and exit
  * equity falls `max_drawdown` below its peak         -> flatten and exit
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import Params, next_position, rolling_hedge_ratio

STATE_DIR = Path("results/live")


# ---------------------------------------------------------------- brokers

class PaperBroker:
    """Simulated broker that persists cash/positions to JSON between restarts."""

    def __init__(self, name: str, starting_cash: float = 100_000.0, cost_bps: float = 5.0):
        self.path = STATE_DIR / f"{name}_broker.json"
        self.cost_bps = cost_bps
        if self.path.exists():
            s = json.loads(self.path.read_text())
            self.cash, self.positions = s["cash"], s["positions"]
        else:
            self.cash, self.positions = starting_cash, {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"cash": self.cash, "positions": self.positions}, indent=2))

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, q in self.positions.items())

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def order(self, symbol: str, qty: float, price: float) -> None:
        """Buy (qty>0) or sell (qty<0) at price, paying cost_bps of notional."""
        slip = 1 + np.sign(qty) * self.cost_bps / 1e4
        self.cash -= qty * price * slip
        self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        if abs(self.positions[symbol]) < 1e-9:
            del self.positions[symbol]
        self._save()


class AlpacaPaperBroker:
    """Alpaca *paper* account (free). Needs `pip install alpaca-py` and env vars
    ALPACA_API_KEY / ALPACA_SECRET_KEY from a PAPER account. paper=True is hardcoded."""

    def __init__(self, crypto: bool = False):
        from alpaca.trading.client import TradingClient
        self.client = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
        self.crypto = crypto

    def _sym(self, s: str) -> str:
        return s.replace("-", "/") if self.crypto else s   # BTC-USD -> BTC/USD

    def equity(self, prices: dict[str, float]) -> float:
        return float(self.client.get_account().equity)

    def get_positions(self) -> dict[str, float]:
        out = {}
        for p in self.client.get_all_positions():
            sym = p.symbol.replace("/", "-")
            if self.crypto and "-" not in sym and sym.endswith("USD"):
                sym = sym[:-3] + "-USD"          # BTCUSD -> BTC-USD
            out[sym] = float(p.qty)
        return out

    def order(self, symbol: str, qty: float, price: float) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        if not self.crypto:
            qty = float(int(qty))   # whole shares keeps it simple for equities
            if qty == 0:
                return
        self.client.submit_order(MarketOrderRequest(
            symbol=self._sym(symbol), qty=round(abs(qty), 6),
            side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if self.crypto else TimeInForce.DAY,
        ))


# ---------------------------------------------------------------- loop

@dataclass
class LiveConfig:
    y: str
    x: str
    params: Params = field(default_factory=Params)
    interval: str = "1h"            # yfinance bar size: 1m,5m,15m,1h,1d
    history_period: str = "60d"     # how much history to pull each poll
    poll_seconds: int = 3600
    allocation: float = 0.5         # fraction of equity committed (gross) to the pair
    max_drawdown: float = 0.15      # kill switch
    min_trade_value: float = 50.0   # skip tiny rebalances
    broker: str = "paper"           # paper | alpaca
    cost_bps: float = 5.0


def _fetch(cfg: LiveConfig) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download([cfg.y, cfg.x], period=cfg.history_period, interval=cfg.interval,
                      auto_adjust=True, progress=False)
    return raw["Close"][[cfg.y, cfg.x]].dropna()


def _log(name: str, row: dict) -> None:
    path = STATE_DIR / f"{name}_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def step(cfg: LiveConfig, broker, state: dict, prices: pd.DataFrame | None = None) -> dict:
    """Run one decision cycle. Returns the updated state. Pure enough to unit-test."""
    p = cfg.params
    prices = _fetch(cfg) if prices is None else prices
    need = max(p.hedge_window, p.z_window, p.coint_window) + 2
    if len(prices) < need:
        raise ValueError(f"Need {need} bars of history, got {len(prices)}; increase history_period.")

    y, x = prices[cfg.y], prices[cfg.x]
    betas = rolling_hedge_ratio(y, x, p.hedge_window)
    spread = y - betas * x                      # same construction as the backtest
    window = spread.iloc[-p.z_window:]
    beta = float(betas.iloc[-1])
    z = float((spread.iloc[-1] - window.mean()) / window.std())

    allow = True
    if p.coint_window:
        from statsmodels.tsa.stattools import coint
        allow = coint(y.iloc[-p.coint_window:], x.iloc[-p.coint_window:])[1] <= p.coint_pvalue
    new_pos, held = next_position(state.get("position", 0.0), z, state.get("held", 0), p, allow)
    py, px = float(y.iloc[-1]), float(x.iloc[-1])
    last = {cfg.y: py, cfg.x: px}
    eq = broker.equity(last)
    peak = max(state.get("peak_equity", eq), eq)

    if eq < peak * (1 - cfg.max_drawdown) or Path("STOP").exists():
        new_pos, state["halted"] = 0.0, True

    # Dollar-neutral-ish target: gross notional = allocation * equity
    gross = py + abs(beta) * px
    units = new_pos * cfg.allocation * eq / gross
    target = {cfg.y: units, cfg.x: -units * beta}
    current = broker.get_positions()
    for sym, tgt in target.items():
        delta = tgt - current.get(sym, 0.0)
        if abs(delta * last[sym]) >= cfg.min_trade_value or (tgt == 0 and current.get(sym)):
            broker.order(sym, delta, last[sym])

    row = {"time": pd.Timestamp.now(tz="UTC").isoformat(), "bar": str(prices.index[-1]),
           cfg.y: py, cfg.x: px, "beta": round(beta, 4), "z": round(z, 3),
           "position": new_pos, "equity": round(broker.equity(last), 2),
           "halted": state.get("halted", False)}
    _log(f"{cfg.y}_{cfg.x}", row)
    state.update(position=new_pos, held=held, peak_equity=peak, last=row)
    return state


def run_forever(cfg: LiveConfig) -> None:
    name = f"{cfg.y}_{cfg.x}"
    state_path = STATE_DIR / f"{name}_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if cfg.broker == "alpaca":
        broker = AlpacaPaperBroker(crypto=cfg.y.endswith("-USD"))
    else:
        broker = PaperBroker(name, cost_bps=cfg.cost_bps)

    print(f"Paper trading {cfg.y}/{cfg.x} every {cfg.poll_seconds}s. Create a file named STOP to halt.")
    while True:
        try:
            state = step(cfg, broker, state)
            print(state["last"])
        except Exception as e:  # keep running through flaky data feeds
            print(f"[warn] {type(e).__name__}: {e}")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, default=str))
        if state.get("halted"):
            print("Halted (kill switch). Positions flattened. Delete STOP / reset state to resume.")
            return
        time.sleep(cfg.poll_seconds)
