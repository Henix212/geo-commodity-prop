"""VectorBT smoke backtest from tension → signals."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd

from backend import config
from backend.quant.market_data import load_price_frame, refresh_prices
from backend.quant.strategy import tension_to_signal

logger = logging.getLogger(__name__)


def _synthetic_tension_series(close: pd.Series, *, seed: int = 0) -> pd.Series:
    """Proxy tension from rolling vol z-score mapped to [0, 1] (smoke path)."""
    rng = np.random.default_rng(seed)
    ret = close.pct_change().fillna(0.0)
    vol = ret.rolling(20, min_periods=5).std().fillna(ret.std() or 0.01)
    z = (ret.abs() / (vol + 1e-8)).clip(0, 5) / 5.0
    noise = rng.normal(0, 0.05, size=len(z))
    tension = (0.5 * z + 0.5 * (0.5 + noise)).clip(0.0, 1.0)
    return pd.Series(tension, index=close.index, name="tension")


def run_backtest(
    commodity: str,
    *,
    tension: pd.Series | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    if refresh:
        refresh_prices(commodity, period="2y")
    df = load_price_frame(commodity, start=start, end=end)
    if df.empty or "close" not in df.columns:
        raise RuntimeError(f"No prices for {commodity}; run market_data refresh first")

    close = df["close"].astype(float)
    if tension is None:
        tension = _synthetic_tension_series(close)
    else:
        tension = tension.reindex(close.index).ffill().fillna(0.5)

    positions = []
    for ts, score in tension.items():
        sig = tension_to_signal({commodity: float(score)}, commodity=commodity)
        positions.append(float(sig["position"]) if sig else 0.0)
    pos = pd.Series(positions, index=close.index, name="position")

    try:
        import os

        os.environ.setdefault("PLOTLY_RENDERER", "json")
        import vectorbt as vbt

        entries = pos > 0
        exits = pos <= 0
        short_entries = pos < 0
        short_exits = pos >= 0
        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            freq="1D",
            init_cash=100_000.0,
            fees=0.0,
        )
        stats = pf.stats()
        sharpe = float(stats.get("Sharpe Ratio", np.nan)) if stats is not None else float("nan")
        max_dd = float(stats.get("Max Drawdown [%]", np.nan)) if stats is not None else float("nan")
        total_ret = float(stats.get("Total Return [%]", np.nan)) if stats is not None else float("nan")
        turnover = float(pos.diff().abs().sum())
        return {
            "commodity": commodity,
            "ticker": config.TICKERS.get(commodity),
            "n_bars": len(close),
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "total_return_pct": total_ret,
            "turnover": turnover,
            "engine": "vectorbt",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("vectorbt failed (%s); using numpy fallback", exc)
        rets = close.pct_change().fillna(0.0)
        strat = pos.shift(1).fillna(0.0) * rets
        equity = (1.0 + strat).cumprod()
        mu, sigma = float(strat.mean()), float(strat.std() or 1e-9)
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0.0
        peak = equity.cummax()
        dd = ((equity / peak) - 1.0).min()
        return {
            "commodity": commodity,
            "ticker": config.TICKERS.get(commodity),
            "n_bars": len(close),
            "sharpe": float(sharpe),
            "max_drawdown_pct": float(dd * 100.0),
            "total_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
            "turnover": float(pos.diff().abs().sum()),
            "engine": "numpy",
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VectorBT tension smoke backtest")
    parser.add_argument("--commodity", default="copper")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(run_backtest(args.commodity, refresh=args.refresh))


if __name__ == "__main__":
    main()
