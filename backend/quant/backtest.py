"""Smoke backtest: VectorBT when available, else numpy/pandas metrics."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd

from backend import config
from backend.logging_setup import configure_logging
from backend.quant.market_data import load_price_frame
from backend.quant.strategy import tension_to_side

logger = logging.getLogger(__name__)


def _positions_from_tension_series(tension: pd.Series) -> pd.Series:
    sides = tension.map(tension_to_side)
    return sides.map({"long": 1.0, "short": -1.0, "flat": 0.0}).astype(float)


def _numpy_backtest(close: pd.Series, positions: pd.Series) -> dict[str, Any]:
    """Sharpe / max DD / turnover without VectorBT (fallback)."""
    pos = positions.reindex(close.index).fillna(0.0)
    rets = close.pct_change().fillna(0.0)
    # position known at close_t applied to return_{t+1}
    strat = (pos.shift(1).fillna(0.0) * rets) - 0.0005 * pos.diff().abs().fillna(0.0)
    equity = (1.0 + strat).cumprod()
    mu = float(strat.mean())
    sigma = float(strat.std(ddof=0)) or 1e-12
    sharpe = mu / sigma * np.sqrt(252.0)
    peak = equity.cummax()
    dd = (equity / peak - 1.0).min()
    turnover = float((pos.diff().fillna(0.0).abs() > 0).mean())
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0
    return {
        "engine": "numpy",
        "n": int(len(close)),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd) * 100.0,
        "total_return": float(total_return) * 100.0,
        "turnover": turnover,
    }


def _vectorbt_backtest(close: pd.Series, positions: pd.Series) -> dict[str, Any]:
    import vectorbt as vbt

    entries = positions == 1
    exits = positions != 1
    short_entries = positions == -1
    short_exits = positions != -1
    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        freq="1D",
        init_cash=100_000,
        fees=0.0005,
    )
    stats = pf.stats()
    turnover = float((positions.diff().fillna(0).abs() > 0).mean())
    return {
        "engine": "vectorbt",
        "n": int(len(close)),
        "sharpe": _stat(stats, "Sharpe Ratio"),
        "max_drawdown": _stat(stats, "Max Drawdown [%]"),
        "total_return": _stat(stats, "Total Return [%]"),
        "turnover": turnover,
    }


def _stat(stats: pd.Series, key: str) -> float | None:
    if key in stats.index:
        try:
            return float(stats[key])
        except (TypeError, ValueError):
            return None
    for idx in stats.index:
        if key.lower() in str(idx).lower():
            try:
                return float(stats[idx])
            except (TypeError, ValueError):
                return None
    return None


def backtest_commodity(
    commodity: str,
    *,
    venue: str | None = None,
    constant_tension: float | None = None,
    tension_series: pd.Series | None = None,
) -> dict[str, Any]:
    prices = load_price_frame(commodity, venue=venue)
    if prices.empty or "close" not in prices.columns:
        return {"commodity": commodity, "error": "no_prices", "n": 0}

    close = prices["close"].astype(float).dropna()
    if constant_tension is not None:
        tension = pd.Series(constant_tension, index=close.index)
    elif tension_series is not None:
        tension = tension_series.reindex(close.index).ffill().fillna(0.5)
    else:
        ret = close.pct_change().abs()
        tension = ret.rolling(20, min_periods=5).mean().fillna(0.0)
        mx = float(tension.max()) or 1.0
        tension = (tension / mx).clip(0, 1)

    positions = _positions_from_tension_series(tension)
    try:
        metrics = _vectorbt_backtest(close, positions)
    except Exception as exc:  # noqa: BLE001
        logger.warning("VectorBT unavailable (%s) — using numpy fallback", exc)
        metrics = _numpy_backtest(close, positions)

    return {
        "commodity": commodity,
        "ticker": config.TICKERS.get(commodity),
        **metrics,
    }


def run_smoke_backtest(
    commodities: list[str] | None = None,
    *,
    tension: float = 0.7,
) -> list[dict[str, Any]]:
    commodities = commodities or ["copper", "gold", "oil"]
    results = []
    for c in commodities:
        try:
            results.append(backtest_commodity(c, constant_tension=tension))
        except Exception as exc:  # noqa: BLE001
            logger.warning("backtest failed for %s: %s", c, exc)
            results.append({"commodity": c, "error": str(exc)})
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke backtest (VectorBT or numpy)")
    parser.add_argument("--commodity", default="copper")
    parser.add_argument("--tension", type=float, default=0.7)
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    result = backtest_commodity(args.commodity, constant_tension=args.tension)
    print(result)


if __name__ == "__main__":
    main()
