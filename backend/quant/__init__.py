"""Market data fetch, signals, and backtests."""

__all__ = [
    "fetch_ohlcv",
    "load_price_frame",
    "refresh_all_metals",
    "refresh_prices",
    "resolve_ticker",
    "make_signals",
    "primary_signal",
    "backtest_commodity",
    "run_scenario",
]


def __getattr__(name: str):
    if name in {
        "fetch_ohlcv",
        "load_price_frame",
        "refresh_all_metals",
        "refresh_prices",
        "resolve_ticker",
    }:
        from backend.quant import market_data

        return getattr(market_data, name)
    if name in {"make_signals", "primary_signal"}:
        from backend.quant import strategy

        return getattr(strategy, name)
    if name in {"backtest_commodity"}:
        from backend.quant import backtest

        return getattr(backtest, name)
    if name in {"run_scenario"}:
        from backend.quant import scenarios

        return getattr(scenarios, name)
    raise AttributeError(name)
