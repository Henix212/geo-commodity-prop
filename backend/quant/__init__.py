"""Market data fetch, signals, and backtests."""

__all__ = [
    "fetch_ohlcv",
    "load_price_frame",
    "refresh_all_metals",
    "refresh_core_futures",
    "refresh_prices",
    "resolve_ticker",
    "tension_to_signal",
    "run_backtest",
    "run_scenario",
]


def __getattr__(name: str):
    if name in {
        "fetch_ohlcv",
        "load_price_frame",
        "refresh_all_metals",
        "refresh_core_futures",
        "refresh_prices",
        "resolve_ticker",
    }:
        from backend.quant import market_data

        return getattr(market_data, name)
    if name == "tension_to_signal":
        from backend.quant.strategy import tension_to_signal

        return tension_to_signal
    if name == "run_backtest":
        from backend.quant.backtest import run_backtest

        return run_backtest
    if name == "run_scenario":
        from backend.quant.scenarios import run_scenario

        return run_scenario
    raise AttributeError(name)
