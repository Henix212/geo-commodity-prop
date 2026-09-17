"""Market data fetch, signals, and backtests."""

__all__ = [
    "fetch_ohlcv",
    "load_price_frame",
    "refresh_all_metals",
    "refresh_prices",
    "resolve_ticker",
]


def __getattr__(name: str):
    if name in __all__:
        from backend.quant import market_data

        return getattr(market_data, name)
    raise AttributeError(name)
