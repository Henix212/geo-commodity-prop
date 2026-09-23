"""Fetch & cache LME / SHFE / COMEX (and peers) price time series via yfinance."""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from backend import config
from backend.database import init_db, list_prices, upsert_prices

logger = logging.getLogger(__name__)


def resolve_ticker(commodity: str, venue: str | None = None) -> tuple[str, str]:
    """Return (yahoo_ticker, venue). Default venue = first configured for commodity."""
    venues = config.VENUE_TICKERS.get(commodity) or {}
    if venue:
        ticker = venues.get(venue) or config.TICKERS.get(commodity)
        if not ticker:
            raise KeyError(f"No ticker for {commodity}/{venue}")
        return ticker, venue
    if venues:
        v0, t0 = next(iter(venues.items()))
        return t0, v0
    ticker = config.TICKERS.get(commodity)
    if not ticker:
        raise KeyError(f"No ticker for {commodity}")
    return ticker, ""


def fetch_ohlcv(
    ticker: str,
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV from yfinance; index = dates, columns Open/High/Low/Close/Volume."""
    import os

    import yfinance as yf

    cache = config.DATA_DIR / "yfinance_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YF_CACHE_DIR", str(cache))

    kwargs: dict[str, Any] = {"auto_adjust": False, "progress": False}
    if period and not start:
        kwargs["period"] = period
    else:
        kwargs["start"] = start or (date.today() - timedelta(days=365 * 2)).isoformat()
        if end:
            kwargs["end"] = end

    logger.info("yfinance download %s %s", ticker, kwargs)
    df = yf.download(ticker, **kwargs)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    out = df[keep].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.dropna(how="all")


def dataframe_to_price_rows(
    df: pd.DataFrame,
    *,
    ticker: str,
    commodity: str,
    venue: str,
    source: str = "yfinance",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        day = pd.Timestamp(ts).date().isoformat()
        rows.append(
            {
                "ticker": ticker,
                "venue": venue,
                "commodity": commodity,
                "date": day,
                "open": _f(row.get("Open")),
                "high": _f(row.get("High")),
                "low": _f(row.get("Low")),
                "close": _f(row.get("Close")),
                "volume": _f(row.get("Volume")),
                "source": source,
            }
        )
    return rows


def _f(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def refresh_prices(
    commodity: str,
    *,
    venues: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> dict[str, Any]:
    """Fetch venue series for one commodity and upsert into SQLite ``prices``.

    Identical yahoo tickers shared across venues (LME/SHFE proxies) are downloaded
    once and stored once per venue label.
    """
    init_db()
    venue_map = config.VENUE_TICKERS.get(commodity) or {"": config.TICKERS[commodity]}
    selected = venues or list(venue_map.keys())

    # ticker -> list of venues using it
    by_ticker: dict[str, list[str]] = {}
    for venue in selected:
        ticker = venue_map.get(venue) or config.TICKERS.get(commodity)
        if not ticker:
            logger.warning("skip %s/%s — no ticker", commodity, venue)
            continue
        by_ticker.setdefault(ticker, []).append(venue)

    summary: dict[str, Any] = {"commodity": commodity, "venues": {}}
    for ticker, venue_list in by_ticker.items():
        df = fetch_ohlcv(ticker, start=start, end=end, period=period)
        if df.empty:
            for venue in venue_list:
                summary["venues"][venue] = {"ticker": ticker, "rows": 0}
            continue
        for venue in venue_list:
            rows = dataframe_to_price_rows(
                df, ticker=ticker, commodity=commodity, venue=venue
            )
            result = upsert_prices(rows)
            summary["venues"][venue] = {
                "ticker": ticker,
                "rows": len(rows),
                "inserted": result.inserted,
                "updated": result.updated,
            }
            logger.info(
                "%s/%s %s → %d bars (+%d ~%d)",
                commodity,
                venue,
                ticker,
                len(rows),
                result.inserted,
                result.updated,
            )
    return summary


def refresh_all_metals(
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "2y",
) -> list[dict[str, Any]]:
    out = []
    for commodity in ("copper", "aluminum", "gold", "silver"):
        out.append(
            refresh_prices(
                commodity,
                start=start,
                end=end,
                period=period if not start else None,
            )
        )
    return out


CORE_FUTURES = ("copper", "oil", "gold", "silver")  # HG=F, CL=F, GC=F, SI=F


def refresh_core_futures(
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "2y",
) -> list[dict[str, Any]]:
    """Refresh HG=F / CL=F / GC=F / SI=F proxies."""
    out = []
    for commodity in CORE_FUTURES:
        out.append(
            refresh_prices(
                commodity,
                start=start,
                end=end,
                period=period if not start else None,
            )
        )
    return out


def refresh_all_commodities(
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "2y",
) -> list[dict[str, Any]]:
    out = []
    for commodity in config.TICKERS:
        out.append(
            refresh_prices(
                commodity,
                start=start,
                end=end,
                period=period if not start else None,
            )
        )
    return out


def load_price_frame(
    commodity: str,
    *,
    venue: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load cached prices as a DataFrame indexed by date."""
    ticker, venue_resolved = resolve_ticker(commodity, venue)
    rows = list_prices(
        ticker=ticker,
        commodity=commodity,
        venue=venue or venue_resolved or None,
        start=start,
        end=end,
        limit=100000,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch/cache commodity price series")
    parser.add_argument(
        "--commodity",
        default="copper",
        help="Commodity key, or 'metals' | 'core' (HG/CL/GC/SI) | 'all'",
    )
    parser.add_argument("--venue", default=None, help="COMEX | LME | SHFE (optional)")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--period", default="2y", help="yfinance period if --start omitted")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.commodity == "metals":
        summary = refresh_all_metals(start=args.start, end=args.end, period=args.period)
        print(summary)
        return
    if args.commodity == "core":
        summary = refresh_core_futures(
            start=args.start, end=args.end, period=args.period
        )
        print(summary)
        return
    if args.commodity == "all":
        summary = refresh_all_commodities(
            start=args.start, end=args.end, period=args.period
        )
        print(summary)
        return

    venues = [args.venue] if args.venue else None
    summary = refresh_prices(
        args.commodity,
        venues=venues,
        start=args.start,
        end=args.end,
        period=args.period if not args.start else None,
    )
    print(summary)


if __name__ == "__main__":
    main()
