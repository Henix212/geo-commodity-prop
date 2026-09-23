"""Align events × prices → supervised tension / direction labels."""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from backend import config
from backend.database import list_events
from backend.quant.market_data import load_price_frame

logger = logging.getLogger(__name__)


def forward_return(
    commodity: str,
    as_of: str | pd.Timestamp,
    *,
    days: int | None = None,
    venue: str | None = None,
) -> float | None:
    """Close-to-close return over ``days`` trading bars after ``as_of``."""
    days = config.FORWARD_RETURN_DAYS if days is None else days
    df = load_price_frame(commodity, venue=venue)
    if df.empty or "close" not in df.columns:
        return None
    ts = pd.Timestamp(str(as_of)[:10])
    idx = int(df.index.searchsorted(ts))
    if idx >= len(df) - 1:
        return None
    if idx < len(df) and df.index[idx] < ts:
        idx += 1
    j = min(idx + days, len(df) - 1)
    if j <= idx:
        return None
    c0 = float(df["close"].iloc[idx])
    c1 = float(df["close"].iloc[j])
    if c0 == 0 or pd.isna(c0) or pd.isna(c1):
        return None
    return (c1 - c0) / c0


def return_to_tension(ret: float, *, scale: float = 20.0) -> float:
    """Map return → [0, 1] tension."""
    return float(1.0 / (1.0 + math.exp(-scale * ret)))


def labels_from_events(
    *,
    limit: int = 500,
    days: int | None = None,
) -> list[dict[str, Any]]:
    """Build labeled rows from persisted events + cached prices."""
    days = config.FORWARD_RETURN_DAYS if days is None else days
    rows = list_events(limit=limit)
    out: list[dict[str, Any]] = []
    for ev in rows:
        commodity = ev.get("commodity")
        extracted = ev.get("extracted_at") or ""
        if not commodity or not extracted:
            continue
        ret = forward_return(commodity, extracted, days=days)
        if ret is None:
            continue
        direction = ev.get("direction") or "neutral"
        signed = ret
        if direction in ("supply_up", "demand_down"):
            signed = -ret
        out.append(
            {
                "event": ev,
                "commodity": commodity,
                "forward_return": ret,
                "tension_label": return_to_tension(signed),
                "direction_label": 1 if signed > 0 else (-1 if signed < 0 else 0),
            }
        )
    logger.info("labels_from_events: %d/%d usable", len(out), len(rows))
    return out


def commodity_label_vector(
    commodity: str,
    tension: float,
    order: list[str] | None = None,
) -> list[float]:
    order = order or list(config.TICKERS.keys())
    return [float(tension) if c == commodity else 0.0 for c in order]
