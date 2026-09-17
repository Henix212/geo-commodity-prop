"""Map commodity tension scores to long / short / flat signals."""

from __future__ import annotations

from typing import Any

from backend import config


def tension_to_side(tension: float) -> str:
    if tension >= config.TENSION_LONG_THRESHOLD:
        return "long"
    if tension <= config.TENSION_SHORT_THRESHOLD:
        return "short"
    return "flat"


def make_signals(tensions: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Return per-commodity signal dicts with ticker + side."""
    out: dict[str, dict[str, Any]] = {}
    for commodity, tension in tensions.items():
        side = tension_to_side(float(tension))
        out[commodity] = {
            "commodity": commodity,
            "tension": float(tension),
            "side": side,
            "ticker": config.TICKERS.get(commodity),
            "long_threshold": config.TENSION_LONG_THRESHOLD,
            "short_threshold": config.TENSION_SHORT_THRESHOLD,
        }
    return out


def primary_signal(
    tensions: dict[str, float],
    commodity: str | None = None,
) -> dict[str, Any] | None:
    commodity = commodity or config.DEFAULT_COMMODITY
    if not tensions:
        return None
    if commodity in tensions:
        return make_signals({commodity: tensions[commodity]})[commodity]
    # fall back to highest tension commodity
    best = max(tensions.items(), key=lambda x: x[1])
    return make_signals({best[0]: best[1]})[best[0]]
