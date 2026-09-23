"""GNN tension → long/short/flat trade signals."""

from __future__ import annotations

from typing import Any

from backend import config


def tension_to_signal(
    tensions: dict[str, float],
    *,
    commodity: str | None = None,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
) -> dict[str, Any] | None:
    """Map commodity tension scores to a trade signal.

    High tension (supply stress) → long commodity futures.
    Low tension → short / flat.
    """
    if not tensions:
        return None
    long_th = config.TENSION_LONG_THRESHOLD if long_threshold is None else long_threshold
    short_th = config.TENSION_SHORT_THRESHOLD if short_threshold is None else short_threshold
    commodity = commodity or config.DEFAULT_COMMODITY

    if commodity in tensions:
        score = float(tensions[commodity])
    else:
        score = float(max(tensions.values()))
        commodity = max(tensions, key=tensions.get)  # type: ignore[arg-type]

    if score >= long_th:
        side = "long"
        position = 1.0
    elif score <= short_th:
        side = "short"
        position = -1.0
    else:
        side = "flat"
        position = 0.0

    size = min(1.0, abs(score - 0.5) * 2.0)
    return {
        "commodity": commodity,
        "ticker": config.TICKERS.get(commodity),
        "tension": score,
        "side": side,
        "position": position,
        "size": float(size * abs(position)),
        "long_threshold": long_th,
        "short_threshold": short_th,
        "all_tensions": dict(tensions),
    }


def tensions_to_position_series(
    tension_by_date: dict[str, float],
    *,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
) -> dict[str, float]:
    """Date → position in {-1, 0, 1}."""
    out: dict[str, float] = {}
    for day, score in sorted(tension_by_date.items()):
        sig = tension_to_signal(
            {"_": float(score)},
            commodity="_",
            long_threshold=long_threshold,
            short_threshold=short_threshold,
        )
        out[day] = float(sig["position"]) if sig else 0.0
    return out
