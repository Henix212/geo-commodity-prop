"""Persist dashboard snapshot for the Streamlit UI (read-only consumer)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config

logger = logging.getLogger(__name__)


def dashboard_state_path() -> Path:
    config.ensure_dirs()
    return Path(config.DASHBOARD_STATE_PATH)


def write_dashboard_state(
    *,
    sector: str,
    network: dict,
    commodity_tensions: dict[str, float],
    node_tensions: dict[str, float] | None = None,
    signal: dict[str, Any] | None = None,
    signals: dict[str, Any] | None = None,
    inference_backend: str = "diffusion",
    events: list[dict] | None = None,
    backtests: dict[str, Any] | None = None,
    compute_backtests: bool = True,
) -> Path:
    """Write ``data/dashboard_state.json`` for the Streamlit frontend."""
    from backend.quant.strategy import make_signals

    node_tensions = node_tensions or {}
    shocked = []
    for node in network.get("nodes") or []:
        sev = abs(float(node.get("event_severity") or 0.0))
        if sev <= 0 and node_tensions.get(node["id"], 0.0) <= 0.05:
            continue
        shocked.append(
            {
                "node_id": node["id"],
                "type": node.get("type"),
                "commodity": node.get("commodity"),
                "event_severity": float(node.get("event_severity") or 0.0),
                "event_type": node.get("event_type"),
                "tension": float(node_tensions.get(node["id"], 0.0)),
            }
        )
    shocked.sort(key=lambda r: (-abs(r["event_severity"]), -r["tension"]))

    if backtests is None and compute_backtests:
        backtests = _compute_backtest_metrics(commodity_tensions)

    payload = {
        "sector": sector,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "inference_backend": inference_backend,
        "commodities": list(network.get("supported_commodities") or []),
        "commodity_tensions": {k: float(v) for k, v in commodity_tensions.items()},
        "node_tensions": {k: float(v) for k, v in node_tensions.items()},
        "signal": signal,
        "signals": signals or make_signals(commodity_tensions),
        "shocked_nodes": shocked[:40],
        "n_nodes": len(network.get("nodes") or []),
        "n_edges": len(network.get("edges") or []),
        "n_events": len(events or []),
        "event_severities": {
            n["id"]: float(n.get("event_severity") or 0.0)
            for n in (network.get("nodes") or [])
            if abs(float(n.get("event_severity") or 0.0)) > 0
        },
        "backtests": backtests or {},
    }

    path = dashboard_state_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Dashboard state written → %s", path)
    return path


def _compute_backtest_metrics(commodity_tensions: dict[str, float]) -> dict[str, Any]:
    """Run constant-tension smoke backtests for snapshot commodities."""
    from backend.quant.backtest import backtest_commodity

    out: dict[str, Any] = {}
    for commodity, tension in commodity_tensions.items():
        if commodity not in config.TICKERS:
            out[commodity] = {"commodity": commodity, "error": "no_ticker", "n": 0}
            continue
        try:
            m = backtest_commodity(commodity, constant_tension=float(tension))
            out[commodity] = {
                k: m.get(k)
                for k in (
                    "commodity",
                    "ticker",
                    "engine",
                    "n",
                    "sharpe",
                    "max_drawdown",
                    "total_return",
                    "turnover",
                    "error",
                )
                if k in m or k == "error"
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("snapshot backtest failed for %s: %s", commodity, exc)
            out[commodity] = {"commodity": commodity, "error": str(exc)}
    return out


def load_dashboard_state() -> dict[str, Any] | None:
    path = dashboard_state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Corrupt dashboard state at %s", path)
        return None
