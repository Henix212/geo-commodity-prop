"""Periodic refresh jobs: prices, production seed, health snapshot."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from backend.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_job_run(record: dict[str, Any]) -> None:
    config.ensure_dirs()
    path = config.DATA_DIR / "job_runs.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_health(updates: dict[str, Any]) -> Path:
    config.ensure_dirs()
    path = config.DATA_DIR / "health.json"
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(updates)
    current["updated_at"] = _now()
    current["db_path"] = str(config.DB_PATH)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return path


def job_refresh_prices(*, period: str = "5d") -> dict[str, Any]:
    from backend.quant.market_data import refresh_all_metals

    started = _now()
    try:
        summary = refresh_all_metals(period=period)
        record = {
            "job": "prices",
            "ok": True,
            "started_at": started,
            "finished_at": _now(),
            "summary": summary,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("price refresh failed")
        record = {
            "job": "prices",
            "ok": False,
            "started_at": started,
            "finished_at": _now(),
            "error": str(exc),
        }
    _append_job_run(record)
    write_health({"last_price_refresh": record})
    return record


def job_refresh_production() -> dict[str, Any]:
    from backend.database.seed_data import seed_production

    started = _now()
    try:
        summary = seed_production()
        record = {
            "job": "production",
            "ok": True,
            "started_at": started,
            "finished_at": _now(),
            "summary": summary,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("production refresh failed")
        record = {
            "job": "production",
            "ok": False,
            "started_at": started,
            "finished_at": _now(),
            "error": str(exc),
        }
    _append_job_run(record)
    write_health({"last_production_refresh": record})
    return record


def job_sync_graph(sector: str | None = None) -> dict[str, Any]:
    from backend.database import sync_graph_mirror
    from backend.main import build_graph

    sector = sector or config.DEFAULT_SECTOR
    started = _now()
    try:
        network = build_graph(sector, apply_reference=True)
        result = sync_graph_mirror(network)
        record = {
            "job": "sync_graph",
            "ok": True,
            "started_at": started,
            "finished_at": _now(),
            "sector": sector,
            "summary": {
                "inserted": result.inserted,
                "total": result.total,
                "n_nodes": len(network["nodes"]),
                "n_edges": len(network["edges"]),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("graph sync failed")
        record = {
            "job": "sync_graph",
            "ok": False,
            "started_at": started,
            "finished_at": _now(),
            "error": str(exc),
        }
    _append_job_run(record)
    write_health({"last_graph_sync": record})
    return record


def job_snapshot(
    sector: str | None = None,
    *,
    inference_backend: str | None = None,
) -> dict[str, Any]:
    """Inject persisted events, run inference, write dashboard_state.json (no LLM)."""
    from backend.main import run

    sector = sector or config.DEFAULT_SECTOR
    started = _now()
    try:
        result = run(
            sector=sector,
            skip_ingest=True,
            inference_backend=inference_backend or "diffusion",
        )
        record = {
            "job": "snapshot",
            "ok": True,
            "started_at": started,
            "finished_at": _now(),
            "sector": sector,
            "summary": {
                "n_events": result.get("n_events"),
                "tensions": result.get("tensions"),
                "signal": result.get("signal"),
                "inference_backend": result.get("inference_backend"),
                "dashboard_state": str(config.DASHBOARD_STATE_PATH),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("dashboard snapshot failed")
        record = {
            "job": "snapshot",
            "ok": False,
            "started_at": started,
            "finished_at": _now(),
            "error": str(exc),
        }
    _append_job_run(record)
    write_health({"last_dashboard_snapshot": record})
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Periodic geo-commodity jobs")
    parser.add_argument(
        "job",
        choices=("prices", "production", "sync_graph", "snapshot", "all"),
    )
    parser.add_argument("--sector", default=config.DEFAULT_SECTOR)
    parser.add_argument("--period", default="5d")
    parser.add_argument(
        "--inference",
        choices=("auto", "diffusion", "gat"),
        default="diffusion",
    )
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    results = []
    if args.job in ("prices", "all"):
        results.append(job_refresh_prices(period=args.period))
    if args.job in ("production", "all"):
        results.append(job_refresh_production())
    if args.job in ("sync_graph", "all"):
        results.append(job_sync_graph(args.sector))
    if args.job in ("snapshot", "all"):
        results.append(job_snapshot(args.sector, inference_backend=args.inference))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
