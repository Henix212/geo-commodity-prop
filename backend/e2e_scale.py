"""Large-scale realistic e2e: multi-shock events, 2y prices, multi-sector tension, backtests.

Usage:
  uv run python -m backend.e2e_scale
  uv run python -m backend.e2e_scale --skip-prices   # reuse cached OHLCV
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from backend import config
from backend.dashboard_state import write_dashboard_state
from backend.database import connect, init_db, list_events
from backend.database.seed_data import seed_all, seed_events
from backend.ingestion.parser_llm import inject_events_into_network
from backend.logging_setup import configure_logging
from backend.main import build_graph
from backend.models.baseline_diffusion import run_diffusion
from backend.models.centrality import single_points_of_failure
from backend.quant.backtest import backtest_commodity
from backend.quant.strategy import make_signals

logger = logging.getLogger(__name__)

SECTORS = ("metals", "energy", "agriculture")
PRICE_COMMODITIES = ("copper", "aluminum", "gold", "silver", "oil", "lng", "wheat", "corn")


def _db_counts() -> dict[str, int]:
    conn = connect()
    init_db(conn)
    out = {
        "articles": conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
        "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        "prices": conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0],
        "graph_nodes": conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0],
        "graph_edges": conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0],
    }
    conn.close()
    return out


def refresh_prices_scale(*, period: str = "2y") -> list[dict[str, Any]]:
    from backend.quant.market_data import refresh_prices

    summaries = []
    for commodity in PRICE_COMMODITIES:
        try:
            summaries.append(refresh_prices(commodity, period=period))
        except Exception as exc:  # noqa: BLE001
            logger.warning("price refresh failed for %s: %s", commodity, exc)
            summaries.append({"commodity": commodity, "error": str(exc)})
    return summaries


def run_sector_inference(sector: str) -> dict[str, Any]:
    network = build_graph(sector, apply_reference=True, sync_mirror=True)
    events = list_events(limit=500)
    injected = inject_events_into_network(network, events, min_severity=0.0)
    result = run_diffusion(network)
    return {
        "sector": sector,
        "network": network,
        "events": events,
        "injected_nodes": injected,
        "commodity_tensions": result["commodity_tensions"],
        "node_tensions": result["node_tensions"],
        "n_nodes": len(network["nodes"]),
        "n_edges": len(network["edges"]),
    }


def merge_tensions(sector_results: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    commodity: dict[str, float] = {}
    nodes: dict[str, float] = {}
    for res in sector_results:
        for k, v in res["commodity_tensions"].items():
            commodity[k] = max(float(v), float(commodity.get(k, 0.0)))
        for k, v in res["node_tensions"].items():
            nodes[k] = max(float(v), float(nodes.get(k, 0.0)))
    return commodity, nodes


def run_scale_test(
    *,
    skip_prices: bool = False,
    price_period: str = "2y",
    train_epochs: int = 0,
) -> dict[str, Any]:
    started = time.time()
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {},
        "ok": True,
        "failures": [],
    }

    # 1) Reference + multi-shock events
    seeds = seed_all(include_events=True)
    report["steps"]["seed"] = seeds

    # 2) Prices at scale
    if not skip_prices:
        report["steps"]["prices"] = refresh_prices_scale(period=price_period)
    else:
        report["steps"]["prices"] = {"skipped": True}

    # 3) Multi-sector diffusion with all DB events injected
    sector_results = [run_sector_inference(s) for s in SECTORS]
    commodity_tensions, node_tensions = merge_tensions(sector_results)
    report["steps"]["inference"] = {
        s["sector"]: {
            "n_nodes": s["n_nodes"],
            "n_edges": s["n_edges"],
            "injected_nodes": s["injected_nodes"],
            "commodity_tensions": s["commodity_tensions"],
        }
        for s in sector_results
    }
    report["commodity_tensions"] = commodity_tensions

    # Primary map = metals (richest topology for dashboard)
    metals = next(s for s in sector_results if s["sector"] == "metals")
    signals = make_signals(commodity_tensions)
    write_dashboard_state(
        sector="metals",
        network=metals["network"],
        commodity_tensions=commodity_tensions,
        node_tensions=node_tensions,
        signal=signals.get(config.DEFAULT_COMMODITY),
        signals=signals,
        inference_backend="diffusion",
        events=metals["events"],
    )
    report["steps"]["dashboard_state"] = str(config.DASHBOARD_STATE_PATH)

    # 4) Optional short GAT train on metals
    if train_epochs > 0:
        from backend.models.train import train_gat

        report["steps"]["train"] = train_gat(metals["network"], epochs=train_epochs)

    # 5) Centrality SPOF
    report["steps"]["spof"] = {
        "top_nodes": single_points_of_failure(metals["network"], top_k=10)["nodes"]
    }

    # 6) Backtests driven by inferred commodity tension
    bt = []
    for commodity in ("copper", "oil", "gold", "wheat", "aluminum", "lng"):
        tension = float(commodity_tensions.get(commodity, 0.5))
        # map mid tensions toward actionable long for smoke (keep real score)
        try:
            bt.append(backtest_commodity(commodity, constant_tension=max(tension, 0.5)))
        except Exception as exc:  # noqa: BLE001
            bt.append({"commodity": commodity, "error": str(exc)})
    report["steps"]["backtests"] = bt

    # 7) Counts + assertions
    counts = _db_counts()
    report["counts"] = counts
    shocked = [
        n["id"]
        for n in metals["network"]["nodes"]
        if abs(float(n.get("event_severity") or 0.0)) >= 0.3
    ]
    report["shocked_node_ids"] = shocked
    report["n_shocked_nodes"] = len(shocked)

    checks = {
        "events_ge_15": counts["events"] >= 15,
        "prices_ge_500": counts["prices"] >= 500,
        "graph_nodes_ge_100": counts["graph_nodes"] >= 100,
        "copper_tension_ge_0_5": float(commodity_tensions.get("copper", 0)) >= 0.5,
        "oil_or_hormuz_tension": float(commodity_tensions.get("oil", 0)) >= 0.2
        or float(node_tensions.get("chokepoint_hormuz", 0)) >= 0.2,
        "shocked_nodes_ge_5": len(shocked) >= 5,
        "backtests_ran": sum(1 for r in bt if "error" not in r) >= 3,
    }
    report["checks"] = checks
    report["ok"] = all(checks.values())
    report["failures"] = [k for k, v in checks.items() if not v]
    report["elapsed_sec"] = round(time.time() - started, 2)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    config.ensure_dirs()
    out_path = config.DATA_DIR / "scale_test_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Large-scale realistic e2e test")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--price-period", default="2y")
    parser.add_argument("--train-epochs", type=int, default=0, help="Optional GAT epochs on metals")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    report = run_scale_test(
        skip_prices=args.skip_prices,
        price_period=args.price_period,
        train_epochs=args.train_epochs,
    )
    print(json.dumps({
        "ok": report["ok"],
        "failures": report["failures"],
        "counts": report["counts"],
        "commodity_tensions": report["commodity_tensions"],
        "n_shocked_nodes": report["n_shocked_nodes"],
        "elapsed_sec": report["elapsed_sec"],
        "report_path": report["report_path"],
        "backtests": [
            {k: r.get(k) for k in ("commodity", "engine", "n", "sharpe", "max_drawdown", "total_return", "error") if k in r or k == "error"}
            for r in report["steps"]["backtests"]
        ],
    }, indent=2, default=str))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
