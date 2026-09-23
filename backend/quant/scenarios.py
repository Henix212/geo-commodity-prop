"""Stress scenarios: inject preset shocks → tension report."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from backend import config
from backend.graph_core import load_sector
from backend.ingestion.inject_demo import inject_demo
from backend.models.infer import infer_tensions
from backend.quant.strategy import tension_to_signal

logger = logging.getLogger(__name__)

SCENARIOS: dict[str, dict[str, Any]] = {
    "escondida_outage": {
        "sector": "metals",
        "node": "mine_escondida",
        "commodity": "copper",
        "severity": 0.95,
        "event_type": "outage",
        "direction": "supply_down",
        "description": "Escondida mine outage",
    },
    "panama_blockade": {
        "sector": "metals",
        "node": "chokepoint_panama",
        "commodity": "copper",
        "severity": 0.9,
        "event_type": "blockade",
        "direction": "transit_block",
        "description": "Panama Canal blockade",
    },
    "suez_closure": {
        "sector": "energy",
        "node": "chokepoint_suez",
        "commodity": "oil",
        "severity": 0.92,
        "event_type": "blockade",
        "direction": "transit_block",
        "description": "Suez Canal closure",
    },
}


def run_scenario(
    name: str,
    *,
    use_gat: bool = False,
    persist: bool = False,
) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}; choose from {list(SCENARIOS)}")
    spec = SCENARIOS[name]
    network = load_sector(spec["sector"], validate=True)
    network, event, injected = inject_demo(
        sector=spec["sector"],
        node_id=spec["node"],
        commodity=spec["commodity"],
        severity=spec["severity"],
        event_type=spec["event_type"],
        direction=spec["direction"],
        persist=persist,
        network=network,
    )
    tensions = infer_tensions(network, use_gat=use_gat)
    signal = tension_to_signal(tensions, commodity=spec["commodity"])
    report = {
        "scenario": name,
        "description": spec["description"],
        "node": spec["node"],
        "commodity": spec["commodity"],
        "injected": injected,
        "event": event.to_dict(),
        "tensions": tensions,
        "signal": signal,
    }
    logger.info("scenario %s tensions=%s signal=%s", name, tensions, signal)
    return report


def run_all(*, use_gat: bool = False) -> list[dict[str, Any]]:
    return [run_scenario(name, use_gat=use_gat) for name in SCENARIOS]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stress scenario runner")
    parser.add_argument(
        "--scenario",
        default="all",
        help="escondida_outage | panama_blockade | suez_closure | all",
    )
    parser.add_argument("--use-gat", action="store_true")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.scenario == "all":
        print(run_all(use_gat=args.use_gat))
    else:
        print(run_scenario(args.scenario, use_gat=args.use_gat, persist=args.persist))


if __name__ == "__main__":
    main()
