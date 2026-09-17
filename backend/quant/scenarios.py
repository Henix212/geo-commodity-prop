"""Named stress scenarios: inject shocks → tension → signal."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Any

from backend import config
from backend.ingestion.parser_llm import ShockEvent, inject_events_into_network
from backend.logging_setup import configure_logging
from backend.models.baseline_diffusion import run_diffusion
from backend.models.gat import load_checkpoint, run_gat
from backend.quant.strategy import make_signals

logger = logging.getLogger(__name__)

SCENARIOS: dict[str, dict[str, Any]] = {
    "escondida_outage": {
        "description": "Escondida mine force majeure / strike",
        "events": [
            {
                "entity_text": "Escondida",
                "node_id": "mine_escondida",
                "event_type": "force_majeure",
                "severity": 0.9,
                "commodity": "copper",
                "direction": "supply_down",
                "confidence": 0.95,
            }
        ],
    },
    "panama_blockade": {
        "description": "Panama Canal transit blockade / severe congestion",
        "events": [
            {
                "entity_text": "Panama Canal",
                "node_id": "chokepoint_panama",
                "event_type": "blockade",
                "severity": 0.85,
                "commodity": "copper",
                "direction": "transit_block",
                "confidence": 0.9,
            }
        ],
    },
    "suez_closure": {
        "description": "Suez Canal closure",
        "events": [
            {
                "entity_text": "Suez Canal",
                "node_id": "chokepoint_suez",
                "event_type": "blockade",
                "severity": 0.9,
                "commodity": "oil",
                "direction": "transit_block",
                "confidence": 0.95,
            }
        ],
    },
    "compound_crisis": {
        "description": "Concurrent copper + energy + grain shockwave",
        "events": [
            {
                "entity_text": "Escondida",
                "node_id": "mine_escondida",
                "event_type": "strike",
                "severity": 0.9,
                "commodity": "copper",
                "direction": "supply_down",
                "confidence": 0.95,
            },
            {
                "entity_text": "Panama Canal",
                "node_id": "chokepoint_panama",
                "event_type": "blockade",
                "severity": 0.8,
                "commodity": "copper",
                "direction": "transit_block",
                "confidence": 0.9,
            },
            {
                "entity_text": "Hormuz",
                "node_id": "chokepoint_hormuz",
                "event_type": "blockade",
                "severity": 0.92,
                "commodity": "oil",
                "direction": "transit_block",
                "confidence": 0.95,
            },
            {
                "entity_text": "Black Sea wheat",
                "node_id": "region_black_sea_wheat",
                "event_type": "sanction",
                "severity": 0.7,
                "commodity": "wheat",
                "direction": "supply_down",
                "confidence": 0.8,
            },
        ],
    },
}


def _resolve_node_id(network: dict, preferred: str, substrings: list[str]) -> str | None:
    ids = {n["id"] for n in network.get("nodes") or []}
    if preferred in ids:
        return preferred
    for nid in ids:
        low = nid.lower()
        if any(s in low for s in substrings):
            return nid
    return None


def build_scenario_events(network: dict, name: str) -> list[ShockEvent]:
    spec = SCENARIOS[name]
    now = datetime.now(timezone.utc).isoformat()
    events: list[ShockEvent] = []
    for raw in spec["events"]:
        node_id = raw.get("node_id")
        if node_id and node_id not in {n["id"] for n in network.get("nodes") or []}:
            # fuzzy resolve chokepoints
            if "panama" in (node_id or ""):
                node_id = _resolve_node_id(network, node_id, ["panama"])
            elif "suez" in (node_id or ""):
                node_id = _resolve_node_id(network, node_id, ["suez"])
            elif "escondida" in (node_id or ""):
                node_id = _resolve_node_id(network, node_id, ["escondida"])
        events.append(
            ShockEvent(
                entity_text=raw["entity_text"],
                event_type=raw["event_type"],
                severity=float(raw["severity"]),
                commodity=raw["commodity"],
                direction=raw["direction"],
                confidence=float(raw["confidence"]),
                article_uid=f"scenario:{name}",
                node_id=node_id,
                match_score=1.0,
                extracted_at=now,
            )
        )
    return events


def run_scenario(
    network: dict,
    name: str,
    *,
    backend: str = "diffusion",
) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name}; choose from {list(SCENARIOS)}")
    events = build_scenario_events(network, name)
    injected = inject_events_into_network(network, events, min_severity=0.0)
    if backend == "gat":
        model = load_checkpoint()
        scores = run_gat(network, model=model)
    else:
        scores = run_diffusion(network)
    signals = make_signals(scores["commodity_tensions"])
    return {
        "scenario": name,
        "description": SCENARIOS[name]["description"],
        "injected_nodes": injected,
        "events": [e.to_dict() for e in events],
        "commodity_tensions": scores["commodity_tensions"],
        "signals": signals,
    }


def main(argv: list[str] | None = None) -> None:
    from backend.main import build_graph

    parser = argparse.ArgumentParser(description="Run stress scenarios")
    parser.add_argument(
        "--scenario",
        default="escondida_outage",
        choices=list(SCENARIOS),
    )
    parser.add_argument("--sector", default=None, help="Default sector from scenario commodity")
    parser.add_argument("--backend", choices=("diffusion", "gat"), default="diffusion")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    sector = args.sector
    if sector is None:
        commodity = SCENARIOS[args.scenario]["events"][0]["commodity"]
        if commodity in ("oil", "lng"):
            sector = "energy"
        elif commodity in ("wheat", "corn"):
            sector = "agriculture"
        else:
            sector = "metals"

    network = build_graph(sector)
    result = run_scenario(network, args.scenario, backend=args.backend)
    print(result["scenario"], result["description"])
    print("tensions:", result["commodity_tensions"])
    print("signals:", {k: v["side"] for k, v in result["signals"].items()})


if __name__ == "__main__":
    main()
