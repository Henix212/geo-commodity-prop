"""Inject a synthetic shock event onto a graph node (no LLM)."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from backend import config
from backend.database import init_db, upsert_events
from backend.graph_core import load_sector
from backend.ingestion.parser_llm import ShockEvent, inject_events_into_network

logger = logging.getLogger(__name__)

DEFAULTS = {
    "node": "mine_escondida",
    "commodity": "copper",
    "severity": 0.9,
    "event_type": "outage",
    "direction": "supply_down",
    "confidence": 1.0,
}


def make_demo_event(
    *,
    node_id: str = DEFAULTS["node"],
    commodity: str = DEFAULTS["commodity"],
    severity: float = DEFAULTS["severity"],
    event_type: str = DEFAULTS["event_type"],
    direction: str = DEFAULTS["direction"],
    confidence: float = DEFAULTS["confidence"],
    article_uid: str | None = None,
) -> ShockEvent:
    uid = article_uid or f"demo:{node_id}:{commodity}:{event_type}"
    return ShockEvent(
        entity_text=node_id,
        event_type=event_type,
        severity=float(severity),
        commodity=commodity,
        direction=direction,
        confidence=float(confidence),
        article_uid=uid,
        node_id=node_id,
        match_score=1.0,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def inject_demo(
    *,
    sector: str | None = None,
    node_id: str = DEFAULTS["node"],
    commodity: str = DEFAULTS["commodity"],
    severity: float = DEFAULTS["severity"],
    event_type: str = DEFAULTS["event_type"],
    direction: str = DEFAULTS["direction"],
    persist: bool = True,
    network: dict | None = None,
) -> tuple[dict, ShockEvent, int]:
    """Build (or reuse) network, upsert demo event, inject severity onto node."""
    config.ensure_dirs()
    init_db()
    sector = sector or config.DEFAULT_SECTOR
    network = network if network is not None else load_sector(sector, validate=True)
    node_ids = {n["id"] for n in network.get("nodes") or []}
    if node_id not in node_ids:
        raise KeyError(f"node_id={node_id!r} not in sector={sector!r}")

    event = make_demo_event(
        node_id=node_id,
        commodity=commodity,
        severity=severity,
        event_type=event_type,
        direction=direction,
    )
    if persist:
        upsert_events([event])
    injected = inject_events_into_network(network, [event], min_severity=0.0)
    logger.info(
        "demo inject node=%s commodity=%s severity=%.2f injected=%d",
        node_id,
        commodity,
        severity,
        injected,
    )
    return network, event, injected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inject synthetic shock onto a node")
    parser.add_argument("--sector", default=config.DEFAULT_SECTOR)
    parser.add_argument("--node", default=DEFAULTS["node"])
    parser.add_argument("--commodity", default=DEFAULTS["commodity"])
    parser.add_argument("--severity", type=float, default=DEFAULTS["severity"])
    parser.add_argument("--event-type", default=DEFAULTS["event_type"])
    parser.add_argument("--direction", default=DEFAULTS["direction"])
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    network, event, injected = inject_demo(
        sector=args.sector,
        node_id=args.node,
        commodity=args.commodity,
        severity=args.severity,
        event_type=args.event_type,
        direction=args.direction,
        persist=not args.no_persist,
    )
    node = next(n for n in network["nodes"] if n["id"] == event.node_id)
    print(
        f"demo ok sector={args.sector} node={event.node_id} "
        f"severity={node.get('event_severity')} injected={injected} "
        f"persist={not args.no_persist}"
    )


if __name__ == "__main__":
    main()
