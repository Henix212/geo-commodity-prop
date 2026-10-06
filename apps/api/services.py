"""Shared helpers for dashboard API routes."""

from __future__ import annotations

from typing import Any

from geo_commodity.store import list_events
from geo_commodity.graph import load_network
from geo_commodity.events.parser_llm import inject_events_into_network


def load_live_network(sector: str = "all") -> tuple[dict, int]:
    """Load sector/merged network and inject persisted events (with decay)."""
    network = load_network(sector, validate=False)
    rows = list_events(limit=2000)
    injected = inject_events_into_network(network, rows)
    return network, len(rows)


def serialize_graph(network: dict) -> dict[str, Any]:
    nodes_out: list[dict[str, Any]] = []
    for n in network.get("nodes") or []:
        lat, lon = n.get("lat"), n.get("lon")
        if lat is None or lon is None:
            continue
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        nodes_out.append(
            {
                "id": n.get("id"),
                "type": n.get("type") or "unknown",
                "lat": lat_f,
                "lon": lon_f,
                "event_severity": float(n.get("event_severity") or 0.0),
                "commodity": n.get("event_commodity") or n.get("commodity"),
                "event_type": n.get("event_type"),
            }
        )
    id_set = {n["id"] for n in nodes_out}
    edges_out: list[dict[str, Any]] = []
    for e in network.get("edges") or []:
        src, tgt = e.get("source"), e.get("target")
        if src not in id_set or tgt not in id_set:
            continue
        edges_out.append(
            {
                "source": src,
                "target": tgt,
                "flow_type": e.get("flow_type") or "unknown",
            }
        )
    return {
        "sector": network.get("sector"),
        "network_name": network.get("network_name"),
        "commodities": network.get("supported_commodities") or [],
        "nodes": nodes_out,
        "edges": edges_out,
    }


def top_stressed_nodes(network: dict, *, k: int = 10) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for n in network.get("nodes") or []:
        sev = abs(float(n.get("event_severity") or 0.0))
        if sev <= 0:
            continue
        scored.append(
            {
                "id": n.get("id"),
                "severity": sev,
                "type": n.get("type"),
                "commodity": n.get("event_commodity"),
            }
        )
    scored.sort(key=lambda x: x["severity"], reverse=True)
    return scored[:k]
