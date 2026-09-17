"""Ablation helpers: drop scrap hubs and/or maritime chokepoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _is_scrap(node: dict[str, Any]) -> bool:
    t = str(node.get("type") or "")
    nid = str(node.get("id") or "")
    return t == "scrap_hub" or nid.startswith("scrap_")


def _is_chokepoint(node: dict[str, Any]) -> bool:
    t = str(node.get("type") or "")
    nid = str(node.get("id") or "")
    return t == "bottleneck" or nid.startswith("chokepoint_") or "panama" in nid or "suez" in nid or "hormuz" in nid or "malacca" in nid or "bosphorus" in nid


def ablate_network(
    network: dict,
    *,
    drop_scrap: bool = False,
    drop_chokepoints: bool = False,
) -> dict:
    """Return a deep-copied network with selected node types removed (and dangling edges)."""
    if not drop_scrap and not drop_chokepoints:
        return network

    out = deepcopy(network)
    drop_ids: set[str] = set()
    kept_nodes = []
    for node in out.get("nodes") or []:
        if drop_scrap and _is_scrap(node):
            drop_ids.add(node["id"])
            continue
        if drop_chokepoints and _is_chokepoint(node):
            drop_ids.add(node["id"])
            continue
        kept_nodes.append(node)
    out["nodes"] = kept_nodes
    out["edges"] = [
        e
        for e in (out.get("edges") or [])
        if e.get("source") not in drop_ids and e.get("target") not in drop_ids
    ]
    return out
