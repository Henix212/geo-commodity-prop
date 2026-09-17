"""Betweenness centrality → single points of failure."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import networkx as nx

from backend import config
from backend.graph_core import to_networkx

logger = logging.getLogger(__name__)


def node_betweenness(network: dict, *, top_k: int = 15) -> list[dict[str, Any]]:
    g = to_networkx(network, validate=False)
    if g.number_of_nodes() == 0:
        return []
    # Use undirected for SPOF-style bottlenecks; weight by inverse edge weight if present
    u = g.to_undirected()
    raw = nx.betweenness_centrality(u, weight=None, normalized=True)
    ranked = sorted(raw.items(), key=lambda x: -x[1])[:top_k]
    by_id = {n["id"]: n for n in network.get("nodes") or []}
    out = []
    for nid, score in ranked:
        meta = by_id.get(nid) or {}
        out.append(
            {
                "node_id": nid,
                "betweenness": float(score),
                "type": meta.get("type"),
                "commodity": meta.get("commodity"),
                "country": meta.get("country"),
            }
        )
    return out


def edge_betweenness(network: dict, *, top_k: int = 15) -> list[dict[str, Any]]:
    g = to_networkx(network, validate=False)
    if g.number_of_edges() == 0:
        return []
    u = g.to_undirected()
    raw = nx.edge_betweenness_centrality(u, normalized=True)
    ranked = sorted(raw.items(), key=lambda x: -x[1])[:top_k]
    out = []
    for edge, score in ranked:
        if isinstance(edge, tuple) and len(edge) >= 2:
            src, dst = edge[0], edge[1]
        else:
            continue
        out.append({"source": src, "target": dst, "betweenness": float(score)})
    return out


def single_points_of_failure(network: dict, *, top_k: int = 15) -> dict[str, Any]:
    return {
        "nodes": node_betweenness(network, top_k=top_k),
        "edges": edge_betweenness(network, top_k=top_k),
    }


def main(argv: list[str] | None = None) -> None:
    from backend.logging_setup import configure_logging
    from backend.main import build_graph

    parser = argparse.ArgumentParser(description="Betweenness / SPOF analysis")
    parser.add_argument("--sector", default=config.DEFAULT_SECTOR)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    network = build_graph(args.sector, apply_reference=True)
    result = single_points_of_failure(network, top_k=args.top_k)
    print("top_nodes:")
    for row in result["nodes"]:
        print(
            f"  {row['betweenness']:.4f}  {row['node_id']}  "
            f"type={row.get('type')} commodity={row.get('commodity')}"
        )
    print("top_edges:")
    for row in result["edges"][:10]:
        print(f"  {row['betweenness']:.4f}  {row['source']} -> {row['target']}")


if __name__ == "__main__":
    main()
