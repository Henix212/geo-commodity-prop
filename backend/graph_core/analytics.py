"""NetworkX diffusion / random-walk baseline and centrality (SPOF)."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import networkx as nx
import numpy as np

from backend import config
from backend.graph_core import load_sector, to_networkx

logger = logging.getLogger(__name__)


def _edge_conductance(attrs: dict[str, Any]) -> float:
    weight = float(attrs.get("weight") or 1.0)
    transit = float(attrs.get("transit_days") or 0.0)
    return max(weight, 1e-6) / (1.0 + max(transit, 0.0))


def diffuse_severity(
    g: nx.MultiDiGraph | nx.DiGraph,
    *,
    steps: int | None = None,
    decay: float | None = None,
    attr: str = "event_severity",
) -> dict[str, float]:
    """Propagate node ``attr`` along outgoing edges (weighted diffusion)."""
    steps = config.DIFFUSION_STEPS if steps is None else steps
    decay = config.DIFFUSION_DECAY if decay is None else decay
    seeds = {n: float(g.nodes[n].get(attr) or 0.0) for n in g.nodes}
    scores = dict(seeds)
    if not scores:
        return {}

    for _ in range(steps):
        nxt = dict(scores)
        for u, v, _key, data in g.edges(keys=True, data=True):
            cond = _edge_conductance(data)
            # normalize conductance softly so a single hop does not explode
            contrib = decay * min(cond, 1.0) * scores.get(u, 0.0)
            # keep strongest absolute shock at destination
            if abs(nxt.get(v, 0.0) + contrib) >= abs(nxt.get(v, 0.0)):
                nxt[v] = nxt.get(v, 0.0) + contrib
        scores = {k: max(-1.0, min(1.0, v)) for k, v in nxt.items()}

    # never weaken original seed below its injected severity
    for n, seed in seeds.items():
        if abs(seed) >= abs(scores.get(n, 0.0)):
            scores[n] = seed
    return scores


def random_walk_tension(
    g: nx.MultiDiGraph | nx.DiGraph,
    *,
    steps: int | None = None,
    restart: float = 0.15,
    attr: str = "event_severity",
) -> dict[str, float]:
    """Personalized PageRank seeded by |event_severity|."""
    steps = config.DIFFUSION_STEPS if steps is None else steps
    seeds = {
        n: abs(float(g.nodes[n].get(attr) or 0.0))
        for n in g.nodes
        if abs(float(g.nodes[n].get(attr) or 0.0)) > 1e-9
    }
    if not seeds:
        return {n: 0.0 for n in g.nodes}
    total = sum(seeds.values())
    personalization = {n: seeds.get(n, 0.0) / total for n in g.nodes}
    # SimpleDiGraph view for pagerank
    simple = nx.DiGraph()
    for u, v, data in g.edges(data=True):
        w = _edge_conductance(data)
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += w
        else:
            simple.add_edge(u, v, weight=w)
    for n in g.nodes:
        simple.add_node(n)
    try:
        pr = nx.pagerank(
            simple,
            alpha=1.0 - restart,
            personalization=personalization,
            weight="weight",
            max_iter=max(100, steps * 20),
        )
    except nx.PowerIterationFailedConvergence:
        pr = personalization
    # sign by dominant seed direction
    signed_seed = sum(float(g.nodes[n].get(attr) or 0.0) for n in seeds)
    sign = 1.0 if signed_seed >= 0 else -1.0
    return {n: sign * float(pr.get(n, 0.0)) for n in g.nodes}


def aggregate_commodity_tension(
    network: dict,
    node_scores: dict[str, float],
    *,
    commodities: list[str] | None = None,
) -> dict[str, float]:
    """Map node tensions → per-commodity scores in [0, 1].

    Prefer nodes tagged with ``event_commodity``; else top abs scores on
    commodity-owned nodes; else 0.
    """
    commodities = commodities or list(network.get("supported_commodities") or [])
    by_id = {n["id"]: n for n in network.get("nodes") or []}
    out: dict[str, float] = {}
    for c in commodities:
        event_hit = [
            abs(node_scores.get(nid, 0.0))
            for nid, node in by_id.items()
            if node.get("event_commodity") == c
        ]
        if event_hit:
            top = sorted(event_hit, reverse=True)[:5]
            score = float(max(top))
        else:
            owned = [
                abs(node_scores.get(nid, 0.0))
                for nid, node in by_id.items()
                if c in (node.get("commodities") or []) or c == node.get("commodity")
            ]
            # only count meaningful bleed (>0.05) so idle commodities stay ~0
            owned = [v for v in owned if v >= 0.05]
            score = float(max(owned)) if owned else 0.0
        out[c] = float(max(0.0, min(1.0, score)))
    return out


def tension_from_network(
    network: dict,
    *,
    method: str = "diffusion",
    steps: int | None = None,
    decay: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (commodity_tensions, node_scores)."""
    g = to_networkx(network, validate=False)
    if method == "random_walk":
        node_scores = random_walk_tension(g, steps=steps)
    else:
        node_scores = diffuse_severity(g, steps=steps, decay=decay)
    commodity = aggregate_commodity_tension(network, node_scores)
    return commodity, node_scores


def betweenness_spof(
    g: nx.MultiDiGraph | nx.DiGraph,
    *,
    top_k: int = 15,
    weight_key: str = "weight",
) -> list[tuple[str, float]]:
    """Betweenness centrality → single points of failure ranking."""
    simple = nx.DiGraph()
    for u, v, data in g.edges(data=True):
        # lower conductance → longer "distance"
        dist = 1.0 / max(_edge_conductance(data), 1e-6)
        if simple.has_edge(u, v):
            simple[u][v][weight_key] = min(simple[u][v][weight_key], dist)
        else:
            simple.add_edge(u, v, **{weight_key: dist})
    for n in g.nodes:
        simple.add_node(n)
    bc = nx.betweenness_centrality(simple, weight=weight_key, normalized=True)
    ranked = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def degree_spof(
    g: nx.MultiDiGraph | nx.DiGraph,
    *,
    top_k: int = 15,
) -> list[tuple[str, float]]:
    deg = dict(g.degree())
    ranked = sorted(deg.items(), key=lambda x: x[1], reverse=True)
    return [(n, float(d)) for n, d in ranked[:top_k]]


def analyze_sector(sector: str, *, top_k: int = 15) -> dict[str, Any]:
    network = load_sector(sector, validate=True)
    g = to_networkx(network, validate=False)
    commodity_t, node_scores = tension_from_network(network, method="diffusion")
    return {
        "sector": sector,
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "tensions": commodity_t,
        "betweenness": betweenness_spof(g, top_k=top_k),
        "degree": degree_spof(g, top_k=top_k),
        "max_node_score": max((abs(v) for v in node_scores.values()), default=0.0),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diffusion + centrality analytics")
    parser.add_argument("--sector", default="metals", help="metals | energy | agriculture")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--method", choices=("diffusion", "random_walk"), default="diffusion")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    network = load_sector(args.sector, validate=True)
    tensions, _ = tension_from_network(network, method=args.method)
    g = to_networkx(network, validate=False)
    bc = betweenness_spof(g, top_k=args.top_k)
    print(f"[{args.sector}] tensions={tensions}")
    print(f"top betweenness SPOF:")
    for nid, score in bc:
        print(f"  {nid}: {score:.4f}")


if __name__ == "__main__":
    main()
