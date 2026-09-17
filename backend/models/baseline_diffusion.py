"""NetworkX heat diffusion baseline for shock → tension."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import networkx as nx
import numpy as np

from backend import config
from backend.graph_core import to_networkx

logger = logging.getLogger(__name__)


def node_tensions_from_diffusion(
    network: dict,
    *,
    steps: int | None = None,
    alpha: float | None = None,
) -> dict[str, float]:
    """Diffuse |event_severity| along the undirected projection; return node scores."""
    steps = config.DIFFUSION_STEPS if steps is None else steps
    alpha = config.DIFFUSION_ALPHA if alpha is None else alpha

    g = to_networkx(network, validate=False)
    if g.number_of_nodes() == 0:
        return {}

    undirected = g.to_undirected()
    nodes = list(undirected.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    heat = np.zeros(n, dtype=float)
    for nid, data in g.nodes(data=True):
        sev = abs(float(data.get("event_severity") or 0.0))
        if sev > 0:
            heat[index[nid]] = sev

    if float(heat.sum()) == 0.0:
        return {nid: 0.0 for nid in nodes}

    # Row-normalized adjacency (random-walk / heat kernel step)
    adj = nx.to_numpy_array(undirected, nodelist=nodes, weight="weight", dtype=float)
    # fallback weight 1
    if not np.any(adj):
        adj = nx.to_numpy_array(undirected, nodelist=nodes, dtype=float)
    deg = adj.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    trans = adj / deg

    state = heat.copy()
    for _ in range(max(1, steps)):
        state = (1.0 - alpha) * heat + alpha * (trans.T @ state)

    # normalize to [0, 1]
    mx = float(state.max()) if state.size else 1.0
    if mx > 0:
        state = state / mx
    return {nid: float(state[index[nid]]) for nid in nodes}


def commodity_tensions(
    network: dict,
    node_scores: dict[str, float] | None = None,
    *,
    steps: int | None = None,
    alpha: float | None = None,
) -> dict[str, float]:
    """Max tension per commodity (nodes tagged with commodity)."""
    scores = node_scores or node_tensions_from_diffusion(network, steps=steps, alpha=alpha)
    by_c: dict[str, list[float]] = {}
    for node in network.get("nodes") or []:
        c = node.get("commodity")
        if not c:
            continue
        by_c.setdefault(c, []).append(scores.get(node["id"], 0.0))
    out = {c: float(max(vals)) if vals else 0.0 for c, vals in by_c.items()}
    # include sector commodities with 0 if missing
    for c in network.get("supported_commodities") or []:
        out.setdefault(c, 0.0)
    return out


def run_diffusion(network: dict) -> dict[str, Any]:
    node_scores = node_tensions_from_diffusion(network)
    commodity = commodity_tensions(network, node_scores)
    return {"node_tensions": node_scores, "commodity_tensions": commodity}


def main(argv: list[str] | None = None) -> None:
    from backend.logging_setup import configure_logging
    from backend.main import build_graph, inject_persisted_events

    parser = argparse.ArgumentParser(description="NetworkX diffusion tension baseline")
    parser.add_argument("--sector", default=config.DEFAULT_SECTOR)
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    network = build_graph(args.sector)
    if not args.skip_events:
        inject_persisted_events(network)
    result = run_diffusion(network)
    print("commodity_tensions:", result["commodity_tensions"])
    top = sorted(result["node_tensions"].items(), key=lambda x: -x[1])[:10]
    print("top_nodes:", top)


if __name__ == "__main__":
    main()
