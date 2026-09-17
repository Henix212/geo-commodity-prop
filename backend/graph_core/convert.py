"""Convert merged network dicts to NetworkX and PyTorch Geometric."""

from __future__ import annotations

from typing import Any

import networkx as nx

from backend.graph_core.validate import assert_valid

NODE_TYPE_VOCAB = [
    "mine",
    "field",
    "producing_region",
    "port",
    "terminal",
    "smelter",
    "alumina_refinery",
    "refinery",
    "liquefaction",
    "regas",
    "mill",
    "processor",
    "silo",
    "warehouse",
    "scrap_hub",
    "transshipment",
    "bottleneck",
    "consumer",
    "exchange",
    "unknown",
]

FLOW_TYPE_VOCAB = [
    "rail_road",
    "shipping",
    "pipeline",
    "inland_transport",
    "plant_internal",
    "coastal_shipping",
    "slurry_pipeline",
    "barge",
    "warehouse",
    "delivery",
    "commercial",
    "transshipment",
    "unknown",
]

PRODUCT_FORM_VOCAB = [
    "concentrate",
    "bauxite",
    "alumina",
    "primary_ingot",
    "blister",
    "cathode",
    "crude",
    "products",
    "lng",
    "gas",
    "grain",
    "flour",
    "feed",
    "ethanol",
    "bullion",
    "doré",
    "scrap",
    "ore_oxide",
    "unknown",
]


def _one_hot(value: str | None, vocab: list[str]) -> list[float]:
    key = value if value in vocab else "unknown"
    return [1.0 if v == key else 0.0 for v in vocab]


def _f(node: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if node.get(k) is not None:
            try:
                return float(node[k])
            except (TypeError, ValueError):
                continue
    return default


def to_networkx(network: dict, *, validate: bool = True) -> nx.DiGraph:
    if validate:
        assert_valid(network)
    g = nx.MultiDiGraph(
        network_name=network.get("network_name"),
        sector=network.get("sector"),
        commodities=network.get("supported_commodities"),
    )
    for node in network.get("nodes") or []:
        nid = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        attrs.setdefault("event_severity", 0.0)
        g.add_node(nid, **attrs)
    for edge in network.get("edges") or []:
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        g.add_edge(edge["source"], edge["target"], **attrs)
    return g


def node_feature_vector(node: dict[str, Any]) -> list[float]:
    feats: list[float] = []
    feats.extend(_one_hot(node.get("type"), NODE_TYPE_VOCAB))
    feats.append(_f(node, "capacity_kt", "capacity_kbd", "capacity_mtpa", "capacity_kbbl"))
    feats.append(_f(node, "stock_level_kt", "stock_level_kbbl"))
    feats.append(_f(node, "lat"))
    feats.append(_f(node, "lon"))
    feats.append(_f(node, "event_severity"))
    feats.append(_f(node, "capacity_ships_per_day"))
    feats.append(_f(node, "utilization"))
    feats.append(_f(node, "production_kt"))
    feats.append(_f(node, "policy_severity"))
    return feats


def edge_feature_vector(edge: dict[str, Any]) -> list[float]:
    feats: list[float] = []
    feats.extend(_one_hot(edge.get("flow_type"), FLOW_TYPE_VOCAB))
    feats.extend(_one_hot(edge.get("product_form"), PRODUCT_FORM_VOCAB))
    feats.append(_f(edge, "weight", default=1.0))
    feats.append(_f(edge, "transit_days"))
    feats.append(_f(edge, "transit_days_min"))
    feats.append(_f(edge, "transit_days_max"))
    # Normalize rough TC/RC scales into ~[0,1] for GNN stability
    feats.append(min(_f(edge, "tc_usd_per_dmt") / 100.0, 2.0))
    feats.append(min(_f(edge, "rc_usc_per_lb") / 10.0, 2.0))
    return feats


def to_pyg(network: dict, *, validate: bool = True):
    """Homogeneous PyG Data with node/edge feature matrices."""
    import torch
    from torch_geometric.data import Data

    if validate:
        assert_valid(network)

    nodes = network.get("nodes") or []
    edges = network.get("edges") or []
    node_ids = [n["id"] for n in nodes]
    index = {nid: i for i, nid in enumerate(node_ids)}

    x = torch.tensor(
        [node_feature_vector(n) for n in nodes],
        dtype=torch.float,
    )

    if edges:
        src = [index[e["source"]] for e in edges]
        dst = [index[e["target"]] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(
            [edge_feature_vector(e) for e in edges],
            dtype=torch.float,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(edge_feature_vector({}))), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_ids=node_ids,
        sector=network.get("sector"),
        network_name=network.get("network_name"),
        commodities=network.get("supported_commodities"),
        num_nodes=len(nodes),
    )


def feature_dims() -> dict[str, int]:
    return {
        "node": len(node_feature_vector({"type": "unknown"})),
        "edge": len(edge_feature_vector({})),
    }
