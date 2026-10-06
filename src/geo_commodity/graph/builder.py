from pathlib import Path

import yaml

from geo_commodity.config import NETWORK_DATA_DIR
from geo_commodity.graph.convert import feature_dims, to_networkx, to_pyg
from geo_commodity.graph.validate import assert_valid, validate_network

__all__ = [
    "load_shared_nodes",
    "load_sector",
    "load_merged_network",
    "load_network",
    "load_all_networks",
    "load_commodity",
    "validate_network",
    "assert_valid",
    "to_networkx",
    "to_pyg",
    "feature_dims",
]


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_shared_nodes() -> list[dict]:
    nodes: list[dict] = []
    shared_dir = NETWORK_DATA_DIR / "shared"
    if not shared_dir.exists():
        return nodes
    for path in sorted(shared_dir.glob("*.yaml")):
        data = _load_yaml(path)
        nodes.extend(data.get("nodes", []))
    return nodes


def load_sector(sector: str, *, validate: bool = False) -> dict:
    """Merge shared nodes with all commodity YAMLs under a sector."""
    sector_dir = NETWORK_DATA_DIR / sector
    nodes = load_shared_nodes()
    edges: list[dict] = []
    commodities: list[str] = []

    if sector_dir.exists():
        for path in sorted(sector_dir.glob("*.yaml")):
            data = _load_yaml(path)
            commodity = data.get("commodity") or path.stem
            commodities.append(commodity)
            nodes.extend(data.get("nodes", []))
            edges.extend(data.get("edges", []))

    network = {
        "network_name": f"global_{sector}_network",
        "sector": sector,
        "supported_commodities": commodities,
        "nodes": nodes,
        "edges": edges,
    }
    if validate:
        assert_valid(network)
    return network


def load_commodity(sector: str, commodity: str, *, validate: bool = False) -> dict:
    """Load one commodity YAML merged with shared bottlenecks."""
    path = NETWORK_DATA_DIR / sector / f"{commodity}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    data = _load_yaml(path)
    network = {
        "network_name": f"{commodity}_network",
        "sector": sector,
        "supported_commodities": [data.get("commodity") or commodity],
        "nodes": load_shared_nodes() + list(data.get("nodes") or []),
        "edges": list(data.get("edges") or []),
    }
    if validate:
        assert_valid(network)
    return network


def load_all_networks(*, validate: bool = False) -> dict[str, dict]:
    networks: dict[str, dict] = {}
    for path in sorted(NETWORK_DATA_DIR.iterdir()):
        if path.is_dir() and path.name != "shared":
            networks[path.name] = load_sector(path.name, validate=validate)
    return networks


def load_merged_network(*, validate: bool = False) -> dict:
    """Merge metals + energy + agriculture (+ shared) into one network.

    Nodes are deduped by ``id`` (shared bottlenecks appear once). Used for the
    global LLM gazetteer so oil/wheat/copper entities all resolve.
    """
    by_id: dict[str, dict] = {}
    edges: list[dict] = []
    commodities: list[str] = []
    sectors: list[str] = []

    for name, net in load_all_networks(validate=False).items():
        sectors.append(name)
        commodities.extend(net.get("supported_commodities") or [])
        for node in net.get("nodes") or []:
            nid = node.get("id")
            if nid and nid not in by_id:
                by_id[nid] = node
        edges.extend(net.get("edges") or [])

    # Dedupe edges on endpoints + flow/product (shared appear once per sector merge)
    seen_e: set[tuple] = set()
    uniq_edges: list[dict] = []
    for e in edges:
        key = (
            e.get("source"),
            e.get("target"),
            e.get("flow_type"),
            e.get("product_form"),
        )
        if key in seen_e:
            continue
        seen_e.add(key)
        uniq_edges.append(e)

    network = {
        "network_name": "global_all_sectors",
        "sector": "all",
        "sectors": sectors,
        "supported_commodities": sorted(set(commodities)),
        "nodes": list(by_id.values()),
        "edges": uniq_edges,
    }
    if validate:
        assert_valid(network)
    return network


def load_network(sector: str | None = "all", *, validate: bool = False) -> dict:
    """Load one sector, or ``all`` / empty → merged multi-sector network."""
    key = (sector or "all").strip().lower()
    if key in {"all", "*"}:
        return load_merged_network(validate=validate)
    return load_sector(key, validate=validate)
