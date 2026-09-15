from pathlib import Path

import yaml

from backend.config import NETWORK_DATA_DIR
from backend.graph_core.convert import feature_dims, to_networkx, to_pyg
from backend.graph_core.validate import assert_valid, validate_network

__all__ = [
    "load_shared_nodes",
    "load_sector",
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
