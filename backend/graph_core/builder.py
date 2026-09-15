from pathlib import Path

import yaml

NETWORK_DATA_DIR = Path(__file__).parent / "network_data"


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


def load_sector(sector: str) -> dict:
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

    return {
        "network_name": f"global_{sector}_network",
        "sector": sector,
        "supported_commodities": commodities,
        "nodes": nodes,
        "edges": edges,
    }


def load_all_networks() -> dict[str, dict]:
    networks: dict[str, dict] = {}
    for path in sorted(NETWORK_DATA_DIR.iterdir()):
        if path.is_dir() and path.name != "shared":
            networks[path.name] = load_sector(path.name)
    return networks
