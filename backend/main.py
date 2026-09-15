"""Entry point: build graph → (events) → (inference) → (signal).

Later phases fill ingestion / GNN / quant; Phase 0 wires the skeleton.
"""

from __future__ import annotations

import argparse
import logging

from backend import config
from backend.graph_core import feature_dims, load_sector, to_networkx, to_pyg, validate_network


logger = logging.getLogger(__name__)


def build_graph(sector: str) -> dict:
    network = load_sector(sector, validate=True)
    logger.info(
        "Loaded %s: %d nodes, %d edges, commodities=%s",
        network["network_name"],
        len(network["nodes"]),
        len(network["edges"]),
        network["supported_commodities"],
    )
    issues = validate_network(network)
    if issues:
        logger.warning("Validation issues: %s", issues[:5])
    return network


def graph_tensors(network: dict) -> tuple:
    """Build NetworkX + PyG views (Phase 1)."""
    g = to_networkx(network, validate=False)
    data = to_pyg(network, validate=False)
    dims = feature_dims()
    logger.info(
        "Graph views: nx_nodes=%d nx_edges=%d pyg_x=%s pyg_e=%s dims=%s",
        g.number_of_nodes(),
        g.number_of_edges(),
        tuple(data.x.shape),
        tuple(data.edge_attr.shape),
        dims,
    )
    return g, data


def ingest_events(_network: dict) -> list[dict]:
    """Phase 2 stub — scrapers + LLM triples."""
    logger.info("Event ingestion not implemented yet (Phase 2)")
    return []


def run_inference(_network: dict, _events: list[dict]) -> dict[str, float]:
    """Phase 3 stub — GNN tension scores by commodity."""
    logger.info("GNN inference not implemented yet (Phase 3)")
    return {}


def make_signal(_tensions: dict[str, float]) -> dict | None:
    """Phase 4 stub — tension → trade signal."""
    logger.info("Strategy / signal not implemented yet (Phase 4)")
    return None


def run(sector: str | None = None) -> dict:
    config.ensure_dirs()
    sector = sector or config.DEFAULT_SECTOR

    network = build_graph(sector)
    _nx_graph, _pyg_data = graph_tensors(network)
    events = ingest_events(network)
    tensions = run_inference(network, events)
    signal = make_signal(tensions)

    return {
        "sector": sector,
        "commodity_default": config.DEFAULT_COMMODITY,
        "ticker": config.TICKERS.get(config.DEFAULT_COMMODITY),
        "n_nodes": len(network["nodes"]),
        "n_edges": len(network["edges"]),
        "commodities": network["supported_commodities"],
        "feature_dims": feature_dims(),
        "n_events": len(events),
        "tensions": tensions,
        "signal": signal,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Geo Commodity Prop pipeline")
    parser.add_argument(
        "--sector",
        default=config.DEFAULT_SECTOR,
        help="Sector to load (metals | energy | agriculture)",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        help="Logging level (DEBUG, INFO, ...)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = run(sector=args.sector)
    print(
        f"[{result['sector']}] nodes={result['n_nodes']} edges={result['n_edges']} "
        f"commodities={result['commodities']} ticker={result['ticker']}"
    )


if __name__ == "__main__":
    main()
