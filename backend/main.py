"""Entry point: build graph → events → inference → signal."""

from __future__ import annotations

import argparse
import logging

from backend import config
from backend.graph_core import feature_dims, load_sector, to_networkx, to_pyg, validate_network
from backend.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def build_graph(sector: str, *, apply_reference: bool = True, sync_mirror: bool = True) -> dict:
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
    if apply_reference:
        from backend.database import (
            apply_policies_to_network,
            apply_production_to_network,
            apply_tc_rc_to_network,
        )

        n_prod = apply_production_to_network(network)
        n_tc = apply_tc_rc_to_network(network)
        n_pol = apply_policies_to_network(network)
        logger.info(
            "Reference data applied: production_nodes=%d tc_rc_edges=%d policy_hits=%d",
            n_prod,
            n_tc,
            n_pol,
        )
    if sync_mirror:
        from backend.database import sync_graph_mirror

        result = sync_graph_mirror(network)
        logger.info(
            "Graph mirror synced for %s: rows=%d",
            network.get("sector"),
            result.total,
        )
    return network


def graph_tensors(network: dict) -> tuple:
    """Build NetworkX + PyG views (call after event inject)."""
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


def ingest_events(network: dict, *, parse_limit: int | None = None) -> list[dict]:
    """Scrape articles, LLM-extract shocks, map to node_ids, persist + inject."""
    from backend.database import upsert_events
    from backend.ingestion.entity_resolver import build_gazetteer
    from backend.ingestion.parser_llm import (
        EventParser,
        filter_events,
        inject_events_into_network,
    )
    from backend.ingestion.scrapers import save_articles, scrape_all

    articles = scrape_all(
        gdelt_timespan=config.GDELT_TIMESPAN,
        gdelt_maxrecords=config.GDELT_MAXRECORDS,
    )
    path = save_articles(articles, persist_db=True)
    logger.info("Ingested %d articles (batch) → %s, db=%s", len(articles), path, config.DB_PATH)

    limit = config.LLM_PARSE_LIMIT if parse_limit is None else parse_limit
    gazetteer = build_gazetteer(network.get("nodes") or [])
    parser = EventParser(max_new_tokens=config.LLM_MAX_NEW_TOKENS)

    raw_events = []
    for art in articles[:limit]:
        raw_events.extend(
            parser.parse_article(
                title=art.title,
                summary=art.summary,
                article_uid=art.uid,
                gazetteer=gazetteer,
            )
        )
    events = filter_events(raw_events, require_node=True)
    db_result = upsert_events(events)
    injected = inject_events_into_network(network, events)
    logger.info(
        "LLM events: raw=%d kept=%d db=+%d/~%d injected_nodes=%d",
        len(raw_events),
        len(events),
        db_result.inserted,
        db_result.updated,
        injected,
    )
    return [e.to_dict() for e in events]


def ingest_events_from_db(
    network: dict,
    *,
    parse_limit: int | None = None,
) -> list[dict]:
    """Parse articles already in SQLite (no re-scrape)."""
    from backend.database import list_articles, upsert_events
    from backend.ingestion.entity_resolver import build_gazetteer
    from backend.ingestion.parser_llm import (
        EventParser,
        filter_events,
        inject_events_into_network,
    )

    limit = config.LLM_PARSE_LIMIT if parse_limit is None else parse_limit
    articles = list_articles(limit=limit)
    if not articles:
        logger.warning("No articles in DB — run scrapers first")
        return []

    gazetteer = build_gazetteer(network.get("nodes") or [])
    parser = EventParser(max_new_tokens=config.LLM_MAX_NEW_TOKENS)
    raw_events = []
    for i, art in enumerate(articles, 1):
        logger.info("LLM parse %d/%d: %s", i, len(articles), (art.get("title") or "")[:80])
        raw_events.extend(
            parser.parse_article(
                title=art.get("title") or "",
                summary=art.get("summary") or "",
                article_uid=art.get("uid") or "",
                gazetteer=gazetteer,
            )
        )
    events = filter_events(raw_events, require_node=True)
    db_result = upsert_events(events)
    injected = inject_events_into_network(network, events)
    logger.info(
        "LLM events (from DB): raw=%d kept=%d db=+%d/~%d injected_nodes=%d",
        len(raw_events),
        len(events),
        db_result.inserted,
        db_result.updated,
        injected,
    )
    return [e.to_dict() for e in events]


def inject_persisted_events(network: dict, *, limit: int = 1000) -> list[dict]:
    """Load events from SQLite and inject with time decay (no LLM)."""
    from backend.database import list_events
    from backend.ingestion.parser_llm import inject_events_into_network

    rows = list_events(limit=limit)
    injected = inject_events_into_network(network, rows)
    logger.info(
        "Injected %d persisted events onto %d nodes (half-life=%sh)",
        len(rows),
        injected,
        config.EVENT_DECAY_HALF_LIFE_HOURS,
    )
    return rows


def run_inference(
    network: dict,
    _events: list[dict] | None = None,
    *,
    backend: str | None = None,
) -> dict:
    """Run diffusion or GAT; return commodity + node tensions."""
    backend = (backend or config.INFERENCE_BACKEND).lower()
    ckpt = config.GNN_CHECKPOINT_DIR / "shock_gat.pt"

    if backend == "auto":
        backend = "gat" if ckpt.exists() else "diffusion"

    if backend == "gat":
        from backend.models.gat import load_checkpoint, run_gat

        model = load_checkpoint()
        result = run_gat(network, model=model)
        logger.info("GAT commodity tensions: %s", result["commodity_tensions"])
        return {
            "backend": "gat",
            "commodity_tensions": result["commodity_tensions"],
            "node_tensions": result["node_tensions"],
        }

    from backend.models.baseline_diffusion import run_diffusion

    result = run_diffusion(network)
    logger.info("Diffusion commodity tensions: %s", result["commodity_tensions"])
    return {
        "backend": "diffusion",
        "commodity_tensions": result["commodity_tensions"],
        "node_tensions": result["node_tensions"],
    }


def make_signal(tensions: dict[str, float]) -> dict | None:
    """Tension → trade signal for default commodity."""
    from backend.quant.strategy import primary_signal

    signal = primary_signal(tensions)
    if signal:
        logger.info(
            "Signal %s side=%s tension=%.3f ticker=%s",
            signal["commodity"],
            signal["side"],
            signal["tension"],
            signal.get("ticker"),
        )
    else:
        logger.info("No signal (empty tensions)")
    return signal


def run(
    sector: str | None = None,
    *,
    skip_ingest: bool = False,
    skip_llm: bool = False,
    from_db: bool = False,
    parse_limit: int | None = None,
    inference_backend: str | None = None,
) -> dict:
    config.ensure_dirs()
    sector = sector or config.DEFAULT_SECTOR

    network = build_graph(sector)

    if skip_ingest and not from_db:
        events = inject_persisted_events(network)
    elif from_db:
        events = ingest_events_from_db(network, parse_limit=parse_limit)
    elif skip_llm:
        from backend.ingestion.scrapers import save_articles, scrape_all

        articles = scrape_all(
            gdelt_timespan=config.GDELT_TIMESPAN,
            gdelt_maxrecords=config.GDELT_MAXRECORDS,
        )
        save_articles(articles, persist_db=True)
        events = inject_persisted_events(network)
        logger.info("Scraped %d articles (LLM skipped)", len(articles))
    else:
        events = ingest_events(network, parse_limit=parse_limit)

    # Rebuild tensors AFTER inject so event_severity lands in PyG features
    nx_graph, pyg_data = graph_tensors(network)

    inference = run_inference(network, events, backend=inference_backend)
    tensions = inference["commodity_tensions"]
    signal = make_signal(tensions)

    from backend.dashboard_state import write_dashboard_state
    from backend.quant.strategy import make_signals

    write_dashboard_state(
        sector=sector,
        network=network,
        commodity_tensions=tensions,
        node_tensions=inference.get("node_tensions") or {},
        signal=signal,
        signals=make_signals(tensions),
        inference_backend=inference.get("backend") or "diffusion",
        events=events,
    )

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
        "nx_nodes": nx_graph.number_of_nodes(),
        "pyg_x": tuple(pyg_data.x.shape),
        "inference_backend": inference.get("backend"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Geo Commodity Prop pipeline")
    parser.add_argument(
        "--sector",
        default=config.DEFAULT_SECTOR,
        help="Sector to load (metals | energy | agriculture)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip RSS/GDELT scraping and LLM parsing; inject persisted events",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Scrape articles but skip LLM event extraction",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Parse articles already in SQLite (skip scrape; recommended for LLM)",
    )
    parser.add_argument(
        "--parse-limit",
        type=int,
        default=None,
        help="Max articles to send to the LLM (default: GCP_LLM_PARSE_LIMIT)",
    )
    parser.add_argument(
        "--inference",
        choices=("auto", "diffusion", "gat"),
        default=None,
        help="Tension backend (default: GCP_INFERENCE_BACKEND)",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        help="Logging level (DEBUG, INFO, ...)",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    result = run(
        sector=args.sector,
        skip_ingest=args.skip_ingest,
        skip_llm=args.skip_llm,
        from_db=args.from_db,
        parse_limit=args.parse_limit,
        inference_backend=args.inference,
    )
    print(
        f"[{result['sector']}] nodes={result['n_nodes']} edges={result['n_edges']} "
        f"commodities={result['commodities']} events={result['n_events']} "
        f"ticker={result['ticker']} tensions={result['tensions']} "
        f"signal={result['signal']}"
    )


if __name__ == "__main__":
    main()
