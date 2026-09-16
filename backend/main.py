"""Entry point: build graph → events → (inference) → (signal)."""

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


def ingest_events(network: dict, *, parse_limit: int | None = None) -> list[dict]:
    """Scrape articles, LLM-extract shocks, map to node_ids, persist + inject."""
    from backend.database import upsert_events
    from backend.ingestion.parser_llm import (
        EventParser,
        filter_events,
        inject_events_into_network,
    )
    from backend.ingestion.entity_resolver import build_gazetteer
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


def run_inference(_network: dict, _events: list[dict]) -> dict[str, float]:
    """Phase 3 stub — GNN tension scores by commodity."""
    logger.info("GNN inference not implemented yet (Phase 3)")
    return {}


def make_signal(_tensions: dict[str, float]) -> dict | None:
    """Phase 4 stub — tension → trade signal."""
    logger.info("Strategy / signal not implemented yet (Phase 4)")
    return None


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


def run(
    sector: str | None = None,
    *,
    skip_ingest: bool = False,
    skip_llm: bool = False,
    from_db: bool = False,
    parse_limit: int | None = None,
) -> dict:
    config.ensure_dirs()
    sector = sector or config.DEFAULT_SECTOR

    network = build_graph(sector)
    _nx_graph, _pyg_data = graph_tensors(network)

    if skip_ingest and not from_db:
        events: list[dict] = []
    elif from_db:
        events = ingest_events_from_db(network, parse_limit=parse_limit)
    elif skip_llm:
        from backend.ingestion.scrapers import save_articles, scrape_all

        articles = scrape_all(
            gdelt_timespan=config.GDELT_TIMESPAN,
            gdelt_maxrecords=config.GDELT_MAXRECORDS,
        )
        save_articles(articles, persist_db=True)
        events = []
        logger.info("Scraped %d articles (LLM skipped)", len(articles))
    else:
        events = ingest_events(network, parse_limit=parse_limit)

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
        "--skip-ingest",
        action="store_true",
        help="Skip RSS/GDELT scraping and LLM parsing",
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
        "--log-level",
        default=config.LOG_LEVEL,
        help="Logging level (DEBUG, INFO, ...)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = run(
        sector=args.sector,
        skip_ingest=args.skip_ingest,
        skip_llm=args.skip_llm,
        from_db=args.from_db,
        parse_limit=args.parse_limit,
    )
    print(
        f"[{result['sector']}] nodes={result['n_nodes']} edges={result['n_edges']} "
        f"commodities={result['commodities']} events={result['n_events']} "
        f"ticker={result['ticker']}"
    )


if __name__ == "__main__":
    main()
