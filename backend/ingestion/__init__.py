"""News ingestion and LLM event extraction."""

__all__ = [
    "Article",
    "scrape_all",
    "save_articles",
    "ShockEvent",
    "EventParser",
    "parse_articles",
    "filter_events",
    "inject_events_into_network",
    "effective_severity",
    "decay_factor",
    "build_gazetteer",
    "resolve_entity",
]


def __getattr__(name: str):
    if name in {"Article", "scrape_all", "save_articles"}:
        from backend.ingestion import scrapers

        return getattr(scrapers, name)
    if name in {
        "ShockEvent",
        "EventParser",
        "parse_articles",
        "filter_events",
        "inject_events_into_network",
        "effective_severity",
        "decay_factor",
    }:
        from backend.ingestion import parser_llm

        return getattr(parser_llm, name)
    if name in {"build_gazetteer", "resolve_entity"}:
        from backend.ingestion import entity_resolver

        return getattr(entity_resolver, name)
    raise AttributeError(name)
