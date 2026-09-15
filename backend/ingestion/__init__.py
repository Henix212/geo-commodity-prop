"""News ingestion and LLM event extraction."""

__all__ = ["Article", "scrape_all", "save_articles"]


def __getattr__(name: str):
    if name in __all__:
        from backend.ingestion import scrapers

        return getattr(scrapers, name)
    raise AttributeError(name)
