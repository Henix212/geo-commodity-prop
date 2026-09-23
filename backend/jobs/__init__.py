"""Periodic refresh jobs (cron-friendly CLIs)."""

__all__ = ["refresh_prices_job", "refresh_production_job", "run_all", "main"]


def __getattr__(name: str):
    if name in __all__:
        from backend.jobs import refresh as _refresh

        return getattr(_refresh, name)
    raise AttributeError(name)
