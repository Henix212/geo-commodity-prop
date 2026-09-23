"""Cronable price + production refresh jobs."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from backend import config
from backend.database import init_db
from backend.database.seed_data import seed_production
from backend.logging_setup import setup_logging
from backend.quant.market_data import refresh_all_commodities, refresh_core_futures

logger = logging.getLogger(__name__)


def refresh_prices_job(*, core_only: bool = False) -> list[dict[str, Any]]:
    init_db()
    if core_only:
        summary = refresh_core_futures(period="2y")
    else:
        summary = refresh_all_commodities(period="2y")
    logger.info("refresh_prices_job done items=%d", len(summary))
    return summary


def refresh_production_job() -> dict[str, int]:
    """Re-upsert production seed YAML (live feed placeholder)."""
    init_db()
    result = seed_production()
    logger.info("refresh_production_job %s", result)
    return result


def run_all(*, core_only: bool = False) -> dict[str, Any]:
    return {
        "prices": refresh_prices_job(core_only=core_only),
        "production": refresh_production_job(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh prices / production jobs")
    parser.add_argument(
        "--job",
        choices=("prices", "production", "all"),
        default="all",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only HG=F CL=F GC=F SI=F when refreshing prices",
    )
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if args.job == "prices":
        print(refresh_prices_job(core_only=args.core_only))
    elif args.job == "production":
        print(refresh_production_job())
    else:
        print(run_all(core_only=args.core_only))


if __name__ == "__main__":
    main()
