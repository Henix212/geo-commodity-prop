"""Load reference YAML seeds (production, TC/RC, policies) into SQLite."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from backend import config
from backend.database import (
    init_db,
    upsert_policies,
    upsert_production,
    upsert_tc_rc,
)

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def seed_production(path: Path | None = None) -> dict[str, int]:
    path = path or (config.SEED_DIR / "production_copper.yaml")
    data = _load_yaml(path)
    commodity = data.get("commodity") or "copper"
    year = int(data["year"])
    rows = []
    for row in data.get("rows") or []:
        rows.append(
            {
                "node_id": row["node_id"],
                "commodity": row.get("commodity") or commodity,
                "year": int(row.get("year") or year),
                "production_kt": float(row["production_kt"]),
                "capacity_kt": row.get("capacity_kt"),
                "source": row.get("source") or data.get("source") or "",
                "as_of": row.get("as_of") or data.get("as_of") or "",
                "notes": row.get("notes") or "",
            }
        )
    result = upsert_production(rows)
    logger.info("production seed %s → +%d ~%d total=%d", path.name, result.inserted, result.updated, result.total)
    return {"inserted": result.inserted, "updated": result.updated, "total": result.total}


def seed_tc_rc(path: Path | None = None) -> dict[str, int]:
    path = path or (config.SEED_DIR / "tc_rc.yaml")
    data = _load_yaml(path)
    commodity = data.get("commodity") or "copper"
    rows = []
    for row in data.get("rows") or []:
        rows.append(
            {
                "commodity": row.get("commodity") or commodity,
                "from_node": row["from_node"],
                "to_node": row["to_node"],
                "tc_usd_per_dmt": row.get("tc_usd_per_dmt"),
                "rc_usc_per_lb": row.get("rc_usc_per_lb"),
                "effective_from": row["effective_from"],
                "effective_to": row.get("effective_to"),
                "source": row.get("source") or data.get("source") or "",
                "notes": row.get("notes") or "",
            }
        )
    result = upsert_tc_rc(rows)
    logger.info("tc_rc seed %s → +%d ~%d total=%d", path.name, result.inserted, result.updated, result.total)
    return {"inserted": result.inserted, "updated": result.updated, "total": result.total}


def seed_policies(path: Path | None = None) -> dict[str, int]:
    path = path or (config.SEED_DIR / "policies.yaml")
    data = _load_yaml(path)
    rows = list(data.get("rows") or [])
    result = upsert_policies(rows)
    logger.info("policies seed %s → +%d ~%d total=%d", path.name, result.inserted, result.updated, result.total)
    return {"inserted": result.inserted, "updated": result.updated, "total": result.total}


def seed_all() -> dict[str, dict[str, int]]:
    init_db()
    return {
        "production": seed_production(),
        "tc_rc": seed_tc_rc(),
        "policies": seed_policies(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed production / TC-RC / policies into SQLite")
    parser.add_argument(
        "--only",
        choices=("production", "tc_rc", "policies", "all"),
        default="all",
    )
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    init_db()
    if args.only == "all":
        out = seed_all()
    elif args.only == "production":
        out = {"production": seed_production()}
    elif args.only == "tc_rc":
        out = {"tc_rc": seed_tc_rc()}
    else:
        out = {"policies": seed_policies()}
    print(out)


if __name__ == "__main__":
    main()
