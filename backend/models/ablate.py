"""Ablations: with/without scrap hubs, with/without chokepoints."""

from __future__ import annotations

import argparse
import copy
import logging
from typing import Any

from backend import config
from backend.graph_core import load_sector
from backend.models.infer import infer_tensions

logger = logging.getLogger(__name__)

SCRAP_TYPES = frozenset({"scrap_hub"})
CHOKEPOINT_TYPES = frozenset({"bottleneck"})
CHOKEPOINT_PREFIX = "chokepoint_"


def _filter_network(
    network: dict,
    *,
    drop_types: set[str] | None = None,
    drop_id_prefix: str | None = None,
) -> dict:
    net = copy.deepcopy(network)
    drop_types = drop_types or set()
    drop_ids = {
        n["id"]
        for n in net.get("nodes") or []
        if n.get("type") in drop_types
        or (drop_id_prefix and str(n.get("id", "")).startswith(drop_id_prefix))
    }
    net["nodes"] = [n for n in net.get("nodes") or [] if n["id"] not in drop_ids]
    net["edges"] = [
        e
        for e in net.get("edges") or []
        if e.get("source") not in drop_ids and e.get("target") not in drop_ids
    ]
    return net


def ablate(
    sector: str,
    *,
    use_gat: bool = False,
    with_demo_shock: bool = True,
) -> dict[str, Any]:
    base = load_sector(sector, validate=True)
    if with_demo_shock:
        from backend.ingestion.inject_demo import inject_demo

        demo_node = "mine_escondida" if sector == "metals" else "chokepoint_hormuz"
        demo_commodity = "copper" if sector == "metals" else "oil"
        try:
            base, _, _ = inject_demo(
                sector=sector,
                node_id=demo_node,
                commodity=demo_commodity,
                persist=False,
                network=base,
            )
        except KeyError:
            logger.warning("demo shock node missing; ablating without shock")

    variants = {
        "full": base,
        "no_scrap": _filter_network(base, drop_types=set(SCRAP_TYPES)),
        "no_chokepoints": _filter_network(
            base, drop_types=set(CHOKEPOINT_TYPES), drop_id_prefix=CHOKEPOINT_PREFIX
        ),
    }
    report: dict[str, Any] = {"sector": sector, "variants": {}}
    for name, net in variants.items():
        tensions = infer_tensions(net, use_gat=use_gat)
        report["variants"][name] = {
            "n_nodes": len(net.get("nodes") or []),
            "n_edges": len(net.get("edges") or []),
            "tensions": tensions,
        }
        logger.info("ablate %s/%s tensions=%s", sector, name, tensions)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GNN / baseline ablations")
    parser.add_argument("--sector", default=config.DEFAULT_SECTOR)
    parser.add_argument("--use-gat", action="store_true")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(ablate(args.sector, use_gat=args.use_gat))


if __name__ == "__main__":
    main()
