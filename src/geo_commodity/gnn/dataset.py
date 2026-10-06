"""PyG snapshot dataset: real events + synthetic severity seeds."""

from __future__ import annotations

import copy
import logging
import random
from typing import Any

import torch
from torch_geometric.data import Data

from geo_commodity import config
from geo_commodity.graph import load_network, to_pyg
from geo_commodity.events.parser_llm import ShockEvent, inject_events_into_network
from geo_commodity.gnn.baseline import tension_from_network
from geo_commodity.gnn.labels import (
    commodity_label_vector,
    forward_return,
    labels_from_events,
    return_to_tension,
)

logger = logging.getLogger(__name__)


def _clone_network(network: dict) -> dict:
    return copy.deepcopy(network)


def _clear_events(network: dict) -> None:
    for node in network.get("nodes") or []:
        node["event_severity"] = 0.0
        for k in (
            "event_type",
            "event_direction",
            "event_commodity",
            "event_extracted_at",
            "event_severity_raw",
            "event_decay_factor",
        ):
            node.pop(k, None)


def network_to_sample(
    network: dict,
    *,
    commodity: str,
    tension_label: float,
) -> Data:
    data = to_pyg(network, validate=False)
    order = list(config.TICKERS.keys())
    y = torch.tensor(
        commodity_label_vector(commodity, tension_label, order),
        dtype=torch.float,
    )
    data.y = y
    data.target_commodity = commodity
    data.tension_label = float(tension_label)
    return data


def synthetic_samples(
    sector: str = "all",
    *,
    n: int = 64,
    seed: int = 42,
) -> list[Data]:
    """Seed random nodes with severity; label from diffusion tension (pseudo).

    ``sector='all'`` uses the merged multi-sector graph (metals+energy+agriculture).
    """
    rng = random.Random(seed)
    base = load_network(sector, validate=True)
    node_ids = [n["id"] for n in base.get("nodes") or []]
    commodities = list(base.get("supported_commodities") or [config.DEFAULT_COMMODITY])
    if not node_ids:
        raise RuntimeError(f"No nodes in network for sector={sector!r}")
    samples: list[Data] = []
    for i in range(n):
        network = _clone_network(base)
        _clear_events(network)
        k = rng.randint(1, min(5, max(1, len(node_ids))))
        chosen = rng.sample(node_ids, k=k)
        commodity = rng.choice(commodities)
        events = []
        for nid in chosen:
            sev = rng.uniform(0.4, 1.0)
            events.append(
                ShockEvent(
                    entity_text=nid,
                    event_type="outage",
                    severity=sev,
                    commodity=commodity,
                    direction="supply_down",
                    confidence=1.0,
                    article_uid=f"synth:{sector}:{i}:{nid}",
                    node_id=nid,
                    match_score=1.0,
                )
            )
        inject_events_into_network(network, events, min_severity=0.0)
        commodity_t, _ = tension_from_network(network, method="diffusion")
        label = float(commodity_t.get(commodity) or 0.0)
        # blend with historical return if available
        ret = forward_return(commodity, "2024-06-01")
        if ret is not None:
            label = 0.5 * label + 0.5 * return_to_tension(ret * (1 if label >= 0 else -1))
        samples.append(network_to_sample(network, commodity=commodity, tension_label=label))
    logger.info(
        "synthetic_samples sector=%s nodes=%d commodities=%s n=%d",
        sector,
        len(node_ids),
        commodities,
        len(samples),
    )
    return samples


def event_samples(sector: str = "all", *, limit: int = 500) -> list[Data]:
    """Build snapshots from DB events with price-based labels when possible."""
    base = load_network(sector, validate=True)
    node_ids = {n["id"] for n in base.get("nodes") or [] if n.get("id")}
    labeled = labels_from_events(limit=limit)
    samples: list[Data] = []
    skipped = 0
    for row in labeled:
        ev = row["event"]
        nid = ev.get("node_id")
        if not nid or nid not in node_ids:
            skipped += 1
            continue
        network = _clone_network(base)
        _clear_events(network)
        inject_events_into_network(network, [ev], min_severity=0.0)
        samples.append(
            network_to_sample(
                network,
                commodity=row["commodity"],
                tension_label=float(row["tension_label"]),
            )
        )
    logger.info(
        "event_samples sector=%s n=%d skipped=%d",
        sector,
        len(samples),
        skipped,
    )
    return samples


def build_dataset(
    sector: str = "all",
    *,
    n_synthetic: int = 64,
    include_events: bool = True,
    event_limit: int = 500,
    seed: int = 42,
) -> list[Data]:
    """Build training set. Prefer ``sector='all'`` for a global multi-commodity GAT."""
    samples = synthetic_samples(sector, n=n_synthetic, seed=seed)
    if include_events:
        samples.extend(event_samples(sector, limit=event_limit))
    return samples


def collate_summary(samples: list[Data]) -> dict[str, Any]:
    return {
        "n": len(samples),
        "commodities": sorted({getattr(s, "target_commodity", "") for s in samples}),
        "mean_label": float(
            sum(float(getattr(s, "tension_label", 0.0)) for s in samples) / max(len(samples), 1)
        ),
    }
