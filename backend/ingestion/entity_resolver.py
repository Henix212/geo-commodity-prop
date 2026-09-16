"""Resolve free-text entity mentions to graph node_ids."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _aliases_for_node(node: dict) -> list[str]:
    nid = str(node.get("id") or "")
    aliases = {nid, nid.replace("_", " ")}

    # strip type prefixes: mine_escondida -> escondida
    for prefix in (
        "mine_",
        "port_",
        "smelter_",
        "refinery_",
        "field_",
        "terminal_",
        "chokepoint_",
        "consumer_",
        "exchange_",
        "warehouse_",
        "vault_",
        "region_",
        "silo_",
        "hub_",
        "scrap_hub_",
        "lng_",
        "regas_",
        "mill_",
        "ethanol_",
        "feed_mill_",
        "alumina_",
        "maritime_route_",
    ):
        if nid.startswith(prefix):
            aliases.add(nid[len(prefix) :])
            aliases.add(nid[len(prefix) :].replace("_", " "))

    for key in ("country", "region", "operator", "description"):
        val = node.get(key)
        if isinstance(val, str) and len(val) >= 3:
            aliases.add(val)

    return [_normalize(a) for a in aliases if a and len(_normalize(a)) >= 2]


def build_gazetteer(nodes: list[dict]) -> dict[str, str]:
    """Map normalized alias -> node_id (first wins on collisions)."""
    gazetteer: dict[str, str] = {}
    for node in nodes:
        nid = node.get("id")
        if not nid:
            continue
        for alias in _aliases_for_node(node):
            gazetteer.setdefault(alias, nid)
        gazetteer.setdefault(_normalize(nid), nid)
    return gazetteer


def resolve_entity(
    entity_text: str,
    gazetteer: dict[str, str],
    *,
    min_ratio: float = 0.72,
) -> tuple[str | None, float]:
    """Return (node_id, match_score)."""
    needle = _normalize(entity_text)
    if not needle:
        return None, 0.0

    if needle in gazetteer:
        return gazetteer[needle], 1.0

    # substring containment (Escondida in mine_escondida aliases)
    for alias, nid in gazetteer.items():
        if len(needle) >= 4 and (needle in alias or alias in needle):
            score = min(len(needle), len(alias)) / max(len(needle), len(alias))
            if score >= 0.5:
                return nid, max(score, 0.8)

    best_id: str | None = None
    best = 0.0
    for alias, nid in gazetteer.items():
        ratio = SequenceMatcher(None, needle, alias).ratio()
        if ratio > best:
            best = ratio
            best_id = nid

    if best >= min_ratio:
        return best_id, best
    return None, best
