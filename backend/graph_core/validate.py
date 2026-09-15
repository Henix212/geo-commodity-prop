"""Validate merged network dicts before graph conversion."""

from __future__ import annotations


def validate_network(network: dict, *, strict_forms: bool = True) -> list[str]:
    """Return list of validation issues (empty = OK)."""
    issues: list[str] = []
    nodes = network.get("nodes") or []
    edges = network.get("edges") or []

    ids = [n.get("id") for n in nodes]
    if any(i is None for i in ids):
        issues.append("node missing id")
    id_set = set(ids)
    if len(ids) != len(id_set):
        seen: set[str] = set()
        for i in ids:
            if i in seen:
                issues.append(f"duplicate node id: {i}")
            seen.add(i)

    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src not in id_set:
            issues.append(f"edge source missing: {src} -> {tgt}")
        if tgt not in id_set:
            issues.append(f"edge target missing: {src} -> {tgt}")

        if not strict_forms:
            continue
        if src not in id_set or tgt not in id_set:
            continue
        node_by_id = {n["id"]: n for n in nodes if n.get("id")}
        tgt_node = node_by_id[tgt]
        accepts = tgt_node.get("accepts_forms")
        pf = e.get("product_form")
        if (
            accepts
            and pf
            and pf not in accepts
            and tgt_node.get("type") in ("smelter", "refinery", "alumina_refinery", "transshipment", "mill", "processor", "regas")
        ):
            issues.append(
                f"product_form mismatch: {src} --{pf}--> {tgt} accepts {accepts}"
            )

    return issues


def assert_valid(network: dict, *, strict_forms: bool = True) -> dict:
    issues = validate_network(network, strict_forms=strict_forms)
    if issues:
        preview = "; ".join(issues[:8])
        more = f" (+{len(issues) - 8} more)" if len(issues) > 8 else ""
        raise ValueError(f"Invalid network ({len(issues)} issues): {preview}{more}")
    return network
