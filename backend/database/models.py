"""CRUD for persisted ingestion records."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from backend.database.db import connect, init_db

if TYPE_CHECKING:
    from backend.ingestion.parser_llm import ShockEvent
    from backend.ingestion.scrapers import Article


@dataclass
class UpsertResult:
    inserted: int
    updated: int
    total: int


def upsert_articles(
    articles: list[Article],
    conn: sqlite3.Connection | None = None,
) -> UpsertResult:
    """Insert new articles or refresh last_seen_at for known uids."""
    owns = conn is None
    conn = conn or connect()
    init_db(conn)

    inserted = 0
    updated = 0

    for art in articles:
        row = conn.execute(
            "SELECT uid FROM articles WHERE uid = ?",
            (art.uid,),
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO articles (
                    uid, source, title, url, published, summary,
                    first_seen_at, last_seen_at, fetch_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    art.uid,
                    art.source,
                    art.title,
                    art.url or "",
                    art.published,
                    art.summary or "",
                    art.fetched_at,
                    art.fetched_at,
                ),
            )
            inserted += 1
        else:
            conn.execute(
                """
                UPDATE articles SET
                    source = ?,
                    title = ?,
                    url = ?,
                    published = COALESCE(?, published),
                    summary = CASE WHEN ? != '' THEN ? ELSE summary END,
                    last_seen_at = ?,
                    fetch_count = fetch_count + 1
                WHERE uid = ?
                """,
                (
                    art.source,
                    art.title,
                    art.url or "",
                    art.published,
                    art.summary or "",
                    art.summary or "",
                    art.fetched_at,
                    art.uid,
                ),
            )
            updated += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    if owns:
        conn.close()

    return UpsertResult(inserted=inserted, updated=updated, total=total)


def list_articles(
    *,
    limit: int = 100,
    source: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)

    if source:
        rows = conn.execute(
            """
            SELECT uid, source, title, url, published, summary,
                   first_seen_at, last_seen_at, fetch_count
            FROM articles
            WHERE source = ?
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (source, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT uid, source, title, url, published, summary,
                   first_seen_at, last_seen_at, fetch_count
            FROM articles
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = [dict(row) for row in rows]
    if owns:
        conn.close()
    return result


def upsert_events(
    events: list[ShockEvent] | list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> UpsertResult:
    """Insert or update shock events (unique on article+entity+type+commodity+direction)."""
    owns = conn is None
    conn = conn or connect()
    init_db(conn)

    inserted = 0
    updated = 0

    for ev in events:
        if hasattr(ev, "to_dict"):
            row = ev.to_dict()
        else:
            row = dict(ev)

        existing = conn.execute(
            """
            SELECT id FROM events
            WHERE article_uid = ?
              AND entity_text = ?
              AND event_type = ?
              AND commodity = ?
              AND direction = ?
            """,
            (
                row.get("article_uid") or "",
                row["entity_text"],
                row["event_type"],
                row["commodity"],
                row["direction"],
            ),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO events (
                    article_uid, entity_text, node_id, event_type, severity,
                    commodity, direction, confidence, match_score, extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("article_uid") or "",
                    row["entity_text"],
                    row.get("node_id"),
                    row["event_type"],
                    float(row["severity"]),
                    row["commodity"],
                    row["direction"],
                    float(row["confidence"]),
                    float(row.get("match_score") or 0.0),
                    row.get("extracted_at") or "",
                ),
            )
            inserted += 1
        else:
            conn.execute(
                """
                UPDATE events SET
                    node_id = ?,
                    severity = ?,
                    confidence = ?,
                    match_score = ?,
                    extracted_at = ?
                WHERE id = ?
                """,
                (
                    row.get("node_id"),
                    float(row["severity"]),
                    float(row["confidence"]),
                    float(row.get("match_score") or 0.0),
                    row.get("extracted_at") or "",
                    existing["id"],
                ),
            )
            updated += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if owns:
        conn.close()
    return UpsertResult(inserted=inserted, updated=updated, total=total)


def list_events(
    *,
    limit: int = 100,
    commodity: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)

    if commodity:
        rows = conn.execute(
            """
            SELECT id, article_uid, entity_text, node_id, event_type, severity,
                   commodity, direction, confidence, match_score, extracted_at
            FROM events
            WHERE commodity = ?
            ORDER BY extracted_at DESC
            LIMIT ?
            """,
            (commodity, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, article_uid, entity_text, node_id, event_type, severity,
                   commodity, direction, confidence, match_score, extracted_at
            FROM events
            ORDER BY extracted_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = [dict(row) for row in rows]
    if owns:
        conn.close()
    return result


# ---------------------------------------------------------------------------
# Node production (actual vs nameplate)
# ---------------------------------------------------------------------------


def upsert_production(
    rows: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> UpsertResult:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    inserted = updated = 0

    for row in rows:
        node_id = row["node_id"]
        commodity = row["commodity"]
        year = int(row["year"])
        production_kt = float(row["production_kt"])
        capacity_kt = row.get("capacity_kt")
        capacity_kt = float(capacity_kt) if capacity_kt is not None else None
        util = row.get("utilization")
        if util is None and capacity_kt and capacity_kt > 0:
            util = production_kt / capacity_kt
        util = float(util) if util is not None else None

        existing = conn.execute(
            "SELECT id FROM node_production WHERE node_id = ? AND commodity = ? AND year = ?",
            (node_id, commodity, year),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO node_production (
                    node_id, commodity, year, production_kt, capacity_kt,
                    utilization, source, as_of, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    commodity,
                    year,
                    production_kt,
                    capacity_kt,
                    util,
                    row.get("source") or "",
                    row.get("as_of") or "",
                    row.get("notes") or "",
                ),
            )
            inserted += 1
        else:
            conn.execute(
                """
                UPDATE node_production SET
                    production_kt = ?, capacity_kt = ?, utilization = ?,
                    source = ?, as_of = ?, notes = ?
                WHERE id = ?
                """,
                (
                    production_kt,
                    capacity_kt,
                    util,
                    row.get("source") or "",
                    row.get("as_of") or "",
                    row.get("notes") or "",
                    existing["id"],
                ),
            )
            updated += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM node_production").fetchone()[0]
    if owns:
        conn.close()
    return UpsertResult(inserted=inserted, updated=updated, total=total)


def list_production(
    *,
    commodity: str | None = None,
    year: int | None = None,
    node_id: str | None = None,
    limit: int = 500,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    clauses: list[str] = []
    params: list[Any] = []
    if commodity:
        clauses.append("commodity = ?")
        params.append(commodity)
    if year is not None:
        clauses.append("year = ?")
        params.append(year)
    if node_id:
        clauses.append("node_id = ?")
        params.append(node_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT id, node_id, commodity, year, production_kt, capacity_kt,
               utilization, source, as_of, notes
        FROM node_production
        {where}
        ORDER BY year DESC, node_id
        LIMIT ?
        """,
        params,
    ).fetchall()
    result = [dict(r) for r in rows]
    if owns:
        conn.close()
    return result


def apply_production_to_network(
    network: dict,
    *,
    year: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Set node production_kt / utilization from DB (latest year if unspecified)."""
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    if year is None:
        row = conn.execute("SELECT MAX(year) AS y FROM node_production").fetchone()
        year = int(row["y"]) if row and row["y"] is not None else None
    if year is None:
        if owns:
            conn.close()
        return 0

    by_id = {n["id"]: n for n in network.get("nodes") or [] if n.get("id")}
    touched = 0
    for rec in list_production(year=year, limit=5000, conn=conn):
        node = by_id.get(rec["node_id"])
        if not node:
            continue
        node["production_kt"] = rec["production_kt"]
        if rec.get("capacity_kt") is not None:
            node["capacity_kt"] = rec["capacity_kt"]
        cap = float(node.get("capacity_kt") or 0) or None
        util = rec.get("utilization")
        if util is None and cap:
            util = float(rec["production_kt"]) / cap
        if util is not None:
            node["utilization"] = float(util)
        node["production_year"] = year
        touched += 1
    if owns:
        conn.close()
    return touched


# ---------------------------------------------------------------------------
# TC / RC by route
# ---------------------------------------------------------------------------


def upsert_tc_rc(
    rows: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> UpsertResult:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    inserted = updated = 0

    for row in rows:
        key = (
            row["commodity"],
            row["from_node"],
            row["to_node"],
            row["effective_from"],
        )
        existing = conn.execute(
            """
            SELECT id FROM tc_rc
            WHERE commodity = ? AND from_node = ? AND to_node = ? AND effective_from = ?
            """,
            key,
        ).fetchone()
        tc = row.get("tc_usd_per_dmt")
        rc = row.get("rc_usc_per_lb")
        vals = (
            float(tc) if tc is not None else None,
            float(rc) if rc is not None else None,
            row.get("effective_to"),
            row.get("source") or "",
            row.get("notes") or "",
        )
        if existing is None:
            conn.execute(
                """
                INSERT INTO tc_rc (
                    commodity, from_node, to_node, tc_usd_per_dmt, rc_usc_per_lb,
                    effective_from, effective_to, source, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*key[:3], vals[0], vals[1], key[3], vals[2], vals[3], vals[4]),
            )
            inserted += 1
        else:
            conn.execute(
                """
                UPDATE tc_rc SET
                    tc_usd_per_dmt = ?, rc_usc_per_lb = ?,
                    effective_to = ?, source = ?, notes = ?
                WHERE id = ?
                """,
                (*vals, existing["id"]),
            )
            updated += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM tc_rc").fetchone()[0]
    if owns:
        conn.close()
    return UpsertResult(inserted=inserted, updated=updated, total=total)


def list_tc_rc(
    *,
    commodity: str | None = None,
    as_of: str | None = None,
    limit: int = 500,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    clauses: list[str] = []
    params: list[Any] = []
    if commodity:
        clauses.append("commodity = ?")
        params.append(commodity)
    if as_of:
        clauses.append("effective_from <= ?")
        params.append(as_of)
        clauses.append("(effective_to IS NULL OR effective_to = '' OR effective_to >= ?)")
        params.append(as_of)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT id, commodity, from_node, to_node, tc_usd_per_dmt, rc_usc_per_lb,
               effective_from, effective_to, source, notes
        FROM tc_rc
        {where}
        ORDER BY effective_from DESC, from_node, to_node
        LIMIT ?
        """,
        params,
    ).fetchall()
    result = [dict(r) for r in rows]
    if owns:
        conn.close()
    return result


def apply_tc_rc_to_network(
    network: dict,
    *,
    as_of: str | None = None,
    commodity: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Annotate matching edges with tc_usd_per_dmt / rc_usc_per_lb."""
    from datetime import date

    as_of = as_of or date.today().isoformat()
    records = list_tc_rc(commodity=commodity, as_of=as_of, limit=5000, conn=conn)
    by_route = {(r["from_node"], r["to_node"]): r for r in reversed(records)}
    touched = 0
    for edge in network.get("edges") or []:
        key = (edge.get("source"), edge.get("target"))
        rec = by_route.get(key)
        if not rec:
            continue
        if rec.get("tc_usd_per_dmt") is not None:
            edge["tc_usd_per_dmt"] = rec["tc_usd_per_dmt"]
        if rec.get("rc_usc_per_lb") is not None:
            edge["rc_usc_per_lb"] = rec["rc_usc_per_lb"]
        edge["tc_rc_as_of"] = as_of
        touched += 1
    return touched


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def upsert_prices(
    rows: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> UpsertResult:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    inserted = updated = 0

    for row in rows:
        ticker = row["ticker"]
        day = row["date"]
        existing = conn.execute(
            "SELECT ticker FROM prices WHERE ticker = ? AND venue = ? AND date = ?",
            (ticker, row.get("venue") or "", day),
        ).fetchone()
        vals = (
            row.get("venue") or "",
            row.get("commodity") or "",
            row.get("open"),
            row.get("high"),
            row.get("low"),
            row.get("close"),
            row.get("volume"),
            row.get("source") or "yfinance",
        )
        if existing is None:
            conn.execute(
                """
                INSERT INTO prices (
                    ticker, venue, commodity, date, open, high, low, close, volume, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, vals[0], vals[1], day, *vals[2:]),
            )
            inserted += 1
        else:
            conn.execute(
                """
                UPDATE prices SET
                    commodity = ?, open = ?, high = ?, low = ?,
                    close = ?, volume = ?, source = ?
                WHERE ticker = ? AND venue = ? AND date = ?
                """,
                (
                    vals[1],
                    vals[2],
                    vals[3],
                    vals[4],
                    vals[5],
                    vals[6],
                    vals[7],
                    ticker,
                    vals[0],
                    day,
                ),
            )
            updated += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    if owns:
        conn.close()
    return UpsertResult(inserted=inserted, updated=updated, total=total)


def list_prices(
    *,
    ticker: str | None = None,
    commodity: str | None = None,
    venue: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 5000,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    clauses: list[str] = []
    params: list[Any] = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker)
    if commodity:
        clauses.append("commodity = ?")
        params.append(commodity)
    if venue:
        clauses.append("venue = ?")
        params.append(venue)
    if start:
        clauses.append("date >= ?")
        params.append(start)
    if end:
        clauses.append("date <= ?")
        params.append(end)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT ticker, venue, commodity, date, open, high, low, close, volume, source
        FROM prices
        {where}
        ORDER BY date ASC
        LIMIT ?
        """,
        params,
    ).fetchall()
    result = [dict(r) for r in rows]
    if owns:
        conn.close()
    return result


# ---------------------------------------------------------------------------
# Trade-policy constraints
# ---------------------------------------------------------------------------


def upsert_policies(
    rows: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> UpsertResult:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    inserted = updated = 0

    for row in rows:
        pid = row["id"]
        existing = conn.execute("SELECT id FROM policies WHERE id = ?", (pid,)).fetchone()
        applies = row.get("applies_to_nodes") or ""
        if isinstance(applies, list):
            applies = ",".join(applies)
        vals = (
            row["jurisdiction"],
            row.get("commodity") or "",
            row["policy_type"],
            row["title"],
            row.get("description") or "",
            row.get("constraint_value"),
            row.get("unit") or "",
            float(row.get("severity") or 0.5),
            row["effective_from"],
            row.get("effective_to"),
            applies,
            row.get("source") or "",
            row.get("notes") or "",
        )
        if existing is None:
            conn.execute(
                """
                INSERT INTO policies (
                    id, jurisdiction, commodity, policy_type, title, description,
                    constraint_value, unit, severity, effective_from, effective_to,
                    applies_to_nodes, source, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (pid, *vals),
            )
            inserted += 1
        else:
            conn.execute(
                """
                UPDATE policies SET
                    jurisdiction = ?, commodity = ?, policy_type = ?, title = ?,
                    description = ?, constraint_value = ?, unit = ?, severity = ?,
                    effective_from = ?, effective_to = ?, applies_to_nodes = ?,
                    source = ?, notes = ?
                WHERE id = ?
                """,
                (*vals, pid),
            )
            updated += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
    if owns:
        conn.close()
    return UpsertResult(inserted=inserted, updated=updated, total=total)


def list_policies(
    *,
    commodity: str | None = None,
    jurisdiction: str | None = None,
    as_of: str | None = None,
    active_only: bool = True,
    limit: int = 500,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    clauses: list[str] = []
    params: list[Any] = []
    if commodity:
        clauses.append("(commodity = ? OR commodity = '' OR commodity = '*')")
        params.append(commodity)
    if jurisdiction:
        clauses.append("jurisdiction = ?")
        params.append(jurisdiction)
    if as_of and active_only:
        clauses.append("effective_from <= ?")
        params.append(as_of)
        clauses.append("(effective_to IS NULL OR effective_to = '' OR effective_to >= ?)")
        params.append(as_of)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT id, jurisdiction, commodity, policy_type, title, description,
               constraint_value, unit, severity, effective_from, effective_to,
               applies_to_nodes, source, notes
        FROM policies
        {where}
        ORDER BY effective_from DESC, jurisdiction
        LIMIT ?
        """,
        params,
    ).fetchall()
    result = [dict(r) for r in rows]
    if owns:
        conn.close()
    return result


def apply_policies_to_network(
    network: dict,
    *,
    as_of: str | None = None,
    commodity: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Attach active policies onto matching nodes (policy_ids / policy_severity)."""
    from datetime import date

    as_of = as_of or date.today().isoformat()
    policies = list_policies(
        commodity=commodity, as_of=as_of, active_only=True, limit=5000, conn=conn
    )
    by_id = {n["id"]: n for n in network.get("nodes") or [] if n.get("id")}
    touched = 0
    for pol in policies:
        targets = [
            t.strip()
            for t in str(pol.get("applies_to_nodes") or "").split(",")
            if t.strip()
        ]
        if not targets:
            jur = (pol.get("jurisdiction") or "").lower()
            targets = [
                nid
                for nid, n in by_id.items()
                if str(n.get("country") or "").lower() == jur
            ]
        for nid in targets:
            node = by_id.get(nid)
            if not node:
                continue
            ids = list(node.get("policy_ids") or [])
            if pol["id"] not in ids:
                ids.append(pol["id"])
            node["policy_ids"] = ids
            prev = float(node.get("policy_severity") or 0.0)
            sev = float(pol.get("severity") or 0.0)
            if sev >= prev:
                node["policy_severity"] = sev
                node["policy_title"] = pol["title"]
            touched += 1
    return touched


def sync_graph_to_db(
    network: dict,
    *,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Persist full graph mirror (nodes + edges) with a shared synced_at stamp."""
    import json
    from datetime import datetime, timezone

    owns = conn is None
    conn = conn or connect()
    init_db(conn)

    synced_at = datetime.now(timezone.utc).isoformat()
    network_name = str(network.get("network_name") or "unknown")
    sector = str(network.get("sector") or "")
    n_nodes = 0
    n_edges = 0

    for node in network.get("nodes") or []:
        nid = node.get("id")
        if not nid:
            continue
        attrs = {k: v for k, v in node.items() if k != "id"}
        conn.execute(
            """
            INSERT INTO node_snapshots (
                network_name, sector, node_id, node_type, commodity,
                capacity, stock, lat, lon, event_severity, attrs_json, synced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                network_name,
                sector,
                nid,
                str(node.get("type") or ""),
                str(node.get("commodity") or node.get("event_commodity") or ""),
                _opt_float(node.get("capacity_kt") or node.get("capacity_kbd")),
                _opt_float(node.get("stock_level_kt") or node.get("stock_level_kbbl")),
                _opt_float(node.get("lat")),
                _opt_float(node.get("lon")),
                float(node.get("event_severity") or 0.0),
                json.dumps(attrs, default=str),
                synced_at,
            ),
        )
        n_nodes += 1

    for edge in network.get("edges") or []:
        src, tgt = edge.get("source"), edge.get("target")
        if not src or not tgt:
            continue
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        conn.execute(
            """
            INSERT INTO edges (
                network_name, sector, source, target, flow_type, product_form,
                weight, transit_days, attrs_json, synced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                network_name,
                sector,
                src,
                tgt,
                str(edge.get("flow_type") or ""),
                str(edge.get("product_form") or ""),
                float(edge.get("weight") or 1.0),
                _opt_float(edge.get("transit_days")),
                json.dumps(attrs, default=str),
                synced_at,
            ),
        )
        n_edges += 1

    conn.commit()
    if owns:
        conn.close()
    return {"nodes": n_nodes, "edges": n_edges, "synced_at": synced_at}

def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_node_snapshots(
    *,
    network_name: str | None = None,
    limit: int = 5000,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    if network_name:
        rows = conn.execute(
            """
            SELECT * FROM node_snapshots
            WHERE network_name = ?
            ORDER BY synced_at DESC, node_id
            LIMIT ?
            """,
            (network_name, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM node_snapshots
            ORDER BY synced_at DESC, node_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    result = [dict(r) for r in rows]
    if owns:
        conn.close()
    return result


def list_graph_edges(
    *,
    network_name: str | None = None,
    limit: int = 10000,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    owns = conn is None
    conn = conn or connect()
    init_db(conn)
    if network_name:
        rows = conn.execute(
            """
            SELECT * FROM edges
            WHERE network_name = ?
            ORDER BY synced_at DESC, source, target
            LIMIT ?
            """,
            (network_name, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM edges
            ORDER BY synced_at DESC, source, target
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    result = [dict(r) for r in rows]
    if owns:
        conn.close()
    return result
