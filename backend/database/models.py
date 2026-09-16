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
