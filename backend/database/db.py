"""SQLite connection and schema initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    uid TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    published TEXT,
    summary TEXT NOT NULL DEFAULT '',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    fetch_count INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_articles_last_seen ON articles(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_uid TEXT NOT NULL DEFAULT '',
    entity_text TEXT NOT NULL,
    node_id TEXT,
    event_type TEXT NOT NULL,
    severity REAL NOT NULL,
    commodity TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    match_score REAL NOT NULL DEFAULT 0,
    extracted_at TEXT NOT NULL,
    UNIQUE(article_uid, entity_text, event_type, commodity, direction)
);

CREATE INDEX IF NOT EXISTS idx_events_node ON events(node_id);
CREATE INDEX IF NOT EXISTS idx_events_commodity ON events(commodity);
CREATE INDEX IF NOT EXISTS idx_events_extracted ON events(extracted_at);
"""


def get_db_path() -> Path:
    config.ensure_dirs()
    return config.DB_PATH


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> None:
    owns = conn is None
    conn = conn or connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        if owns:
            conn.close()
