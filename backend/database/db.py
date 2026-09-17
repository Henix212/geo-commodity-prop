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

CREATE TABLE IF NOT EXISTS node_production (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    commodity TEXT NOT NULL,
    year INTEGER NOT NULL,
    production_kt REAL NOT NULL,
    capacity_kt REAL,
    utilization REAL,
    source TEXT NOT NULL DEFAULT '',
    as_of TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE(node_id, commodity, year)
);

CREATE INDEX IF NOT EXISTS idx_production_node ON node_production(node_id);
CREATE INDEX IF NOT EXISTS idx_production_year ON node_production(year);

CREATE TABLE IF NOT EXISTS tc_rc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity TEXT NOT NULL,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    tc_usd_per_dmt REAL,
    rc_usc_per_lb REAL,
    effective_from TEXT NOT NULL,
    effective_to TEXT,
    source TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE(commodity, from_node, to_node, effective_from)
);

CREATE INDEX IF NOT EXISTS idx_tc_rc_route ON tc_rc(from_node, to_node);
CREATE INDEX IF NOT EXISTS idx_tc_rc_commodity ON tc_rc(commodity);

CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT '',
    commodity TEXT NOT NULL DEFAULT '',
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    source TEXT NOT NULL DEFAULT 'yfinance',
    PRIMARY KEY (ticker, venue, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_commodity ON prices(commodity, date);
CREATE INDEX IF NOT EXISTS idx_prices_venue ON prices(venue, date);

CREATE TABLE IF NOT EXISTS policies (
    id TEXT PRIMARY KEY,
    jurisdiction TEXT NOT NULL,
    commodity TEXT NOT NULL DEFAULT '',
    policy_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    constraint_value REAL,
    unit TEXT NOT NULL DEFAULT '',
    severity REAL NOT NULL DEFAULT 0.5,
    effective_from TEXT NOT NULL,
    effective_to TEXT,
    applies_to_nodes TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_policies_jurisdiction ON policies(jurisdiction);
CREATE INDEX IF NOT EXISTS idx_policies_commodity ON policies(commodity);
CREATE INDEX IF NOT EXISTS idx_policies_effective ON policies(effective_from);

CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id TEXT NOT NULL,
    sector TEXT NOT NULL,
    commodity TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT '',
    country TEXT NOT NULL DEFAULT '',
    lat REAL,
    lon REAL,
    capacity_kt REAL,
    attrs_json TEXT NOT NULL DEFAULT '{}',
    synced_at TEXT NOT NULL,
    PRIMARY KEY (sector, node_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_commodity ON graph_nodes(commodity);

CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sector TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    flow_type TEXT NOT NULL DEFAULT '',
    product_form TEXT NOT NULL DEFAULT '',
    weight REAL,
    transit_days REAL,
    attrs_json TEXT NOT NULL DEFAULT '{}',
    synced_at TEXT NOT NULL,
    UNIQUE(sector, source, target, flow_type, product_form)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_sector ON graph_edges(sector);
CREATE INDEX IF NOT EXISTS idx_graph_edges_endpoints ON graph_edges(source, target);
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
        _migrate_prices_pk(conn)
        conn.commit()
    finally:
        if owns:
            conn.close()


def _migrate_prices_pk(conn: sqlite3.Connection) -> None:
    """Recreate prices table if an older PK (ticker, date) is present."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='prices'"
    ).fetchone()
    if not row or not row[0]:
        return
    sql = " ".join(row[0].split())
    if "PRIMARY KEY (ticker, venue, date)" in sql:
        return
    if "PRIMARY KEY (ticker, date)" not in sql:
        return
    conn.executescript(
        """
        ALTER TABLE prices RENAME TO prices_legacy;
        CREATE TABLE prices (
            ticker TEXT NOT NULL,
            venue TEXT NOT NULL DEFAULT '',
            commodity TEXT NOT NULL DEFAULT '',
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT NOT NULL DEFAULT 'yfinance',
            PRIMARY KEY (ticker, venue, date)
        );
        INSERT OR IGNORE INTO prices (
            ticker, venue, commodity, date, open, high, low, close, volume, source
        )
        SELECT ticker, venue, commodity, date, open, high, low, close, volume, source
        FROM prices_legacy;
        DROP TABLE prices_legacy;
        CREATE INDEX IF NOT EXISTS idx_prices_commodity ON prices(commodity, date);
        CREATE INDEX IF NOT EXISTS idx_prices_venue ON prices(venue, date);
        """
    )
