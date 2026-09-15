"""Project configuration: paths, tickers, thresholds, environment."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

NETWORK_DATA_DIR = BACKEND_DIR / "graph_core" / "network_data"
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = Path(os.getenv("GCP_DATA_DIR", ROOT_DIR / "data"))
DB_PATH = Path(os.getenv("GCP_DB_PATH", DATA_DIR / "geo_commodity.db"))

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEFAULT_SECTOR = os.getenv("GCP_SECTOR", "metals")
DEFAULT_COMMODITY = os.getenv("GCP_COMMODITY", "copper")
LOG_LEVEL = os.getenv("GCP_LOG_LEVEL", "INFO")
DEVICE = os.getenv("GCP_DEVICE", "cpu")  # cpu | cuda | mps

# ---------------------------------------------------------------------------
# Market tickers (proxies — refine per venue later)
# ---------------------------------------------------------------------------
TICKERS: dict[str, str] = {
    "copper": os.getenv("GCP_TICKER_COPPER", "HG=F"),
    "aluminum": os.getenv("GCP_TICKER_ALUMINUM", "ALI=F"),
    "oil": os.getenv("GCP_TICKER_OIL", "CL=F"),
    "lng": os.getenv("GCP_TICKER_LNG", "NG=F"),
    "wheat": os.getenv("GCP_TICKER_WHEAT", "ZW=F"),
    "corn": os.getenv("GCP_TICKER_CORN", "ZC=F"),
}

# ---------------------------------------------------------------------------
# Event / model thresholds
# ---------------------------------------------------------------------------
EVENT_SEVERITY_MIN = float(os.getenv("GCP_EVENT_SEVERITY_MIN", "0.3"))
EVENT_DECAY_HALF_LIFE_HOURS = float(os.getenv("GCP_EVENT_DECAY_HALF_LIFE_HOURS", "72"))
TENSION_LONG_THRESHOLD = float(os.getenv("GCP_TENSION_LONG", "0.65"))
TENSION_SHORT_THRESHOLD = float(os.getenv("GCP_TENSION_SHORT", "0.35"))

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
RSS_FEEDS: list[str] = [
    feed.strip()
    for feed in os.getenv(
        "GCP_RSS_FEEDS",
        "https://www.mining.com/feed/,https://oilprice.com/rss/main",
    ).split(",")
    if feed.strip()
]
LLM_MODEL_NAME = os.getenv("GCP_LLM_MODEL", "local")


def ensure_dirs() -> None:
    """Create runtime directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
