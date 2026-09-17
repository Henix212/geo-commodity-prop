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
    "silver": os.getenv("GCP_TICKER_SILVER", "SI=F"),
    "gold": os.getenv("GCP_TICKER_GOLD", "GC=F"),
    "oil": os.getenv("GCP_TICKER_OIL", "CL=F"),
    "lng": os.getenv("GCP_TICKER_LNG", "NG=F"),
    "wheat": os.getenv("GCP_TICKER_WHEAT", "ZW=F"),
    "corn": os.getenv("GCP_TICKER_CORN", "ZC=F"),
}

# Venue-tagged series for LME / SHFE / COMEX (yfinance proxies where needed).
# Format: commodity -> venue -> yahoo ticker
VENUE_TICKERS: dict[str, dict[str, str]] = {
    "copper": {
        "COMEX": os.getenv("GCP_TICKER_COPPER_COMEX", "HG=F"),
        "LME": os.getenv("GCP_TICKER_COPPER_LME", "HG=F"),  # proxy until dedicated LME feed
        "SHFE": os.getenv("GCP_TICKER_COPPER_SHFE", "HG=F"),  # proxy
    },
    "aluminum": {
        "COMEX": os.getenv("GCP_TICKER_ALUMINUM_COMEX", "ALI=F"),
        "LME": os.getenv("GCP_TICKER_ALUMINUM_LME", "ALI=F"),
        "SHFE": os.getenv("GCP_TICKER_ALUMINUM_SHFE", "ALI=F"),
    },
    "gold": {
        "COMEX": os.getenv("GCP_TICKER_GOLD_COMEX", "GC=F"),
        "LME": os.getenv("GCP_TICKER_GOLD_LME", "GC=F"),
        "SHFE": os.getenv("GCP_TICKER_GOLD_SHFE", "GC=F"),
    },
    "silver": {
        "COMEX": os.getenv("GCP_TICKER_SILVER_COMEX", "SI=F"),
        "LME": os.getenv("GCP_TICKER_SILVER_LME", "SI=F"),
        "SHFE": os.getenv("GCP_TICKER_SILVER_SHFE", "SI=F"),
    },
    "oil": {"NYMEX": os.getenv("GCP_TICKER_OIL", "CL=F")},
    "lng": {"NYMEX": os.getenv("GCP_TICKER_LNG", "NG=F")},
    "wheat": {"CBOT": os.getenv("GCP_TICKER_WHEAT", "ZW=F")},
    "corn": {"CBOT": os.getenv("GCP_TICKER_CORN", "ZC=F")},
}

SEED_DIR = BACKEND_DIR / "database" / "seed"

# ---------------------------------------------------------------------------
# Event / model thresholds
# ---------------------------------------------------------------------------
EVENT_SEVERITY_MIN = float(os.getenv("GCP_EVENT_SEVERITY_MIN", "0.3"))
EVENT_CONFIDENCE_MIN = float(os.getenv("GCP_EVENT_CONFIDENCE_MIN", "0.4"))
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
        ",".join(
            [
                "https://www.mining.com/feed/",
                "https://oilprice.com/rss/main",
                "https://www.eia.gov/rss/todayinenergy.xml",
                "https://news.google.com/rss/search?q=copper+OR+Hormuz+OR+LNG+OR+Escondida&hl=en-US&gl=US&ceid=US:en",
            ]
        ),
    ).split(",")
    if feed.strip()
]

GDELT_QUERY = os.getenv(
    "GCP_GDELT_QUERY",
    "(copper OR aluminum OR aluminium OR silver OR gold OR LNG OR crude OR oil "
    "OR wheat OR corn OR Hormuz OR Panama OR Suez OR Escondida OR Malacca "
    "OR strike OR sanction OR outage OR blockade)",
)
GDELT_TIMESPAN = os.getenv("GCP_GDELT_TIMESPAN", "24h")
GDELT_MAXRECORDS = int(os.getenv("GCP_GDELT_MAXRECORDS", "75"))
LLM_MODEL_PATH = os.getenv(
    "GCP_LLM_MODEL",
    str(MODELS_DIR / "Qwen2.5-7B-Instruct"),
)
LLM_MAX_NEW_TOKENS = int(os.getenv("GCP_LLM_MAX_NEW_TOKENS", "256"))
LLM_PARSE_LIMIT = int(os.getenv("GCP_LLM_PARSE_LIMIT", "20"))


def ensure_dirs() -> None:
    """Create runtime directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
