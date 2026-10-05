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
GNN_DIR = Path(os.getenv("GCP_GNN_DIR", MODELS_DIR / "gnn"))
DATA_DIR = Path(os.getenv("GCP_DATA_DIR", ROOT_DIR / "data"))
LOG_DIR = Path(os.getenv("GCP_LOG_DIR", DATA_DIR / "logs"))
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
DIFFUSION_STEPS = int(os.getenv("GCP_DIFFUSION_STEPS", "8"))
DIFFUSION_DECAY = float(os.getenv("GCP_DIFFUSION_DECAY", "0.85"))
FORWARD_RETURN_DAYS = int(os.getenv("GCP_FORWARD_RETURN_DAYS", "5"))
GNN_HIDDEN = int(os.getenv("GCP_GNN_HIDDEN", "64"))
GNN_HEADS = int(os.getenv("GCP_GNN_HEADS", "4"))
GNN_EPOCHS = int(os.getenv("GCP_GNN_EPOCHS", "30"))
GNN_LR = float(os.getenv("GCP_GNN_LR", "1e-3"))
GNN_CHECKPOINT = Path(
    os.getenv("GCP_GNN_CHECKPOINT", str(GNN_DIR / "best.pt"))
)

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
# Sector-level RSS (broad). Per-commodity Google News queries live in
# COMMODITY_NEWS_QUERIES and are scraped with equal weight.
RSS_FEEDS: list[str] = [
    feed.strip()
    for feed in os.getenv(
        "GCP_RSS_FEEDS",
        ",".join(
            [
                "https://www.mining.com/feed/",
                "https://oilprice.com/rss/main",
                "https://www.eia.gov/rss/todayinenergy.xml",
            ]
        ),
    ).split(",")
    if feed.strip()
]

# Balanced news queries — one slot per tracked commodity.
COMMODITY_NEWS_QUERIES: dict[str, str] = {
    "copper": "(copper OR Escondida OR Codelco) AND (mine OR strike OR outage OR sanction OR port)",
    "aluminum": "(aluminum OR aluminium OR bauxite OR alumina) AND (smelter OR mine OR outage OR sanction)",
    "silver": "(silver mining OR silver mine OR LBMA silver) AND (mine OR strike OR outage)",
    "gold": "(gold mining OR gold mine OR LBMA gold) AND (mine OR strike OR outage OR sanction)",
    "oil": "(crude oil OR Brent OR WTI OR OPEC) AND (pipeline OR refinery OR Hormuz OR sanction OR outage)",
    "lng": "(LNG OR liquefied natural gas) AND (terminal OR Qatar OR Hormuz OR outage OR sanction)",
    "wheat": "(wheat OR Black Sea grain) AND (export OR drought OR blockade OR port OR harvest)",
    "corn": "(corn OR maize) AND (export OR drought OR harvest OR port OR USDA)",
}

GDELT_QUERY = os.getenv(
    "GCP_GDELT_QUERY",
    "(copper OR aluminum OR aluminium OR silver OR gold OR LNG OR crude OR oil "
    "OR wheat OR corn OR Hormuz OR Panama OR Suez OR Escondida OR Malacca "
    "OR strike OR sanction OR outage OR blockade)",
)
GDELT_TIMESPAN = os.getenv("GCP_GDELT_TIMESPAN", "24h")
# Total GDELT budget; split evenly across COMMODITY_NEWS_QUERIES when balanced.
GDELT_MAXRECORDS = int(os.getenv("GCP_GDELT_MAXRECORDS", "80"))
GDELT_PER_COMMODITY = int(
    os.getenv(
        "GCP_GDELT_PER_COMMODITY",
        str(max(5, GDELT_MAXRECORDS // max(1, len(COMMODITY_NEWS_QUERIES)))),
    )
)
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
    GNN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
