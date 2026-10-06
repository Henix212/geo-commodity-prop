# Geo Commodity Prop

GNN for **shock propagation** along physical commodity supply chains  
(mines → ports → chokepoints → smelters → tension / trading signal).

> Research / education prototype — **not** financial advice.

## Architecture

1. **Physical graph** — YAML topology (`datasets/network`) → NetworkX / PyG  
2. **Events** — scrape news → LLM parse → map to `node_id` → inject  
3. **GNN (GAT)** — propagate shocks → commodity tension scores  
4. **Quant** — tension → signal → backtest equity curve  
5. **Dashboard** — Streamlit map + tension + portfolio  

```text
src/geo_commodity/   # library (graph, events, store, gnn, quant)
datasets/            # versioned network YAML + seed reference data
apps/dashboard/      # Streamlit UI
apps/api/            # optional FastAPI
data/                # runtime DB / checkpoints / cache (gitignored)
```

## Quick start

```bash
uv sync
uv run streamlit run apps/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501).

### Pipeline CLI

```bash
uv run geo-commodity --sector energy --skip-ingest
uv run python -m geo_commodity.events.scrapers
uv run python -m geo_commodity.events.parser_llm --limit 1 --sector energy
uv run python -m geo_commodity.store.seed_data
uv run python -m geo_commodity.quant.market_data --commodity oil --period 1y
# GCP_DEVICE=cuda uv run python -m geo_commodity.gnn.train --sector all --epochs 30
```

### Optional API

```bash
uv run uvicorn apps.api.app:app --reload --port 8000
```

## Data layout

| Path | Role |
|------|------|
| `datasets/network/` | Static topology (shared bottlenecks + sector YAML) |
| `datasets/seed/` | Production / TC-RC / policies reference |
| `data/db/` | SQLite events & prices |
| `data/checkpoints/` | GAT `best.pt` |
| `data/cache/` | yfinance + optional LLM weights |

## Disclaimer

This project is a technical demonstration of graph ML on commodity networks.  
Signals and backtests use proxies and synthetic tension series where noted.  
Do not trade on this output without your own research and risk controls.
