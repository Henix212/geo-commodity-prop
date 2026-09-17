# Geo Commodity Prop

GNN for shock propagation along physical commodity supply chains  
(mines → ports → chokepoints → smelters → refiners → tension / alpha).

Detailed roadmap: see `[TODO.md](TODO.md)`.

## Architecture

1. **Physical graph** — nodes (mines, ports, bottlenecks, …) + edges (flows) in YAML → NetworkX / PyG
2. **NLP events** — scrape news → LLM → structured shock JSON → map to `node_id` → inject (+ time decay)
3. **Propagation** — NetworkX diffusion baseline and/or GAT → commodity tension
4. **Quant** — tension → long/short signal → VectorBT backtest / stress scenarios
5. **Dashboard** — Streamlit reads SQLite + `dashboard_state.json` (PyVis map, event feed, Plotly)

```text
backend/
  graph_core/     # builder + network_data YAML
  ingestion/      # scrapers + LLM parser
  database/       # production, TC/RC, policies, events, graph mirror
  quant/          # market_data + strategy + backtest + scenarios
  models/         # GAT + diffusion + centrality (+ LLM weights dir)
  jobs.py         # periodic price / production / graph sync / snapshot
  dashboard_state.py
frontend/
  app.py          # Streamlit UI
  viz.py          # PyVis + Plotly helpers
```

## Network data layout

```text
backend/graph_core/network_data/
  shared/          # bottlenecks (Hormuz, Suez, Malacca, Bosphorus, …)
  metals/          # copper, aluminum, silver, gold
  energy/          # oil, lng
  agriculture/     # wheat, corn
```

YAML = static topology. Prices, production, TC/RC, and policies live in `database/` + `quant/`.

## Run

```bash
uv sync

# Seed reference tables + graph mirror + inject events + diffusion/GAT tension
uv run python -m backend.database.seed_data
uv run python -m backend.main --sector metals --skip-ingest --inference diffusion

# Scrapers / LLM
uv run python -m backend.ingestion.scrapers
uv run python -m backend.ingestion.parser_llm --limit 5 --sector metals
# GCP_DEVICE=cuda uv run python -m backend.main --sector metals --from-db --parse-limit 5

# Train GAT (weak labels: diffusion teacher ± price reactions)
uv run python -m backend.models.train --sector metals --epochs 20 --ablate

# Centrality / SPOF
uv run python -m backend.models.centrality --sector metals

# Stress scenarios
uv run python -m backend.quant.scenarios --scenario escondida_outage --sector metals
uv run python -m backend.quant.scenarios --scenario suez_closure --sector energy

# Prices + VectorBT smoke backtest
uv run python -m backend.quant.market_data --commodity copper --venue COMEX --period 1y
uv run python -m backend.quant.backtest --commodity copper --tension 0.7

# Periodic jobs
uv run python -m backend.jobs prices
uv run python -m backend.jobs production
uv run python -m backend.jobs sync_graph --sector metals
uv run python -m backend.jobs snapshot --sector metals --inference diffusion

# Dashboard (read-only — refresh snapshot first)
uv run streamlit run frontend/app.py

# Large-scale realistic e2e (multi-shock events + 2y prices + multi-sector + backtests)
uv run python -m backend.e2e_scale
# uv run python -m backend.e2e_scale --skip-prices
# uv run python -m backend.database.seed_data --only events
```

### Crontab examples

```cron
0 6 * * 1-5  cd /path/to/geo-commodity-prop && uv run python -m backend.jobs prices --period 5d
0 7 * * 1    cd /path/to/geo-commodity-prop && uv run python -m backend.jobs production
0 8 * * 1    cd /path/to/geo-commodity-prop && uv run python -m backend.jobs sync_graph --sector metals
15 8 * * 1-5 cd /path/to/geo-commodity-prop && uv run python -m backend.jobs snapshot --sector metals
```

Logs: `data/app.log`. Job history: `data/job_runs.jsonl`. Health: `data/health.json`.  
Dashboard snapshot: `data/dashboard_state.json`.  
GAT checkpoints: `data/checkpoints/shock_gat.pt`.  
Half-life: `GCP_EVENT_DECAY_HALF_LIFE_HOURS` (default 72).

## Data sources (topology)

- [USGS Copper](https://www.usgs.gov/centers/national-minerals-information-center/copper-statistics-and-information)
- [ICSG](https://icsg.org)
- [UN Comtrade](https://comtradeplus.un.org)
- [Cochilco](https://www.cochilco.cl)
- Company reports (BHP, Freeport, Codelco, Glencore, …)
