# TODO — Geo Commodity Prop

GNN for shock propagation along physical commodity supply chains  
(mines → ports → chokepoints → smelters → refiners → price / alpha).

Target stack: PyTorch Geometric · NetworkX · transformers / local LLM · VectorBT · yfinance.

---

## Phase 0 — Project foundations

- [x] `pyproject.toml` + `uv` (torch, torch-geometric, networkx, pandas, numpy, transformers, requests, beautifulsoup4, yfinance, vectorbt, pyyaml)
- [x] `backend/` layout (`graph_core`, `ingestion`, `database`, `quant`, `models`)
- [x] YAML structure `network_data/{shared,metals,energy,agriculture}/`
- [x] `config.py` — paths, tickers, thresholds, env
- [x] `main.py` — entry point (build graph → events → inference → signal)
- [x] Python packages (`__init__.py`) + clean imports

---



## Phase 1 — Graph modeling (physical network)

Files: `backend/graph_core/` + `network_data/`

### Static YAML topology

- [x] Copper: mines, ports, smelters, refiners, scrap, consumers, exchanges
- [x] Aluminum — bauxite → alumina → smelter
- [x] Silver / Gold — mines → refiners → LBMA/COMEX/SGE vaults
- [x] Oil — fields → terminals → Hormuz/Malacca/Suez → refineries
- [x] LNG — liquefaction → chokepoints → regas → hubs (TTF/JKM/HH)
- [x] Wheat / Corn — regions → silos/ports → Bosphorus / Gulf → mills
- [x] Shared bottlenecks (+ Hormuz, Bosphorus, Gibraltar)



### Builder → GNN tensors

- [x] `builder.py` — merge shared + sector YAML (+ `load_commodity`)
- [x] NetworkX export (`MultiDiGraph`) from merged network
- [x] PyG conversion (`torch_geometric.data.Data`)
- [x] Node features: type one-hot, capacity, stock, lat/lon, event severity
- [x] Edge features: flow_type, weight, product_form, transit_days
- [x] Validation: edge endpoints + `accepts_forms` consistency



### Quant data (outside YAML — DB / market)

- [x] Actual mine production (vs nameplate `capacity_kt`) → `database/`
- [x] TC/RC by route → `database/`
- [x] LME / SHFE / COMEX price time series → `quant/market_data.py`
- [x] Trade-policy constraints (Indonesia ore rules, Chile taxes, …) → dated table / config
- [x] Periodic jobs: refresh prices, refresh production

---



## Phase 2 — Event injection (NLP engine)

Files: `backend/ingestion/`

- [x] `scrapers.py` — RSS (Mining.com, OilPrice, EIA, Google News) + GDELT + NASA EONET
- [x] `parser_llm.py` — LLM → structured shock JSON (+ commodity/direction/confidence)
- [x] Text entity → graph `node_id` mapping (gazetteer / fuzzy)
- [x] Event types: strike, accident, sanction, weather, congestion, force majeure, …
- [x] Dynamic injection: temporary node `event_severity` feature
- [x] Persist events in DB (`database/models.py`)
- [x] Event severity time decay (half-life from config)

---



## Phase 3 — GNN propagation (model core)

Files: `backend/models/` + training

- [x] Baseline: NetworkX diffusion / random walk (sanity check without deep learning)
- [x] GAT architecture (Graph Attention) — PyG
- [x] Input: graph + event features → output: tension score per commodity / node
- [x] Labels: historical post-shock price reactions (supervised / semi-supervised)
- [x] Training, checkpoints, eval (tension MAE, price directionality)
- [x] Ablations: with/without scrap, with/without chokepoints

---



## Phase 4 — Quant / alpha / backtest

Files: `backend/quant/`

- [x] `market_data.py` — fetch & cache prices (yfinance / futures APIs)
- [x] `strategy.py` — GNN tension → long/short signal
- [x] VectorBT backtest (sharpe, drawdown, turnover)
- [x] Stress scenarios: Escondida outage, Panama blockade, Suez closure
- [x] Centrality analysis (betweenness) → single points of failure

---



## Phase 5 — Database & ops

Files: `backend/database/`

- [x] `db.py` — SQLite connection + schema init
- [x] `models.py` — article upsert/list (scraper history preserved by uid)
- [x] `models.py` — shock events upsert/list
- [x] `models.py` — production, TC/RC, prices, policies (+ seed YAML)
- [x] `models.py` — node snapshots, edges (full graph mirror)
- [x] Periodic jobs: refresh prices, refresh production
- [x] Basic logging / monitoring

---



## Smoke commands (suggested steps — done)

```bash
# 1. Synthetic event inject (no LLM)
uv run python -m backend.ingestion.inject_demo --node mine_escondida --commodity copper
uv run python -m backend.main --demo-event --baseline --skip-ingest --sync-graph

# 2. Diffusion / centrality on copper & oil sectors
uv run python -m backend.graph_core.analytics --sector metals --top-k 10
uv run python -m backend.graph_core.analytics --sector energy --top-k 10

# 3. Core futures HG=F CL=F GC=F SI=F
uv run python -m backend.quant.market_data --commodity core

# 4. Minimal GAT → tension → VectorBT
uv run python -m backend.models.train --sector metals --epochs 5 --n-synthetic 24
uv run python -m backend.main --demo-event --use-gat --skip-ingest
uv run python -m backend.quant.backtest --commodity copper --refresh
uv run python -m backend.quant.scenarios --scenario escondida_outage
```
