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

- [ ] Actual mine production (vs nameplate `capacity_kt`) → `database/`
- [ ] TC/RC by route → `database/`
- [ ] LME / SHFE / COMEX price time series → `quant/market_data.py`
- [ ] Trade-policy constraints (Indonesia ore rules, Chile taxes, …) → dated table / config

---



## Phase 2 — Event injection (NLP engine)

Files: `backend/ingestion/`

- [x] `scrapers.py` — RSS (Mining.com, OilPrice, EIA, Google News) + GDELT + NASA EONET
- [ ] `parser_llm.py` — LLM → triplet `[entity, event_type, severity 0–1]`
- [ ] Text entity → graph `node_id` mapping (alias / fuzzy / gazetteer)
- [ ] Event types: strike, accident, sanction, weather, congestion, force majeure
- [ ] Dynamic injection: temporary node feature (time decay)
- [ ] Persist events in DB (`database/models.py`)

---



## Phase 3 — GNN propagation (model core)

Files: `backend/models/` + training

- [ ] Baseline: NetworkX diffusion / random walk (sanity check without deep learning)
- [ ] GAT architecture (Graph Attention) — PyG
- [ ] Input: graph + event features → output: tension score per commodity / node
- [ ] Labels: historical post-shock price reactions (supervised / semi-supervised)
- [ ] Training, checkpoints, eval (tension MAE, price directionality)
- [ ] Ablations: with/without scrap, with/without chokepoints

---



## Phase 4 — Quant / alpha / backtest

Files: `backend/quant/`

- [ ] `market_data.py` — fetch & cache prices (yfinance / futures APIs)
- [ ] `strategy.py` — GNN tension → long/short signal
- [ ] VectorBT backtest (sharpe, drawdown, turnover)
- [ ] Stress scenarios: Escondida outage, Panama blockade, Suez closure
- [ ] Centrality analysis (betweenness) → single points of failure

---



## Phase 5 — Database & ops

Files: `backend/database/`

- [x] `db.py` — SQLite connection + schema init
- [x] `models.py` — article upsert/list (scraper history preserved by uid)
- [ ] `models.py` — node snapshots, edges, events, prices, production, policies
- [ ] Periodic jobs: refresh prices, refresh production
- [ ] Basic logging / monitoring

---



## Suggested next steps

1. RSS scraper + LLM parser → inject one event onto a node (Phase 2)
2. NetworkX diffusion / centrality baseline on copper & oil
3. `market_data.py` — HG=F, CL=F, GC=F, SI=F series
4. Minimal GAT → tension score → VectorBT smoke backtest

