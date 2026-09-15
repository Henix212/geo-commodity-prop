# TODO — Geo Commodity Prop

GNN pour la propagation de chocs sur les chaînes physiques commodities  
(mines → ports → chokepoints → smelters → raffineries → prix / alpha).

Stack cible : PyTorch Geometric · NetworkX · transformers / LLM local · VectorBT · yfinance.

---

## Phase 0 — Fondations projet

- [x] `pyproject.toml` + `uv` (torch, torch-geometric, networkx, pandas, numpy, transformers, requests, beautifulsoup4, yfinance, vectorbt, pyyaml)
- [x] Layout `backend/` (`graph_core`, `ingestion`, `database`, `quant`, `models`)
- [x] Structure YAML `network_data/{shared,metals,energy,agriculture}/`
- [x] `config.py` — chemins, tickers, seuils, env
- [x] `main.py` — point d’entrée (build graph → events → inference → signal)
- [x] Packages Python (`__init__.py`) + imports propres

---

## Phase 1 — Modélisation du graphe (réseau physique)

Fichiers : `backend/graph_core/` + `network_data/`

### Topologie YAML (statique)
- [x] Copper : mines, ports, smelters, refiners, scrap, consumers, exchanges
- [x] Aluminum — bauxite → alumina → smelter
- [x] Silver / Gold — mines → refiners → vaults LBMA/COMEX/SGE
- [x] Oil — fields → terminals → Hormuz/Malacca/Suez → refineries
- [x] LNG — liquefaction → chokepoints → regas → hubs (TTF/JKM/HH)
- [x] Wheat / Corn — régions → silos/ports → Bosphore / Gulf → mills
- [x] Shared bottlenecks (+ Ormuz, Bosphore, Gibraltar)

### Builder → tenseurs GNN
- [x] `builder.py` — merge shared + sector YAML (+ `load_commodity`)
- [x] Export NetworkX (`nx.DiGraph`) depuis le merge
- [x] Conversion PyG (`torch_geometric.data.Data`)
- [x] Features nœud : type one-hot, capacity, stock, lat/lon, event severity
- [x] Features arête : flow_type, weight, product_form, transit_days
- [x] Validation : endpoints + cohérence `accepts_forms`

### Données quant (hors YAML — DB / market)
- [ ] Production réelle par mine (vs `capacity_kt`) → `database/`
- [ ] TC/RC par route → `database/`
- [ ] Prix LME / SHFE / COMEX (time series) → `quant/market_data.py`
- [ ] Policies commerciales (ban Indonésie, taxes Chili, …) → table / config datée

---

## Phase 2 — Injection d’événements (moteur NLP)

Fichiers : `backend/ingestion/`

- [ ] `scrapers.py` — RSS / news / rapports (USGS, maritime, energy ministries)
- [ ] `parser_llm.py` — LLM → triplet `[entité, type_event, sévérité 0–1]`
- [ ] Mapping entité texte → `node_id` du graphe (alias / fuzzy / gazetteer)
- [ ] Types d’événements : grève, accident, sanction, météo, congestion, force majeure
- [ ] Injection dynamique : feature temporaire sur le nœud (decay temporel)
- [ ] Persistance événements en DB (`database/models.py`)

---

## Phase 3 — Propagation GNN (cœur modèle)

Fichiers : `backend/models/` + entraînement

- [ ] Baseline : diffusion NetworkX / random walk (sanity check sans deep learning)
- [ ] Architecture GAT (Graph Attention) — PyG
- [ ] Input : graphe + features events → output : score de tension par commodity / nœud
- [ ] Labels : réactions prix historiques post-choc (pour supervised / semi-supervised)
- [ ] Entraînement, checkpoints, eval (MAE tension, directionnalité prix)
- [ ] Ablations : avec/sans scrap, avec/sans chokepoints

---

## Phase 4 — Quant / alpha / backtest

Fichiers : `backend/quant/`

- [ ] `market_data.py` — fetch & cache prix (yfinance / APIs futures)
- [ ] `strategy.py` — tension GNN → signal long/short
- [ ] Backtest VectorBT (sharpe, drawdown, turnover)
- [ ] Stress scenarios : coupe Escondida, blocage Panama, fermeture Suez
- [ ] Analyse centralité (betweenness) → single points of failure

---

## Phase 5 — Database & ops

Fichiers : `backend/database/`

- [ ] `models.py` — nodes snapshot, edges, events, prices, production, policies
- [ ] `db.py` — connexion, migrations, CRUD
- [ ] Jobs périodiques : scrape news, refresh prices, refresh production
- [ ] Logging / monitoring basique

---

## Ordre de travail suggéré (prochaines actions)

1. Scraper RSS + parser LLM → event injecté sur un nœud (Phase 2)
2. Baseline diffusion NetworkX / centralité sur copper & oil
3. `market_data.py` — séries HG=F, CL=F, GC=F, SI=F
4. GAT minimal → score tension → VectorBT smoke backtest
