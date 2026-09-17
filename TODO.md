# TODO — Geo Commodity Prop

GNN for shock propagation along physical commodity supply chains  
(mines → ports → chokepoints → smelters → refiners → price / alpha).

Target stack: PyTorch Geometric · NetworkX · transformers / local LLM · VectorBT · yfinance · Streamlit.

---

## Phase 0 — Project foundations

- [x] `pyproject.toml` + `uv`
- [x] `backend/` layout
- [x] YAML structure `network_data/{shared,metals,energy,agriculture}/`
- [x] `config.py` — paths, tickers, thresholds, env
- [x] `main.py` — entry point (build graph → events → inference → signal)
- [x] Python packages (`__init__.py`) + clean imports

---

## Phase 1 — Graph modeling

- [x] Commodity YAML topologies (metals / energy / agriculture) + shared bottlenecks
- [x] Builder → NetworkX / PyG + features + validation
- [x] Production, TC/RC, prices, policies in `database/`
- [x] Periodic jobs: refresh prices, refresh production

---

## Phase 2 — Event injection

- [x] Scrapers (RSS / GDELT / EONET)
- [x] LLM parser + entity → node_id
- [x] Persist events + time decay

---

## Phase 3 — GNN propagation

- [x] NetworkX diffusion baseline
- [x] GAT + weak-label train / eval / checkpoint
- [x] Ablations (scrap / chokepoints)
- [x] Centrality / SPOF

---

## Phase 4 — Quant

- [x] `market_data.py` — yfinance cache
- [x] `strategy.py` — tension → signal
- [x] Backtest (VectorBT / numpy fallback)
- [x] Stress scenarios (+ `compound_crisis`)

---

## Phase 5 — Database & ops

- [x] Articles / events / production / TC-RC / prices / policies
- [x] Graph mirror (`graph_nodes` / `graph_edges`)
- [x] Jobs + logging / health

---

## Phase 6 — Dashboard (Streamlit)

- [x] `dashboard_state.json` bridge
- [x] PyVis supply-chain map
- [x] Live event feed
- [x] Plotly alpha / signal charts

---

## Scale / e2e

- [x] Realistic multi-shock event seed (`events_realistic.yaml`)
- [x] `python -m backend.e2e_scale` — multi-sector + 2y prices + backtests + report

---

## Suggested next steps

1. `uv run python -m backend.e2e_scale` then `uv run streamlit run frontend/app.py`
2. Cron `jobs snapshot` after daily price refresh
3. Optional: tension history time series for richer alpha overlays
4. Optional: live USGS/ICSG production scrapers
