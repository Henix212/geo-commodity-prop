# Geo Commodity Prop

GNN de propagation de chocs sur les chaînes physiques commodities  
(mines → ports → chokepoints → smelters → raffineries → tension / alpha).

Roadmap détaillée : voir [`TODO.md`](TODO.md).

## Architecture

1. **Graphe physique** — nœuds (mines, ports, bottlenecks, …) + arêtes (flux) en YAML → NetworkX / PyG  
2. **Événements NLP** — scrape news → LLM → `[entité, type, sévérité]` injecté sur le graphe  
3. **GNN (GAT)** — propagation de l’onde de choc → score de tension par commodity  
4. **Quant** — signal → backtest VectorBT

```text
backend/
  graph_core/     # builder + network_data YAML
  ingestion/      # scrapers + parser LLM
  database/       # prod, stocks, TC/RC, policies, events
  quant/          # market_data + strategy + backtest
  models/         # checkpoints GNN
```

## Data sources (topologie)

- [USGS Copper](https://www.usgs.gov/centers/national-minerals-information-center/copper-statistics-and-information)
- [ICSG](https://icsg.org)
- [UN Comtrade](https://comtradeplus.un.org)
- [Cochilco](https://www.cochilco.cl)
- Company reports (BHP, Freeport, Codelco, Glencore, …)
- Port / shipping stats (MarineTraffic, port authorities)

## Network data layout

```text
backend/graph_core/network_data/
  shared/          # bottlenecks (Hormuz, Suez, Malacca, Bosphore, …)
  metals/          # copper, aluminum, silver, gold
  energy/          # oil, lng
  agriculture/     # wheat, corn
```

YAML = topologie statique. Prix, prod réelle, TC/RC, policies → `database/` + `quant/` (pas les YAML).

## Run (Phase 0)

```bash
uv sync
uv run python -m backend.main --sector metals
# ou: PYTHONPATH=. python3 -m backend.main --sector metals
```
