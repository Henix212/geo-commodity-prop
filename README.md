# Geo Commodity Prop

GNN for shock propagation along physical commodity supply chains  
(mines → ports → chokepoints → smelters → refiners → tension / alpha).

Detailed roadmap: see [`TODO.md`](TODO.md).

## Architecture

1. **Physical graph** — nodes (mines, ports, bottlenecks, …) + edges (flows) in YAML → NetworkX / PyG  
2. **NLP events** — scrape news → LLM → `[entity, type, severity]` injected into the graph  
3. **GNN (GAT)** — shock-wave propagation → tension score per commodity  
4. **Quant** — signal → VectorBT backtest

```text
backend/
  graph_core/     # builder + network_data YAML
  ingestion/      # scrapers + LLM parser
  database/       # production, stocks, TC/RC, policies, events
  quant/          # market_data + strategy + backtest
  models/         # GNN checkpoints
```

## Data sources (topology)

- [USGS Copper](https://www.usgs.gov/centers/national-minerals-information-center/copper-statistics-and-information)
- [ICSG](https://icsg.org)
- [UN Comtrade](https://comtradeplus.un.org)
- [Cochilco](https://www.cochilco.cl)
- Company reports (BHP, Freeport, Codelco, Glencore, …)
- Port / shipping stats (MarineTraffic, port authorities)

## Network data layout

```text
backend/graph_core/network_data/
  shared/          # bottlenecks (Hormuz, Suez, Malacca, Bosphorus, …)
  metals/          # copper, aluminum, silver, gold
  energy/          # oil, lng
  agriculture/     # wheat, corn
```

YAML = static topology. Prices, actual production, TC/RC, and policies belong in `database/` + `quant/` (not in the YAML files).

## Run

```bash
uv sync
uv run python -m backend.main --sector metals
# or: PYTHONPATH=. python3 -m backend.main --sector metals
```
