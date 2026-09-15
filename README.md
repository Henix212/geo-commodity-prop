# Geo Commodity Prop

Physical commodity flow graphs for prop-trading research (metals, energy, agriculture).

## Data sources

Network topology and capacity hints are compiled from public sources (approximate; not live feeds):

- [USGS Copper Statistics and Information](https://www.usgs.gov/centers/national-minerals-information-center/copper-statistics-and-information) — production by country
- [ICSG](https://icsg.org) — concentrate / blister / refined copper flows
- [UN Comtrade](https://comtradeplus.un.org) — trade flows (HS 2603, 7403)
- [Cochilco](https://www.cochilco.cl) — Chile copper exports and market data
- Company reports — BHP (Escondida), Freeport, Codelco, Glencore, Ivanhoe, Aurubis, Jiangxi Copper
- Port / shipping context — MarineTraffic and port authority statistics

## Network data layout

```text
backend/graph_core/network_data/
  shared/          # cross-commodity bottlenecks (canals, straits, routes)
  metals/          # copper, aluminum, ...
  energy/          # oil, lng, ...
  agriculture/     # wheat, corn, ...
```
