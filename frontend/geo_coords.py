"""Fallback lat/lon for nodes missing coords in YAML (exchanges, hubs, aggregates)."""

from __future__ import annotations

# Approximate WGS84 positions for map display when YAML has no lat/lon.
GEO_FALLBACKS: dict[str, tuple[float, float]] = {
    # China hubs
    "alumina_guinea_coast": (9.50, -13.70),
    "smelter_china_shandong": (36.65, 117.00),
    "scrap_hub_alu_china": (31.23, 121.47),
    "scrap_hub_china": (31.23, 121.47),
    "hub_qingdao_transship": (36.07, 120.38),
    "hub_shanghai_transship": (31.23, 121.47),
    "consumer_alu_china_construction": (31.23, 121.47),
    "consumer_china_grid": (39.90, 116.40),
    "consumer_china_ev": (31.23, 121.47),
    "consumer_alu_ev_autos": (31.23, 121.47),
    "exchange_shfe": (31.23, 121.47),
    "warehouse_shfe": (31.23, 121.47),
    "mine_china_au": (37.50, 105.00),
    "mine_china_ag": (37.50, 105.00),
    "refinery_china_au": (31.23, 121.47),
    "refinery_china_ag": (31.23, 121.47),
    "vault_shanghai_au": (31.23, 121.47),
    "exchange_sge": (31.23, 121.47),
    "consumer_au_jewelry_china": (31.23, 121.47),
    # EU / UK
    "scrap_hub_alu_eu": (51.50, 4.30),
    "scrap_hub_eu": (51.50, 4.30),
    "consumer_alu_eu_packaging": (50.85, 4.35),
    "consumer_eu_industrial": (50.11, 8.68),
    "exchange_lme_alu": (51.51, -0.09),
    "exchange_lme": (51.51, -0.09),
    "warehouse_lme_alu": (51.51, -0.09),
    "warehouse_lme_asia": (1.29, 103.85),
    "vault_lbma_au": (51.51, -0.09),
    "vault_lbma_ag": (51.51, -0.09),
    "exchange_lbma_au": (51.51, -0.09),
    "exchange_lbma_ag": (51.51, -0.09),
    "refinery_pamp_suisse": (46.01, 8.96),
    # US
    "scrap_hub_us": (40.71, -74.01),
    "consumer_us_construction": (40.71, -74.01),
    "warehouse_comex": (40.71, -74.01),
    "exchange_comex": (40.71, -74.01),
    "vault_comex_au": (40.71, -74.01),
    "vault_comex_ag": (40.71, -74.01),
    "exchange_comex_au": (40.71, -74.01),
    "exchange_comex_ag": (40.71, -74.01),
    "refinery_us_west": (39.53, -119.81),
    # LatAm / others
    "mine_grasberg_au": (-4.05, 137.12),
    "mine_antamina_ag": (-9.55, -77.05),
    "mine_kachkanar_ag": (58.70, 59.48),
    "refinery_perth_mint_au": (-31.95, 115.86),
    "refinery_perth_mint_ag": (-31.95, 115.86),
    "refinery_rand_refinery": (-26.20, 28.05),
    "refinery_fresnillo": (23.17, -102.87),
    "refinery_peru_cajamarquilla": (-11.98, -76.92),
    "consumer_korea_semis": (37.57, 126.98),
    "consumer_au_jewelry_india": (19.08, 72.88),
    "consumer_au_investment": (40.71, -74.01),
    "scrap_hub_au": (51.51, -0.09),
    "scrap_hub_ag": (40.71, -74.01),
    "consumer_ag_solar": (31.23, 121.47),
    "consumer_ag_electronics": (37.57, 126.98),
    "consumer_ag_jewelry": (19.08, 72.88),
}


def resolve_lat_lon(node: dict) -> tuple[float, float] | None:
    lat, lon = node.get("lat"), node.get("lon")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
    fb = GEO_FALLBACKS.get(node.get("id") or "")
    return fb
