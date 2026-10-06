"""GET /api/tensions — commodity tension scores + trade signal."""

from __future__ import annotations

from fastapi import APIRouter, Query

from apps.api.schemas import TensionsResponse
from apps.api.services import load_live_network, top_stressed_nodes
from geo_commodity.gnn.infer import infer_tensions
from geo_commodity.quant.strategy import tension_to_signal

router = APIRouter(tags=["tensions"])


@router.get("/tensions", response_model=TensionsResponse)
def get_tensions(
    sector: str = Query("energy", description="all | metals | energy | agriculture"),
    use_gat: bool = Query(True),
) -> TensionsResponse:
    network, n_events = load_live_network(sector)
    tensions = infer_tensions(network, use_gat=use_gat)
    commodities = network.get("supported_commodities") or []
    commodity = commodities[0] if commodities else None
    signal = tension_to_signal(tensions, commodity=commodity)
    return TensionsResponse(
        sector=sector,
        tensions=tensions,
        signal=signal,
        n_events=n_events,
        top_nodes=top_stressed_nodes(network),
        use_gat=use_gat,
    )
