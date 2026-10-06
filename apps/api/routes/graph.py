"""GET /api/graph — geolocated network with live event severity."""

from __future__ import annotations

from fastapi import APIRouter, Query

from apps.api.schemas import GraphResponse
from apps.api.services import load_live_network, serialize_graph

router = APIRouter(tags=["graph"])


@router.get("/graph", response_model=GraphResponse)
def get_graph(
    sector: str = Query("all", description="all | metals | energy | agriculture"),
) -> GraphResponse:
    network, n_events = load_live_network(sector)
    payload = serialize_graph(network)
    return GraphResponse(**payload, n_events=n_events)
