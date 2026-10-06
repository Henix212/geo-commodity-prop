"""GET /api/portfolio — equity curve + backtest stats."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from apps.api.schemas import PortfolioResponse
from geo_commodity.quant.backtest import run_backtest

router = APIRouter(tags=["portfolio"])


@router.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio(
    commodity: str = Query("oil"),
    refresh: bool = Query(False),
) -> PortfolioResponse:
    try:
        result = run_backtest(commodity, refresh=refresh, include_curve=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PortfolioResponse(**result)
