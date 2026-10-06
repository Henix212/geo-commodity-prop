"""Pydantic response models for the dashboard API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    type: str = "unknown"
    lat: float
    lon: float
    event_severity: float = 0.0
    commodity: str | None = None
    event_type: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    flow_type: str = "unknown"


class GraphResponse(BaseModel):
    sector: str | None = None
    network_name: str | None = None
    commodities: list[str] = Field(default_factory=list)
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    n_events: int = 0


class SignalModel(BaseModel):
    commodity: str | None = None
    ticker: str | None = None
    tension: float | None = None
    side: str | None = None
    position: float | None = None
    size: float | None = None
    all_tensions: dict[str, float] | None = None


class TopNode(BaseModel):
    id: str | None = None
    severity: float
    type: str | None = None
    commodity: str | None = None


class TensionsResponse(BaseModel):
    sector: str
    tensions: dict[str, float]
    signal: dict[str, Any] | None = None
    n_events: int = 0
    top_nodes: list[TopNode] = Field(default_factory=list)
    use_gat: bool = False


class EquityPoint(BaseModel):
    date: str
    value: float


class PortfolioResponse(BaseModel):
    commodity: str
    ticker: str | None = None
    n_bars: int = 0
    sharpe: float | None = None
    max_drawdown_pct: float | None = None
    total_return_pct: float | None = None
    turnover: float | None = None
    engine: str | None = None
    last_tension: float | None = None
    equity_curve: list[EquityPoint] = Field(default_factory=list)
