"""Visualization helpers for the Streamlit dashboard (no ML)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _heat_color(score: float) -> str:
    """Map [0,1] score to hex (gray → amber → red)."""
    s = max(0.0, min(1.0, float(score)))
    if s < 0.15:
        return "#94a3b8"
    if s < 0.4:
        return "#f59e0b"
    if s < 0.7:
        return "#f97316"
    return "#ef4444"


def build_pyvis_html(
    network: dict,
    *,
    node_tensions: dict[str, float] | None = None,
    event_severities: dict[str, float] | None = None,
    height: str = "620px",
) -> str:
    """Return PyVis HTML string for the supply-chain graph."""
    from pyvis.network import Network

    node_tensions = node_tensions or {}
    event_severities = event_severities or {}

    g = nx.DiGraph()
    for node in network.get("nodes") or []:
        g.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in network.get("edges") or []:
        g.add_edge(edge["source"], edge["target"], **{
            k: v for k, v in edge.items() if k not in ("source", "target")
        })

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#0f172a",
        font_color="#e2e8f0",
        notebook=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    for nid, data in g.nodes(data=True):
        sev = abs(float(event_severities.get(nid) or data.get("event_severity") or 0.0))
        tension = float(node_tensions.get(nid, 0.0))
        score = max(sev, tension)
        size = 12 + 28 * score
        title = (
            f"<b>{nid}</b><br>type={data.get('type')}<br>"
            f"commodity={data.get('commodity')}<br>"
            f"event_severity={sev:.2f}<br>tension={tension:.2f}"
        )
        net.add_node(
            nid,
            label=nid.replace("_", "\n") if score >= 0.25 else " ",
            title=title,
            color=_heat_color(score),
            size=size,
            borderWidth=2 if score >= 0.4 else 1,
        )

    for u, v, data in g.edges(data=True):
        w = float(data.get("weight") or 1.0)
        net.add_edge(u, v, value=max(0.5, w), color="#334155", arrows="to")

    tmp = Path(tempfile.gettempdir()) / "gcp_pyvis_graph.html"
    net.save_graph(str(tmp))
    return tmp.read_text(encoding="utf-8")


def event_feed_rows(
    events: list[dict],
    articles_by_uid: dict[str, dict],
) -> list[dict[str, Any]]:
    rows = []
    for ev in events:
        uid = ev.get("article_uid") or ""
        art = articles_by_uid.get(uid) or {}
        title = art.get("title") or uid or "(no article)"
        rows.append(
            {
                "headline": title,
                "node": ev.get("node_id") or "—",
                "type": ev.get("event_type"),
                "severity": ev.get("severity"),
                "commodity": ev.get("commodity"),
                "direction": ev.get("direction"),
                "entity": ev.get("entity_text"),
                "extracted_at": ev.get("extracted_at"),
            }
        )
    return rows


def price_signal_figure(
    prices: pd.DataFrame,
    *,
    commodity: str,
    tension: float | None = None,
    side: str | None = None,
    ticker: str | None = None,
) -> go.Figure:
    """Plotly price chart with signal annotation."""
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    if prices is None or prices.empty:
        fig.update_layout(
            title=f"{commodity}: no price data",
            template="plotly_dark",
            height=420,
        )
        return fig

    df = prices.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    close = df["close"].astype(float) if "close" in df.columns else df.iloc[:, 0].astype(float)

    if {"open", "high", "low", "close"}.issubset(df.columns):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=ticker or commodity,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(x=close.index, y=close.values, mode="lines", name="close")
        )

    # Latest tension as annotation / marker on last bar
    if tension is not None and len(close):
        last_x = close.index[-1]
        last_y = float(close.iloc[-1])
        label = f"tension={tension:.2f}"
        if side:
            label += f" → {side.upper()}"
        fig.add_annotation(
            x=last_x,
            y=last_y,
            text=label,
            showarrow=True,
            arrowhead=2,
            bgcolor="#1e293b",
            font=dict(color="#f8fafc"),
        )

    title = f"{commodity.upper()}"
    if ticker:
        title += f" ({ticker})"
    if side:
        title += f"  |  signal: {side.upper()}"
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=420,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def tension_bar_figure(commodity_tensions: dict[str, float]) -> go.Figure:
    items = sorted(commodity_tensions.items(), key=lambda x: -x[1])
    fig = go.Figure(
        go.Bar(
            x=[k for k, _ in items],
            y=[v for _, v in items],
            marker_color=[_heat_color(v) for _, v in items],
        )
    )
    fig.update_layout(
        title="Commodity tension scores",
        template="plotly_dark",
        height=320,
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
