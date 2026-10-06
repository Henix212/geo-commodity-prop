"""Streamlit dashboard — map + tensions + portfolio."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.api.services import load_live_network, serialize_graph, top_stressed_nodes
from geo_commodity.gnn.infer import infer_tensions
from geo_commodity.quant.backtest import run_backtest
from geo_commodity.quant.strategy import tension_to_signal

SECTORS = ("all", "metals", "energy", "agriculture")
COMMODITIES = ("copper", "aluminum", "gold", "silver", "oil", "lng", "wheat", "corn")


def severity_color(severity: float) -> str:
    s = abs(float(severity))
    if s >= 0.65:
        return "#c45c4a"
    if s >= 0.35:
        return "#c9a227"
    return "#3d9a7a"


def build_map(nodes: list[dict], edges: list[dict]) -> go.Figure:
    by_id = {n["id"]: n for n in nodes}
    fig = go.Figure()

    for e in edges:
        a, b = by_id.get(e["source"]), by_id.get(e["target"])
        if not a or not b:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=[a["lon"], b["lon"], None],
                lat=[a["lat"], b["lat"], None],
                mode="lines",
                line=dict(width=1, color="rgba(74,93,112,0.55)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if nodes:
        df = pd.DataFrame(nodes)
        df["sev"] = df["event_severity"].abs()
        df["color"] = df["sev"].map(severity_color)
        df["size"] = 6 + (df["sev"] * 18).clip(upper=18)
        fig.add_trace(
            go.Scattergeo(
                lon=df["lon"],
                lat=df["lat"],
                mode="markers",
                marker=dict(
                    size=df["size"],
                    color=df["color"],
                    line=dict(width=0.5, color="#0c1218"),
                    opacity=0.9,
                ),
                text=df.apply(
                    lambda r: (
                        f"<b>{r['id']}</b><br>type: {r['type']}"
                        f"<br>severity: {r['sev']:.3f}"
                        + (f"<br>commodity: {r['commodity']}" if r.get("commodity") else "")
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#1a2430",
        showocean=True,
        oceancolor="#0c1218",
        showlakes=False,
        showcountries=True,
        countrycolor="#243040",
        bgcolor="#0c1218",
        fitbounds="locations" if nodes else False,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0c1218",
        height=560,
    )
    return fig


@st.cache_data(ttl=30, show_spinner=False)
def load_graph_and_tensions(sector: str, use_gat: bool) -> dict:
    network, n_events = load_live_network(sector)
    graph = serialize_graph(network)
    tensions = infer_tensions(network, use_gat=use_gat)
    commodities = network.get("supported_commodities") or []
    commodity = commodities[0] if commodities else None
    signal = tension_to_signal(tensions, commodity=commodity)
    return {
        "graph": graph,
        "n_events": n_events,
        "tensions": tensions,
        "signal": signal,
        "top_nodes": top_stressed_nodes(network),
        "use_gat": use_gat,
        "sector": sector,
    }


@st.cache_data(ttl=60, show_spinner=False)
def load_portfolio(commodity: str) -> dict:
    # Numpy path — skip vectorbt import (slow + noisy under Streamlit)
    return run_backtest(commodity, include_curve=True, prefer_numpy=True)


st.set_page_config(
    page_title="Geo Commodity Prop",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #0c1218; color: #e8eef4; }
      [data-testid="stSidebar"] { background: #141c24; }
      h1, h2, h3 { color: #e8eef4 !important; }
      .metric-label { color: #8a9aab; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption("SUPPLY-CHAIN GNN")
    st.title("Geo Commodity Prop")
    sector = st.selectbox("Sector", SECTORS, index=SECTORS.index("energy"))
    commodity = st.selectbox("Commodity", COMMODITIES, index=COMMODITIES.index("oil"))
    use_gat = st.checkbox("GAT", value=True)
    if st.button("Refresh", width="stretch"):
        load_graph_and_tensions.clear()
        load_portfolio.clear()
        st.rerun()

left, right = st.columns([1.4, 0.9], gap="medium")

with st.spinner("Loading network…"):
    bundle = load_graph_and_tensions(sector, use_gat)

graph = bundle["graph"]
nodes, edges = graph["nodes"], graph["edges"]

with left:
    st.subheader("Network")
    st.caption(f"{len(nodes)} nodes · {len(edges)} edges · events {bundle['n_events']}")
    st.plotly_chart(build_map(nodes, edges), width="stretch")

with right:
    st.subheader("Tension")
    signal = bundle["signal"] or {}
    side = (signal.get("side") or "—").upper()
    c1, c2, c3 = st.columns(3)
    c1.metric("Signal", side)
    c2.metric("Ticker", signal.get("ticker") or "—")
    c3.metric("Tension", f"{float(signal.get('tension') or 0):.2f}")

    t_df = pd.DataFrame(
        [{"commodity": k, "tension": v} for k, v in sorted(bundle["tensions"].items())]
    )
    if not t_df.empty:
        t_df["color"] = t_df["tension"].map(severity_color)
        fig_t = px.bar(
            t_df,
            x="tension",
            y="commodity",
            orientation="h",
            color="color",
            color_discrete_map="identity",
            range_x=[0, 1],
        )
        fig_t.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="#141c24",
            plot_bgcolor="#141c24",
            font_color="#e8eef4",
            height=220,
            showlegend=False,
            xaxis=dict(gridcolor="#243040"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_t, width="stretch")

    top = bundle["top_nodes"]
    if top:
        st.caption("Top stressed nodes")
        st.dataframe(
            pd.DataFrame(top)[["id", "severity", "type", "commodity"]],
            hide_index=True,
            width="stretch",
            height=160,
        )

    st.subheader("Portfolio")
    try:
        with st.spinner("Backtest…"):
            port = load_portfolio(commodity)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        port = None

    if port:
        m1, m2, m3 = st.columns(3)
        m1.metric("Sharpe", f"{port.get('sharpe') or 0:.2f}")
        m2.metric("Max DD %", f"{port.get('max_drawdown_pct') or 0:.2f}")
        m3.metric("Return %", f"{port.get('total_return_pct') or 0:.2f}")
        st.caption(
            f"{port.get('commodity')} · {port.get('ticker')} · {port.get('engine')} · "
            f"bars={port.get('n_bars')}"
        )
        curve = port.get("equity_curve") or []
        if curve:
            eq = pd.DataFrame(curve)
            last = float(eq["value"].iloc[-1])
            lt = float(port.get("last_tension") or 0.5)
            proj_val = last * (1 + (lt - 0.5) * 0.04)
            fig_p = go.Figure()
            fig_p.add_trace(
                go.Scatter(
                    x=eq["date"],
                    y=eq["value"],
                    mode="lines",
                    name="equity",
                    line=dict(color="#3d9a7a", width=1.8),
                )
            )
            fig_p.add_trace(
                go.Scatter(
                    x=[eq["date"].iloc[-1], "proj"],
                    y=[last, proj_val],
                    mode="lines",
                    name="indicative",
                    line=dict(color="#c9a227", width=1.4, dash="dash"),
                )
            )
            fig_p.update_layout(
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="#141c24",
                plot_bgcolor="#141c24",
                font_color="#e8eef4",
                height=260,
                legend=dict(orientation="h", y=1.12),
                xaxis=dict(gridcolor="#243040", showgrid=True),
                yaxis=dict(gridcolor="#243040", showgrid=True),
            )
            st.plotly_chart(fig_p, width="stretch")
