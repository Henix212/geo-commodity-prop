"""Geo Commodity Prop — Streamlit dashboard (read-only).

Run:
  uv run streamlit run frontend/app.py

Refresh snapshot first:
  uv run python -m backend.jobs snapshot --sector metals
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run frontend/app.py` from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components

from backend import config
from backend.dashboard_state import load_dashboard_state
from backend.database import init_db, list_articles, list_events
from backend.graph_core import load_sector
from backend.quant.market_data import load_price_frame
from frontend.viz import (
    build_pyvis_html,
    event_feed_rows,
    price_signal_figure,
    tension_bar_figure,
)

st.set_page_config(
    page_title="Geo Commodity Prop",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_network(sector: str) -> dict:
    return load_sector(sector, validate=False)


@st.cache_data(ttl=30)
def _cached_events(limit: int = 50):
    init_db()
    return list_events(limit=limit)


@st.cache_data(ttl=30)
def _cached_articles(limit: int = 200):
    init_db()
    arts = list_articles(limit=limit)
    return {a["uid"]: a for a in arts}


@st.cache_data(ttl=60)
def _cached_prices(commodity: str):
    return load_price_frame(commodity)


def main() -> None:
    state = load_dashboard_state()

    with st.sidebar:
        st.title("Geo Commodity Prop")
        st.caption("Supply-chain shock → tension → signal")
        sectors = ["metals", "energy", "agriculture"]
        snap_sector = (state or {}).get("sector") or config.DEFAULT_SECTOR
        sector_idx = sectors.index(snap_sector) if snap_sector in sectors else 0
        sector = st.selectbox("Sector", sectors, index=sector_idx)
        commodities = list((state or {}).get("commodities") or list(config.TICKERS.keys()))
        default_c = config.DEFAULT_COMMODITY
        if default_c not in commodities and commodities:
            default_c = commodities[0]
        commodity = st.selectbox(
            "Commodity",
            commodities,
            index=commodities.index(default_c) if default_c in commodities else 0,
        )
        auto = st.checkbox("Auto-refresh (15s)", value=False)
        if st.button("Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        if auto:
            st.markdown(
                '<meta http-equiv="refresh" content="15">',
                unsafe_allow_html=True,
            )

        if state:
            st.success(f"Snapshot: {state.get('updated_at', '')[:19]}")
            st.write(f"Backend: **{state.get('inference_backend', '?')}**")
            st.write(f"Events in snapshot: **{state.get('n_events', 0)}**")
        else:
            st.warning(
                "No `data/dashboard_state.json` yet.\n\n"
                "Run: `uv run python -m backend.jobs snapshot --sector metals`"
            )

    # --- Header metrics ---
    tensions = (state or {}).get("commodity_tensions") or {}
    signals = (state or {}).get("signals") or {}
    sig = signals.get(commodity) or (state or {}).get("signal") or {}
    tension = float(tensions.get(commodity, sig.get("tension") or 0.0))
    side = sig.get("side") if sig.get("commodity") == commodity else (
        signals.get(commodity, {}) or {}
    ).get("side", "flat")
    ticker = config.TICKERS.get(commodity)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Commodity", commodity.upper())
    c2.metric("Tension", f"{tension:.3f}")
    c3.metric("Signal", str(side).upper())
    c4.metric("Ticker", ticker or "—")

    # --- Layout: map | feed ---
    left, right = st.columns([1.55, 1.0], gap="medium")

    with left:
        st.subheader("Supply chain map")
        try:
            network = _load_network(sector)
            html = build_pyvis_html(
                network,
                node_tensions=(state or {}).get("node_tensions") or {},
                event_severities=(state or {}).get("event_severities") or {},
            )
            components.html(html, height=640, scrolling=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Graph render failed: {exc}")

        if state and state.get("shocked_nodes"):
            st.caption("Top shocked / high-tension nodes")
            st.dataframe(
                state["shocked_nodes"][:15],
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.subheader("Live event feed")
        events = _cached_events(60)
        articles = _cached_articles(300)
        rows = event_feed_rows(events, articles)
        if not rows:
            st.info("No parsed events in SQLite yet.")
        else:
            for row in rows[:25]:
                st.markdown(
                    f"**{row['headline'][:110]}**  \n"
                    f"`{row['node']}` · {row['type']} · sev={row['severity']} · "
                    f"{row['commodity']} · {row['direction']}"
                )
                st.divider()

    # --- Alpha charts ---
    st.subheader("Alpha & strategy")
    chart_l, chart_r = st.columns([1.6, 1.0])
    with chart_l:
        prices = _cached_prices(commodity)
        fig = price_signal_figure(
            prices,
            commodity=commodity,
            tension=tension,
            side=side,
            ticker=ticker,
        )
        st.plotly_chart(fig, use_container_width=True)
    with chart_r:
        if tensions:
            st.plotly_chart(tension_bar_figure(tensions), use_container_width=True)
        else:
            st.info("No commodity tensions in snapshot.")


if __name__ == "__main__":
    main()
