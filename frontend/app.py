"""Geo Commodity Prop — Streamlit quant desk (read-only).

Run:
  uv run streamlit run frontend/app.py

Refresh snapshot:
  uv run python -m backend.jobs snapshot --sector metals
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from backend import config
from backend.dashboard_state import load_dashboard_state
from backend.database import init_db, list_articles, list_events
from backend.graph_core import load_sector
from backend.quant.backtest import backtest_commodity
from backend.quant.market_data import load_price_frame
from frontend.viz import (
    alpha_metric_cards_html,
    build_pyvis_html,
    equity_drawdown_figure,
    event_cards_html,
    event_feed_rows,
    glossary_html,
    kpi_strip_html,
    map_legend_html,
    metrics_table_rows,
    portfolio_rows,
    portfolio_summary_html,
    price_signal_figure,
    relative_time,
    signal_book_rows,
    strategy_example_html,
    strategy_flow_html,
    strategy_pipeline_html,
    strategy_rules_html,
    style_portfolio_book,
    style_shocked_table,
    style_signal_book,
    supply_chain_geo_figure,
    tension_attribution_html,
    tension_bar_figure,
    tension_gauge_html,
    tension_hist_figure,
    why_signal_html,
)

st.set_page_config(
    page_title="Geo Commodity Prop",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


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


@st.cache_data(ttl=60)
def _cached_backtest(commodity: str, tension: float):
    return backtest_commodity(commodity, constant_tension=tension, return_series=True)


@st.cache_data(ttl=120)
def _cached_spof(sector: str):
    from backend.models.centrality import node_betweenness

    network = _load_network(sector)
    return node_betweenness(network, top_k=15)


def _snapshot_age(iso: str | None) -> str:
    if not iso:
        return "no snapshot"
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = int((datetime.now(timezone.utc) - dt).total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        return f"{secs // 3600}h ago"
    except (TypeError, ValueError):
        return str(iso)[:19]


def _trim_prices(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if df is None or df.empty or lookback == "all":
        return df
    days = {"3m": 90, "6m": 180, "1y": 365, "2y": 730}.get(lookback)
    if not days:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        cutoff = out["date"].max() - pd.Timedelta(days=days)
        return out[out["date"] >= cutoff]
    idx = pd.to_datetime(out.index)
    cutoff = idx.max() - pd.Timedelta(days=days)
    return out.loc[idx >= cutoff]


@st.cache_data(ttl=60)
def _cached_last_prices(commodities: tuple[str, ...]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for c in commodities:
        try:
            df = load_price_frame(c)
            if df is None or df.empty or "close" not in df.columns:
                out[c] = None
            else:
                out[c] = float(df["close"].astype(float).iloc[-1])
        except Exception:  # noqa: BLE001
            out[c] = None
    return out


def _drivers_for_commodity(shocked: list[dict], commodity: str) -> list[dict]:
    related = [
        r
        for r in shocked
        if str(r.get("commodity") or "").lower()
        in {commodity.lower(), "cross_metal", "cross_commodity", "cross_energy", "cross_agri"}
    ]
    related.sort(
        key=lambda r: (-float(r.get("tension") or 0), -abs(float(r.get("event_severity") or 0)))
    )
    return related


def _top_shock(shocked: list[dict], commodity: str) -> dict | None:
    drivers = _drivers_for_commodity(shocked, commodity)
    return drivers[0] if drivers else (shocked[0] if shocked else None)


def main() -> None:
    state = load_dashboard_state()
    long_th = config.TENSION_LONG_THRESHOLD
    short_th = config.TENSION_SHORT_THRESHOLD

    with st.sidebar:
        st.markdown('<p class="desk-brand">GEO COMMODITY PROP</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="desk-sub">shock → tension → signal → alpha</p>',
            unsafe_allow_html=True,
        )
        sectors = ["metals", "energy", "agriculture"]
        snap_sector = (state or {}).get("sector") or config.DEFAULT_SECTOR
        sector_idx = sectors.index(snap_sector) if snap_sector in sectors else 0
        sector = st.selectbox(
            "Sector",
            sectors,
            index=sector_idx,
            help="Réseau supply-chain à charger (metals / energy / agriculture).",
        )

        commodities = list((state or {}).get("commodities") or list(config.TICKERS.keys()))
        for k in ((state or {}).get("commodity_tensions") or {}):
            if k not in commodities:
                commodities.append(k)
        default_c = config.DEFAULT_COMMODITY
        if default_c not in commodities and commodities:
            default_c = commodities[0]
        commodity = st.selectbox(
            "Commodity",
            commodities,
            index=commodities.index(default_c) if default_c in commodities else 0,
            help="Commodity suivie pour le signal et le backtest.",
        )
        lookback = st.selectbox(
            "Lookback prix",
            ["3m", "6m", "1y", "2y", "all"],
            index=2,
            help="Fenêtre affichée sur les charts ALPHA.",
        )
        auto = st.checkbox("Auto-refresh 15s", value=False)
        if st.button("Refresh", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        if auto:
            st.markdown(
                '<meta http-equiv="refresh" content="15">',
                unsafe_allow_html=True,
            )

        if state:
            age = _snapshot_age(state.get("updated_at"))
            st.markdown(
                f'<span class="desk-badge ok">SNAP {age}</span>'
                f'<span class="desk-badge">{(state.get("inference_backend") or "?").upper()}</span>'
                f'<span class="desk-badge">{state.get("n_events", 0)} EVT</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"nodes={state.get('n_nodes')} · edges={state.get('n_edges')}"
            )
        else:
            st.markdown('<span class="desk-badge warn">NO SNAPSHOT</span>', unsafe_allow_html=True)
            st.caption("`uv run python -m backend.jobs snapshot --sector metals`")

    tensions = (state or {}).get("commodity_tensions") or {}
    signals = (state or {}).get("signals") or {}
    backtests_snap = (state or {}).get("backtests") or {}
    shocked = (state or {}).get("shocked_nodes") or []
    sig = signals.get(commodity) or {}
    tension = float(tensions.get(commodity, sig.get("tension") or 0.0))
    side = str(sig.get("side") or "flat")
    ticker = config.TICKERS.get(commodity) or sig.get("ticker")
    backend = str((state or {}).get("inference_backend") or "—")

    bt = backtests_snap.get(commodity) or {}
    if not bt or bt.get("error"):
        if commodity in config.TICKERS:
            live = _cached_backtest(commodity, tension)
            bt = {k: live.get(k) for k in (
                "sharpe", "max_drawdown", "total_return", "turnover", "n", "engine", "error"
            )}
            bt_full = live
        else:
            bt_full = {}
    else:
        bt_full = _cached_backtest(commodity, tension) if commodity in config.TICKERS else {}

    sharpe = bt.get("sharpe")
    max_dd = bt.get("max_drawdown")
    top = _top_shock(shocked, commodity)
    shock_label = "—"
    if top:
        shock_label = (
            f"{top.get('node_id')} · sev={float(top.get('event_severity') or 0):.2f}"
        )

    # --- Header (slim) ---
    st.markdown(
        f'<p class="desk-sub" style="margin-bottom:0.2rem;">'
        f"BOOK · {sector.upper()} · focus {commodity.upper()}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        strategy_pipeline_html(
            shock_label=shock_label,
            backend=backend,
            n_shocked=len(shocked),
            tension=tension,
            side=side,
            sharpe=sharpe,
            max_dd=max_dd,
            active={1, 2, 3, 4, 5},
            compact=True,
        ),
        unsafe_allow_html=True,
    )

    tab_pf, tab_st, tab_ov, tab_map, tab_alpha, tab_risk = st.tabs(
        ["PORTFOLIO", "STRATEGY", "OVERVIEW", "MAP", "ALPHA", "RISK"]
    )

    # ---- PORTFOLIO ----
    with tab_pf:
        tradable = [c for c, s in signals.items() if s.get("ticker")]
        last_px = _cached_last_prices(tuple(sorted(tradable)))
        book = portfolio_rows(
            signals,
            shocked=shocked,
            last_prices=last_px,
            long_th=long_th,
            short_th=short_th,
        )
        n_long = sum(1 for r in book if r["side"] == "LONG")
        n_short = sum(1 for r in book if r["side"] == "SHORT")
        n_flat = sum(1 for r in book if r["side"] == "FLAT")
        snap_age = _snapshot_age((state or {}).get("updated_at")) if state else "—"
        st.markdown(
            portfolio_summary_html(
                n_long=n_long,
                n_short=n_short,
                n_flat=n_flat,
                n_live_events=int((state or {}).get("n_events") or len(_cached_events(50))),
                snap_age=snap_age,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="callout info">'
            "Portefeuille = signaux <b>actuels</b> du snapshot (pas encore de fills live). "
            "Prix = dernier close en cache · News = event strip à droite. "
            "Refresh snapshot / prices pour coller au marché."
            "</div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.35, 1.0], gap="medium")
        with left:
            st.markdown('<div class="section-title">Open book</div>', unsafe_allow_html=True)
            if book:
                st.dataframe(
                    style_portfolio_book(pd.DataFrame(book)),
                    width="stretch",
                    hide_index=True,
                    height=360,
                )
            else:
                st.caption("Aucun ticker dans les signaux — lance un snapshot.")

            st.markdown(
                tension_attribution_html(
                    commodity=commodity,
                    tension=tension,
                    drivers=_drivers_for_commodity(shocked, commodity),
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                tension_gauge_html(tension, long_th=long_th, short_th=short_th, side=side),
                unsafe_allow_html=True,
            )

        with right:
            st.markdown('<div class="section-title">News / events (live feed)</div>', unsafe_allow_html=True)
            events = _cached_events(40)
            articles = _cached_articles(300)
            rows = event_feed_rows(events, articles)
            components.html(event_cards_html(rows, limit=16), height=560, scrolling=True)

    # ---- STRATEGY ----
    with tab_st:
        st.markdown(
            '<div class="callout">'
            "Un choc sur un <b>nœud physique</b> (mine, port, chokepoint) se propage dans le "
            "graphe supply-chain → <b>tension</b> commodity → position "
            "<span style='color:#22c55e'>LONG</span> / "
            "<span style='color:#ef4444'>SHORT</span> / FLAT → backtest = alpha."
            "<br/><br/>"
            "<b>Astuce lecture :</b> après diffusion, les scores sont <span class='mono'>normalisés</span> "
            "→ le nœud le plus chaud vaut toujours <span class='mono'>1.0</span>. "
            "Donc copper à 1.0 = au moins une mine/port cuivre (souvent Escondida) au max du graphe, "
            "pas « 100% du marché mondial »."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(strategy_flow_html(), unsafe_allow_html=True)
        c1, c2 = st.columns([1.1, 1.0])
        with c1:
            st.markdown(
                strategy_rules_html(long_th=long_th, short_th=short_th),
                unsafe_allow_html=True,
            )
            st.markdown(
                strategy_example_html(
                    top_node=(top or {}).get("node_id"),
                    sev=float((top or {}).get("event_severity") or 0) if top else None,
                    commodity=commodity,
                    tension=tension,
                    side=side,
                    ticker=ticker,
                    sharpe=sharpe,
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                glossary_html(half_life_h=config.EVENT_DECAY_HALF_LIFE_HOURS),
                unsafe_allow_html=True,
            )

    # ---- Overview ----
    with tab_ov:
        left, mid, right = st.columns([1.0, 1.15, 1.0], gap="medium")
        with left:
            st.markdown('<div class="section-title">Why this signal</div>', unsafe_allow_html=True)
            st.markdown(
                why_signal_html(
                    commodity=commodity,
                    side=side,
                    tension=tension,
                    shocked=shocked,
                    ticker=ticker,
                ),
                unsafe_allow_html=True,
            )
            if shocked:
                st.markdown('<div class="section-title">Shocked nodes</div>', unsafe_allow_html=True)
                cols = [
                    c
                    for c in ("node_id", "type", "commodity", "event_severity", "tension")
                    if shocked and c in shocked[0]
                ]
                st.dataframe(
                    style_shocked_table(pd.DataFrame(shocked[:8])[cols]),
                    width="stretch",
                    hide_index=True,
                    height=260,
                )

        with mid:
            st.markdown('<div class="section-title">Commodity tensions</div>', unsafe_allow_html=True)
            if tensions:
                st.plotly_chart(tension_bar_figure(tensions), width="stretch")
            else:
                st.info("No commodity tensions in snapshot.")
            book = signal_book_rows(signals, long_th=long_th, short_th=short_th)
            st.markdown('<div class="section-title">Signal book</div>', unsafe_allow_html=True)
            if book:
                st.dataframe(
                    style_signal_book(pd.DataFrame(book)),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("Empty book — run a snapshot.")

        with right:
            st.markdown('<div class="section-title">Event strip</div>', unsafe_allow_html=True)
            events = _cached_events(40)
            articles = _cached_articles(300)
            rows = event_feed_rows(events, articles)
            components.html(event_cards_html(rows, limit=14), height=520, scrolling=True)

    # ---- Map ----
    with tab_map:
        st.markdown(map_legend_html(), unsafe_allow_html=True)
        map_c1, map_c2 = st.columns([1, 1])
        with map_c1:
            # Default ON when commodity is a real traded key in tickers
            filter_commodity = st.checkbox(
                "Filtrer commodity sélectionnée",
                value=commodity in config.TICKERS,
                help="Garde les nœuds de la commodity + bottlenecks partagés.",
            )
        with map_c2:
            show_graph = st.checkbox("Graphe réseau (PyVis)", value=False)
        network = None
        try:
            network = _load_network(sector)
            fig_geo = supply_chain_geo_figure(
                network,
                node_tensions=(state or {}).get("node_tensions") or {},
                event_severities=(state or {}).get("event_severities") or {},
                commodity_filter=commodity if filter_commodity else None,
                height=620,
            )
            st.plotly_chart(fig_geo, width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Geo map failed: {exc}")

        if show_graph and network is not None:
            try:
                html = build_pyvis_html(
                    network,
                    node_tensions=(state or {}).get("node_tensions") or {},
                    event_severities=(state or {}).get("event_severities") or {},
                    height="420px",
                )
                components.html(html, height=440, scrolling=False)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Graph render failed: {exc}")

        if shocked:
            st.markdown(
                '<div class="section-title">Shocked / high-tension nodes</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                style_shocked_table(pd.DataFrame(shocked)),
                width="stretch",
                hide_index=True,
            )

    # ---- Alpha ----
    with tab_alpha:
        st.markdown(
            f'<div class="callout info">'
            f"Si on avait tenu le signal <b>{side.upper()}</b> "
            f"(tension constante = {tension:.2f}) sur la fenêtre <b>{lookback}</b>…"
            f"</div>",
            unsafe_allow_html=True,
        )
        prices = _trim_prices(_cached_prices(commodity), lookback) if commodity in config.TICKERS else pd.DataFrame()

        chart_l, chart_r = st.columns([1.55, 1.0])
        with chart_l:
            fig = price_signal_figure(
                prices,
                commodity=commodity,
                tension=tension,
                side=side,
                ticker=ticker,
                positions=bt_full.get("positions") if isinstance(bt_full, dict) else None,
                long_th=long_th,
                short_th=short_th,
            )
            st.plotly_chart(fig, width="stretch")
        with chart_r:
            equity = bt_full.get("equity") if isinstance(bt_full, dict) else None
            dd = bt_full.get("drawdown") if isinstance(bt_full, dict) else None
            if equity is not None and dd is not None and not getattr(equity, "empty", True):
                st.plotly_chart(
                    equity_drawdown_figure(
                        equity,
                        dd,
                        title=f"Equity · {commodity.upper()} @ tension={tension:.2f}",
                    ),
                    width="stretch",
                )
            else:
                st.info("Pas de courbe equity (prix manquants ou commodity non tradable).")

            st.markdown(alpha_metric_cards_html(bt), unsafe_allow_html=True)

    # ---- Risk ----
    with tab_risk:
        st.markdown(
            '<div class="callout info">'
            "<b>SPOF</b> = nœuds dont la coupure casse le réseau (betweenness élevée). "
            "Le book backtest compare le ratio risque/rendement par commodity."
            "</div>",
            unsafe_allow_html=True,
        )
        r1, r2 = st.columns([1.2, 1.0])
        with r1:
            st.markdown('<div class="section-title">Backtest book</div>', unsafe_allow_html=True)
            all_bt = dict(backtests_snap)
            if commodity in config.TICKERS and (
                commodity not in all_bt or all_bt.get(commodity, {}).get("error")
            ):
                all_bt[commodity] = bt
            for c, t in tensions.items():
                if c not in config.TICKERS:
                    continue
                if c not in all_bt or all_bt[c].get("error"):
                    try:
                        live = _cached_backtest(c, float(t))
                        all_bt[c] = {
                            k: live.get(k)
                            for k in (
                                "commodity", "ticker", "sharpe", "max_drawdown",
                                "total_return", "turnover", "n", "engine", "error",
                            )
                        }
                    except Exception:  # noqa: BLE001
                        pass
            st.dataframe(
                pd.DataFrame(metrics_table_rows(all_bt)),
                width="stretch",
                hide_index=True,
            )

            node_t = (state or {}).get("node_tensions") or {}
            st.plotly_chart(tension_hist_figure(node_t), width="stretch")

        with r2:
            st.markdown(
                '<div class="section-title">SPOF · betweenness</div>',
                unsafe_allow_html=True,
            )
            try:
                spof = _cached_spof(sector)
                if spof:
                    st.dataframe(pd.DataFrame(spof), width="stretch", hide_index=True)
                else:
                    st.caption("No centrality ranks.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Centrality failed: {exc}")

            st.markdown('<div class="section-title">Live events</div>', unsafe_allow_html=True)
            events = _cached_events(50)
            articles = _cached_articles(300)
            rows = event_feed_rows(events, articles)
            components.html(event_cards_html(rows, limit=15), height=420, scrolling=True)


if __name__ == "__main__":
    main()
