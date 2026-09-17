"""Visualization helpers for the Streamlit quant desk."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Desk palette
BG = "#0b1220"
PANEL = "#111827"
GRID = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#64748b"
AMBER = "#f59e0b"
LONG = "#22c55e"
SHORT = "#ef4444"
FLAT = "#94a3b8"


def _heat_color(score: float) -> str:
    s = max(0.0, min(1.0, float(score)))
    if s < 0.15:
        return "#64748b"
    if s < 0.4:
        return "#f59e0b"
    if s < 0.7:
        return "#f97316"
    return "#ef4444"


def _side_color(side: str) -> str:
    s = (side or "flat").lower()
    if s == "long":
        return LONG
    if s == "short":
        return SHORT
    return FLAT


def desk_layout(fig: go.Figure, *, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title or "", font=dict(family="IBM Plex Mono", size=13, color=TEXT)),
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        height=height,
        margin=dict(l=48, r=24, t=44 if title else 24, b=36),
        font=dict(family="IBM Plex Mono", size=11, color=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    return fig


def supply_chain_geo_figure(
    network: dict,
    *,
    node_tensions: dict[str, float] | None = None,
    event_severities: dict[str, float] | None = None,
    commodity_filter: str | None = None,
    show_edges: bool = True,
    height: int = 640,
) -> go.Figure:
    """Geographic supply-chain map: nodes by type, red rings by tension."""
    from frontend.geo_coords import resolve_lat_lon

    node_tensions = node_tensions or {}
    event_severities = event_severities or {}

    # type → plotly marker symbol
    symbols = {
        "mine": "diamond",
        "field": "diamond",
        "producing_region": "diamond-wide",
        "port": "circle",
        "terminal": "circle",
        "smelter": "square",
        "alumina_refinery": "square",
        "refinery": "square",
        "liquefaction": "hexagon",
        "regas": "hexagon",
        "mill": "square",
        "processor": "square",
        "silo": "triangle-down",
        "warehouse": "triangle-down",
        "scrap_hub": "x",
        "transshipment": "hexagon-open",
        "bottleneck": "hexagram",
        "consumer": "triangle-up",
        "exchange": "star",
    }

    positions: dict[str, tuple[float, float]] = {}
    meta: dict[str, dict] = {}
    for node in network.get("nodes") or []:
        nid = node["id"]
        if commodity_filter:
            c = (node.get("commodity") or "").lower()
            if c and c not in {
                commodity_filter.lower(),
                "cross_metal",
                "cross_commodity",
                "cross_energy",
                "cross_agri",
            }:
                # keep bottlenecks / shared always
                if node.get("type") != "bottleneck":
                    continue
        xy = resolve_lat_lon(node)
        if xy is None:
            continue
        positions[nid] = xy
        sev = abs(float(event_severities.get(nid) or node.get("event_severity") or 0.0))
        tension = float(node_tensions.get(nid, 0.0))
        meta[nid] = {
            **node,
            "lat": xy[0],
            "lon": xy[1],
            "sev": sev,
            "tension": tension,
            "score": max(sev, tension),
        }

    fig = go.Figure()

    # Edges (great-circle-ish straight lines on geo)
    if show_edges:
        lon_lines: list[float | None] = []
        lat_lines: list[float | None] = []
        for edge in network.get("edges") or []:
            u, v = edge.get("source"), edge.get("target")
            if u not in positions or v not in positions:
                continue
            lat_lines.extend([positions[u][0], positions[v][0], None])
            lon_lines.extend([positions[u][1], positions[v][1], None])
        if lon_lines:
            fig.add_trace(
                go.Scattergeo(
                    lon=lon_lines,
                    lat=lat_lines,
                    mode="lines",
                    line=dict(width=0.7, color="#475569"),
                    hoverinfo="skip",
                    name="flows",
                    opacity=0.45,
                    showlegend=True,
                )
            )

    # Group nodes by type for legend
    by_type: dict[str, list[str]] = {}
    for nid, m in meta.items():
        t = str(m.get("type") or "unknown")
        by_type.setdefault(t, []).append(nid)

    type_order = [
        "bottleneck",
        "mine",
        "field",
        "producing_region",
        "port",
        "terminal",
        "smelter",
        "refinery",
        "alumina_refinery",
        "liquefaction",
        "regas",
        "exchange",
        "warehouse",
        "silo",
        "consumer",
        "scrap_hub",
        "transshipment",
        "mill",
        "processor",
    ]
    ordered_types = [t for t in type_order if t in by_type] + [
        t for t in by_type if t not in type_order
    ]

    for ntype in ordered_types:
        ids = by_type[ntype]
        lats = [meta[i]["lat"] for i in ids]
        lons = [meta[i]["lon"] for i in ids]
        scores = [meta[i]["score"] for i in ids]
        sizes = [8 + 14 * s for s in scores]
        colors = [_heat_color(s) if s >= 0.15 else "#94a3b8" for s in scores]
        texts = [
            (
                f"<b>{i}</b><br>type={meta[i].get('type')}"
                f"<br>commodity={meta[i].get('commodity')}"
                f"<br>country={meta[i].get('country')}"
                f"<br>tension={meta[i]['tension']:.2f}"
                f"<br>event={meta[i]['sev']:.2f}"
            )
            for i in ids
        ]
        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="markers",
                name=ntype,
                text=texts,
                hoverinfo="text",
                marker=dict(
                    size=sizes,
                    color=colors,
                    symbol=symbols.get(ntype, "circle"),
                    line=dict(width=0.8, color="#0f172a"),
                    opacity=0.95,
                ),
            )
        )

    # Tension / shock rings (open circles)
    ring_ids = [nid for nid, m in meta.items() if m["score"] >= 0.2]
    if ring_ids:
        fig.add_trace(
            go.Scattergeo(
                lon=[meta[i]["lon"] for i in ring_ids],
                lat=[meta[i]["lat"] for i in ring_ids],
                mode="markers",
                name="tension ring",
                hoverinfo="skip",
                marker=dict(
                    size=[18 + 42 * meta[i]["score"] for i in ring_ids],
                    color="rgba(0,0,0,0)",
                    symbol="circle",
                    line=dict(
                        width=2.5,
                        color=[
                            SHORT if meta[i]["score"] >= 0.55 else AMBER
                            for i in ring_ids
                        ],
                    ),
                    opacity=0.9,
                ),
                showlegend=True,
            )
        )

    n_plotted = len(meta)
    n_total = len(network.get("nodes") or [])
    title = f"Supply chain map · {n_plotted}/{n_total} geolocated"
    if commodity_filter:
        title += f" · {commodity_filter}"

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#1e293b",
        showocean=True,
        oceancolor="#0b1220",
        showlakes=True,
        lakecolor="#0b1220",
        showcountries=True,
        countrycolor="#334155",
        showcoastlines=True,
        coastlinecolor="#475569",
        bgcolor=BG,
        resolution=110,
        fitbounds="locations",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(family="IBM Plex Mono", size=13, color=TEXT)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=height,
        margin=dict(l=0, r=0, t=44, b=0),
        font=dict(family="IBM Plex Mono", size=10, color=MUTED),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            x=0.01,
            bgcolor="rgba(17,24,39,0.85)",
            bordercolor="#1e293b",
            font=dict(size=9),
        ),
        geo=dict(bgcolor=BG),
    )
    return fig


def build_pyvis_html(
    network: dict,
    *,
    node_tensions: dict[str, float] | None = None,
    event_severities: dict[str, float] | None = None,
    height: str = "620px",
) -> str:
    from pyvis.network import Network

    node_tensions = node_tensions or {}
    event_severities = event_severities or {}

    g = nx.DiGraph()
    for node in network.get("nodes") or []:
        g.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in network.get("edges") or []:
        g.add_edge(
            edge["source"],
            edge["target"],
            **{k: v for k, v in edge.items() if k not in ("source", "target")},
        )

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor=BG,
        font_color=TEXT,
        notebook=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-9000, central_gravity=0.25, spring_length=140)

    for nid, data in g.nodes(data=True):
        sev = abs(float(event_severities.get(nid) or data.get("event_severity") or 0.0))
        tension = float(node_tensions.get(nid, 0.0))
        score = max(sev, tension)
        size = 10 + 30 * score
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
            borderWidthSelected=3,
        )

    for u, v, data in g.edges(data=True):
        w = float(data.get("weight") or 1.0)
        net.add_edge(u, v, value=max(0.4, w), color="#475569", arrows="to", width=max(0.5, w))

    tmp = Path(tempfile.gettempdir()) / "gcp_pyvis_graph.html"
    net.save_graph(str(tmp))
    return tmp.read_text(encoding="utf-8")


def relative_time(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m"
        if secs < 86400:
            return f"{secs // 3600}h"
        return f"{secs // 86400}d"
    except (TypeError, ValueError):
        return str(iso)[:16]


def _event_headline(ev: dict, art: dict) -> str:
    title = (art.get("title") or "").strip()
    if title:
        return title
    entity = (ev.get("entity_text") or "").strip()
    etype = (ev.get("event_type") or "").strip()
    commodity = (ev.get("commodity") or "").strip()
    if entity and etype:
        base = f"{entity} · {etype}"
        return f"{base} · {commodity}" if commodity else base
    if entity:
        return entity
    uid = (ev.get("article_uid") or "").strip()
    if uid.startswith("scale:"):
        return uid.replace("scale:", "").replace("_", " ")
    return uid or "(untitled)"


def event_feed_rows(
    events: list[dict],
    articles_by_uid: dict[str, dict],
) -> list[dict[str, Any]]:
    rows = []
    for ev in events:
        uid = ev.get("article_uid") or ""
        art = articles_by_uid.get(uid) or {}
        rows.append(
            {
                "headline": _event_headline(ev, art),
                "node": ev.get("node_id") or "—",
                "type": ev.get("event_type"),
                "severity": float(ev.get("severity") or 0.0),
                "commodity": ev.get("commodity"),
                "direction": ev.get("direction"),
                "entity": ev.get("entity_text"),
                "extracted_at": ev.get("extracted_at"),
                "age": relative_time(ev.get("extracted_at")),
            }
        )
    return rows


def event_cards_html(rows: list[dict], *, limit: int = 20) -> str:
    """Self-contained HTML (inline CSS) for Streamlit components.html."""
    style = (
        "<style>"
        "body{margin:0;background:transparent;color:#e2e8f0;"
        "font-family:'IBM Plex Sans',sans-serif;}"
        ".card{background:#111827;border-left:3px solid #334155;"
        "padding:0.45rem 0.65rem;margin-bottom:0.35rem;font-size:0.8rem;}"
        ".sev-hi{border-left-color:#ef4444;}"
        ".sev-mid{border-left-color:#f59e0b;}"
        ".sev-lo{border-left-color:#64748b;}"
        ".meta{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;"
        "color:#94a3b8;margin-top:0.2rem;}"
        ".chip{display:inline-block;font-family:'IBM Plex Mono',monospace;"
        "font-size:0.62rem;padding:0.05rem 0.35rem;margin:0.1rem 0.25rem 0 0;"
        "background:#1e293b;color:#cbd5e1;border-radius:2px;}"
        ".empty{font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#64748b;}"
        "</style>"
    )
    if not rows:
        return f"{style}<div class='empty'>No parsed events</div>"
    parts = [style]
    for row in rows[:limit]:
        sev = abs(float(row.get("severity") or 0.0))
        cls = "sev-hi" if sev >= 0.7 else ("sev-mid" if sev >= 0.4 else "sev-lo")
        headline = (row.get("headline") or "")[:120]
        parts.append(
            f"<div class='card {cls}'>"
            f"<div><strong>{headline}</strong></div>"
            f"<div class='meta'>"
            f"<span class='chip'>sev {sev:.2f}</span>"
            f"<span class='chip'>{row.get('node')}</span>"
            f"<span class='chip'>{row.get('type')}</span>"
            f"<span class='chip'>{row.get('commodity') or '—'}</span>"
            f"<span class='chip'>{row.get('direction') or '—'}</span>"
            f"<span class='chip'>{row.get('age')}</span>"
            f"</div></div>"
        )
    return "\n".join(parts)


def tension_heatmap_figure(commodity_tensions: dict[str, float]) -> go.Figure:
    items = sorted(commodity_tensions.items(), key=lambda x: -x[1])
    if not items:
        fig = go.Figure()
        return desk_layout(fig, height=280, title="Tension heatmap")
    labels = [k.upper() for k, _ in items]
    values = [[v for _, v in items]]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=["TENSION"],
            colorscale=[
                [0.0, "#1e293b"],
                [0.25, "#64748b"],
                [0.5, "#f59e0b"],
                [0.75, "#f97316"],
                [1.0, "#ef4444"],
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(thickness=12, len=0.8, tickfont=dict(size=10)),
            text=[[f"{v:.2f}" for _, v in items]],
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=11, color="#f8fafc"),
            hovertemplate="%{x}: %{z:.3f}<extra></extra>",
        )
    )
    return desk_layout(fig, height=200, title="Commodity tension heatmap")


def tension_bar_figure(commodity_tensions: dict[str, float]) -> go.Figure:
    items = sorted(commodity_tensions.items(), key=lambda x: -x[1])
    fig = go.Figure(
        go.Bar(
            x=[k.upper() for k, _ in items],
            y=[v for _, v in items],
            marker_color=[_heat_color(v) for _, v in items],
            text=[f"{v:.2f}" for _, v in items],
            textposition="outside",
            textfont=dict(size=10, color=MUTED),
        )
    )
    fig.update_yaxes(range=[0, 1.15])
    return desk_layout(fig, height=320, title="Commodity tensions")


def signal_book_rows(
    signals: dict[str, dict],
    *,
    long_th: float,
    short_th: float,
) -> list[dict[str, Any]]:
    rows = []
    for commodity, sig in signals.items():
        tension = float(sig.get("tension") or 0.0)
        side = str(sig.get("side") or "flat")
        if side == "long":
            dist = tension - long_th
        elif side == "short":
            dist = short_th - tension
        else:
            mid = (long_th + short_th) / 2
            dist = -(abs(tension - mid))
        rows.append(
            {
                "commodity": commodity.upper(),
                "ticker": sig.get("ticker") or "—",
                "side": side.upper(),
                "tension": round(tension, 3),
                "dist_to_edge": round(dist, 3),
            }
        )
    rows.sort(key=lambda r: -abs(float(r["tension"])))
    return rows


def style_signal_book(df: pd.DataFrame):
    """Pandas styler: green LONG / red SHORT."""
    if df is None or df.empty:
        return df

    def _side(val: str) -> str:
        s = str(val).upper()
        if s == "LONG":
            return f"color: {LONG}; font-weight: 600"
        if s == "SHORT":
            return f"color: {SHORT}; font-weight: 600"
        return f"color: {FLAT}"

    return df.style.map(_side, subset=["side"]).format(
        {"tension": "{:.3f}", "dist_to_edge": "{:.3f}"}
    )


def style_shocked_table(df: pd.DataFrame):
    """Dark heat on severity / tension columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ("event_severity", "tension"):
        if col in out.columns:
            out[col] = out[col].astype(float).round(3)

    def _heat(val: float) -> str:
        try:
            s = float(val)
        except (TypeError, ValueError):
            return ""
        return f"background-color: {_heat_color(s)}22; color: {_heat_color(s)}; font-weight: 600"

    styler = out.style
    subset = [c for c in ("event_severity", "tension") if c in out.columns]
    if subset:
        styler = styler.map(_heat, subset=subset)
    return styler


def kpi_strip_html(
    *,
    commodity: str,
    tension: float,
    side: str,
    ticker: str | None,
    sharpe: float | None,
    max_dd: float | None,
    long_th: float = 0.65,
    short_th: float = 0.35,
) -> str:
    side_u = (side or "flat").upper()
    side_c = _side_color(side)
    sharpe_s = f"{sharpe:.2f}" if sharpe is not None else "—"
    dd_s = f"{max_dd:.1f}%" if max_dd is not None else "—"
    dd_c = SHORT if (max_dd is not None and max_dd < 0) else TEXT
    cells = [
        ("COMMODITY", commodity.upper(), TEXT, "selected"),
        ("TENSION", f"{tension:.3f}", AMBER, "0–1 stress score"),
        ("SIGNAL", side_u, side_c, f"≥{long_th:.2f} long · ≤{short_th:.2f} short"),
        ("TICKER", ticker or "—", TEXT, "venue instrument"),
        ("SHARPE", sharpe_s, LONG if (sharpe or 0) > 0 else TEXT, "backtest @ this tension"),
        ("MAX DD", dd_s, dd_c, "worst peak-to-trough"),
    ]
    parts = [
        "<div style='display:flex;gap:0.5rem;flex-wrap:wrap;margin:0.2rem 0 0.55rem 0;'>"
    ]
    for label, value, color, hint in cells:
        parts.append(
            "<div style='flex:1;min-width:110px;background:#111827;border:1px solid #1e293b;"
            "padding:0.55rem 0.75rem;'>"
            f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:0.62rem;"
            f"color:#64748b;letter-spacing:0.06em;\">{label}</div>"
            f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:1.15rem;"
            f"color:{color};margin-top:0.12rem;\">{value}</div>"
            f"<div style=\"font-family:'IBM Plex Sans',sans-serif;font-size:0.65rem;"
            f"color:#64748b;margin-top:0.2rem;\">{hint}</div>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def strategy_pipeline_html(
    *,
    shock_label: str,
    backend: str,
    n_shocked: int,
    tension: float,
    side: str,
    sharpe: float | None,
    max_dd: float | None,
    active: set[int] | None = None,
    compact: bool = False,
) -> str:
    """Live 5-step strategy banner."""
    active = active or {1, 2, 3, 4, 5}
    side_u = (side or "flat").upper()
    side_cls = side_u.lower() if side_u in {"LONG", "SHORT", "FLAT"} else "flat"
    alpha_bits = []
    if sharpe is not None:
        alpha_bits.append(f"Sharpe {sharpe:.2f}")
    if max_dd is not None:
        alpha_bits.append(f"DD {max_dd:.1f}%")
    alpha_val = " · ".join(alpha_bits) if alpha_bits else "—"

    steps = [
        (1, "SHOCK", shock_label or "no event"),
        (2, "PROPAGATE", f"{(backend or '?').upper()} · {n_shocked} nodes"),
        (3, "TENSION", f"{tension:.3f}"),
        (4, "SIGNAL", side_u),
        (5, "ALPHA", alpha_val),
    ]
    wrap = "pipeline compact" if compact else "pipeline"
    parts = [f'<div class="{wrap}">']
    for n, label, val in steps:
        cls = "step active" if n in active else "step"
        val_cls = f"val {side_cls}" if n == 4 else "val"
        parts.append(
            f'<div class="{cls}">'
            f'<div class="n">STEP {n}</div>'
            f'<div class="label">{label}</div>'
            f'<div class="{val_cls}">{val}</div>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def tension_gauge_html(
    tension: float,
    *,
    long_th: float,
    short_th: float,
    side: str,
) -> str:
    """Horizontal gauge showing tension vs long/short thresholds."""
    t = max(0.0, min(1.0, float(tension)))
    pct = t * 100
    long_pct = long_th * 100
    short_pct = short_th * 100
    side_c = _side_color(side)
    return f"""
<div class="callout" style="padding:0.65rem 0.85rem;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#94a3b8;margin-bottom:0.35rem;">
    SIGNAL = f(TENSION) · current <span style="color:{side_c}">{(side or 'flat').upper()}</span>
  </div>
  <div style="position:relative;height:14px;background:#1e293b;border-radius:2px;overflow:hidden;">
    <div style="position:absolute;left:0;top:0;bottom:0;width:{pct}%;background:{AMBER};"></div>
    <div style="position:absolute;left:{short_pct}%;top:0;bottom:0;width:2px;background:{SHORT};"></div>
    <div style="position:absolute;left:{long_pct}%;top:0;bottom:0;width:2px;background:{LONG};"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;
       font-size:0.65rem;color:#64748b;margin-top:0.3rem;">
    <span>0 SHORT≤{short_th:.2f}</span>
    <span>FLAT</span>
    <span>LONG≥{long_th:.2f} 1</span>
  </div>
</div>
"""


def why_signal_html(
    *,
    commodity: str,
    side: str,
    tension: float,
    shocked: list[dict],
    ticker: str | None = None,
) -> str:
    """Explain why the current signal exists from top shocked nodes."""
    side_u = (side or "flat").upper()
    side_c = _side_color(side)
    related = [
        r
        for r in shocked
        if str(r.get("commodity") or "").lower()
        in {commodity.lower(), "cross_metal", "cross_commodity", "cross_energy", "cross_agri", ""}
    ][:5]
    if not related:
        related = list(shocked)[:5]

    if side_u == "LONG":
        meaning = (
            f"Supply-chain stress is elevated for <span class='mono'>{commodity}</span> "
            f"→ model expects upside pressure on <span class='mono'>{ticker or commodity}</span>."
        )
    elif side_u == "SHORT":
        meaning = (
            f"Tension is low for <span class='mono'>{commodity}</span> "
            f"→ model stays short / fade the stress narrative."
        )
    else:
        meaning = (
            f"Tension mid-range for <span class='mono'>{commodity}</span> "
            f"→ no edge vs thresholds → <span class='mono'>FLAT</span>."
        )

    rows = ""
    for r in related:
        rows += (
            f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:0.72rem;"
            f"color:#cbd5e1;margin:0.2rem 0;\">"
            f"· {r.get('node_id')} · sev={float(r.get('event_severity') or 0):.2f} "
            f"· {r.get('event_type') or r.get('type') or '—'}"
            f"</div>"
        )
    if not rows:
        rows = "<div style='color:#64748b;font-size:0.8rem;'>No shocked nodes in snapshot.</div>"

    return f"""
<div class="callout why">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#94a3b8;margin-bottom:0.25rem;">
    WHY THIS SIGNAL · <span style="color:{side_c}">{side_u}</span> · tension={tension:.3f}
  </div>
  <div>{meaning}</div>
  <div style="margin-top:0.55rem;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#64748b;">
    TOP SHOCKS FEEDING THE GRAPH
  </div>
  {rows}
</div>
"""


def map_legend_html() -> str:
    return """
<div class="map-legend">
  <span class="item">◆ mine / field</span>
  <span class="item">● port / terminal</span>
  <span class="item">■ smelter / refinery</span>
  <span class="item">✦ bottleneck</span>
  <span class="item">★ exchange</span>
  <span class="item">▲ consumer</span>
  <span class="item"><span class="ring"></span>tension / event ring</span>
  <span class="item"><span class="legend-dot" style="background:#64748b"></span>idle</span>
  <span class="item"><span class="legend-dot" style="background:#f59e0b"></span>elevated</span>
  <span class="item"><span class="legend-dot" style="background:#ef4444"></span>critical</span>
</div>
<p class="desk-sub" style="margin-top:0;">Anneaux = tension ou sévérité d’événement · Traits = flux physiques (shipping / rail / pipe)</p>
"""


def strategy_flow_html() -> str:
    chips = [
        "Event",
        "Node",
        "Graph diffusion / GAT",
        "Commodity tension",
        "Side",
        "Trade / backtest",
    ]
    parts = ['<div class="flow-row">']
    for i, c in enumerate(chips):
        if i:
            parts.append('<span class="flow-arrow">→</span>')
        parts.append(f'<span class="flow-chip">{c}</span>')
    parts.append("</div>")
    return "".join(parts)


def strategy_rules_html(*, long_th: float, short_th: float) -> str:
    return f"""
<div class="callout info">
  <div class="section-title" style="margin-top:0;">Règles de signal</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;line-height:1.7;">
    <div><span style="color:{LONG}">tension ≥ {long_th:.2f}</span> → <b style="color:{LONG}">LONG</b> — stress supply / upside prix</div>
    <div><span style="color:{SHORT}">tension ≤ {short_th:.2f}</span> → <b style="color:{SHORT}">SHORT</b> — stress bas / fade</div>
    <div><span style="color:{FLAT}">sinon</span> → <b style="color:{FLAT}">FLAT</b> — pas de trade</div>
  </div>
</div>
"""


def strategy_example_html(
    *,
    top_node: str | None,
    sev: float | None,
    commodity: str,
    tension: float,
    side: str,
    ticker: str | None,
    sharpe: float | None,
) -> str:
    node = top_node or "—"
    sev_s = f"{sev:.2f}" if sev is not None else "—"
    sh = f"{sharpe:.2f}" if sharpe is not None else "—"
    side_c = _side_color(side)
    return f"""
<div class="callout">
  <div class="section-title" style="margin-top:0;">Exemple live · {commodity.upper()}</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;line-height:1.6;">
    <span class="mono">{node}</span> sev={sev_s}
    → tension <span class="mono">{commodity}={tension:.2f}</span>
    → <span style="color:{side_c}">{(side or 'flat').upper()}</span>
    <span class="mono">{ticker or '—'}</span>
    → Sharpe {sh}
  </div>
</div>
"""


def glossary_html(*, half_life_h: float = 72.0) -> str:
    return f"""
<div class="section-title">Glossaire</div>
<dl class="glossary">
  <dt>event_severity</dt><dd>Intensité du choc sur un nœud (0–1), après decay temporel.</dd>
  <dt>tension</dt><dd>Stress agrégé (nœud ou commodity) après propagation dans le graphe.</dd>
  <dt>decay</dt><dd>Demi-vie {half_life_h:.0f}h — un choc ancien pèse moins.</dd>
  <dt>shocked node</dt><dd>Nœud avec événement actif ou tension élevée.</dd>
  <dt>SPOF</dt><dd>Single point of failure — centralité (betweenness) élevée.</dd>
  <dt>signal</dt><dd>LONG / SHORT / FLAT dérivé des seuils de tension.</dd>
  <dt>alpha</dt><dd>Performance simulée si on tient ce signal sur l’historique prix.</dd>
</dl>
"""


def alpha_metric_cards_html(bt: dict[str, Any]) -> str:
    sharpe = bt.get("sharpe")
    ret = bt.get("total_return")
    dd = bt.get("max_drawdown")
    turn = bt.get("turnover")
    cards = [
        (
            "Sharpe",
            f"{sharpe:.2f}" if sharpe is not None else "—",
            "Rendement / risque annualisé",
        ),
        (
            "Return",
            f"{ret:.1f}%" if ret is not None else "—",
            "P&L total sur la fenêtre",
        ),
        (
            "Max DD",
            f"{dd:.1f}%" if dd is not None else "—",
            "Pire perte peak → trough",
        ),
        (
            "Turnover",
            f"{turn:.2f}" if turn is not None else "—",
            "Fréquence de changement de position",
        ),
    ]
    parts = ['<div class="metric-cards">']
    for k, v, h in cards:
        parts.append(
            f'<div class="metric-card"><div class="k">{k}</div>'
            f'<div class="v">{v}</div><div class="h">{h}</div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def tension_attribution_html(
    *,
    commodity: str,
    tension: float,
    drivers: list[dict[str, Any]],
) -> str:
    """Explain why commodity tension is high (max node after diffusion + normalize)."""
    rows = ""
    for i, d in enumerate(drivers[:8]):
        bar = max(0.0, min(1.0, float(d.get("tension") or 0.0))) * 100
        mark = " ← max (=1.0 after normalize)" if i == 0 else ""
        rows += (
            f"<div class='attr-row'>"
            f"<div class='attr-lab'>{d.get('node_id')}</div>"
            f"<div class='attr-bar'><span style='width:{bar}%'></span></div>"
            f"<div class='attr-meta'>sev={float(d.get('event_severity') or 0):.2f} "
            f"· ten={float(d.get('tension') or 0):.2f} · {d.get('event_type') or d.get('type') or '—'}"
            f"{mark}</div>"
            f"</div>"
        )
    if not rows:
        rows = "<div style='color:#64748b;font-size:0.8rem;'>Aucun choc lié à cette commodity.</div>"

    return f"""
<div class="callout why">
  <div class="section-title" style="margin-top:0;">Pourquoi {commodity.upper()} = {tension:.2f} ?</div>
  <div style="font-size:0.85rem;line-height:1.45;margin-bottom:0.55rem;">
    La diffusion propage les chocs, puis <b>normalise</b> pour que le nœud le plus chaud = <span class="mono">1.0</span>.
    La tension commodity = <span class="mono">max</span> des nœuds tagués {commodity}
    (et chokepoints partagés qui y sont connectés).
  </div>
  {rows}
</div>
"""


def portfolio_summary_html(
    *,
    n_long: int,
    n_short: int,
    n_flat: int,
    n_live_events: int,
    snap_age: str,
) -> str:
    return f"""
<div class="metric-cards">
  <div class="metric-card"><div class="k">LONG</div><div class="v" style="color:{LONG}">{n_long}</div><div class="h">positions ouvertes</div></div>
  <div class="metric-card"><div class="k">SHORT</div><div class="v" style="color:{SHORT}">{n_short}</div><div class="h">positions ouvertes</div></div>
  <div class="metric-card"><div class="k">FLAT</div><div class="v" style="color:{FLAT}">{n_flat}</div><div class="h">pas de trade</div></div>
  <div class="metric-card"><div class="k">NEWS / EVT</div><div class="v">{n_live_events}</div><div class="h">snapshot {snap_age}</div></div>
</div>
"""


def portfolio_rows(
    signals: dict[str, dict],
    *,
    shocked: list[dict],
    last_prices: dict[str, float | None],
    long_th: float,
    short_th: float,
) -> list[dict[str, Any]]:
    """Build book rows for tradable commodities only."""
    rows = []
    for commodity, sig in signals.items():
        ticker = sig.get("ticker")
        if not ticker:
            continue
        tension = float(sig.get("tension") or 0.0)
        side = str(sig.get("side") or "flat").upper()
        drivers = [
            r
            for r in shocked
            if str(r.get("commodity") or "").lower()
            in {commodity.lower(), "cross_metal", "cross_commodity", "cross_energy", "cross_agri"}
        ]
        drivers.sort(
            key=lambda r: (
                -float(r.get("tension") or 0),
                -abs(float(r.get("event_severity") or 0)),
            )
        )
        top = drivers[0] if drivers else None
        why = "—"
        if top:
            why = (
                f"{top.get('node_id')} ({top.get('event_type') or top.get('type')}) "
                f"sev={float(top.get('event_severity') or 0):.2f}"
            )
        elif side == "LONG":
            why = f"tension {tension:.2f} ≥ {long_th:.2f}"
        elif side == "SHORT":
            why = f"tension {tension:.2f} ≤ {short_th:.2f}"
        else:
            why = "entre seuils → flat"

        px = last_prices.get(commodity)
        rows.append(
            {
                "commodity": commodity.upper(),
                "ticker": ticker,
                "side": side,
                "tension": round(tension, 3),
                "last": round(float(px), 2) if px is not None else None,
                "why": why,
            }
        )
    # actives first
    order = {"LONG": 0, "SHORT": 1, "FLAT": 2}
    rows.sort(key=lambda r: (order.get(r["side"], 9), -r["tension"]))
    return rows


def style_portfolio_book(df: pd.DataFrame):
    if df is None or df.empty:
        return df

    out = df.copy()
    if "last" in out.columns:
        out["last"] = out["last"].apply(
            lambda x: "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{float(x):,.2f}"
        )

    def _side(val: str) -> str:
        s = str(val).upper()
        if s == "LONG":
            return f"color: {LONG}; font-weight: 600"
        if s == "SHORT":
            return f"color: {SHORT}; font-weight: 600"
        return f"color: {FLAT}"

    return out.style.map(_side, subset=["side"]).format({"tension": "{:.3f}"})


def price_signal_figure(
    prices: pd.DataFrame,
    *,
    commodity: str,
    tension: float | None = None,
    side: str | None = None,
    ticker: str | None = None,
    positions: pd.Series | None = None,
    long_th: float = 0.65,
    short_th: float = 0.35,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )
    if prices is None or prices.empty:
        return desk_layout(fig, height=420, title=f"{commodity}: no price data")

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
                increasing_line_color=LONG,
                decreasing_line_color=SHORT,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=close.index,
                y=close.values,
                mode="lines",
                name="close",
                line=dict(color=AMBER, width=1.5),
            ),
            row=1,
            col=1,
        )

    if positions is not None and len(positions):
        pos = positions.reindex(close.index).fillna(0.0)
        long_mask = pos == 1.0
        short_mask = pos == -1.0
        if long_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=close.index[long_mask],
                    y=close.values[long_mask],
                    mode="markers",
                    name="LONG",
                    marker=dict(symbol="triangle-up", size=7, color=LONG),
                ),
                row=1,
                col=1,
            )
        if short_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=close.index[short_mask],
                    y=close.values[short_mask],
                    mode="markers",
                    name="SHORT",
                    marker=dict(symbol="triangle-down", size=7, color=SHORT),
                ),
                row=1,
                col=1,
            )

    t_val = float(tension) if tension is not None else 0.0
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=[t_val] * len(close),
            mode="lines",
            name="tension",
            line=dict(color=AMBER, width=2),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.15)",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=long_th,
        line_dash="dot",
        line_color=LONG,
        line_width=1,
        row=2,
        col=1,
        annotation_text="long",
        annotation_position="right",
    )
    fig.add_hline(
        y=short_th,
        line_dash="dot",
        line_color=SHORT,
        line_width=1,
        row=2,
        col=1,
        annotation_text="short",
        annotation_position="right",
    )
    fig.update_yaxes(range=[0, 1], title_text="tension", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False)

    title = f"{commodity.upper()}"
    if ticker:
        title += f"  {ticker}"
    if side:
        title += f"  ·  {side.upper()}"
    fig = desk_layout(fig, height=440, title=title)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def equity_drawdown_figure(
    equity: pd.Series,
    drawdown: pd.Series,
    *,
    title: str = "Equity & drawdown",
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
    )
    if equity is None or equity.empty:
        return desk_layout(fig, height=360, title=title)

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="equity",
            line=dict(color=AMBER, width=1.6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="drawdown %",
            line=dict(color=SHORT, width=1),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.2)",
        ),
        row=2,
        col=1,
    )
    return desk_layout(fig, height=380, title=title)


def tension_hist_figure(node_tensions: dict[str, float]) -> go.Figure:
    vals = [float(v) for v in node_tensions.values() if float(v) > 0]
    fig = go.Figure(
        go.Histogram(
            x=vals or [0.0],
            nbinsx=20,
            marker_color=AMBER,
            opacity=0.85,
        )
    )
    fig.update_xaxes(title_text="node tension", range=[0, 1])
    fig.update_yaxes(title_text="count")
    return desk_layout(fig, height=300, title="Node tension distribution")


def metrics_table_rows(backtests: dict[str, dict]) -> list[dict[str, Any]]:
    rows = []
    for commodity, m in sorted(backtests.items()):
        if m.get("error"):
            rows.append(
                {
                    "commodity": commodity.upper(),
                    "sharpe": None,
                    "return_%": None,
                    "max_dd_%": None,
                    "turnover": None,
                    "n": m.get("n"),
                    "error": m.get("error"),
                }
            )
            continue
        rows.append(
            {
                "commodity": commodity.upper(),
                "ticker": m.get("ticker"),
                "sharpe": round(float(m["sharpe"]), 3) if m.get("sharpe") is not None else None,
                "return_%": round(float(m["total_return"]), 2)
                if m.get("total_return") is not None
                else None,
                "max_dd_%": round(float(m["max_drawdown"]), 2)
                if m.get("max_drawdown") is not None
                else None,
                "turnover": round(float(m["turnover"]), 3)
                if m.get("turnover") is not None
                else None,
                "n": m.get("n"),
                "engine": m.get("engine"),
            }
        )
    return rows
