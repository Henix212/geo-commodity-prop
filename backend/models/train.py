"""Weak-label GAT training: price forward returns or diffusion teacher."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import torch
import torch.nn.functional as F

from backend import config
from backend.graph_core import to_pyg
from backend.logging_setup import configure_logging
from backend.models.baseline_diffusion import node_tensions_from_diffusion
from backend.models.gat import (
    build_model,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


def _parse_day(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def weak_node_targets(network: dict, *, horizon_days: int | None = None) -> torch.Tensor:
    """Build per-node regression targets in [0, 1].

    Prefer |forward return| mapped through events' commodities; fallback to diffusion.
    """
    from backend.database import list_events, list_prices

    horizon = config.LABEL_HORIZON_DAYS if horizon_days is None else horizon_days
    nodes = network.get("nodes") or []
    n = len(nodes)
    teacher = node_tensions_from_diffusion(network)
    targets = torch.tensor([teacher.get(node["id"], 0.0) for node in nodes], dtype=torch.float)

    # Overlay commodity-level weak labels from prices around events
    events = list_events(limit=500)
    if not events:
        return targets

    commodity_scores: dict[str, list[float]] = {}
    for ev in events:
        commodity = ev.get("commodity")
        if not commodity:
            continue
        day = _parse_day(ev.get("extracted_at"))
        if day is None:
            continue
        start = day.date().isoformat()
        end = (day + timedelta(days=horizon)).date().isoformat()
        rows = list_prices(commodity=commodity, start=start, end=end, limit=50)
        if len(rows) < 2:
            continue
        closes = [r["close"] for r in rows if r.get("close") is not None]
        if len(closes) < 2 or closes[0] == 0:
            continue
        ret = abs((closes[-1] - closes[0]) / closes[0])
        # map return magnitude into [0,1] with soft cap
        score = min(1.0, ret * 20.0)
        # blend with event severity
        score = max(score, float(ev.get("severity") or 0.0) * 0.5)
        commodity_scores.setdefault(commodity, []).append(score)

    if not commodity_scores:
        return targets

    commodity_target = {
        c: float(sum(vals) / len(vals)) for c, vals in commodity_scores.items()
    }
    blended = []
    for node in nodes:
        base = teacher.get(node["id"], 0.0)
        c = node.get("commodity")
        if c and c in commodity_target:
            blended.append(0.5 * base + 0.5 * commodity_target[c])
        else:
            blended.append(base)
    return torch.tensor(blended, dtype=torch.float)


def train_gat(
    network: dict,
    *,
    epochs: int | None = None,
    lr: float | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    epochs = config.GNN_EPOCHS if epochs is None else epochs
    lr = config.GNN_LR if lr is None else lr
    device = device or config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    data = to_pyg(network, validate=False).to(device)
    y = weak_node_targets(network).to(device)
    model = build_model(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            mae = float(torch.mean(torch.abs(pred - y)).item())
            # directionality vs mean
            pred_dir = pred > pred.mean()
            y_dir = y > y.mean()
            direction_acc = float((pred_dir == y_dir).float().mean().item())
        history.append({"epoch": epoch, "loss": float(loss.item()), "mae": mae, "direction_acc": direction_acc})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "epoch %d/%d loss=%.4f mae=%.4f dir_acc=%.3f",
                epoch,
                epochs,
                loss.item(),
                mae,
                direction_acc,
            )

    path = save_checkpoint(
        model,
        meta={"epochs": epochs, "final": history[-1] if history else {}},
    )
    return {"checkpoint": str(path), "history": history, "final": history[-1] if history else {}}


def evaluate_ablations(network: dict) -> dict[str, Any]:
    from backend.models.ablate import ablate_network
    from backend.models.baseline_diffusion import commodity_tensions

    variants = {
        "full": network,
        "no_scrap": ablate_network(network, drop_scrap=True),
        "no_chokepoints": ablate_network(network, drop_chokepoints=True),
        "no_scrap_no_chokepoints": ablate_network(
            network, drop_scrap=True, drop_chokepoints=True
        ),
    }
    out = {}
    for name, net in variants.items():
        out[name] = {
            "n_nodes": len(net.get("nodes") or []),
            "n_edges": len(net.get("edges") or []),
            "commodity_tensions": commodity_tensions(net),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    from backend.main import build_graph, inject_persisted_events

    parser = argparse.ArgumentParser(description="Train ShockGAT with weak labels")
    parser.add_argument("--sector", default=config.DEFAULT_SECTOR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ablate", action="store_true", help="Also print ablation table")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    network = build_graph(args.sector)
    inject_persisted_events(network)
    result = train_gat(network, epochs=args.epochs)
    print("train:", {k: result[k] for k in ("checkpoint", "final")})
    if args.ablate:
        print("ablations:", evaluate_ablations(network))


if __name__ == "__main__":
    main()
