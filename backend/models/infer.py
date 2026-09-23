"""Load GAT checkpoint → commodity tension scores."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from backend import config
from backend.graph_core import to_pyg
from backend.graph_core.convert import feature_dims
from backend.models.baseline import tension_from_network
from backend.models.gat import CommodityGAT

logger = logging.getLogger(__name__)


def checkpoint_exists(path: Path | None = None) -> bool:
    path = path or config.GNN_CHECKPOINT
    return path.is_file()


def load_gat(path: Path | None = None, device: str | None = None) -> CommodityGAT:
    path = path or config.GNN_CHECKPOINT
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    dims = ckpt.get("dims") or feature_dims()
    model = CommodityGAT(
        in_channels=dims["node"],
        edge_dim=dims["edge"],
        n_commodities=len(ckpt.get("commodity_order") or config.TICKERS),
    )
    model.load_state_dict(ckpt["model"])
    if ckpt.get("commodity_order"):
        model.commodity_order = list(ckpt["commodity_order"])
    model.eval()
    return model


def _blend(
    gat: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    keys = set(gat) | set(baseline)
    # under-trained GAT often collapses near 0 — keep baseline floor
    gat_peak = max(gat.values(), default=0.0)
    if gat_peak < 0.05:
        return dict(baseline)
    return {k: max(float(gat.get(k, 0.0)), float(baseline.get(k, 0.0))) for k in keys}


def infer_tensions(
    network: dict,
    *,
    use_gat: bool | None = None,
    method: str = "diffusion",
    checkpoint: Path | None = None,
) -> dict[str, float]:
    """Commodity → tension in [0, 1]. Prefers GAT when checkpoint present."""
    ckpt = checkpoint or config.GNN_CHECKPOINT
    prefer_gat = checkpoint_exists(ckpt) if use_gat is None else use_gat
    baseline, _ = tension_from_network(network, method=method)
    if prefer_gat and checkpoint_exists(ckpt):
        try:
            model = load_gat(ckpt)
            data = to_pyg(network, validate=False)
            scores = model.commodity_scores(data)
            supported = set(network.get("supported_commodities") or scores.keys())
            gat = {k: v for k, v in scores.items() if k in supported or not supported}
            return _blend(gat, baseline)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GAT infer failed (%s); falling back to diffusion", exc)
    return baseline
