"""GAT shock-propagation model (PyTorch Geometric)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from backend import config
from backend.graph_core import feature_dims, to_pyg

logger = logging.getLogger(__name__)


class ShockGAT(nn.Module):
    """2-layer GAT → per-node tension in [0, 1]."""

    def __init__(
        self,
        in_channels: int,
        hidden: int | None = None,
        heads: int | None = None,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = config.GNN_HIDDEN if hidden is None else hidden
        heads = config.GNN_HEADS if heads is None else heads
        dims = feature_dims()
        edge_dim = dims["edge"] if edge_dim is None else edge_dim

        self.conv1 = GATConv(
            in_channels,
            hidden,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=0.1,
        )
        self.conv2 = GATConv(
            hidden * heads,
            hidden,
            heads=1,
            edge_dim=edge_dim,
            concat=False,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        out = torch.sigmoid(self.head(x)).view(-1)
        return out


def build_model(device: str | None = None) -> ShockGAT:
    dims = feature_dims()
    model = ShockGAT(in_channels=dims["node"], edge_dim=dims["edge"])
    device = device or config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return model.to(device)


def default_checkpoint_path() -> Path:
    config.ensure_dirs()
    return config.GNN_CHECKPOINT_DIR / "shock_gat.pt"


def save_checkpoint(model: ShockGAT, path: Path | None = None, meta: dict | None = None) -> Path:
    path = path or default_checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "feature_dims": feature_dims(),
        "meta": meta or {},
    }
    torch.save(payload, path)
    logger.info("Saved GAT checkpoint → %s", path)
    return path


def load_checkpoint(path: Path | None = None, device: str | None = None) -> ShockGAT | None:
    path = path or default_checkpoint_path()
    if not path.exists():
        logger.warning("No GAT checkpoint at %s", path)
        return None
    device = device or config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(device=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_node_tensions(
    network: dict,
    model: ShockGAT | None = None,
    *,
    device: str | None = None,
) -> dict[str, float]:
    device = device or config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = model or load_checkpoint(device=device)
    if model is None:
        model = build_model(device=device)
        model.eval()

    data = to_pyg(network, validate=False).to(device)
    scores = model(data).detach().cpu().tolist()
    node_ids = list(data.node_ids)
    return {nid: float(s) for nid, s in zip(node_ids, scores)}


def commodity_tensions_from_nodes(
    network: dict, node_scores: dict[str, float]
) -> dict[str, float]:
    by_c: dict[str, list[float]] = {}
    for node in network.get("nodes") or []:
        c = node.get("commodity")
        if not c:
            continue
        by_c.setdefault(c, []).append(node_scores.get(node["id"], 0.0))
    out = {c: float(max(vals)) if vals else 0.0 for c, vals in by_c.items()}
    for c in network.get("supported_commodities") or []:
        out.setdefault(c, 0.0)
    return out


def run_gat(network: dict, model: ShockGAT | None = None) -> dict[str, Any]:
    node_scores = predict_node_tensions(network, model=model)
    return {
        "node_tensions": node_scores,
        "commodity_tensions": commodity_tensions_from_nodes(network, node_scores),
    }
