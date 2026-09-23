"""Graph Attention Network for shock → tension scores."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from backend import config
from backend.graph_core.convert import feature_dims


class CommodityGAT(nn.Module):
    """Node features (+ optional edge_attr) → node tension → commodity scores."""

    def __init__(
        self,
        in_channels: int | None = None,
        edge_dim: int | None = None,
        hidden: int | None = None,
        heads: int | None = None,
        n_commodities: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = feature_dims()
        in_channels = in_channels if in_channels is not None else dims["node"]
        edge_dim = edge_dim if edge_dim is not None else dims["edge"]
        hidden = config.GNN_HIDDEN if hidden is None else hidden
        heads = config.GNN_HEADS if heads is None else heads
        n_commodities = n_commodities or len(config.TICKERS)

        self.n_commodities = n_commodities
        self.commodity_order = list(config.TICKERS.keys())
        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels,
            hidden,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
        )
        self.conv2 = GATConv(
            hidden * heads,
            hidden,
            heads=1,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False,
        )
        self.node_head = nn.Linear(hidden, 1)
        self.commodity_head = nn.Linear(hidden, n_commodities)

    def encode(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (node_tension [N], commodity_logits [C] in (-inf,inf))."""
        h = self.encode(data)
        node_t = torch.tanh(self.node_head(h)).squeeze(-1)
        pooled = h.mean(dim=0)
        commodity_logits = self.commodity_head(pooled)
        return node_t, commodity_logits

    def commodity_scores(self, data: Data) -> dict[str, float]:
        self.eval()
        with torch.no_grad():
            _, logits = self.forward(data)
            probs = torch.sigmoid(logits).cpu().tolist()
        return {
            name: float(probs[i]) if i < len(probs) else 0.0
            for i, name in enumerate(self.commodity_order)
        }
