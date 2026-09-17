"""GNN models: diffusion baseline, GAT, centrality, training.

Lazy exports — keep torch/transformers out of the Streamlit import path.
"""

__all__ = [
    "ablate_network",
    "commodity_tensions",
    "node_tensions_from_diffusion",
    "run_diffusion",
    "single_points_of_failure",
    "node_betweenness",
    "ShockGAT",
    "load_checkpoint",
    "predict_node_tensions",
    "run_gat",
    "save_checkpoint",
]


def __getattr__(name: str):
    if name == "ablate_network":
        from backend.models.ablate import ablate_network

        return ablate_network
    if name in {
        "commodity_tensions",
        "node_tensions_from_diffusion",
        "run_diffusion",
    }:
        from backend.models import baseline_diffusion

        return getattr(baseline_diffusion, name)
    if name in {"single_points_of_failure", "node_betweenness"}:
        from backend.models import centrality

        return getattr(centrality, name)
    if name in {
        "ShockGAT",
        "load_checkpoint",
        "predict_node_tensions",
        "run_gat",
        "save_checkpoint",
    }:
        from backend.models import gat

        return getattr(gat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
