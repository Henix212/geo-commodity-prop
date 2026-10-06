"""Train CommodityGAT; save checkpoint under data/checkpoints/."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from geo_commodity import config
from geo_commodity.graph.convert import feature_dims
from geo_commodity.gnn.dataset import build_dataset, collate_summary
from geo_commodity.gnn.gat import CommodityGAT

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    name = (config.DEVICE or "cpu").lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_gat(
    sector: str = "all",
    *,
    epochs: int | None = None,
    lr: float | None = None,
    n_synthetic: int | None = None,
    event_limit: int = 500,
    checkpoint: Path | None = None,
) -> dict:
    config.ensure_dirs()
    sector = (sector or "all").strip().lower()
    epochs = config.GNN_EPOCHS if epochs is None else epochs
    lr = config.GNN_LR if lr is None else lr
    # More synthetic shocks when training the full multi-sector graph
    if n_synthetic is None:
        n_synthetic = 200 if sector in {"all", "*"} else 48
    checkpoint = checkpoint or config.GNN_CHECKPOINT
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    samples = build_dataset(
        sector,
        n_synthetic=n_synthetic,
        include_events=True,
        event_limit=event_limit,
    )
    if not samples:
        raise RuntimeError("No training samples")
    logger.info("dataset %s", collate_summary(samples))

    device = _device()
    dims = feature_dims()
    model = CommodityGAT(
        in_channels=dims["node"],
        edge_dim=dims["edge"],
        n_commodities=len(config.TICKERS),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    history: list[float] = []
    n_dir_ok = 0
    n_dir = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for data in samples:
            data = data.to(device)
            opt.zero_grad()
            _, logits = model(data)
            target = data.y.to(device)
            loss = F.mse_loss(torch.sigmoid(logits), target)
            loss.backward()
            opt.step()
            total += float(loss.item())

            # directionality: predicted max commodity vs label commodity
            pred_idx = int(torch.argmax(torch.sigmoid(logits)).item())
            true_idx = int(torch.argmax(target).item())
            if target.max() > 0.05:
                n_dir += 1
                n_dir_ok += int(pred_idx == true_idx)

        avg = total / len(samples)
        history.append(avg)
        logger.info("epoch %d/%d loss=%.4f", epoch, epochs, avg)
        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "sector": sector,
                    "dims": dims,
                    "commodity_order": list(config.TICKERS.keys()),
                    "epoch": epoch,
                    "loss": best_loss,
                    "n_samples": len(samples),
                },
                checkpoint,
            )

    dir_acc = n_dir_ok / n_dir if n_dir else 0.0
    return {
        "checkpoint": str(checkpoint),
        "sector": sector,
        "best_loss": best_loss,
        "epochs": epochs,
        "direction_acc": dir_acc,
        "n_samples": len(samples),
        "n_synthetic": n_synthetic,
        "history": history[-5:],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CommodityGAT")
    parser.add_argument(
        "--sector",
        default="all",
        help="all (merged metals+energy+agriculture) | metals | energy | agriculture",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=None,
        help="Synthetic shock samples (default: 200 for all, 48 otherwise)",
    )
    parser.add_argument(
        "--event-limit",
        type=int,
        default=500,
        help="Max DB events to turn into labeled snapshots",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = train_gat(
        args.sector,
        epochs=args.epochs,
        lr=args.lr,
        n_synthetic=args.n_synthetic,
        event_limit=args.event_limit,
    )
    print(result)


if __name__ == "__main__":
    main()
