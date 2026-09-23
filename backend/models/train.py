"""Train CommodityGAT; save checkpoint under backend/models/gnn/."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from backend import config
from backend.graph_core.convert import feature_dims
from backend.models.dataset import build_dataset, collate_summary
from backend.models.gat import CommodityGAT

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    name = (config.DEVICE or "cpu").lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_gat(
    sector: str,
    *,
    epochs: int | None = None,
    lr: float | None = None,
    n_synthetic: int = 48,
    checkpoint: Path | None = None,
) -> dict:
    config.ensure_dirs()
    epochs = config.GNN_EPOCHS if epochs is None else epochs
    lr = config.GNN_LR if lr is None else lr
    checkpoint = checkpoint or config.GNN_CHECKPOINT
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    samples = build_dataset(sector, n_synthetic=n_synthetic, include_events=True)
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
                },
                checkpoint,
            )

    dir_acc = n_dir_ok / n_dir if n_dir else 0.0
    return {
        "checkpoint": str(checkpoint),
        "best_loss": best_loss,
        "epochs": epochs,
        "direction_acc": dir_acc,
        "n_samples": len(samples),
        "history": history[-5:],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CommodityGAT")
    parser.add_argument("--sector", default=config.DEFAULT_SECTOR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-synthetic", type=int, default=48)
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
    )
    print(result)


if __name__ == "__main__":
    main()
