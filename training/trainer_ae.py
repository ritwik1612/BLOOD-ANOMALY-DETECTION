from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.autoencoder import ConvAutoencoder
from training.dataset_loader import CellCropDataset
from utils.config import PROJECT_ROOT, resolve_path
from utils.reproducibility import seed_everything


def _device(value: str | None) -> torch.device:
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def _run_epoch(model: ConvAutoencoder, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: Adam | None) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            reconstruction, _ = model(images)
            loss = criterion(reconstruction, images)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
    return total_loss / max(1, len(loader.dataset))


def train_autoencoder(config: dict) -> Path:
    seed_everything(int(config["seed"]))
    device = _device(config.get("device"))
    crops_root = PROJECT_ROOT / "dataset" / "crops"
    train_data = CellCropDataset(crops_root, "train", augment=True)
    val_data = CellCropDataset(crops_root, "val")
    if not train_data or not val_data:
        raise RuntimeError("Cell crops are missing. Run `python dataset/extract_crops.py` first.")
    loader_args = {"batch_size": config["batch_size"], "num_workers": config["num_workers"], "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    model = ConvAutoencoder(int(config["latent_dim"])).to(device)
    optimizer = Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    criterion = nn.MSELoss()
    best_loss, stale_epochs = float("inf"), 0
    checkpoint_path = resolve_path(config["checkpoint_path"])
    last_checkpoint_path = resolve_path(config["last_checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(resolve_path(config["tensorboard_dir"])))
    try:
        for epoch in range(1, int(config["epochs"]) + 1):
            train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = _run_epoch(model, val_loader, criterion, device, None)
            writer.add_scalars("loss", {"train": train_loss, "validation": val_loss}, epoch)
            state = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_mse": val_loss, "config": config}
            torch.save(state, last_checkpoint_path)
            if val_loss < best_loss:
                best_loss, stale_epochs = val_loss, 0
                torch.save(state, checkpoint_path)
            else:
                stale_epochs += 1
            print(f"Epoch {epoch:03d}: train_mse={train_loss:.6f}, val_mse={val_loss:.6f}")
            if stale_epochs >= int(config["early_stopping_patience"]):
                print("Early stopping reached.")
                break
    finally:
        writer.close()
    return checkpoint_path
