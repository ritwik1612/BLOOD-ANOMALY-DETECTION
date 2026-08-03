from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from models.autoencoder import ConvAutoencoder
from training.dataset_loader import CellCropDataset
from utils.config import PROJECT_ROOT


def load_autoencoder(checkpoint_path: Path, device: torch.device) -> ConvAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ConvAutoencoder(int(checkpoint["config"]["latent_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


@torch.inference_mode()
def extract_features(checkpoint_path: Path, split: str, batch_size: int = 128, device_name: str | None = None) -> Path:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_autoencoder(checkpoint_path, device)
    dataset = CellCropDataset(PROJECT_ROOT / "dataset" / "crops", split)
    if not dataset:
        raise RuntimeError(f"No {split} cell crops found. Run dataset/extract_crops.py first.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    latent_rows, mse_rows, labels, paths = [], [], [], []
    for images, batch_labels, batch_paths in loader:
        images = images.to(device)
        reconstruction, latent = model(images)
        mse = functional.mse_loss(reconstruction, images, reduction="none").mean(dim=(1, 2, 3))
        latent_rows.append(latent.cpu().numpy())
        mse_rows.append(mse.cpu().numpy())
        labels.append(batch_labels.numpy())
        paths.extend(batch_paths)
    output = PROJECT_ROOT / "outputs" / "weights" / f"features_{split}.npz"
    np.savez_compressed(output, latent=np.concatenate(latent_rows), mse=np.concatenate(mse_rows), label=np.concatenate(labels), path=np.array(paths))
    return output
