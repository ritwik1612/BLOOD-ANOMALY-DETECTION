from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def compute_centroids(features_path: Path, output_path: Path) -> Path:
    data = np.load(features_path, allow_pickle=False)
    latent, labels = data["latent"], data["label"]
    centroids = {int(class_id): torch.from_numpy(latent[labels == class_id].mean(axis=0)).float() for class_id in np.unique(labels)}
    if len(centroids) != 3:
        raise ValueError("Expected all three TXL-PBC normal classes in feature data.")
    torch.save({"centroids": centroids, "source": str(features_path)}, output_path)
    return output_path
