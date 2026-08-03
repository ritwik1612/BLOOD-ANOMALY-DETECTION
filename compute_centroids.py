from __future__ import annotations

from anomaly_detection.centroid_calculator import compute_centroids
from utils.config import PROJECT_ROOT


if __name__ == "__main__":
    source = PROJECT_ROOT / "outputs" / "weights" / "features_train.npz"
    target = PROJECT_ROOT / "outputs" / "weights" / "centroids.pt"
    print(f"Saved: {compute_centroids(source, target)}")
