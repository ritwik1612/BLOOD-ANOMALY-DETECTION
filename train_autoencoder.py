from __future__ import annotations

from utils.config import PROJECT_ROOT, load_yaml
from training.trainer_ae import train_autoencoder


if __name__ == "__main__":
    print(f"Best autoencoder saved to: {train_autoencoder(load_yaml(PROJECT_ROOT / 'configs' / 'autoencoder.yaml'))}")
