from __future__ import annotations

import argparse
from pathlib import Path

from feature_extraction.extract_latent import extract_features
from utils.config import PROJECT_ROOT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract autoencoder features from normal cell crops.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "outputs" / "weights" / "autoencoder_best.pt")
    args = parser.parse_args()
    print(f"Saved: {extract_features(args.weights, args.split)}")
