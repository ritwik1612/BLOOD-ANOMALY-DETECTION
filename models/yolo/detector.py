from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def load_detector(weights: str | Path) -> YOLO:
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    return YOLO(str(weights_path))
