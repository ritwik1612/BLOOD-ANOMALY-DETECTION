from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def xywhn_to_xyxy(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    center_x, center_y, box_width, box_height = box
    left = round((center_x - box_width / 2) * width)
    top = round((center_y - box_height / 2) * height)
    right = round((center_x + box_width / 2) * width)
    bottom = round((center_y + box_height / 2) * height)
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def crop_and_resize(image: np.ndarray, xyxy: tuple[int, int, int, int], size: int = 64) -> np.ndarray | None:
    left, top, right, bottom = xyxy
    if right <= left or bottom <= top:
        return None
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def read_yolo_labels(label_path: Path) -> list[tuple[int, list[float]]]:
    if not label_path.exists():
        return []
    entries = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) == 5:
            entries.append((int(float(values[0])), [float(value) for value in values[1:]]))
    return entries


def sanitize_yolo_box(box: list[float]) -> list[float]:
    """Clip negligible annotation rounding overshoots to the valid YOLO range."""
    center_x, center_y, width, height = box
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)
    center_x = min(max(center_x, width / 2), 1.0 - width / 2)
    center_y = min(max(center_y, height / 2), 1.0 - height / 2)
    return [center_x, center_y, width, height]
