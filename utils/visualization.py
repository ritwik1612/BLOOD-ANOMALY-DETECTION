from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw_open_set_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for item in detections:
        left, top, right, bottom = (int(value) for value in item["xyxy"])
        color = (0, 170, 0) if item["status"] == "Normal" else (0, 0, 220)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        cv2.putText(canvas, item["display_label"], (left, max(18, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def save_reconstruction_grid(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    prioritized_rows = sorted(
        rows,
        key=lambda item: (item.get("status") == "Abnormal", item.get("anomaly_score", 0.0)),
        reverse=True,
    )
    count = min(12, len(prioritized_rows))
    figure, axes = plt.subplots(count, 2, figsize=(5, 2.5 * count))
    axes = np.atleast_2d(axes)
    for index, item in enumerate(prioritized_rows[:count]):
        axes[index, 0].imshow(cv2.cvtColor(item["patch"], cv2.COLOR_BGR2RGB))
        axes[index, 0].set_title(f"Input: {item['display_label']}")
        axes[index, 1].imshow(item["reconstruction"])
        axes[index, 1].set_title("Reconstruction")
        for axis in axes[index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
