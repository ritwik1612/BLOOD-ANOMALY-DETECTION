from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anomaly_detection.scoring import cosine_distance, fit_calibration
from inference.pipeline import OpenSetPipeline
from utils.config import PROJECT_ROOT, load_yaml, resolve_path
from utils.image_ops import read_yolo_labels, xywhn_to_xyxy


def iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / union if union else 0.0


def calibrate(anomaly_config: dict, dataset_config: dict, yolo_weights: Path, ae_weights: Path, centroids: Path) -> Path:
    dataset_root = resolve_path(dataset_config["dataset_root"])
    pipeline = OpenSetPipeline(yolo_weights, ae_weights, centroids, anomaly_config)
    values_by_class = {class_id: {"mse": [], "cosine": [], "confidence": []} for class_id in anomaly_config["class_labels"]}
    for image_path in sorted((dataset_root / "images" / "val").iterdir()):
        image, predictions = pipeline.analyze_path(image_path)
        ground_truth = []
        for class_id, box in read_yolo_labels(dataset_root / "labels" / "val" / f"{image_path.stem}.txt"):
            ground_truth.append((class_id, xywhn_to_xyxy(box, image.shape[1], image.shape[0])))
        used_truth = set()
        for prediction in predictions:
            candidates = [(index, iou(prediction["xyxy"], truth_box)) for index, (truth_class, truth_box) in enumerate(ground_truth) if truth_class == prediction["class_id"] and index not in used_truth]
            if not candidates:
                continue
            matched_index, matched_iou = max(candidates, key=lambda item: item[1])
            if matched_iou >= 0.5:
                used_truth.add(matched_index)
                class_values = values_by_class[prediction["class_id"]]
                class_values["mse"].append(prediction["mse"])
                class_values["cosine"].append(prediction["cosine_distance"])
                class_values["confidence"].append(prediction["confidence"])
    populated = {class_id: values for class_id, values in values_by_class.items() if values["mse"]}
    if not populated:
        raise RuntimeError("No correct validation detections available for calibration. Check YOLO weights.")
    calibration_by_class = {
        class_id: fit_calibration(np.asarray(values["mse"]), np.asarray(values["cosine"]), np.asarray(values["confidence"]), anomaly_config)
        for class_id, values in populated.items()
    }
    combined_mse = np.concatenate([np.asarray(values["mse"]) for values in populated.values()])
    combined_cosine = np.concatenate([np.asarray(values["cosine"]) for values in populated.values()])
    combined_confidence = np.concatenate([np.asarray(values["confidence"]) for values in populated.values()])
    calibration = fit_calibration(combined_mse, combined_cosine, combined_confidence, anomaly_config)
    output = resolve_path(anomaly_config["artifacts"]["calibration"])
    output.write_text(
        json.dumps(
            {
                "calibration": calibration.to_dict(),
                "calibration_by_class": {str(class_id): calibration.to_dict() for class_id, calibration in calibration_by_class.items()},
                "normal_validation_cells": int(sum(len(values["mse"]) for values in populated.values())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate anomaly score threshold from correctly detected normal validation cells.")
    parser.add_argument("--yolo", type=Path, default=PROJECT_ROOT / "outputs" / "weights" / "yolov8n_txl_pbc_best.pt")
    parser.add_argument("--autoencoder", type=Path, default=PROJECT_ROOT / "outputs" / "weights" / "autoencoder_best.pt")
    parser.add_argument("--centroids", type=Path, default=PROJECT_ROOT / "outputs" / "weights" / "centroids.pt")
    args = parser.parse_args()
    print(calibrate(load_yaml(PROJECT_ROOT / "configs" / "anomaly.yaml"), load_yaml(PROJECT_ROOT / "configs" / "dataset.yaml"), args.yolo, args.autoencoder, args.centroids))
