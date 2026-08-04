from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as functional

from anomaly_detection.scoring import AnomalyScorer, Calibration, cosine_distance
from feature_extraction.extract_latent import load_autoencoder
from models.yolo import load_detector
from utils.config import PROJECT_ROOT, load_yaml
from utils.image_ops import crop_and_resize


class OpenSetPipeline:
    """YOLOv8 detection followed by normal-morphology reconstruction scoring."""

    def __init__(self, yolo_weights: Path, autoencoder_weights: Path, centroids_path: Path, anomaly_config: dict, calibration_path: Path | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = load_detector(yolo_weights)
        self.autoencoder = load_autoencoder(autoencoder_weights, self.device)
        saved = torch.load(centroids_path, map_location="cpu", weights_only=False)
        self.centroids = {int(class_id): value.numpy() for class_id, value in saved["centroids"].items()}
        self.class_labels = {int(key): value for key, value in anomaly_config["class_labels"].items()}
        self.threshold = None
        self.scorer = None
        self.scorers: dict[int, AnomalyScorer] = {}
        self.thresholds: dict[int, float] = {}
        self.minimum_abnormal_confidence = float(anomaly_config["threshold"].get("minimum_detection_confidence_for_abnormal", 0.0))
        self.use_class_specific_calibration = bool(anomaly_config["threshold"].get("use_class_specific_calibration", False))
        if calibration_path is not None:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration = Calibration.from_dict(payload["calibration"])
            self.scorer = AnomalyScorer(anomaly_config["weights"], calibration, anomaly_config["normalization"]["epsilon"])
            class_calibrations = payload.get("calibration_by_class", {})
            for class_id, values in class_calibrations.items():
                class_calibration = Calibration.from_dict(values)
                numeric_class_id = int(class_id)
                self.scorers[numeric_class_id] = AnomalyScorer(anomaly_config["weights"], class_calibration, anomaly_config["normalization"]["epsilon"])
                self.thresholds[numeric_class_id] = class_calibration.threshold
            self.threshold = max(self.thresholds.values(), default=calibration.threshold) if self.use_class_specific_calibration else calibration.threshold

    @staticmethod
    def _tag(class_id: int, abnormal: bool) -> str:
        normal_tags = {0: "WBC", 1: "RBC", 2: "PLAT"}
        abnormal_tags = {0: "AWBC", 1: "ARBC", 2: "APLAT"}
        return (abnormal_tags if abnormal else normal_tags)[class_id]

    @staticmethod
    def _contains_cell_foreground(patch: np.ndarray) -> bool:
        """Reject mostly blank, low-stain boxes before anomaly scoring."""
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        center = hsv[8:56, 8:56]
        stained_pixels = (center[:, :, 1] >= 18) & (center[:, :, 2] <= 248)
        return float(stained_pixels.mean()) >= 0.25

    @torch.inference_mode()
    def analyze_array(self, image: np.ndarray, confidence: float = 0.15, iou: float = 0.45) -> list[dict]:
        result = self.detector.predict(
            source=image,
            conf=confidence,
            iou=iou,
            imgsz=640,
            verbose=False,
        )[0]
        detections: list[dict] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in self.centroids:
                continue
            xyxy = tuple(int(round(value)) for value in box.xyxy[0].tolist())
            patch = crop_and_resize(image, xyxy)
            if patch is None or not self._contains_cell_foreground(patch):
                continue
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb_patch).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
            reconstruction, latent = self.autoencoder(tensor)
            mse = float(functional.mse_loss(reconstruction, tensor).item())
            latent_vector = latent.squeeze(0).cpu().numpy()
            cosine = cosine_distance(latent_vector, self.centroids[class_id])
            detection = {
                "class_id": class_id,
                "class_label": self.class_labels[class_id],
                "xyxy": xyxy,
                "confidence": float(box.conf.item()),
                "mse": mse,
                "cosine_distance": cosine,
                "patch": patch,
                "reconstruction": reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy(),
            }
            if self.scorer is not None and self.threshold is not None:
                scorer = self.scorers.get(class_id, self.scorer) if self.use_class_specific_calibration else self.scorer
                threshold = self.thresholds.get(class_id, self.threshold) if self.use_class_specific_calibration else self.threshold
                detection.update(scorer.score(mse, cosine, detection["confidence"]))
                is_abnormal = detection["confidence"] >= self.minimum_abnormal_confidence and detection["anomaly_score"] > threshold
                detection["status"] = "Abnormal" if is_abnormal else "Normal"
                detection["display_label"] = self._tag(class_id, detection["status"] == "Abnormal")
            detections.append(detection)
        return detections

    def analyze_path(self, path: Path, confidence: float = 0.15, iou: float = 0.45) -> tuple[np.ndarray, list[dict]]:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return image, self.analyze_array(image, confidence, iou)


def default_pipeline(calibrated: bool = True) -> OpenSetPipeline:
    config = load_yaml(PROJECT_ROOT / "configs" / "anomaly.yaml")
    weights = PROJECT_ROOT / "outputs" / "weights"
    return OpenSetPipeline(
        weights / "yolov8n_txl_pbc_best.pt", weights / "autoencoder_best.pt", weights / "centroids.pt", config,
        weights / "anomaly_calibration.json" if calibrated else None,
    )
