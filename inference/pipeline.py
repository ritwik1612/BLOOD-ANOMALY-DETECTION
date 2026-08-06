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

    def __init__(
        self,
        yolo_weights: Path,
        autoencoder_weights: Path,
        centroids_path: Path,
        anomaly_config: dict,
        calibration_path: Path | None = None,
        detector_config: dict | None = None,
    ) -> None:
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
        detector_config = detector_config or {}
        inference_config = detector_config.get("inference", {})
        self.inference_confidence = float(detector_config.get("confidence", 0.15))
        self.inference_iou = float(detector_config.get("iou", 0.65))
        self.tile_enabled = bool(inference_config.get("enabled", True))
        self.tile_size = int(inference_config.get("tile_size", 640))
        self.tile_overlap = float(inference_config.get("tile_overlap", 0.25))
        self.tile_min_dimension = int(inference_config.get("tile_min_dimension", 720))
        self.max_det = int(inference_config.get("max_det", 1000))
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

    @staticmethod
    def _is_rbc_cluster_candidate(candidate: dict, detections: list[dict]) -> bool:
        """Reject a cell box when it demonstrably encloses several detected RBCs.

        This prevents a group of overlapping red cells from being presented as a
        single large abnormal cell without changing individual RBC detections.
        """
        if candidate["class_id"] not in {0, 1}:
            return False
        red_cells = [item for item in detections if item["class_id"] == 1]
        if len(red_cells) < 3:
            return False

        left, top, right, bottom = candidate["xyxy"]
        candidate_area = max(1, (right - left) * (bottom - top))
        red_cell_areas = [
            max(1, (item["xyxy"][2] - item["xyxy"][0]) * (item["xyxy"][3] - item["xyxy"][1]))
            for item in red_cells
        ]
        median_red_area = float(np.median(red_cell_areas))
        area_multiplier = 3.0 if candidate["class_id"] == 1 else 12.0
        minimum_enclosed = 3 if candidate["class_id"] == 1 else 10
        if candidate_area < area_multiplier * median_red_area:
            return False

        enclosed_red_cells = 0
        for red_cell in red_cells:
            red_left, red_top, red_right, red_bottom = red_cell["xyxy"]
            center_x = (red_left + red_right) / 2
            center_y = (red_top + red_bottom) / 2
            red_area = max(1, (red_right - red_left) * (red_bottom - red_top))
            if left < center_x < right and top < center_y < bottom and red_area < 0.6 * candidate_area:
                enclosed_red_cells += 1
        return enclosed_red_cells >= minimum_enclosed

    @staticmethod
    def _iou(left: dict, right: dict) -> float:
        left_x1, left_y1, left_x2, left_y2 = left["xyxy"]
        right_x1, right_y1, right_x2, right_y2 = right["xyxy"]
        intersection_width = max(0, min(left_x2, right_x2) - max(left_x1, right_x1))
        intersection_height = max(0, min(left_y2, right_y2) - max(left_y1, right_y1))
        intersection = intersection_width * intersection_height
        left_area = max(1, (left_x2 - left_x1) * (left_y2 - left_y1))
        right_area = max(1, (right_x2 - right_x1) * (right_y2 - right_y1))
        return intersection / max(1, left_area + right_area - intersection)

    @classmethod
    def _deduplicate_detections(cls, detections: list[dict]) -> list[dict]:
        """Remove duplicate boxes produced where overlapping tiles meet."""
        kept: list[dict] = []
        for candidate in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            duplicate = False
            for existing in kept:
                if candidate["class_id"] != existing["class_id"]:
                    continue
                if candidate.get("source_id") == existing.get("source_id"):
                    continue
                if cls._iou(candidate, existing) >= 0.70:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    @staticmethod
    def _tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = max(1, int(tile_size * (1.0 - overlap)))
        starts = list(range(0, length - tile_size + 1, stride))
        final_start = length - tile_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def _predict_detections(self, image: np.ndarray, confidence: float, iou: float) -> list[dict]:
        """Run a full-image pass plus overlapping tiles for crowded smears."""
        height, width = image.shape[:2]
        use_tiles = self.tile_enabled and max(height, width) >= self.tile_min_dimension
        sources: list[tuple[np.ndarray, int, int]] = [(image, 0, 0)]
        if use_tiles:
            for top in self._tile_starts(height, self.tile_size, self.tile_overlap):
                for left in self._tile_starts(width, self.tile_size, self.tile_overlap):
                    if left == 0 and top == 0 and width <= self.tile_size and height <= self.tile_size:
                        continue
                    sources.append((image[top:min(top + self.tile_size, height), left:min(left + self.tile_size, width)], left, top))

        detections: list[dict] = []
        for source_id, (source, offset_x, offset_y) in enumerate(sources):
            result = self.detector.predict(
                source=source,
                conf=confidence,
                iou=iou,
                imgsz=self.tile_size if use_tiles else 640,
                max_det=self.max_det,
                verbose=False,
            )[0]
            for box in result.boxes:
                left, top, right, bottom = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class_id": int(box.cls.item()),
                        "confidence": float(box.conf.item()),
                        "source_id": source_id,
                        "xyxy": (
                            int(round(left + offset_x)),
                            int(round(top + offset_y)),
                            int(round(right + offset_x)),
                            int(round(bottom + offset_y)),
                        ),
                    }
                )
        return self._deduplicate_detections(detections)

    @torch.inference_mode()
    def analyze_array(self, image: np.ndarray, confidence: float | None = None, iou: float | None = None) -> list[dict]:
        confidence = self.inference_confidence if confidence is None else confidence
        iou = self.inference_iou if iou is None else iou
        raw_detections = self._predict_detections(image, confidence, iou)
        detections: list[dict] = []
        for raw_detection in raw_detections:
            class_id = raw_detection["class_id"]
            if class_id not in self.centroids:
                continue
            if self._is_rbc_cluster_candidate(raw_detection, raw_detections):
                continue
            xyxy = raw_detection["xyxy"]
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
                "confidence": raw_detection["confidence"],
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

    def analyze_path(self, path: Path, confidence: float | None = None, iou: float | None = None) -> tuple[np.ndarray, list[dict]]:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return image, self.analyze_array(image, confidence, iou)


def default_pipeline(calibrated: bool = True) -> OpenSetPipeline:
    config = load_yaml(PROJECT_ROOT / "configs" / "anomaly.yaml")
    detector_config = load_yaml(PROJECT_ROOT / "configs" / "yolo.yaml")
    weights = PROJECT_ROOT / "outputs" / "weights"
    return OpenSetPipeline(
        weights / "yolov8n_txl_pbc_best.pt", weights / "autoencoder_best.pt", weights / "centroids.pt", config,
        weights / "anomaly_calibration.json" if calibrated else None,
        detector_config,
    )
