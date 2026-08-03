from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from inference.pipeline import default_pipeline
from utils.config import PROJECT_ROOT, load_yaml
from utils.visualization import draw_open_set_detections, save_reconstruction_grid


def find_images(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return [input_path] if input_path.is_file() else sorted(path for path in input_path.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detection and open-set abnormal cell identification.")
    parser.add_argument("--input", type=Path, required=True, help="One smear image or a directory of images.")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    args = parser.parse_args()
    yolo_config = load_yaml(PROJECT_ROOT / "configs" / "yolo.yaml")
    confidence = args.confidence if args.confidence is not None else yolo_config["confidence"]
    iou = args.iou if args.iou is not None else yolo_config["iou"]
    pipeline = default_pipeline()
    prediction_dir = PROJECT_ROOT / "outputs" / "predictions"
    reconstruction_dir = PROJECT_ROOT / "outputs" / "reconstructions"
    records = []
    for image_path in find_images(args.input):
        image, detections = pipeline.analyze_path(image_path, confidence, iou)
        cv2.imwrite(str(prediction_dir / f"{image_path.stem}_open_set.png"), draw_open_set_detections(image, detections))
        save_reconstruction_grid(detections, reconstruction_dir / f"{image_path.stem}_reconstructions.png")
        for detection in detections:
            records.append({key: value for key, value in detection.items() if key not in {"patch", "reconstruction"}} | {"image_path": str(image_path), "threshold": pipeline.threshold})
    csv_path = prediction_dir / "open_set_predictions.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"Saved {len(records)} detections to {csv_path}")


if __name__ == "__main__":
    main()
