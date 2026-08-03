from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from dataset.prepare import prepare_dataset
from utils.config import PROJECT_ROOT, load_yaml, resolve_path
from utils.reproducibility import seed_everything


def train_yolo(config: dict, data_yaml: Path) -> Path:
    seed_everything(int(config["seed"]))
    model = YOLO(config["model"])
    result = model.train(
        data=str(data_yaml), epochs=int(config["epochs"]), batch=int(config["batch"]), imgsz=int(config["imgsz"]),
        lr0=float(config["lr0"]), optimizer=config["optimizer"], weight_decay=float(config["weight_decay"]),
        workers=int(config["workers"]), patience=int(config["patience"]), seed=int(config["seed"]),
        device=config.get("device"), project=str(resolve_path(config["project"])), name=config["name"], exist_ok=True,
        pretrained=True, plots=True,
    )
    best_source = Path(result.save_dir) / "weights" / "best.pt"
    target = PROJECT_ROOT / "outputs" / "weights" / "yolov8n_txl_pbc_best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_source, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 Nano on the local TXL-PBC dataset.")
    parser.add_argument("--prepare", action="store_true", help="Rebuild the processed, augmented YOLO dataset first.")
    args = parser.parse_args()
    data_yaml = PROJECT_ROOT / "dataset" / "processed" / "data.yaml"
    if args.prepare or not data_yaml.exists():
        data_yaml = prepare_dataset(PROJECT_ROOT / "configs" / "dataset.yaml", force=args.prepare)
    print(f"Best detector saved to: {train_yolo(load_yaml(PROJECT_ROOT / 'configs' / 'yolo.yaml'), data_yaml)}")


if __name__ == "__main__":
    main()
