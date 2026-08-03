from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import yaml

from dataset.augment import detection_augmentation
from utils.config import PROJECT_ROOT, class_names, load_yaml, resolve_path
from utils.image_ops import read_yolo_labels, sanitize_yolo_box
from utils.reproducibility import seed_everything


def write_labels(path: Path, boxes: list[list[float]], labels: list[int]) -> None:
    lines = [f"{label} " + " ".join(f"{value:.10f}" for value in box) for box, label in zip(boxes, labels)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_source_split(source: Path, destination: Path, split: str) -> None:
    image_target = destination / "images" / split
    label_target = destination / "labels" / split
    image_target.mkdir(parents=True, exist_ok=True)
    label_target.mkdir(parents=True, exist_ok=True)
    for source_file in (source / "images" / split).iterdir():
        shutil.copy2(source_file, image_target / source_file.name)
    for source_file in (source / "labels" / split).iterdir():
        annotations = read_yolo_labels(source_file)
        write_labels(label_target / source_file.name, [sanitize_yolo_box(box) for _, box in annotations], [class_id for class_id, _ in annotations])


def augment_training_images(source: Path, destination: Path, config: dict) -> int:
    settings = config["augmentation"]
    transform = detection_augmentation(settings["crop_height"], settings["crop_width"], settings["min_visibility"])
    image_dir = source / "images" / "train"
    label_dir = source / "labels" / "train"
    output_images = destination / "images" / "train"
    output_labels = destination / "labels" / "train"
    count = 0
    for image_path in sorted(image_dir.iterdir()):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        annotations = read_yolo_labels(label_dir / f"{image_path.stem}.txt")
        boxes = [sanitize_yolo_box(box) for _, box in annotations]
        labels = [label for label, _ in annotations]
        for index in range(settings["copies_per_image"]):
            result = transform(image=image, bboxes=boxes, class_labels=labels)
            if not result["bboxes"]:
                continue
            name = f"{image_path.stem}_aug{index}.jpg"
            cv2.imwrite(str(output_images / name), result["image"])
            write_labels(output_labels / f"{Path(name).stem}.txt", list(result["bboxes"]), list(result["class_labels"]))
            count += 1
    return count


def prepare_dataset(config_path: Path, force: bool = False) -> Path:
    config = load_yaml(config_path)
    seed_everything(config["augmentation"]["random_seed"])
    source = resolve_path(config["dataset_root"])
    if not source.exists():
        raise FileNotFoundError(f"TXL-PBC data not found: {source}")
    destination = PROJECT_ROOT / "dataset" / "processed"
    if force and destination.exists():
        shutil.rmtree(destination)
    for split in config["splits"]:
        copy_source_split(source, destination, split)
    if config["augmentation"]["enabled"]:
        augment_training_images(source, destination, config)
    data_yaml = {
        "path": str(destination.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(config["classes"]),
        "names": class_names(config),
    }
    yaml_path = destination / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TXL-PBC YOLO dataset with bbox-safe augmentation.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--force", action="store_true", help="Replace the generated processed dataset.")
    args = parser.parse_args()
    print(f"Dataset ready: {prepare_dataset(args.config, force=args.force)}")


if __name__ == "__main__":
    main()
