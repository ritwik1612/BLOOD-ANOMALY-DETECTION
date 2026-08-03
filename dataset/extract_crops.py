from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from utils.config import PROJECT_ROOT, class_names, load_yaml, resolve_path
from utils.image_ops import crop_and_resize, read_yolo_labels, xywhn_to_xyxy


def extract_normal_crops(config_path: Path, patch_size: int = 64) -> dict[str, int]:
    config = load_yaml(config_path)
    source = resolve_path(config["dataset_root"])
    names = class_names(config)
    output_root = PROJECT_ROOT / "dataset" / "crops"
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_count = 0
        for image_path in sorted((source / "images" / split).iterdir()):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            labels = read_yolo_labels(source / "labels" / split / f"{image_path.stem}.txt")
            for annotation_index, (class_id, box) in enumerate(labels):
                crop = crop_and_resize(image, xywhn_to_xyxy(box, width, height), patch_size)
                if crop is None:
                    continue
                class_dir = output_root / split / f"{class_id}_{names[class_id]}"
                class_dir.mkdir(parents=True, exist_ok=True)
                output_path = class_dir / f"{image_path.stem}_{annotation_index:03d}.png"
                cv2.imwrite(str(output_path), crop)
                split_count += 1
        counts[split] = split_count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract normal 64x64 cell crops from TXL-PBC ground-truth annotations.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--patch-size", type=int, default=64)
    args = parser.parse_args()
    print(extract_normal_crops(args.config, args.patch_size))


if __name__ == "__main__":
    main()
