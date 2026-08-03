from __future__ import annotations

import albumentations as A


def detection_augmentation(crop_height: int, crop_width: int, min_visibility: float) -> A.Compose:
    """BBox-safe transformations required by the research protocol."""
    return A.Compose(
        [
            A.Rotate(limit=25, border_mode=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(p=0.25),
            A.RandomSizedBBoxSafeCrop(height=crop_height, width=crop_width, p=0.5),
            A.Resize(height=640, width=640),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=min_visibility),
    )
