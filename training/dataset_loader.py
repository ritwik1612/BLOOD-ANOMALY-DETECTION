from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CellCropDataset(Dataset):
    """Normal single-cell crops grouped by the original TXL-PBC class ID."""

    def __init__(self, crops_root: str | Path, split: str, augment: bool = False) -> None:
        self.root = Path(crops_root) / split
        self.samples: list[tuple[Path, int]] = []
        for class_dir in sorted(self.root.iterdir() if self.root.exists() else [], key=lambda path: path.name):
            if not class_dir.is_dir():
                continue
            class_id = int(class_dir.name.split("_", maxsplit=1)[0])
            self.samples.extend((path, class_id) for path in sorted(class_dir.glob("*.png")))
        operations = []
        if augment:
            operations += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20)]
        operations += [transforms.Resize((64, 64)), transforms.ToTensor()]
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, class_id = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), class_id, str(path)
