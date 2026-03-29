import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DigitDataset(Dataset):
    """
    Generic image dataset for digit classification.

    Expected structure (recommended):
        data_root_dir/
            train/
                0/*.png
                1/*.png
                ...
            val/
                0/*.png
                1/*.png
                ...
            test/
                0/*.png
                1/*.png
                ...

    If split folders are not present, class folders can live directly under data_root_dir,
    and the DataModule can create train/val/test splits by index.
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    CSV_EXTENSIONS = {".csv"}

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = "train",
        transform=None,
        indices: Optional[Sequence[int]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        image_mode: str = "L",
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_mode = image_mode
        self.data_format: str = "csv"
        self.csv_path: Optional[Path] = None

        self.samples: List[Tuple[Path, int]] = []
        self.csv_images: Optional[np.ndarray] = None
        self.csv_labels: Optional[np.ndarray] = None
        self._csv_side: Optional[int] = None

        self.data_dir = self._resolve_data_source(self.root_dir, split)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.data_dir}")

        if self.data_format == "csv":
            raw_labels, raw_pixels = self._load_csv_rows(self.csv_path)
            self.class_to_idx = class_to_idx or self._discover_csv_classes(raw_labels)

            encoded_labels = np.array(
                [self.class_to_idx[str(label)] for label in raw_labels], dtype=np.int64
            )
            pixels = np.array(raw_pixels, dtype=np.uint8)

            if indices is not None:
                pixels = pixels[list(indices)]
                encoded_labels = encoded_labels[list(indices)]

            if len(pixels) == 0:
                raise ValueError(f"No rows found in CSV dataset: {self.csv_path}")

            n_pixels = pixels.shape[1]
            side = int(math.sqrt(n_pixels))
            if side * side != n_pixels:
                raise ValueError(
                    f"CSV rows must contain a square number of pixel columns, got {n_pixels}."
                )

            self._csv_side = side
            self.csv_images = pixels
            self.csv_labels = encoded_labels
            return

        self.class_to_idx = class_to_idx or self._discover_classes(self.data_dir)
        self.samples = self._build_samples(self.data_dir, self.class_to_idx)

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.data_dir}")

    def _resolve_data_source(self, root_dir: Path, split: Optional[str]) -> Path:
        if root_dir.is_file() and root_dir.suffix.lower() in self.CSV_EXTENSIONS:
            self.data_format = "csv"
            self.csv_path = root_dir
            return root_dir

        if split is not None:
            split_dir = root_dir / split
            split_csv = root_dir / f"{split}.csv"

            if split_dir.exists() and split_dir.is_dir():
                self.data_format = "image_folder"
                return split_dir

            if split_csv.exists() and split_csv.is_file():
                self.data_format = "csv"
                self.csv_path = split_csv
                return split_csv

            if split_dir.exists() and split_dir.is_file() and split_dir.suffix.lower() in self.CSV_EXTENSIONS:
                self.data_format = "csv"
                self.csv_path = split_dir
                return split_dir

            raise FileNotFoundError(
                f"Could not find split '{split}' as directory or CSV under {root_dir}."
            )

        csv_files = sorted(root_dir.glob("*.csv"))
        if len(csv_files) == 1:
            self.data_format = "csv"
            self.csv_path = csv_files[0]
            return csv_files[0]

        self.data_format = "image_folder"
        return root_dir

    @staticmethod
    def _is_numeric_row(row: Sequence[str]) -> bool:
        try:
            _ = float(row[0])
            for value in row[1:]:
                _ = float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _load_csv_rows(self, csv_path: Path) -> Tuple[List[int], List[List[int]]]:
        labels: List[int] = []
        pixels: List[List[int]] = []

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first_data_row_seen = False

            for row in reader:
                if not row:
                    continue

                # Skip header if present
                if not first_data_row_seen and not self._is_numeric_row(row):
                    first_data_row_seen = True
                    continue

                first_data_row_seen = True

                if len(row) < 2:
                    continue

                label = int(float(row[0]))
                pixel_row = [int(float(v)) for v in row[1:]]
                pixel_row = [min(255, max(0, p)) for p in pixel_row]

                labels.append(label)
                pixels.append(pixel_row)

        if not labels:
            raise ValueError(f"No valid data rows found in CSV file: {csv_path}")

        return labels, pixels

    @staticmethod
    def _discover_csv_classes(labels: Sequence[int]) -> Dict[str, int]:
        unique_labels = sorted(set(int(lbl) for lbl in labels))
        return {str(label): idx for idx, label in enumerate(unique_labels)}

    @staticmethod
    def _discover_classes(data_dir: Path) -> Dict[str, int]:
        class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        if not class_dirs:
            raise ValueError(
                f"No class directories found in {data_dir}. Expected subfolders per class label."
            )

        return {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}

    def _build_samples(
        self, data_dir: Path, class_to_idx: Dict[str, int]
    ) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []

        for class_name, class_idx in class_to_idx.items():
            class_path = data_dir / class_name
            if not class_path.exists() or not class_path.is_dir():
                continue

            for image_path in class_path.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in self.IMG_EXTENSIONS:
                    samples.append((image_path, class_idx))

        return samples

    def __len__(self) -> int:
        if self.data_format == "csv":
            return int(len(self.csv_labels))
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.data_format == "csv":
            flat_pixels = self.csv_images[idx]
            label = int(self.csv_labels[idx])

            image_array = flat_pixels.reshape(self._csv_side, self._csv_side).astype(np.uint8)
            image = Image.fromarray(image_array, mode="L")

            if self.image_mode != "L":
                image = image.convert(self.image_mode)

            if self.transform is not None:
                image = self.transform(image)

            return image, label

        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert(self.image_mode)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
