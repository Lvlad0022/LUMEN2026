import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import DigitDataset


class DigitDatasetModule(pl.LightningDataModule):
    """LightningDataModule for loading digit images from ``data_root_dir``."""

    def __init__(
        self,
        data_root_dir: str,
        batch_size: int = 32,
        num_workers: int = 1,
        image_size: int = 28,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
        augment_train: bool = True,
        image_mode: str = "L",
    ):
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.image_mode = image_mode

        self.train_dataset: Optional[DigitDataset] = None
        self.valid_dataset: Optional[DigitDataset] = None
        self.test_dataset: Optional[DigitDataset] = None

    def _build_transforms(self):
        train_steps = [transforms.Resize((self.image_size, self.image_size))]
        if self.augment_train:
            train_steps.extend(
                [
                    transforms.RandomAffine(
                        degrees=15,
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                    )
                ]
            )
        train_steps.append(transforms.ToTensor())

        eval_steps = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        return transforms.Compose(train_steps), transforms.Compose(eval_steps)

    def setup(self, stage: str = None):
        if self.train_dataset is not None and self.valid_dataset is not None and self.test_dataset is not None:
            return

        if not self.data_root_dir.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {self.data_root_dir}")

        logging.info("Setting up digit datasets from %s", self.data_root_dir)
        train_tf, eval_tf = self._build_transforms()

        train_dir = self.data_root_dir / "train"
        val_dir = self.data_root_dir / "val"
        test_dir = self.data_root_dir / "test"
        train_csv = self.data_root_dir / "train.csv"
        val_csv = self.data_root_dir / "val.csv"
        test_csv = self.data_root_dir / "test.csv"

        has_train_split = train_dir.exists() or train_csv.exists()
        has_val_split = val_dir.exists() or val_csv.exists()
        has_test_split = test_dir.exists() or test_csv.exists()

        if any([has_train_split, has_val_split, has_test_split]) and not all(
            [has_train_split, has_val_split, has_test_split]
        ):
            raise ValueError(
                "Partial split detected. Please provide all train/val/test as folders or CSV files."
            )

        if has_train_split and has_val_split and has_test_split:
            self.train_dataset = DigitDataset(
                root_dir=str(self.data_root_dir),
                split="train",
                transform=train_tf,
                image_mode=self.image_mode,
            )
            self.valid_dataset = DigitDataset(
                root_dir=str(self.data_root_dir),
                split="val",
                transform=eval_tf,
                class_to_idx=self.train_dataset.class_to_idx,
                image_mode=self.image_mode,
            )
            self.test_dataset = DigitDataset(
                root_dir=str(self.data_root_dir),
                split="test",
                transform=eval_tf,
                class_to_idx=self.train_dataset.class_to_idx,
                image_mode=self.image_mode,
            )
            return

        full_dataset = DigitDataset(
            root_dir=str(self.data_root_dir),
            split=None,
            transform=None,
            image_mode=self.image_mode,
        )

        n_total = len(full_dataset)
        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_val - n_test

        if n_train <= 0:
            raise ValueError(
                "Invalid split sizes. Increase dataset size or reduce val_split/test_split."
            )

        generator = torch.Generator().manual_seed(self.seed)
        permutation = torch.randperm(n_total, generator=generator).tolist()
        train_indices = permutation[:n_train]
        val_indices = permutation[n_train : n_train + n_val]
        test_indices = permutation[n_train + n_val :]

        self.train_dataset = DigitDataset(
            root_dir=str(self.data_root_dir),
            split=None,
            transform=train_tf,
            indices=train_indices,
            class_to_idx=full_dataset.class_to_idx,
            image_mode=self.image_mode,
        )
        self.valid_dataset = DigitDataset(
            root_dir=str(self.data_root_dir),
            split=None,
            transform=eval_tf,
            indices=val_indices,
            class_to_idx=full_dataset.class_to_idx,
            image_mode=self.image_mode,
        )
        self.test_dataset = DigitDataset(
            root_dir=str(self.data_root_dir),
            split=None,
            transform=eval_tf,
            indices=test_indices,
            class_to_idx=full_dataset.class_to_idx,
            image_mode=self.image_mode,
        )

    def train_dataloader(self):
        self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        self.setup()
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
