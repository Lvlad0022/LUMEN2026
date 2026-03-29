"""Hydra entry-point for training."""
import logging
import re
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from src.model.callbacks import utils


def _next_daily_version(date_str: str) -> int:
    """Return next `vN` index for a given date folder inside `outputs/`.

    Example date folder: outputs/2026-03-16/v1, v2, ...
    """
    base_dir = Path("outputs") / date_str
    if not base_dir.exists():
        return 1

    version_pattern = re.compile(r"^v(\d+)$")
    max_version = 0
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = version_pattern.match(child.name)
        if match:
            max_version = max(max_version, int(match.group(1)))

    return max_version + 1


OmegaConf.register_new_resolver("next_daily_version", _next_daily_version)

def train_model(cfg: DictConfig) -> None:
    """
    Training function
    Args:
        cfg: Full Hydra config, with subgroups for architecture, optimizer, scheduler, and loss.
    """
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    logger = utils.instantiate_loggers(cfg.logger)

    callbacks: List[pl.Callback] = utils.instantiate_callbacks(cfg.callbacks)

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer, logger =logger, callbacks=callbacks
    )

    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=cfg.get("checkpoint", None)
    )

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    print("Training with config:")
    print(OmegaConf.to_yaml(cfg))
    train_model(cfg)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()