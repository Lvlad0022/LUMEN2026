"""LightningModule wiring architecture, optimizer, scheduler, and loss."""

from functools import partial
import logging


import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from .architecture.architecture import CNN

from .loss import build_loss


class CNNModule(pl.LightningModule):
    """Lightning wrapper that builds everything from a Hydra config subtree."""

    def __init__(
        self,
        architecture,
        optimizer,
        scheduler,
        scheduler_frequency: int , 
        loss: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        

        # Store sub-configs for later use (optimizer/scheduler created in configure_optimizers)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.scheduler_frequency = scheduler_frequency
        self.loss_fn = build_loss(loss) if isinstance(loss, DictConfig) else loss

        # Build architecture and loss eagerly
        self.model: CNN = architecture # ovo je CNN soecific

        self.log_metric = partial(
            self.log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs, targets = batch
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)
        self.log_metric("loss_train", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:

        inputs, targets = batch
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)
        self.log("loss_val", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self) :
        params = self.model.parameters()
        optimizer = self.optimizer(params=params)
        scheduler = self.scheduler(optimizer=optimizer)

        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": self.scheduler_frequency
        }
        logging.info(f"Initialized with: {optimizer}, {scheduler}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

