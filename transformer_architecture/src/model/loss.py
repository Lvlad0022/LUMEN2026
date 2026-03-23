"""Factory: instantiate a loss function from Hydra config."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss


def build_loss(cfg: DictConfig) -> _Loss:
    """Create a loss module via ``hydra.utils.instantiate``.

    The config must contain a ``_target_`` (e.g. ``torch.nn.CrossEntropyLoss``).
    """
    loss_fn: _Loss = hydra.utils.instantiate(cfg)
    return loss_fn
