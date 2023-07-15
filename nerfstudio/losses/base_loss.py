"""Base loss class and config.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from torch import Tensor

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.models.base_model import Model


@dataclass
class LossConfig(InstantiateConfig):
    """Configuration for base loss.
    """

    _target: Type = field(default_factory=lambda: Loss)


class Loss(torch.nn.Module):
    """Loss function.

    TODO This will supercede ``nerfstudio/model_components/losses.py``

    Each loss function is it's own class with config. This increases modularity;
    for example, the depth smoothness loss from RegNerf can be applied on other
    models, and can be customized easily.

    Subclasses should:
    - Override ``__init__``; always call ``super().__init__()``
    - Define compute_loss; this computes and returns the loss.

    Args:
        config: Config class instance.
    """

    config: LossConfig

    def __init__(self, config: LossConfig, **kwargs) -> None:
        """Base ``__init__`` function.
        """
        super().__init__()

        self.config = config

    def compute_loss(self, model: Model, step: int, **kwargs) -> Tensor:
        """Compute loss.

        Override this function in subclasses.

        Args:
            model: Model instance to optimize for.
            step: Training step.

        Returns:
            Loss tensor.
        """
        raise NotImplementedError

    def forward(self, model: Model, step: int, **kwargs) -> Tensor:
        """Same as ``compute_loss``.
        """
        return self.compute_loss(model, step, **kwargs)
