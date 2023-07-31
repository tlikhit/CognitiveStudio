"""Depth smoothness loss from RegNerf.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type, Union

import numpy as np
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.losses.base_loss import Loss, LossConfig
from nerfstudio.model_components.randposes import (
    sample_randposes_sphere,
    poses_to_cameras,
    forward_cameras,
)
from nerfstudio.utils import plotly_utils as vis


@dataclass
class DepthSmoothnessLossConfig(LossConfig):
    """Config for depth smoothness loss.
    """

    _target: Type = field(default_factory=lambda: DepthSmoothnessLoss)

    output_names: Tuple[str, ...] = ("depth_fine", "depth_coarse")
    """Keys of ``model.get_outputs`` to use for depth smoothness loss.
    Default is fine/coarse depth (VanillaNerf and variants).
    """

    batch_size: int = 8
    """Number of random poses to process each step.
    """

    s_patch: int = 8
    """Pixel resolution of patch from each random pose.
    See RegNerf paper for details.
    """

    focal: float = 100
    """Focal length of randpose cameras.
    The official RegNerf implementation uses the same focal length as the
    training images.
    Here, it is hardcoded.
    """

    radius: Union[float, Tuple[float, float]] = (2, 5)
    """Radius OR min/max radius to sample from.
    """

    only_upper: bool = False
    """If True, only sample from the upper hemisphere (z >= 0).
    """

    start_weight: float = 1000
    """Weight starts high and linearly decreases to 1.
    This is the beginning weight.
    Multiplicative weight, i.e. loss *= weight
    """

    start_weight_steps: int = 512
    """Number of steps to linearly decrease the weight.
    See ``start_weight``.
    """


class DepthSmoothnessLoss(Loss):
    """Depth smoothness loss from RegNerf.

    Samples random poses from fixed radii pointing towards origin
    (uses ``sample_randposes_sphere`` from ``randposes.py``).

    Note:
    This uses the depth output of the model, which is rendered by the DepthRenderer.
    The DepthRenderer has a "method" argument.
    It must be set to "expected" to propogate gradients properly.

    TODO allow other sampling methods.
    """

    config: DepthSmoothnessLossConfig

    def compute_loss(self, *, model, step, **kwargs):
        assert model.collider is not None, "Collider must be set."

        randposes = sample_randposes_sphere(
            n=self.config.batch_size,
            radius=self.config.radius,
            only_upper=self.config.only_upper,
        )
        cameras = poses_to_cameras(
            poses=randposes,
            resolution=self.config.s_patch,
            focal=self.config.focal,
        ).to(model.device)

        outputs = forward_cameras(model, step, cameras, self.config.s_patch)

        # Compute loss
        loss = 0
        for key in self.config.output_names:
            depth = outputs[key]
            # Compute depth gradients
            loss += F.mse_loss(depth[:, :, 1:, :], depth[:, :, :-1, :])
            loss += F.mse_loss(depth[:, 1:, :, :], depth[:, :-1, :, :])
        loss /= 2 * len(self.config.output_names)

        # Start DS loss at high weight, then decrease to 1. (RegNerf supplementary paper)
        if step < 512:
            scale = np.interp(step, [0, self.config.start_weight_steps], [self.config.start_weight, 1])
        else:
            scale = 1

        return loss * scale
