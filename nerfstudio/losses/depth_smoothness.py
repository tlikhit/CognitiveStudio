"""Depth smoothness loss from RegNerf.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type, Union

import numpy as np
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.losses.base_loss import Loss, LossConfig
from nerfstudio.model_components.randposes import *
from nerfstudio.utils import plotly_utils as vis


@dataclass
class DepthSmoothnessLossConfig(LossConfig):
    """Config for depth smoothness loss.
    """

    _target: Type = field(default_factory=lambda: DepthSmoothnessLoss)

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

    radius: Union[float, Tuple[float, float]] = (2, 5)  #4.03112885717555
    """Radius OR min/max radius to sample from.
    The official RegNerf implementation uses this particular default value.
    """

    only_upper: bool = False
    """If True, only sample from the upper hemisphere (z >= 0).
    """


class DepthSmoothnessLoss(Loss):
    """Depth smoothness loss from RegNerf.

    Samples random poses from fixed radii pointing towards origin
    (uses ``sample_randposes_sphere`` from ``randposes.py``).

    Note:
    This uses the depth output of the model, which is rendered by the DepthRenderer.
    The DepthRenderer has a "method" argument.
    It must be set to "expected" to propogate gradients properly.
    """

    config: DepthSmoothnessLossConfig

    def forward_cameras(self, model, step, cameras: Cameras) -> Dict[str, Tensor]:
        """Gets model outputs for a batch of cameras.

        TODO this should be moved to ``randposes.py``.
        There is a lot of tedious code (e.g. reshaping tensors) which is handled here.

        Returns:
            Dict with same keys as ``model.get_outputs``. Each value is reshaped
            to ``(batch_size, s_patch, s_patch, x)`` where x is 3 for RGB and 1
            for depth/accumulation.
        """
        # Allocate tensors for all rays
        num_px = self.config.s_patch ** 2
        total_ray_count = self.config.batch_size * num_px
        combined_rays = RayBundle(
            origins=torch.empty(total_ray_count, 3, device=model.device),
            directions=torch.empty(total_ray_count, 3, device=model.device),
            pixel_area=torch.empty(total_ray_count, 1, device=model.device),
        )
        # Copy rays generated from each camera.
        for i in range(cameras.size):
            ray_bundle = cameras.generate_rays(i)
            ind = i * num_px
            combined_rays.origins[ind : ind+num_px] = ray_bundle.origins.view(num_px, 3)
            combined_rays.directions[ind : ind+num_px] = ray_bundle.directions.view(num_px, 3)
            combined_rays.pixel_area[ind : ind+num_px] = ray_bundle.pixel_area.view(num_px, 1)

        # Run rays through model
        combined_rays = model.collider(combined_rays, step)
        outputs = model.get_outputs(combined_rays)

        # Reshape outputs
        for k, v in outputs.items():
            outputs[k] = v.view(self.config.batch_size, self.config.s_patch, self.config.s_patch, -1)

        return outputs

    def compute_loss(self, model, step, **kwargs):
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

        outputs = self.forward_cameras(model, step, cameras)

        # Compute loss
        loss = 0
        for depth in (outputs["depth_fine"], outputs["depth_coarse"]):
            # Compute depth gradients
            loss += F.mse_loss(depth[:, :, 1:, :], depth[:, :, :-1, :])
            loss += F.mse_loss(depth[:, 1:, :, :], depth[:, :-1, :, :])
        loss /= 4

        # Start DS loss at high weight, then decrease to 1. (RegNerf paper)
        # TODO this should be in the config
        if step < 512:
            scale = np.interp(step, [0, 512], [1000, 1])
        else:
            scale = 1

        return loss * scale
