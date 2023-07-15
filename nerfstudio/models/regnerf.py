# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of RegNeRF.
"""

from dataclasses import dataclass, field
from typing import Dict, Type

from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.losses.depth_smoothness import DepthSmoothnessLossConfig, DepthSmoothnessLoss
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.model_components.scene_colliders import AnnealedCollider
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.vanilla_nerf import VanillaModelConfig


@dataclass
class RegNerfModelConfig(VanillaModelConfig):
    """Config for RegNeRF.
    """

    _target: Type = field(default_factory=lambda: RegNerfModel)

    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss_coarse": 0.1,
        "rgb_loss_fine": 1.0,
        "depth_smoothness": 0.1,
    })

    collider_params: Dict[str, float] = to_immutable_dict({
        "near_plane": 1,
        "far_plane": 4,
        "anneal_duration": 256,
    })
    """For AnnealedCollider"""

    ds_loss: DepthSmoothnessLossConfig = DepthSmoothnessLossConfig()
    """Depth smoothness loss config."""


class RegNerfModel(MipNerfModel):
    """RegNeRF model.
    """

    collider: AnnealedCollider
    config: RegNerfModelConfig
    ds_loss: DepthSmoothnessLoss

    def populate_modules(self):
        """
        - Sets collider to AnnealedCollider (overrides super).
        - Sets depth renderer method to "expected". This propogates gradient.
        - Setup depth smoothness loss.
        """
        super().populate_modules()

        self.collider = AnnealedCollider(**self.config.collider_params)
        self.ds_loss = self.config.ds_loss.setup()
        self.renderer_depth = DepthRenderer(method="expected")

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        loss_dict["anneal_fac"] = self.collider.get_anneal_fac(self.step)
        loss_dict["depth_smoothness"] = self.ds_loss(self, self.step)
        return loss_dict
