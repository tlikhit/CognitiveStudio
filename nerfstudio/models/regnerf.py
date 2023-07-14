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

"""
Implementation of RegNeRF.
Author: Patrick Huang
"""

from dataclasses import dataclass, field
from typing import Dict, Type

from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.model_components.scene_colliders import AnnealedCollider
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.vanilla_nerf import VanillaModelConfig


@dataclass
class RegNerfModelConfig(VanillaModelConfig):
    """Configuration for RegNeRF.
    """

    _target: Type = field(default_factory=lambda: RegNerfModel)

    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss_coarse": 0.1,
        "rgb_loss_fine": 1.0,
        "depth_smoothness": 0.1,
    })
    collider_params: Dict[str, float] = to_immutable_dict({
        "near_plane": 2.0,
        "far_plane": 6.0,
        "duration": 2000,
        "start": 0.5,
    })
    """For AnnealedCollider"""


class RegNerfModel(MipNerfModel):
    """RegNeRF model.
    """

    collider: AnnealedCollider
    config: RegNerfModelConfig

    def populate_modules(self):
        """
        - Overrides collider to AnnealedCollider.
        """
        super().populate_modules()

        self.collider = AnnealedCollider(**self.config.collider_params)

    def get_outputs(self, ray_bundle):
        #ray_bundle.nears[...] = 0
        #ray_bundle.fars[...] = 0
        return super().get_outputs(ray_bundle)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        loss_dict["anneal_fac"] = self.collider.get_anneal_fac(self.step)
        return loss_dict
