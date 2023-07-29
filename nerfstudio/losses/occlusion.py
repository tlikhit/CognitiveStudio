"""Occlusion regularization from FreeNeRF.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from torch import Tensor

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.losses.base_loss import Loss, LossConfig


@dataclass
class OcclusionLossConfig(LossConfig):
    """Config for depth smoothness loss.
    """

    _target: Type = field(default_factory=lambda: OcclusionLoss)

    output_names: Tuple[str, ...] = ("field_outputs_coarse",)
    """Keys of ``model.get_outputs`` to use.
    Default is coarse output of VanillaNerf and variants.
    TODO check if field_outputs_fine is sorted by distance.
    """

    threshold: int = 10
    """Penalize only the first ``threshold`` field outputs;
    i.e. the closest ``threshold`` points sampled.
    Corresponds to hyperparameter M in the paper.
    """


class OcclusionLoss(Loss):
    """Occlusion regularization loss from FreeNeRF.

    Penalizes high density values near camera origin; i.e. "floaters".

    A typical forward pass is:
    - Sample points along ray.
    - Pass points through field, obtaining density and RGB values for each point.
    - Pass outputs through renderer, aggregating to what the camera would see.
    This loss operates on the 2nd stage, the field outputs.

    The first M points are considered, assuming sorted by distance to ray origin.
    Loss is simply the mean density of them (and scale factor).
    """

    config: OcclusionLossConfig

    def compute_loss(self, *, outputs: Dict[str, Dict[FieldHeadNames, Tensor]], **kwargs):
        loss = 0
        for key in self.config.output_names:
            density_outputs = outputs[key][FieldHeadNames.DENSITY]  # (B, N, 1)
            loss += density_outputs[:, :self.config.threshold, :].mean() / density_outputs.shape[1]
        loss /= len(self.config.output_names)

        return loss
