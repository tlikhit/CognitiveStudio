"""Implementation of FreeNeRF.
"""

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.field_components.encodings import FreeNerfEncoding
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.models.regnerf import RegNerfModel, RegNerfModelConfig


@dataclass
class FreeNerfModelConfig(RegNerfModelConfig):
    """Config for FreeNeRF.
    """

    _target: Type = field(default_factory=lambda: FreeNerfModel)

    freq_reg_duration: int = 10000
    """Number of steps frequency regularization lasts for.
    Max posenc freq is slowly increased while training from start to this step.
    Official FreeNeRF uses a fraction of total iters; here it is hardcoded.
    """


class FreeNerfModel(RegNerfModel):
    """FreeNeRF model.
    """

    config: FreeNerfModelConfig

    def populate_modules(self):
        """
        Uses FreeNerfEncoding
        """
        super().populate_modules()

        self.position_encoding = FreeNerfEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True,
            reg_duration=self.config.freq_reg_duration,
        )
        self.direction_encoding = FreeNerfEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True,
            reg_duration=self.config.freq_reg_duration,
        )
        self.field = NeRFField(
            position_encoding=self.position_encoding,
            direction_encoding=self.direction_encoding,
            use_integrated_encoding=True,
        )

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        loss_dict["freq_reg_index"] = self.position_encoding.get_freq_reg_index(self.step)
        return loss_dict

    def get_outputs(self, ray_bundle):
        """Sets ``step`` for encodings.
        See workaround note in ``FreeNerfEncoding``.
        """
        self.position_encoding.step = self.step
        self.direction_encoding.step = self.step
        return super().get_outputs(ray_bundle)
