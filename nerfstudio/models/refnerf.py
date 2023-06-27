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
# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Ref-NeRF (https://github.com/google-research/multinerf)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

'''
RefNerf Implementation
'''

from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.field_components.encodings import NeRFEncoding, IntegratedDirectionEncoding


class RefNerfModelConfig(VanilaModelConfig):
        # deg_view,
        min_deg_point: int = 0,
        max_deg_point: int = 16,
        netdepth: int = 8,
        netwidth: int = 256,
        bottleneck_width: int = 128,
        netdepth_viewdirs: int = 8,
        netwidth_viewdirs: int = 256,
        # net_activation: Callable[..., Any] = nn.ReLU(),
        skip_layer: int = 4,
        skip_layer_dir: int = 4,
        perturb: float = 1.0,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        num_roughness_channels: int = 1,
        # roughness_activation: Callable[..., Any] = nn.Softplus(),
        roughness_bias: float = -1.0,
        bottleneck_noise: float = 0.0,
        # density_activation: Callable[..., Any] = nn.Softplus(),
        density_bias: float = -1.0,
        density_noise: float = 0.0,
        rgb_premultiplier: float = 1.0,
        # rgb_activation: Callable[..., Any] = nn.Sigmoid(),
        rgb_bias: float = 0.0,
        rgb_padding: float = 0.001,
        num_normal_channels: int = 3,
        num_tint_channels: int = 3,

class RefNerfModel(MipNerfModel):

    def __init__(
        self,
        config: VanillaModelConfig,
        **kwargs,
    ) -> None:
        self.field = None
        assert config.collider_params is not None, "Parent MipNeRF model requires bounding box collider parameters."
        super().__init__(config=config, **kwargs)
        assert self.config.collider_params is not None, "Parent mip-NeRF requires collider parameters to be set."


    def reflect(self, viewdirs, normals):
        """Reflect view directions about normals.

        The reflection of a vector v about a unit vector n is a vector u such that
        dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
        equations is u = 2 dot(n, v) n - v.

        Args:
            viewdirs: [..., 3] array of view directions.
            normals: [..., 3] array of normal directions (assumed to be unit vectors).

        Returns:
            [..., 3] array of reflection directions.
        """
        return (
            2.0 * torch.sum(normals * viewdirs, dim=-1, keepdims=True) * normals - viewdirs
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
        )
        direction_encoding = IntegratedDirectionEncoding(
            in_dim=3, num_frequencies=4
        )

        self.field = NeRFField(
            position_encoding=position_encoding, direction_encoding=direction_encoding, use_integrated_encoding=True
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    