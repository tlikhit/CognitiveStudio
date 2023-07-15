"""Utilities for random camera pose sampling.
Random poses are used in many regularization methods (e.g. RegNerf, DietNerf).
"""

from typing import Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.model_components.ray_generators import RayGenerator


def sample_randposes_sphere(
        n: int,
        radius: Union[float, Tuple[float, float]],
        only_upper: bool = True,
        ) -> Float[Tensor, "batch 3 4"]:
    """Sample random camera poses at some radius looking at the origin.

    Args:
        n: Number of poses to sample.
        radius: Radius OR min/max radius to sample from.
        only_upper: If True, only sample from the upper hemisphere (z >= 0).

    Returns:
        SE3 camera to world matrices.
    """
    def normalize(v):
        """Normalize on last dim with epsilon."""
        return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)

    if isinstance(radius, (int, float)):
        radius = (radius, radius)

    # Get direction unit vectors.
    dirs = normalize(torch.randn(n, 3))
    if only_upper:
        dirs[:, 2] = torch.abs(dirs[:, 2])

    # Get radii.
    radii = torch.rand(n, 1) * (radius[1] - radius[0]) + radius[0]
    origins = dirs * radii

    # Get camera to world rotation.
    front = dirs
    right = normalize(torch.cross(front, torch.tensor([[0, 0, 1.0]])))
    up = normalize(torch.cross(right, front))

    # Final camera to world matrix.
    c2w = torch.stack([right, up, front, origins], dim=-1)

    return c2w


def sample_randposes_interpolate(n: int, *args):
    """Generate random poses by interpolating three ground truth poses.

    Described in DietNerf.

    This is useful for forward facing scenes (e.g. LLFF) where ground truth poses face
    in similar directions.
    """
    raise NotImplementedError


def poses_to_cameras(
        poses: Float[Tensor, "batch 3 4"],
        resolution: Union[int, Tuple[int, int]],
        focal: Union[float, Tuple[float, float]],
        camera_type: CameraType = CameraType.FISHEYE,
        ) -> Cameras:
    """Convert poses to Cameras object.

    Args:
        poses: SE3 camera to world matrices. You can get these from other functions
            in this file.
        resolution: Resolution of the cameras. If an int, assumes square resolution.
        focal: Focal length of the cameras. If a float, assumes square focal length.

    Returns:
        Cameras object.
    """
    if isinstance(resolution, (int, float)):
        resolution = (resolution, resolution)
    if isinstance(focal, (int, float)):
        focal = (focal, focal)

    camera = Cameras(
        camera_type=camera_type,
        camera_to_worlds=poses,
        fx=float(focal[0]),
        fy=float(focal[1]),
        cx=resolution[0] / 2,
        cy=resolution[1] / 2,
    )
    return camera
