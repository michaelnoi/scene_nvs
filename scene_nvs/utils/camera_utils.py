import math

import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras


def get_scaled_intrinsics(K: np.array, scale: float) -> np.array:
    assert math.isclose(scale, 192 / 1440)  # TODO: delete
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return K_scaled


def get_cameras(
    pose_cond: np.array,
    pose_target: np.array,
    K_cond: np.array,
    K_target: np.array,
    image_size: torch.tensor,
    scale: float = 1.0,
) -> tuple[PerspectiveCameras, PerspectiveCameras]:
    if scale != 1.0:
        K_cond = get_scaled_intrinsics(K_cond, scale)
        K_target = get_scaled_intrinsics(K_target, scale)

    pose_cond = torch.tensor(pose_cond).float()
    R_cond = pose_cond[:3, :3].T  # transpose because pytorch3d assmues row vectors
    t_cond = pose_cond[:3, 3]
    R_cond = R_cond.float().unsqueeze(0)
    t_cond = t_cond.float().unsqueeze(0)

    camera_cond = PerspectiveCameras(
        R=R_cond, T=t_cond, K=K_cond, image_size=image_size, in_ndc=False
    )

    pose_target = pose_target.float()
    pose_target[:, [0, 1]] = -pose_target[
        :, [0, 1]
    ]  # switch convention as target pose is used to render
    pose_target = np.linalg.inv(pose_target)  # to render world to cam
    pose_target = torch.tensor(pose_target).float()

    R_target = pose_target[:3, :3].T  # transpose because pytorch3d assmues row vectors
    t_target = pose_target[:3, 3]
    R_target = R_target.float().unsqueeze(0)
    t_target = t_target.float().unsqueeze(0)

    camera_target = PerspectiveCameras(
        R=R_target, T=t_target, K=K_target, image_size=image_size, in_ndc=False
    )

    return camera_cond, camera_target
