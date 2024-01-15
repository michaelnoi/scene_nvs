import torch
from pytorch3d.renderer import PerspectiveCameras


def get_scaled_intrinsics(K: torch.Tensor, scale: float) -> torch.Tensor:
    assert K.dim() == 3
    batch_size = K.shape[0]
    K_scaled = torch.zeros((batch_size, 4, 4), dtype=torch.float32)
    K_scaled[:, 0, 0] = K[:, 0, 0] * scale
    K_scaled[:, 1, 1] = K[:, 1, 1] * scale
    K_scaled[:, 0, 2] = K[:, 0, 2] * scale
    K_scaled[:, 1, 2] = K[:, 1, 2] * scale
    K_scaled[:, 2, 3] = 1
    K_scaled[:, 3, 2] = 1
    return K_scaled


def get_cameras(
    pose_cond: torch.tensor,
    pose_target: torch.tensor,
    K_cond: torch.tensor,
    K_target: torch.tensor,
    image_size: torch.tensor,
) -> tuple[PerspectiveCameras, PerspectiveCameras]:
    R_cond = pose_cond[:, :3, :3].transpose(
        1, 2
    )  # transpose because pytorch3d assmues row vectors
    t_cond = pose_cond[:, :3, 3]
    R_cond = R_cond.float()
    t_cond = t_cond.float()

    camera_cond = PerspectiveCameras(
        R=R_cond, T=t_cond, K=K_cond, image_size=image_size, in_ndc=False
    )

    pose_target[:, :, [0, 1]] = -pose_target[
        :, :, [0, 1]
    ]  # switch convention as target pose is used to render
    pose_target = torch.inverse(pose_target.float())

    R_target = pose_target[:, :3, :3].transpose(
        1, 2
    )  # transpose because pytorch3d assmues row vectors
    t_target = pose_target[:, :3, 3]

    camera_target = PerspectiveCameras(
        R=R_target, T=t_target, K=K_target, image_size=image_size, in_ndc=False
    )
    return camera_cond, camera_target
