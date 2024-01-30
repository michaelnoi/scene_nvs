from typing import Dict, List

import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision
from pytorch3d.structures import Pointclouds
from skimage.transform import resize

from scene_nvs.utils.camera_utils import get_cameras, get_scaled_intrinsics
from scene_nvs.utils.render_utils import CustomRenderer

# from utils.timings import rank_zero_print_log_time


RENDER_W, RENDER_H = 768, 576
SAVE_W, SAVE_H = 85, 64
RENDER_DEVICE = torch.device("cuda:2")


def render_and_save_image_batched(
    data_dict_batch: List[Dict[str, torch.Tensor]], depth_map_path: List[str]
):
    depth_cond = []
    image_cond = []
    for i in range(len(data_dict_batch)):
        depth_cond.append(cv2.imread(depth_map_path[i], cv2.IMREAD_ANYDEPTH))
        image_cond.append(torchvision.io.read_image(data_dict_batch[i]["path_cond"]))
    # convert uint16 to int16, hacky, but depth shouldn't exceed 32.767 m
    depth_cond = torch.from_numpy(np.stack(depth_cond).astype(np.int16))
    image_cond = torch.stack(image_cond)

    images = render_images_batched(
        data_dict_batch, image_cond, depth_cond
    ).half()  # [b, RENDER_H, RENDER_W, 3]

    # resize image
    images_resized = torch.zeros((images.shape[0], SAVE_H, SAVE_W, images.shape[-1]))
    for i in range(images.shape[0]):
        images_resized[i] = torch.from_numpy(
            resize(images[i].detach().cpu().numpy(), (SAVE_H, SAVE_W))
        )  # with resize anti-aliasing looks better, TODO: check if not possible without numpy

    # center crop
    images_resized = images_resized.permute(0, 3, 1, 2)  # [b, 3, SAVE_H, SAVE_W]
    images_cropped = torchvision.transforms.CenterCrop(min(SAVE_H, SAVE_W))(
        images_resized
    )

    # save depth map as jpg image
    images_cropped = images_cropped.detach().cpu().numpy()  # [b, 3, SAVE_H, SAVE_H]
    assert images_cropped.max() <= 255
    assert images_cropped.shape == (images.shape[0], 3, SAVE_H, SAVE_H)
    images_cropped = images_cropped.astype(np.uint8)  # [b, 3, SAVE_H, SAVE_H]

    for i in range(len(data_dict_batch)):
        image_path_proj = data_dict_batch[i]["rgb_cond_path"]
        imageio.imwrite(
            image_path_proj, images_cropped[i].transpose(1, 2, 0), format="jpg"
        )


# @rank_zero_print_log_time
def render_images_batched(
    data_dict_batch: List[Dict[str, torch.Tensor]],
    image_cond: torch.Tensor,
    depth_cond: torch.Tensor,
    device: torch.device = RENDER_DEVICE,
) -> torch.Tensor:
    image_cond = image_cond
    depth_cond = depth_cond
    # as rendered images will also be channel concat in latent space
    # we'll render to a lower resolution
    b = depth_cond.shape[0]
    h, w = RENDER_H, RENDER_W
    h_full_res, w_full_res = image_cond.shape[2:]
    assert image_cond.shape == (b, 3, 1440, 1920)
    assert depth_cond.shape == (b, 192, 256)

    K_cond_batch = torch.from_numpy(
        np.array([data_dict["K_cond"] for data_dict in data_dict_batch])
    )
    K_target_batch = torch.from_numpy(
        np.array([data_dict["K_target"] for data_dict in data_dict_batch])
    )
    pose_cond_batch = torch.from_numpy(
        np.array([data_dict["pose_cond"] for data_dict in data_dict_batch])
    )
    pose_target_batch = torch.from_numpy(
        np.array([data_dict["pose_target"] for data_dict in data_dict_batch])
    )

    K_cond_scaled = get_scaled_intrinsics(K_cond_batch, RENDER_H / h_full_res)
    K_target_scaled = get_scaled_intrinsics(K_target_batch, RENDER_H / h_full_res)

    camera_cond, camera_target = get_cameras(
        pose_cond_batch,
        pose_target_batch,
        K_cond_scaled,
        K_target_scaled,
        torch.tensor([h, w])[None, ...].float(),
    )

    renderer = CustomRenderer(
        camera_target,
        (RENDER_H, RENDER_W),
        device=device,
        radius=0.004,
        points_per_pixel=3,
    )

    # create pointcloud from RGBD image
    image_cond_rgb = F.interpolate(
        image_cond, size=(h, w), mode="nearest"
    )  # [b, 3, h, w]
    image_cond_rgb = image_cond_rgb.permute(0, 2, 3, 1)  # [b, h, w, 3]

    depth_cond = cv2.resize(
        depth_cond.permute(1, 2, 0).numpy(), (w, h), interpolation=cv2.INTER_NEAREST
    )
    if depth_cond.ndim == 2:
        depth_cond = depth_cond[..., None]
    depth_cond = torch.from_numpy(depth_cond).permute(2, 0, 1)  # [b, h, w]
    assert depth_cond.shape == (b, h, w)

    pc_batch = torch.zeros((b, h * w, 3))
    color_batch = torch.zeros((b, h * w, 3))

    for i in range(b):
        rgb_o3d = o3d.geometry.Image(
            np.ascontiguousarray(image_cond_rgb[i].detach().cpu().numpy())
        )  # [3, h, w]
        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_cond[i].detach().cpu().numpy())
        )  # [h, w]
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d,
            depth=depth_o3d,
            depth_scale=1000.0,
            depth_trunc=depth_cond[i].max(),
            convert_rgb_to_intensity=False,
        )

        # create pointcloud from RGBD image
        R_unproject = (
            camera_cond[i].get_world_to_view_transform().get_matrix().numpy()[0][:3, :3]
        )
        t_unproject = (
            camera_cond[i].get_world_to_view_transform().get_matrix().numpy()[0][3, :3]
        )
        pose_unproject = np.eye(4)
        pose_unproject[:3, :3] = R_unproject
        # pose[:3, 3] = t_unproject  # doesn't work for some reason...

        unproject_cond_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                w,
                h,
                K_cond_scaled[i, 0, 0],
                K_cond_scaled[i, 1, 1],
                K_cond_scaled[i, 0, 2],
                K_cond_scaled[i, 1, 2],
            ),
            pose_unproject,
        )
        # offset pointcloud to align with scene
        offset = np.eye(4)
        offset[:3, 3] = t_unproject
        unproject_cond_pointcloud.transform(offset)

        # get pointcloud from pytorch3d
        pc = torch.tensor(np.array(unproject_cond_pointcloud.points)).float()
        colors = torch.tensor(np.array(unproject_cond_pointcloud.colors)).float() * 255

        pc_batch[i] = pc
        color_batch[i] = colors

    # which points to render?
    point_cloud = Pointclouds(points=pc_batch, features=color_batch)
    # print(point_cloud.points_padded().shape)

    # render
    images = renderer.render_depth_or_image(point_cloud)  # [b, RENDER_H, RENDER_W, 3]
    return images
