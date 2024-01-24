from typing import Dict, List

import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision
from pytorch3d.structures import Pointclouds

from scene_nvs.utils.camera_utils import get_cameras, get_scaled_intrinsics
from scene_nvs.utils.render_utils import CustomRenderer

# from utils.timings import rank_zero_print_log_time


def render_and_save_depth_map_batched(
    data_dict_batch: List[Dict[str, torch.Tensor]], depth_map_path: List[str]
):
    depth_cond = []
    image_cond = []
    for i in range(len(data_dict_batch)):
        depth_cond.append(cv2.imread(depth_map_path[i], cv2.IMREAD_ANYDEPTH))
        image_cond.append(torchvision.io.read_image(data_dict_batch[i]["path_cond"]))
    # convert to int16, hacky, but depth shouldn't exceed 32.767 m
    depth_cond = torch.from_numpy(np.stack(depth_cond).astype(np.int16))
    image_cond = torch.stack(image_cond)

    depth_maps = render_depth_maps_batched(
        data_dict_batch, image_cond, depth_cond
    ).half()

    # save depth map as 16-bit png
    depth_maps = depth_maps.detach().cpu().numpy()  # [b, 1, h, h]
    depth_maps = (depth_maps * 1000).clip(0, 65535).astype(np.uint16)  # [b, 1, h, h]

    for i in range(len(data_dict_batch)):
        depth_map_path_proj = data_dict_batch[i]["depth_map_path"]
        imageio.imwrite(
            depth_map_path_proj, depth_maps[i, 0], format="png", bitdepth=16
        )


def render_and_save_depth_map(
    data_dict: Dict[str, torch.Tensor], depth_map_path: str
) -> None:
    depth_cond = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
    # convert to int16, hacky, but depth shouldn't exceed 32.767 m
    depth_cond = depth_cond.astype(np.int16)
    depth_cond = torch.from_numpy(depth_cond)

    image_cond = torchvision.io.read_image(data_dict["path_cond"])

    depth_map = render_depth_map(data_dict, image_cond, depth_cond).half()

    # save depth map as 16-bit png
    depth_map = depth_map.detach().cpu().numpy()  # [h, h]
    depth_map = (depth_map * 1000).clip(0, 65535).astype(np.uint16)  # [h, h]
    # print(f"depth_map.min(): {depth_map.min()}, depth_map.max(): {depth_map.max()}")

    depth_map_path_proj = data_dict["depth_map_path"]
    imageio.imwrite(depth_map_path_proj, depth_map, format="png", bitdepth=16)


# @rank_zero_print_log_time
def render_depth_map(
    data_dict: Dict[str, torch.Tensor],
    image_cond: torch.Tensor,
    depth_cond: torch.Tensor,
    device: torch.device = torch.device("cuda:0"),
) -> torch.Tensor:
    image_cond = image_cond
    depth_cond = depth_cond
    assert image_cond.shape == (3, 1440, 1920)
    assert depth_cond.shape == (192, 256)
    h, w = depth_cond.shape[:2]
    h_full_res, w_full_res = image_cond.shape[1:]

    K_cond = torch.from_numpy(data_dict["K_cond"]).unsqueeze(0)
    K_target = torch.from_numpy(data_dict["K_target"]).unsqueeze(0)
    pose_cond = torch.from_numpy(data_dict["pose_cond"]).unsqueeze(0)
    pose_target = torch.from_numpy(data_dict["pose_target"]).unsqueeze(0)

    K_cond_scaled = get_scaled_intrinsics(K_cond, h / h_full_res)
    K_target_scaled = get_scaled_intrinsics(K_target, h / h_full_res)

    camera_cond, camera_target = get_cameras(
        pose_cond,
        pose_target,
        K_cond_scaled,
        K_target_scaled,
        torch.tensor([h, w])[None, ...].float(),
    )

    renderer = CustomRenderer(camera_target, (h, w), device=device)

    # create pointcloud from RGBD image
    # image_cond [3, h, w]
    image_cond_rgb = F.interpolate(
        image_cond.unsqueeze(0), size=(h, w), mode="nearest"
    ).squeeze(0)
    image_cond_rgb = image_cond_rgb.permute(1, 2, 0)  # [h, w, 3]

    rgb_o3d = o3d.geometry.Image(
        np.ascontiguousarray(image_cond_rgb.detach().cpu().numpy())
    )  # [h, w, 3]
    depth_o3d = o3d.geometry.Image(
        np.ascontiguousarray(depth_cond.detach().cpu().numpy())
    )  # [h, w]
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_o3d,
        depth=depth_o3d,
        depth_scale=1000.0,
        depth_trunc=depth_cond.max(),
        convert_rgb_to_intensity=False,
    )

    # create pointcloud from RGBD image
    R_unproject = (
        camera_cond.get_world_to_view_transform().get_matrix().numpy()[0][:3, :3]
    )
    t_unproject = (
        camera_cond.get_world_to_view_transform().get_matrix().numpy()[0][3, :3]
    )
    pose_unproject = np.eye(4)
    pose_unproject[:3, :3] = R_unproject
    # pose[:3, 3] = t_unproject  # doesn't work for some reason...

    unproject_cond_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            w,
            h,
            K_cond_scaled[0, 0, 0],
            K_cond_scaled[0, 1, 1],
            K_cond_scaled[0, 0, 2],
            K_cond_scaled[0, 1, 2],
        ),
        pose_unproject,
    )
    # offset pointcloud to align with scene
    offset = np.eye(4)
    offset[:3, 3] = t_unproject
    unproject_cond_pointcloud.transform(offset)

    # get pointcloud from pytorch3d
    pc = torch.tensor(np.array(unproject_cond_pointcloud.points)).float().unsqueeze(0)
    colors = (
        torch.tensor(np.array(unproject_cond_pointcloud.colors)).float().unsqueeze(0)
        * 255
    )

    assert pc.shape == (1, h * w, 3)
    assert colors.shape == (1, h * w, 3)

    # which points to render?
    point_cloud = Pointclouds(points=pc, features=colors)

    # render
    depth_map = renderer(point_cloud)

    # ensure that the depth image corresponds to the target image
    depth_map = torchvision.transforms.CenterCrop(min(h, w))(depth_map)  # [1, h, h]
    depth_map = depth_map.squeeze(0)
    assert depth_map.shape == (h, h)
    return depth_map


# @rank_zero_print_log_time
def render_depth_maps_batched(
    data_dict_batch: List[Dict[str, torch.Tensor]],
    image_cond: torch.Tensor,
    depth_cond: torch.Tensor,
    device: torch.device = torch.device("cuda:0"),
) -> torch.Tensor:
    image_cond = image_cond
    depth_cond = depth_cond
    b, h, w = depth_cond.shape[:3]
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

    K_cond_scaled = get_scaled_intrinsics(K_cond_batch, h / h_full_res)
    K_target_scaled = get_scaled_intrinsics(K_target_batch, h / h_full_res)

    camera_cond, camera_target = get_cameras(
        pose_cond_batch,
        pose_target_batch,
        K_cond_scaled,
        K_target_scaled,
        torch.tensor([h, w])[None, ...].float(),
    )

    renderer = CustomRenderer(camera_target, (h, w), device=device)

    # create pointcloud from RGBD image
    # image_cond [b, h, w, 3]
    image_cond_rgb = F.interpolate(
        image_cond, size=(h, w), mode="nearest"
    )  # [b, 3, h, w]
    image_cond_rgb = image_cond_rgb.permute(0, 2, 3, 1)  # [b, h, w, 3]

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

    # render
    depth_map = renderer(point_cloud)

    # ensure that the depth image corresponds to the target image
    depth_map = torchvision.transforms.CenterCrop(min(h, w))(depth_map)  # [b, h, h]
    depth_map = depth_map.unsqueeze(1)  # [b, 1, h, h]

    assert depth_map.shape == (b, 1, h, h)
    return depth_map
