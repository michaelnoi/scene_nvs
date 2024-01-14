import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
import torchvision
import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from scene_nvs.utils.distributed import rank_zero_print
from scene_nvs.utils.timings import rank_zero_print_log_time


class ScannetppIphoneDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        scenes: List[str],
        image_pairs_per_scene: int,
        distance_threshold: float,
        depth_map: bool = True,
        transform: torchvision.transforms = None,
        stage: str = "train",
    ):
        self.root_dir: str = root_dir
        self.scenes: List[str] = scenes
        self.transform: torchvision.transforms = transform
        self.data: List[Dict[str, Union[str, torch.Tensor]]] = []
        self.distance_threshold = distance_threshold
        self.depth_map = depth_map
        self.stage = stage

        for scene in tqdm.tqdm(scenes, desc="Loading scenes"):
            self.data += self.load_data(os.path.join(root_dir, scene, "iphone"))

        # doesnt work from commandline
        # with multiprocessing.Pool(8) as pool:
        #     result = list(tqdm(pool.imap(self.load_data, [os.path.join(
        #         root_dir, scene, "iphone") for scene in scenes]), total=len(scenes)))
        # for data in result:
        #    self.data += data

        scene_counts = Counter(item["scene"] for item in self.data)

        # Find the minimum amount of data for one scene
        min_scene_data_count = min(scene_counts.values())
        rank_zero_print(
            "Minimum number of data points for one scene: ", min_scene_data_count
        )

        if image_pairs_per_scene > min_scene_data_count:
            rank_zero_print(
                "image_pairs_per_scene is larger than the minimum number of data points for one scene. Setting image_pairs_per_scene to minimum number of data points for one scene"
            )
            image_pairs_per_scene = min_scene_data_count

        # Remove data from all_data until all scenes have the same amount of data
        filtered_data = []
        # initialize dict with scene names as keys and 0 as values
        scene_counts_new = {scene: 0 for scene in scene_counts.keys()}
        for item in self.data:
            if scene_counts_new[item["scene"]] < image_pairs_per_scene:
                filtered_data.append(item)
                scene_counts_new[item["scene"]] += 1

        self.data = filtered_data

        # assert that all scenes have the same amount of data
        scene_counts = Counter(item["scene"] for item in self.data)
        assert (
            len(set(scene_counts.values())) == 1
        ), "Not all scenes have the same amount of data"

        # shuffle data randomly
        random.seed(10)
        random.shuffle(self.data)

    def load_data(self, directory) -> List[Dict[str, Union[str, torch.Tensor]]]:
        # Load data (Image + Camera Poses)
        image_folder = os.path.join(directory, "rgb")
        image_names = sorted(os.listdir(image_folder))
        image_files = [
            os.path.join(image_folder, image_name) for image_name in image_names
        ]

        # read the json file pose_intrinsic_imu.json at self.root_dir
        with open(os.path.join(directory, "pose_intrinsic_imu.json")) as f:
            poses = json.load(f)

        poses_c2w = {
            # aligned_pose is aligned with mesh the dataset provides
            frame: np.asarray(poses[frame]["aligned_pose"])
            for frame, pose in poses.items()
            if frame + ".jpg" in image_names
        }

        K = {
            frame: np.asarray(poses[frame]["intrinsic"])
            for frame, pose in poses.items()
            if frame + ".jpg" in image_names
        }

        # check if difference matrix exists
        if os.path.exists(os.path.join(directory, "distance_matrix.npy")):
            distance_matrix = np.load(os.path.join(directory, "distance_matrix.npy"))
            # to torch tensor
            distance_matrix = torch.from_numpy(distance_matrix)
            # rank_zero_print("Loaded distance matrix from file")
        else:
            distance_matrix = self.get_distance_matrix(
                np.asarray(list(poses_c2w.values()))
            )
            # get max
            maximum = torch.max(distance_matrix[~torch.isnan(distance_matrix)])
            # scale to 0-1
            distance_matrix = distance_matrix / maximum
            np.save(os.path.join(directory, "distance_matrix.npy"), distance_matrix)
            # rank_zero_print("Saved distance matrix to file")

        if not torch.is_tensor(distance_matrix):
            distance_matrix = torch.from_numpy(distance_matrix)
        # print("shape of distance matrix: ", distance_matrix.shape)

        mask = torch.logical_and(
            distance_matrix > 0, distance_matrix <= self.distance_threshold
        )

        candidate_indicies = torch.argwhere(mask)
        # get corresponding viewpoint metric for the candidate indicies
        candidate_viewpoint_metric = distance_matrix[mask]
        # bin the viewpoint metric into 10 bins to stratify the data
        bins = np.linspace(0, self.distance_threshold, 5)
        binned_viewpoint_metric = np.digitize(candidate_viewpoint_metric, bins)
        learn, test = train_test_split(
            candidate_indicies,
            test_size=0.2,
            stratify=binned_viewpoint_metric,
            random_state=42,
        )

        # shape of splits: (n, 2)
        train, val = train_test_split(
            learn,
            test_size=0.2,
            stratify=binned_viewpoint_metric[learn],
            random_state=42,
        )

        if self.stage == "train":
            data = [
                {
                    "path_cond": image_files[i],
                    "path_target": image_files[j],
                    "pose_cond": poses_c2w[image_names[i].split(".")[0]],
                    "pose_target": poses_c2w[image_names[j].split(".")[0]],
                    "scene": directory.split("/")[-2],
                    "K_cond": K[image_names[i].split(".")[0]],
                    "K_target": K[image_names[j].split(".")[0]],
                }
                for i, j in train
            ]

        elif self.stage == "val":
            data = [
                {
                    "path_cond": image_files[i],
                    "path_target": image_files[j],
                    "pose_cond": poses_c2w[image_names[i].split(".")[0]],
                    "pose_target": poses_c2w[image_names[j].split(".")[0]],
                    "scene": directory.split("/")[-2],
                    "K_cond": K[image_names[i].split(".")[0]],
                    "K_target": K[image_names[j].split(".")[0]],
                }
                for i, j in val
            ]

        elif self.stage == "test":
            data = [
                {
                    "path_cond": image_files[i],
                    "path_target": image_files[j],
                    "pose_cond": poses_c2w[image_names[i].split(".")[0]],
                    "pose_target": poses_c2w[image_names[j].split(".")[0]],
                    "scene": directory.split("/")[-2],
                    "K_cond": K[image_names[i].split(".")[0]],
                    "K_target": K[image_names[j].split(".")[0]],
                }
                for i, j in test
            ]

        else:
            raise ValueError("stage must be one of train, val, test")

        return data

    def __len__(self) -> int:
        return len(self.data)

    # @log_time
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_dict = self.data[idx]
        image_target = Image.open(data_dict["path_target"])  # shape [3,1920,1440]
        image_cond = torchvision.io.read_image(
            data_dict["path_cond"]
        )  # shape [3,1920,1440]

        T = self.get_relative_pose(
            data_dict["pose_target"], data_dict["pose_cond"]
        )  # shape [7]

        # Overfit DEBUG set target image to be completely white
        # image_target = Image.new('RGB', (512, 512), color = 'white')
        image_cond_vae = None
        if self.transform:
            # apply transformations for VAE only on target image
            image_target = self.transform(image_target)
            image_cond_vae = torchvision.transforms.ToPILImage()(image_cond)
            image_cond_vae = self.transform(image_cond_vae)  # used for DreamPoseADapter

        result = {
            "image_cond": image_cond,
            "image_target": image_target,
            "T": T,
            "path_cond": data_dict["path_cond"],
            "image_cond_vae": image_cond_vae,
            "pose_cond": data_dict["pose_cond"],
            "pose_target": data_dict["pose_target"],
            "K_cond": data_dict["K_cond"],
            "K_target": data_dict["K_target"],
        }

        if self.depth_map:
            depth_map_path = (
                data_dict["path_cond"].replace("rgb", "depth").replace("jpg", "png")
            )
            depth_map = cv2.imread(
                depth_map_path, cv2.IMREAD_ANYDEPTH
            )  # make sure to read the image as 16 bit
            depth_map = depth_map.astype(
                np.int16
            )  # convert to int16, hacky, but depth shouldn't exceed 32.767 m
            # result["raw_depth_map"] = torch.from_numpy(depth_map)  # convert to torch tensor

            # TODO: prerender all possible depth maps and save them to disk
            # TODO: load depth map from disk

        return result

    def _truncate_data(self, n: int) -> None:
        # truncate the data to n points (for debugging)
        self.data = self.data[:n]

        rank_zero_print("Truncated data to length: " + str(self.__len__()))

    @rank_zero_print_log_time
    def get_distance_matrix(self, poses: np.ndarray) -> torch.Tensor:
        n = len(poses)
        distance_matrix = np.zeros((n, n, 2))
        for i in tqdm.tqdm(range(n)):
            for j in range(n):
                rotational_distance = self.get_rotational_distance(poses[i], poses[j])
                translational_distance = self.get_translational_distance(
                    poses[i], poses[j]
                )
                distance_matrix[i, j] = np.array(
                    [rotational_distance, translational_distance]
                )

        distance_matrix = np.sqrt(np.sum(distance_matrix**2, axis=2))
        return torch.from_numpy(distance_matrix)

    @rank_zero_print_log_time
    def get_difference_matrix_old(self, poses: np.ndarray) -> torch.Tensor:
        n = len(poses)
        difference_matrix = torch.zeros((n, n, 4))
        for i in tqdm.tqdm(range(n)):
            for j in range(n):
                difference_matrix[i, j] = self.get_T(poses[i], poses[j])
        return difference_matrix

    @rank_zero_print_log_time
    def get_distance_matrix_old(self, difference_matrix: torch.Tensor) -> torch.Tensor:
        difference_matrix[:, :, 2] = difference_matrix[:, :, 2] - 1  # to map 1 -> 0
        # Normalize
        difference_matrix[:, :, 2] = (
            difference_matrix[:, :, 2] / difference_matrix[:, :, 2].min()
        )
        difference_matrix[:, :, 0] = (
            difference_matrix[:, :, 0] / difference_matrix[:, :, 0].max()
        )
        difference_matrix[:, :, 3] = (
            difference_matrix[:, :, 3] / difference_matrix[:, :, 3].max()
        )

        single_distance_matrix = torch.sqrt(
            torch.sum(difference_matrix**2, dim=2)
        )  # Shape (n,n)

        return single_distance_matrix

    def get_rotational_distance(self, pose_1: np.ndarray, pose_2: np.ndarray) -> float:
        # http://www.boris-belousov.net/2016/12/01/quat-dist/#:~:text=Using%20quaternions%C2%B6&text=The%20difference%20rotation%20quaternion%20that,quaternion%20r%20%3D%20p%20q%20%E2%88%97%20.
        # https://math.stackexchange.com/questions/90081/quaternion-distance

        rotation_1 = Rotation.from_matrix(pose_1[:3, :3]).as_quat()
        rotation_2 = Rotation.from_matrix(pose_2[:3, :3]).as_quat()
        return 2 * np.arccos(np.dot(rotation_1, rotation_2))

    def get_translational_distance(
        self, pose_1: np.ndarray, pose_2: np.ndarray
    ) -> float:
        return np.linalg.norm(pose_1[:3, 3] - pose_2[:3, 3])

    def get_rotational_difference(
        self, rotation_1: Rotation, rotation_2: Rotation
    ) -> np.ndarray:
        # https://stackoverflow.com/questions/22157435/difference-between-the-two-quaternions
        # http://www.boris-belousov.net/2016/12/01/quat-dist/#:~:text=Using%20quaternions%C2%B6&text=The%20difference%20rotation%20quaternion%20that,quaternion%20r%20%3D%20p%20q%20%E2%88%97%20.

        return rotation_2.as_quat() * rotation_1.inv().as_quat()

    def get_translational_difference(
        self, translation_1: np.ndarray, translation_2: np.ndarray
    ) -> np.ndarray:
        return translation_1 - translation_2

    def get_relative_pose(self, pose_1: np.ndarray, pose_2: np.ndarray) -> np.ndarray:
        rotation_1 = Rotation.from_matrix(pose_1[:3, :3])
        rotation_2 = Rotation.from_matrix(pose_2[:3, :3])
        translation_1 = pose_1[:3, 3]
        translation_2 = pose_2[:3, 3]

        return np.concatenate(
            [
                self.get_rotational_difference(rotation_1, rotation_2),
                self.get_translational_difference(translation_1, translation_2),
            ]
        )

    def cartesian_to_spherical(self, xyz: np.ndarray) -> np.ndarray:
        # https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py#L318

        # ptsnew = np.hstack((xyz, np.zeros(xyz.shape))) #what is this for?
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        # for elevation angle defined from Z-axis down
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy))
        # # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT: np.ndarray, cond_RT: np.ndarray) -> torch.Tensor:
        # https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py#L318

        R, T = target_RT[:3, :3], target_RT[:3, -1]  # double check this
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:3, -1]  # double check this
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(
            T_target[None, :]
        )

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor(
            [
                d_theta.item(),
                math.sin(d_azimuth.item()),
                math.cos(d_azimuth.item()),
                d_z.item(),
            ]
        )

        return d_T
