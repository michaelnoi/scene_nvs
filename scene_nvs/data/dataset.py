import json
import math
import os
from typing import Dict, List, Union

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
        distance_threshold: float,
        depth_map: bool = True,
        transform: torchvision.transforms = None,
        stage: str = "train",
    ):
        self.root_dir: str = root_dir
        self.transform: torchvision.transforms = transform

        self.data: List[Dict[str, Union[str, torch.Tensor]]] = []
        self.distance_threshold = distance_threshold
        self.depth_map = depth_map
        self.stage = stage
        self.load_data()

    def load_data(self) -> None:
        # Load data (Image + Camera Poses)
        image_folder = os.path.join(self.root_dir, "rgb")
        image_names = sorted(os.listdir(image_folder))
        image_files = [
            os.path.join(image_folder, image_name) for image_name in image_names
        ]

        # read the json file pose_intrinsic_imu.json at self.root_dir
        with open(os.path.join(self.root_dir, "pose_intrinsic_imu.json")) as f:
            poses = json.load(f)

        poses = {
            frame: np.asarray(poses[frame]["pose"])
            for frame, pose in poses.items()
            if frame + ".jpg" in image_names
        }

        # check if difference matrix exists
        if os.path.exists(os.path.join(self.root_dir, "distance_matrix.npy")):
            distance_matrix = np.load(
                os.path.join(self.root_dir, "distance_matrix.npy")
            )
            # to torch tensor
            distance_matrix = torch.from_numpy(distance_matrix)
            rank_zero_print("Loaded distance matrix from file")
        else:
            distance_matrix = self.get_distance_matrix(np.asarray(list(poses.values())))
            # get max
            maximum = torch.max(distance_matrix[~torch.isnan(distance_matrix)])
            # scale to 0-1
            distance_matrix = distance_matrix / maximum
            np.save(os.path.join(self.root_dir, "distance_matrix.npy"), distance_matrix)
            rank_zero_print("Saved distance matrix to file")

        if not torch.is_tensor(distance_matrix):
            distance_matrix = torch.from_numpy(distance_matrix)
        print("shape of distance matrix: ", distance_matrix.shape)

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
            self.data = [
                {
                    "path_cond": image_files[i],
                    "path_target": image_files[j],
                    "pose_cond": poses[image_names[i].split(".")[0]],
                    "pose_target": poses[image_names[j].split(".")[0]],
                }
                for i, j in train
            ]

        elif self.stage == "val":
            self.data = [
                {
                    "path_cond": image_files[i],
                    "path_target": image_files[j],
                    "pose_cond": poses[image_names[i].split(".")[0]],
                    "pose_target": poses[image_names[j].split(".")[0]],
                }
                for i, j in val
            ]

        elif self.stage == "test":
            self.data = [
                {
                    "path_cond": image_files[i],
                    "path_target": image_files[j],
                    "pose_cond": poses[image_names[i].split(".")[0]],
                    "pose_target": poses[image_names[j].split(".")[0]],
                }
                for i, j in test
            ]

        else:
            raise ValueError("stage must be one of train, val, test")

        print_statement = {
            "train": "Length of train: " + str(len(train)),
            "val": "Length of val: " + str(len(val)),
            "test": "Length of test: " + str(len(test)),
        }[self.stage]
        rank_zero_print(print_statement)

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

        if self.transform:
            # apply transformations for VAE only on target image
            image_target = self.transform(image_target)

        result = {
            "image_cond": image_cond,
            "image_target": image_target,
            "T": T,
            "path_cond": data_dict["path_cond"],
        }

        if self.depth_map:
            depth_map_path = (
                data_dict["path_target"].replace("rgb", "depth").replace("jpg", "png")
            )
            depth_map = Image.open(depth_map_path)
            h, w = depth_map.size
            # ensure that the depth image corresponds to the target image
            depth_map = torchvision.transforms.CenterCrop(min(h, w))(depth_map)
            depth_map = torchvision.transforms.ToTensor()(depth_map)
            result["depth_map"] = depth_map.float()

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
