import json
import math
import os
from typing import Dict, List, Union

import numpy as np
import torch
import torchvision
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from scene_nvs.utils.timings import log_time


class ScannetppIphoneDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: torchvision.transforms = None,
        distance_threshold: float = 1,
        stage: str = "train",
    ):
        self.root_dir: str = root_dir
        self.transform: torchvision.transforms = transform

        self.data: List[Dict[str, Union[str, torch.Tensor]]] = []
        self.distance_threshold = distance_threshold
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

        if os.path.exists(os.path.join(self.root_dir, "difference_matrix.npy")):
            difference_matrix = np.load(
                os.path.join(self.root_dir, "difference_matrix.npy")
            )
            # to torch tensor
            difference_matrix = torch.from_numpy(difference_matrix)
            print("Loaded difference matrix from file")
        else:
            difference_matrix = self.get_difference_matrix(
                np.asarray(list(poses.values()))
            )
            np.save(
                os.path.join(self.root_dir, "difference_matrix.npy"), difference_matrix
            )
            print("Saved difference matrix to file")

        distance_matrix = self.get_distance_matrix(difference_matrix)

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

        print("Length of train, val, test: ", len(train), len(val), len(test))

        if self.stage == "train":
            self.data = [
                {
                    "path_cond": image_files[i],
                    "path_target": image_files[j],
                    "T": difference_matrix[i, j],
                }
                for i, j in train
            ]

        elif self.stage == "val":
            self.data = [
                {
                    "path_cond": image_files[i],
                    "path_target": image_files[j],
                    "T": difference_matrix[i, j],
                }
                for i, j in val
            ]

        elif self.stage == "test":
            self.data = [
                {
                    "path_cond": image_files[i],
                    "path_target": image_files[j],
                    "T": difference_matrix[i, j],
                }
                for i, j in test
            ]

        else:
            raise ValueError("stage must be one of train, val, test")

    @log_time
    def get_difference_matrix(self, poses: np.ndarray) -> torch.Tensor:
        n = len(poses)
        difference_matrix = torch.zeros((n, n, 4))
        for i in tqdm.tqdm(range(n)):
            for j in range(n):
                difference_matrix[i, j] = self.get_T(poses[i], poses[j])
        return difference_matrix

    @log_time
    def get_distance_matrix(self, difference_matrix: torch.Tensor) -> torch.Tensor:
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

    def __len__(self) -> int:
        return len(self.data)

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

    # @log_time
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_dict = self.data[idx]
        image_cond = torchvision.io.read_image(data_dict["path_cond"])
        image_target = torchvision.io.read_image(data_dict["path_target"])
        T = data_dict["T"]

        if self.transform:
            image_cond = self.transform(image_cond)
            image_target = self.transform(image_target)

        return {"image_cond": image_cond, "image_target": image_target, "T": T}
