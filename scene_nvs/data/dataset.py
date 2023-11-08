import os
from typing import Dict, List, Tuple

import torch
import torchvision
from torch.utils.data import Dataset

from ..utils.timings import log_time


class ScannetppIphoneDataset(Dataset):
    def __init__(self, root_dir: str, transform: torchvision.transforms = None):
        self.root_dir: str = root_dir
        self.transform: torchvision.transforms = transform

        self.data: List[Tuple[torch.Tensor, Dict[str, List[str]]]] = []
        self.targets: List[int] = []
        self.load_data()

    def load_data(self) -> None:
        # Load data (Image + Camera Poses)
        image_path = os.path.join(self.root_dir, "resized_images")
        camera_file = os.path.join(self.root_dir, "colmap", "images.txt")
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        with open(camera_file) as f:
            # read line if does not start with #
            lines = [line for line in f.readlines() if not line.startswith("#")]

        # all even lines are image info
        image_info = lines[::2]
        camera_poses = {
            line.split()[-1]: {
                "quaternion": line.split()[1:5],
                "translation": line.split()[5:8],
            }
            for line in image_info
        }

        image_files = sorted(os.listdir(image_path))
        for image_file in image_files:
            image = torchvision.io.read_image(os.path.join(image_path, image_file))
            self.data.append((image, camera_poses[image_file]))

        # Load targets TODO mock targets
        self.targets = [0] * len(self.data)

    def __len__(self) -> int:
        return len(self.data)

    @log_time
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            data = self.transform(data)

        return data, target
