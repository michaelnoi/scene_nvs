import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

# from scene_nvs.utils.timings import log_time


class ScannetppIphoneDataset(Dataset):
    def __init__(self, root_dir: str, transform: torchvision.transforms = None):
        self.root_dir: str = root_dir
        self.transform: torchvision.transforms = transform

        self.data: List[Tuple[torch.Tensor, Dict[str, List[str]]]] = []
        self.targets: List[int] = []
        self.load_data()

    def load_data(self) -> None:
        # Load data (Image + Camera Poses)
        image_paths = os.path.join(self.root_dir, "rgb")
        # read the json file pose_intrinsic_imu.json at self.root_dir
        with open(os.path.join(self.root_dir, "pose_intrinsic_imu.json")) as f:
            poses = json.load(f)

        poses = {
            frame: np.asarray(poses[frame]["pose"]) for frame, pose in poses.items()
        }

        image_files = sorted(os.listdir(image_paths))
        for image_file in image_files:
            self.data.append(
                (
                    os.path.join(image_paths, image_file),
                    poses[image_file.replace(".jpg", "")],
                )
            )

        # Load targets TODO mock targets
        self.targets = [0] * len(self.data)

    def __len__(self) -> int:
        return len(self.data)

    # @log_time
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, pose = self.data[idx]
        image_data = torchvision.io.read_image(image_path)
        target = self.targets[idx]

        if self.transform:
            image_data = self.transform(image_data)

        return {"image": image_data, "pose": pose, "target": target}
