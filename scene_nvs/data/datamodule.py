from typing import List, Optional

import lightning as L
from torchvision import transforms

from .dataloader import Scene_NVSDataLoader
from .dataset import ScannetppIphoneDataset


class Scene_NVSDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        train_scenes: List[str],
        image_pairs_per_scene: int,
        batch_size: int,
        num_workers: int,
        distance_threshold: float,
        val_scenes=None,
        test_scenes=None,
        depth_map=None,
        rendered_rgb_cond: bool = False,
        truncate_data_train: Optional[int] = None,
        truncate_data_val: Optional[int] = None,
        image_size: Optional[int] = 512,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.train_scenes = train_scenes
        self.val_scenes = val_scenes if val_scenes else train_scenes
        self.test_scenes = test_scenes if test_scenes else train_scenes
        self.image_pairs_per_scene = image_pairs_per_scene
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distance_threshold = distance_threshold
        self.truncate_data_train = truncate_data_train
        self.truncate_data_val = truncate_data_val
        self.transformations = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        if depth_map in ["gt", "projected", "partial_gt"]:
            self.depth_map = True
            self.depth_map_type = depth_map
        elif depth_map is None:
            self.depth_map = False
            self.depth_map_type = None
        else:
            raise ValueError(
                f"depth_map must be one of ['gt', 'projected', 'partial_gt', None], got {depth_map}"
            )
        self.rendered_rgb_cond = rendered_rgb_cond

    def setup(self, stage: str = ""):
        if stage == "fit" or stage == "":
            self.train_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.train_scenes,
                self.image_pairs_per_scene,
                self.distance_threshold,
                transform=self.transformations,
                stage="train",
                depth_map_type=self.depth_map_type,
                depth_map=self.depth_map,
                rendered_rgb_cond=self.rendered_rgb_cond,
            )
            self.val_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.val_scenes,
                self.image_pairs_per_scene,
                self.distance_threshold,
                transform=self.transformations,
                stage="val",
                depth_map_type=self.depth_map_type,
                depth_map=self.depth_map,
                rendered_rgb_cond=self.rendered_rgb_cond,
            )
            if self.truncate_data_train:
                self.train_dataset._truncate_data(self.truncate_data_train)
            if self.truncate_data_val:
                self.val_dataset._truncate_data(self.truncate_data_val)
        if stage == "test" or stage == "":
            self.test_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.test_scenes,
                self.image_pairs_per_scene,
                self.distance_threshold,
                transform=self.transformations,
                stage="test",
                depth_map_type=self.depth_map_type,
                depth_map=self.depth_map,
                rendered_rgb_cond=self.rendered_rgb_cond,
            )
            if self.truncate_data_val:
                self.test_dataset._truncate_data(self.truncate_data_val)
        if stage == "validate":
            self.val_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.val_scenes,
                self.image_pairs_per_scene,
                self.distance_threshold,
                transform=self.transformations,
                stage="val",
                depth_map_type=self.depth_map_type,
                depth_map=self.depth_map,
                rendered_rgb_cond=self.rendered_rgb_cond,
            )

            if self.truncate_data_val:
                self.val_dataset._truncate_data(self.truncate_data_val)
        # if stage == "predict": TODO: also handle the last possible stage predict

    def train_dataloader(self):
        return Scene_NVSDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return Scene_NVSDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return Scene_NVSDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
