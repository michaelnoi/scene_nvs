from typing import Optional

import lightning as L

from .dataloader import Scene_NVSDataLoader
from .dataset import ScannetppIphoneDataset


class Scene_NVSDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        num_workers: int,
        # image_size: int,  # TODO: delete together with transforms
        distance_threshold: float,
        truncate_data: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distance_threshold = distance_threshold
        self.truncate_data = truncate_data

        self.train_transforms = None  # self.basic_transforms
        self.val_transforms = None  # self.basic_transforms
        self.test_transforms = None  # self.basic_transforms

    def setup(self, stage: str = ""):
        if stage == "fit" or stage == "":
            self.train_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.distance_threshold,
                transform=self.train_transforms,
                stage="train",
            )
            self.val_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.distance_threshold,
                transform=self.val_transforms,
                stage="val",
            )
            if self.truncate_data:
                self.train_dataset._truncate_data(self.truncate_data)
                self.val_dataset._truncate_data(self.truncate_data)
        if stage == "test" or stage == "":
            self.test_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.distance_threshold,
                transform=self.test_transforms,
                stage="test",
            )
            if self.truncate_data:
                self.test_dataset._truncate_data(self.truncate_data)

        # self.datasets = {
        #     "train": self.train_dataset,
        #     "val": self.val_dataset,
        #     "test": self.test_dataset,
        # }

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
