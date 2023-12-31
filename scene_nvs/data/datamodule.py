from typing import Optional

import lightning as L
from torchvision import transforms

from .dataloader import Scene_NVSDataLoader
from .dataset import ScannetppIphoneDataset


class Scene_NVSDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        num_workers: int,
        distance_threshold: float,
        truncate_data_train: Optional[int] = None,
        truncate_data_val: Optional[int] = None,
        image_size: Optional[int] = 512,
    ):
        super().__init__()
        self.root_dir = root_dir
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

    def setup(self, stage: str = ""):
        if stage == "fit" or stage == "":
            self.train_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.distance_threshold,
                transform=self.transformations,
                stage="train",
            )
            self.val_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.distance_threshold,
                transform=self.transformations,
                stage="val",
            )
            if self.truncate_data_train:
                self.train_dataset._truncate_data(self.truncate_data_train)
            if self.truncate_data_val:
                self.val_dataset._truncate_data(self.truncate_data_val)
        if stage == "test" or stage == "":
            self.test_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.distance_threshold,
                transform=self.transformations,
                stage="test",
            )
            if self.truncate_data_val:
                self.test_dataset._truncate_data(self.truncate_data_val)
        if stage == "validate":
            self.val_dataset = ScannetppIphoneDataset(
                self.root_dir,
                self.distance_threshold,
                transform=self.transformations,
                stage="val",
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
