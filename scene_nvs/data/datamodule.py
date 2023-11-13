import lightning as L
import torchvision
from einops import rearrange

from .dataloader import Scene_NVSDataLoader
from .dataset import ScannetppIphoneDataset


class Scene_NVSDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        num_workers: int,
        image_size: int,
        transforms: torchvision.transforms = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # TODO: load from config and find smarter setup
        self.basic_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
                torchvision.transforms.Resize([image_size, image_size]),
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: rearrange(x, "c h w -> h w c")),
            ]
        )

        self.train_transforms = self.basic_transforms
        self.val_transforms = self.basic_transforms
        self.test_transforms = self.basic_transforms

    def setup(self, stage: str = ""):
        if stage == "fit" or stage == "":
            self.train_dataset = ScannetppIphoneDataset(
                self.root_dir, self.train_transforms, stage="train"
            )
            self.val_dataset = ScannetppIphoneDataset(
                self.root_dir, self.val_transforms, stage="val"
            )
        if stage == "test" or stage == "":
            self.test_dataset = ScannetppIphoneDataset(
                self.root_dir, self.test_transforms, stage="test"
            )

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
