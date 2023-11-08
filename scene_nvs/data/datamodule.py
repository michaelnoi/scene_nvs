import lightning as L
import torchvision

from .dataloader import Scene_NVSDataLoader
from .dataset import ScannetppIphoneDataset


class Scene_NVSDataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, transforms: torchvision.transforms = None):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = None

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ScannetppIphoneDataset(
                self.root_dir, self.transforms
            )  # dummy directory
            self.val_dataset = ScannetppIphoneDataset(
                self.root_dir, self.transforms
            )  # dummy directory
        if stage == "test" or stage is None:
            self.test_dataset = ScannetppIphoneDataset(
                self.root_dir, self.transforms
            )  # dummy directory

    def train_dataloader(self):
        return Scene_NVSDataLoader(self.train_dataset, batch_size=32)

    def val_dataloader(self):
        return Scene_NVSDataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return Scene_NVSDataLoader(self.test_dataset, batch_size=32)
