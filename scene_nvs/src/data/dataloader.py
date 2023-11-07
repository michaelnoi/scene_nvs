from torch.utils.data import DataLoader


class Scene_NVSDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
