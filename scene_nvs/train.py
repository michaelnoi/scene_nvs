import lightning as pl
from data.datamodule import Scene_NVSDataModule
from ldm.model import SceneNVSNet
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(project="zero-shot-nvs")

datamodule = Scene_NVSDataModule(
    root_dir="/home/scannet/data/41b00feddb/iphone/",
    batch_size=1,
    num_workers=0,
    image_size=256,
    truncate_data=10,
)


model = SceneNVSNet()

trainer = pl.Trainer(
    devices=[0, 1],
    accelerator="gpu",
    max_epochs=1,
    precision=16,
    strategy="deepspeed_stage_2",
    enable_progress_bar=True,
    logger=logger,
)
trainer.fit(model, datamodule=datamodule)
