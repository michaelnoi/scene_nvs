import os

import lightning as pl
import torch
from data.datamodule import Scene_NVSDataModule
from ldm.model import SceneNVSNet
from pytorch_lightning.loggers import WandbLogger

# for GPU
torch.set_float32_matmul_precision("medium")

# TODO: change all to proper shared W&B setup, maybe with config files
project_name = "temp_scene_nvs"
run_name = "debug_image_conditioning"
logger = WandbLogger(project=project_name, name=run_name)

datamodule = Scene_NVSDataModule(
    root_dir="/home/scannet/data/41b00feddb/iphone/",
    batch_size=1,
    num_workers=4,
    image_size=256,
    truncate_data=1,
)

model = SceneNVSNet()

trainer = pl.Trainer(
    devices=[0, 1],
    accelerator="gpu",
    max_epochs=10,
    precision="16-mixed",
    strategy="deepspeed_stage_2",
    # strategy="fsdp",
    enable_progress_bar=True,
    logger=logger,
    log_every_n_steps=1,
)
trainer.fit(model, datamodule=datamodule)

# Clean DeepSpeed checkpoints for debugging (they take up a lot of space)
os.system(f"rm -rf {project_name}/")
