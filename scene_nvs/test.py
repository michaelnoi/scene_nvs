import logging

import deepspeed
import hydra
import lightning as pl
import torch
from data.datamodule import Scene_NVSDataModule
from ldm.model import SceneNVSNet
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    # init wandb logger
    project_name = cfg.project_name
    run_name = cfg.run_name
    logger = WandbLogger(
        project=project_name,
        name=run_name,
        # kwargs passed to wandb.init:
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.tags,
    )

    # init datamodule
    datamodule = Scene_NVSDataModule(
        **OmegaConf.to_container(cfg.datamodule, resolve=True)
    )

    # init model from checkpoint
    model = SceneNVSNet(cfg)
    inference_engine = deepspeed.init_inference(
        model,
        dtype=torch.float16,
        checkpoint=cfg.model.from_ckpt_path,
    )
    model = inference_engine.module

    # init trainer, inference always on single GPU
    cfg.trainer.devices = [0]
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True), logger=logger
    )

    # validate calls validation dataset
    # TODO: set up metrics in validation step
    trainer.validate(model, datamodule=datamodule)
    # trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    # Set the logging level for PyTorch distributed to only show warnings or higher
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)

    # for GPU
    torch.set_float32_matmul_precision("medium")

    train()
