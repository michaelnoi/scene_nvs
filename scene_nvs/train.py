import logging

import hydra
import lightning as pl
import torch
from data.datamodule import Scene_NVSDataModule
from ldm.model import SceneNVSNet
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from utils.checkpoint_cleanup import cleanup_checkpoints
from utils.distributed import rank_zero_print


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

    # init model
    model = SceneNVSNet(cfg)

    # estimate memory usage
    # deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(
    #     model, num_gpus_per_node=2, num_nodes=1
    # )

    # Profiler setup; TODO: debug profiler and optimize stable version of training with it
    # profiler = pl.profiler.SimpleProfiler(
    #     schedule=pl.profiler.ProfilerSchedule(wait=5, warmup=2, active=6, repeat=2),
    #     record_shapes=False,  # Optional: To record tensor shapes
    #     profile_memory=False,  # Optional: To profile memory usage
    #     with_stack=False,
    # )

    trainer_conf = OmegaConf.to_container(cfg.trainer, resolve=True)

    # ds_strategy = DeepSpeedStrategy(**trainer_conf["deepspeed_config"])

    # remove deepspeed from trainer_conf
    # del trainer_conf["deepspeed_config"]
    del trainer_conf["optimizer"]
    checkpoint_callback = ModelCheckpoint(every_n_epochs=100)

    # init trainer, profiler has some overhead and might kill runs
    trainer = pl.Trainer(
        **trainer_conf,
        callbacks=[checkpoint_callback],
        logger=logger,
        profiler=SimpleProfiler("/home/tim/", "logging")
        # ds_strategy,
        # profiler=profiler if cfg.logger.activate_profiler else None,
    )

    # if cfg.model.flex_diffuse.enable:
    #    logger.watch(model.linear_flex_diffuse, log="all", log_freq=10)

    trainer.fit(model, datamodule=datamodule)
    rank_zero_print("Finished training")

    cleanup_checkpoints(cfg.project_name, logger.experiment.id)
    rank_zero_print("Finished cleanup")


if __name__ == "__main__":
    # Set the logging level for PyTorch distributed to only show warnings or higher
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)

    # for GPU
    torch.set_float32_matmul_precision("medium")

    train()
