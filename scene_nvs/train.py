import logging

import deepspeed
import hydra
import lightning as pl
import torch
from data.datamodule import Scene_NVSDataModule
from ldm.model import SceneNVSNet
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from utils.checkpoint_cleanup import cleanup_checkpoints
from utils.distributed import rank_zero_print

seed_everything(42, workers=True)


@rank_zero_only
def estimate_memory_usage(model):
    deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(
        model, num_gpus_per_node=2, num_nodes=1
    )
    deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(
        model, num_gpus_per_node=2, num_nodes=1
    )


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
        render_cfg=cfg.render,
        **OmegaConf.to_container(cfg.datamodule, resolve=True),
    )

    # init model
    model = SceneNVSNet(cfg)

    if cfg.image_embeds_to_disk:
        datamodule.setup()
        model.save_all_image_embeddings(datamodule)
        model.del_vision_encoder()
        assert model.vision_encoder is None

    estimate_memory_usage(model)

    trainer_conf = OmegaConf.to_container(cfg.trainer, resolve=True)

    # ds_strategy = DeepSpeedStrategy(**trainer_conf["deepspeed_config"])

    # remove deepspeed from trainer_conf
    # del trainer_conf["deepspeed_config"]
    del trainer_conf["optimizer"]
    checkpoint_callback = ModelCheckpoint(every_n_epochs=1)
    # device_stats = DeviceStatsMonitor(cpu_stats=True)

    # init trainer, profiler has some overhead and might kill runs
    trainer = pl.Trainer(
        **trainer_conf,
        deterministic=True,
        callbacks=[checkpoint_callback],  # , device_stats],
        logger=logger,
        strategy=DeepSpeedStrategy(
            zero_optimization=True,
            stage=2,
            overlap_comm=False,
            contiguous_gradients=True,
            logging_batch_size_per_gpu=1,
        ),
        profiler=(
            SimpleProfiler(cfg.logger.profiling_dir, "simple")
            if cfg.logger.activate_profiler
            else None
        ),
        # ds_strategy,
    )

    # log number of samples in train and val
    # logger.config["train_samples"] = len(datamodule.train_dataloader().dataset)
    # logger.config["val_samples"] = len(datamodule.val_dataloader().dataset)

    if cfg.model.flex_diffuse.enable:
        logger.watch(model.linear_flex_diffuse, log="all", log_freq=50)

    if cfg.model.depth_conditioning.enable:
        logger.watch(model.unet.conv_in, log="all", log_freq=50)
    #    logger.watch(model.depth_feature_extractor, log="all", log_freq=50)

    if cfg.model.dreampose_adapter.enable:
        logger.watch(model.dreampose_adapter, log="all", log_freq=50)

    logger.watch(model.pose_projection, log="all", log_freq=50)

    # ,ckpt_path=cfg.model.from_ckpt_path)
    # estimate memory usage

    train = True
    if train:
        trainer.fit(model, datamodule=datamodule)
        rank_zero_print("Finished training")

        cleanup_checkpoints(cfg.project_name, logger.experiment.id)
        rank_zero_print("Finished cleanup")
    else:
        trainer.validate(
            model, datamodule=datamodule, ckpt_path=cfg.model.from_ckpt_path
        )
        rank_zero_print("Finished Sampling")


if __name__ == "__main__":
    # Set the logging level for PyTorch distributed to only show warnings or higher
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)

    # for GPU
    torch.set_float32_matmul_precision("medium")

    train()
