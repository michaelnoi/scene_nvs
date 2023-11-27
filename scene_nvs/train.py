import datetime
import logging
import os

import hydra
import lightning as pl
import torch
from data.datamodule import Scene_NVSDataModule
from ldm.model import SceneNVSNet
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from utils.distributed import rank_zero_print


@rank_zero_only
def cleanup_checkpoints(project_dir):
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("weights", date_time)
    os.makedirs(save_dir, exist_ok=True)

    # get the latest checkpoint file path
    run_folder = os.path.join(project_dir, os.listdir(project_dir)[0])
    checkpoint_folder = os.path.join(run_folder, os.listdir(run_folder)[0])
    latest_checkpoint = os.path.join(
        checkpoint_folder, os.listdir(checkpoint_folder)[0]
    )

    # get the fp32 model checkpoint only and store it in a new folder
    os.system(
        f"python utils/zero_to_fp32.py {latest_checkpoint}  {save_dir}/model_checkpoint.pt"
    )

    # clean DeepSpeed checkpoints for debugging (they take up a lot of space)
    os.system(f"rm -rf {run_folder}/")


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

    # Profiler setup
    profiler = PyTorchProfiler(
        schedule=torch.profiler.schedule(wait=5, warmup=2, active=6, repeat=2),
        record_shapes=False,  # Optional: To record tensor shapes
        profile_memory=False,  # Optional: To profile memory usage
        with_stack=False,
    )

    # init trainer, profiler has some overhead and might kill runs
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=logger,
        profiler=profiler if cfg.logger.activate_profiler else None,
    )

    trainer.fit(model, datamodule=datamodule)
    rank_zero_print("Finished training")

    # NOTE: unfortunately, we can't change deepspeed checkpointing directory with lightning
    # TODO: sort by creation date and so that the correct checkpoint is saved if previous run was interrupted
    cleanup_checkpoints(cfg.project_name)
    rank_zero_print("Finished cleanup")


if __name__ == "__main__":
    # Set the logging level for PyTorch distributed to only show warnings or higher
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)

    # for GPU
    torch.set_float32_matmul_precision("medium")

    train()
