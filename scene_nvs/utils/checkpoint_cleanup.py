import os

from lightning.pytorch.utilities.rank_zero import rank_zero_only

# import datetime


@rank_zero_only
def cleanup_checkpoints(project_dir, run_id):
    # get the latest checkpoint file path: # zero-shot-nvs/<e.g. b2pv3lmc>/checkpoints
    run_folder = os.path.join(project_dir, run_id, "checkpoints")

    try:
        # /epoch=XXX-step=XXX.ckpt
        latest_checkpoint = os.path.join(run_folder, os.listdir(run_folder)[0])
        os.remove(os.path.join(latest_checkpoint, "zero_to_fp32.py"))

        # -> just delete optim_states to save space:
        ckpt_dir = os.path.join(latest_checkpoint, "checkpoint")
        for file in os.listdir(ckpt_dir):
            if file.endswith("optim_states.pt"):
                file_path = os.path.join(ckpt_dir, file)
                os.remove(file_path)
    except FileNotFoundError:
        print("No checkpoint found under ", run_folder)

    # -> with deepspeed clean-up file:
    # now = datetime.datetime.now()
    # date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    # save_dir = os.path.join("weights", date_time)
    # os.makedirs(save_dir, exist_ok=True)

    # get the fp32 model checkpoint only and store it in a new folder
    # os.system(
    #     f"python utils/zero_to_fp32.py {latest_checkpoint}  {save_dir}/model_checkpoint.pt"
    # )
    # clean DeepSpeed checkpoints for debugging (they take up a lot of space)
    # os.system(f"rm -rf {run_folder}/")
