import os
import shutil

from prepare_iphone_data import main as extract_frames
from subsample_rgb_frames import subsample_frames
from tqdm import tqdm

FULL_DATASET_PATH = "/home/data_hdd/scannet/data"  # cold storage
SCANNET_PATH = "/home/scannet/data"


def main():
    done = []
    for scene in tqdm(os.listdir(FULL_DATASET_PATH)):
        # copy iphone folder to scannet folder
        iphone_path = os.path.join(FULL_DATASET_PATH, scene, "iphone")
        if not os.path.exists(iphone_path):
            raise ValueError("iphone path does not exist")
        try:
            shutil.copytree(iphone_path, os.path.join(SCANNET_PATH, scene, "iphone"))
        except FileExistsError:
            print(f"iphone of scene {scene} folder already exists")
            continue

        # unpack rgb and depth frames
        cfg = {
            "extract_rgb": True,
            "extract_masks": True,
            "extract_depth": True,
            "data_root": "/home/scannet/",
            "scene_ids": [scene],
        }
        # make sure to apply the scannet++ pyenv environment here and have ffmpeg installed
        extract_frames(cfg)

        # subsample rgb and depth frames to every 15th frame
        iphone_path_new = os.path.join(SCANNET_PATH, scene, "iphone")
        subsample_frames(os.path.join(iphone_path_new, "rgb"))

        done.append(scene)

    print("Scenes done: ", done)
    print("Number of scenes transformed:", len(done))


if __name__ == "__main__":
    main()
