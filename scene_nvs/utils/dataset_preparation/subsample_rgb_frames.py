import argparse
import os

import numpy as np
import tqdm


def subsample_frames(data_dir):
    frames = sorted(os.listdir(data_dir))
    max_frame = int(frames[-1].split(".")[0].split("_")[-1])
    # sample every 15th frame: get multiple of 15 until max_frame
    max_frame = max_frame - (max_frame % 15)
    frame_numbers_to_keep = np.arange(0, max_frame + 1, 15)

    # now check if frame number is in frame_numbers_to_keep, if not, delete it
    for frame in tqdm.tqdm(frames):
        frame_number = int(frame.split(".")[0].split("_")[-1])
        if frame_number not in frame_numbers_to_keep:
            os.remove(os.path.join(data_dir, frame))
            os.remove(
                os.path.join(
                    data_dir.replace("rgb", "depth"), frame.replace("jpg", "png")
                )
            )
            print("removed frame", frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    subsample_frames(args.data_dir)
