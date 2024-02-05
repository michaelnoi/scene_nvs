import argparse
import os

import numpy as np
import tqdm


def subsample_frames(data_dir):
    frames = sorted(os.listdir(data_dir))
    max_frame = int(frames[-1].split(".")[0].split("_")[-1])
    depth_max_frame = int(
        sorted(os.listdir(data_dir.replace("rgb", "depth")))[-1]
        .split(".")[0]
        .split("_")[-1]
    )
    # sample every 15th frame: get multiple of 15 until max_frame
    depth_max_frame_multiple = depth_max_frame - (depth_max_frame % 15)
    max_frame_multiple = max_frame - (max_frame % 15)
    frame_numbers_to_keep = np.arange(
        0, min(depth_max_frame_multiple, max_frame_multiple) + 1, 15
    )

    rgb_frames = frames
    depth_frames = sorted(os.listdir(data_dir.replace("rgb", "depth")))

    # now check if frame number is in frame_numbers_to_keep, if not, delete it
    for frame in tqdm.tqdm(rgb_frames):
        frame_number = int(frame.split(".")[0].split("_")[-1])
        if frame_number not in frame_numbers_to_keep:
            os.remove(os.path.join(data_dir, frame))

    for frame in tqdm.tqdm(depth_frames):
        frame_number = int(frame.split(".")[0].split("_")[-1])
        if frame_number not in frame_numbers_to_keep:
            os.remove(
                os.path.join(
                    data_dir.replace("rgb", "depth"), frame.replace("jpg", "png")
                )
            )

    assert len(os.listdir(data_dir)) == len(
        os.listdir(data_dir.replace("rgb", "depth"))
    ), "rgb and depth folder do not have same number of frames"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    subsample_frames(args.data_dir)
