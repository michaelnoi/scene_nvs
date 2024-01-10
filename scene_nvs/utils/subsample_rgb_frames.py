import argparse
import os

import numpy as np
import tqdm


def main(args):
    frames = sorted(os.listdir(args.data_dir))
    max_frame = int(frames[-1].split(".")[0].split("_")[-1])
    # sample every 15th frame
    # get max frame number out of string 'frame_number.jpg'

    frame_numbers_to_keep = np.arange(0, max_frame, 15)

    # now check if frame number is in frame_numbers_to_keep
    # if not, delete it
    for frame in tqdm.tqdm(frames):
        frame_number = int(frame.split(".")[0].split("_")[-1])
        if frame_number not in frame_numbers_to_keep:
            os.remove(os.path.join(args.data_dir, frame))
            os.remove(
                os.path.join(
                    args.data_dir.replace("rgb", "depth"), frame.replace("jpg", "png")
                )
            )

            print("removed frame", frame)

    # get multiple of 15 until max_frame

    print(max_frame)

    # print(frames)
    # for frame in tqdm.tqdm(frames):
    #    os.remove(os.path.join(args.data_dir, frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/home/scannet/data/41b00feddb/iphone/rgb/"
    )
    args = parser.parse_args()
    main(args)
