import os

import pandas as pd
import streamlit as st
import torchvision

# select mode "dslr" or "iphone"

# select scene


# get all directories in /home/scannet/data
data_dir = "/home/scannet/data/"
scenes = os.listdir(data_dir)


# select scene with dropdown
scene = st.selectbox("Select scene", scenes)

# select mode with radio button
mode = st.radio("Select mode", ("dslr", "iphone"))

if mode == "dslr":
    # load camera data from images.txt
    # /home/scannet/data/41b00feddb/dslr/images.txt
    camera_file = os.path.join(data_dir, scene, "dslr", "colmap", "images.txt")
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    with open(camera_file) as f:
        # read line if does not start with #
        lines = [line for line in f.readlines() if not line.startswith("#")]

    # all even lines are image info
    image_info = lines[::2]
    # all odd lines are point info
    point_info = lines[1::2]

    # create dataframe with image info
    image_info_df = pd.DataFrame([line.split() for line in image_info])
    image_info_df.columns = [
        "image_id",
        "qw",
        "qx",
        "qy",
        "qz",
        "tx",
        "ty",
        "tz",
        "camera_id",
        "name",
    ]

    # /home/scannet/data/41b00feddb/dslr/resized_images
    image_dir = os.path.join(data_dir, scene, "dslr", "resized_images")
    images = os.listdir(image_dir)

    # select image with slider
    image_id = st.select_slider("Select image", options=images)

    # display information about image
    image_info = image_info_df[image_info_df["name"] == image_id]

    # write elements of image_info to sidebar
    for i in image_info.columns:
        st.sidebar.write(i, image_info[i].values[0])

    # display image
    image_path = os.path.join(image_dir, image_id)
    image_data = torchvision.io.read_image(image_path).permute(1, 2, 0).numpy()
    st.image(image_data)

else:
    raise NotImplementedError("Only dslr mode is supported at the moment")
