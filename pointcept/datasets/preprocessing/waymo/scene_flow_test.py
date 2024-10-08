"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def create_lidar(frame):
    """Parse and save the lidar data in psd format.
    Args:
        frame (:obj:`Frame`): Open dataset frame proto.
    """
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        keep_polar_features=True,
    )
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1,
        keep_polar_features=True,
    )

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # point labels.

    points_all = np.concatenate([points_all, points_all_ri2], axis=0)

    velodyne = np.c_[points_all[:, 3:6], points_all[:, 1]]
    velodyne = velodyne.reshape((velodyne.shape[0] * velodyne.shape[1]))
    return velodyne


def create_label(frame):
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels
    )
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels, ri_index=1
    )

    # point labels.
    point_labels_all = np.concatenate(point_labels, axis=0)
    point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
    point_labels_all = np.concatenate([point_labels_all, point_labels_all_ri2], axis=0)

    labels = point_labels_all
    return labels


def convert_range_image_to_point_cloud_labels(
    frame, range_images, segmentation_labels, ri_index=0
):
    """Convert segmentation labels from range images to point clouds.

    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
    points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def handle_process(file_path, output_root):
    file = os.path.basename(file_path)
    split = os.path.basename(os.path.dirname(file_path))
    print(f"Parsing {split}/{file_path}")

    data_group = tf.data.TFRecordDataset(file_path, compression_type="")
    count = 0
    for data in data_group:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        range_images, camera_projectsions, seg_labels, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(
            frame
        )
        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projectsions, range_image_top_pose
        )
        print(points[0].shape)
        print("---------------end-----------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--splits",
        required=True,
        nargs="+",
        choices=["train", "valid"],
        help="Splits need to process ([training, validation, testing]).",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    config = parser.parse_args()

    # load file list
    file_list = glob.glob(
        os.path.join(os.path.abspath(config.dataset_root), "*", "*.tfrecord")
    )

    file_list = [
        file
        for file in file_list
        if os.path.basename(os.path.dirname(file)) in config.splits
    ]

    # Preprocess data.
    print("Processing scenes...")
    handle_process(file_list[0], config.output_root)
