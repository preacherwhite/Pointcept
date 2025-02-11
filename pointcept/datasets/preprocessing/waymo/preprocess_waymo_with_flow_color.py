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
from waymo_open_dataset import dataset_pb2
from tqdm import tqdm

def parse_range_image_and_camera_projection_flow(
    frame: dataset_pb2.Frame):
  """Parse range images and camera projections given a frame.

  Args:
    frame: open dataset frame proto

  Returns:
    range_images: A dict of {laser_name,
      [range_image_first_return, range_image_second_return]}.
    camera_projections: A dict of {laser_name,
      [camera_projection_from_first_return,
      camera_projection_from_second_return]}.
    seg_labels: segmentation labels, a dict of {laser_name,
      [seg_label_first_return, seg_label_second_return]}
    range_image_top_pose: range image pixel pose for top lidar.
  """
  range_images = {}
  camera_projections = {}
  seg_labels = {}
  range_image_top_pose: dataset_pb2.MatrixFloat = dataset_pb2.MatrixFloat()
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_flow_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return1.range_image_flow_compressed, 'ZLIB')
      ri = dataset_pb2.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name] = [ri]

      if laser.name == dataset_pb2.LaserName.TOP:
        range_image_top_pose_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = dataset_pb2.MatrixFloat()
        range_image_top_pose.ParseFromString(
            bytearray(range_image_top_pose_str_tensor.numpy()))

      camera_projection_str_tensor = tf.io.decode_compressed(
          laser.ri_return1.camera_projection_compressed, 'ZLIB')
      cp = dataset_pb2.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name] = [cp]

      if len(laser.ri_return1.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
        seg_label_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.segmentation_label_compressed, 'ZLIB')
        seg_label = dataset_pb2.MatrixInt32()
        seg_label.ParseFromString(bytearray(seg_label_str_tensor.numpy()))
        seg_labels[laser.name] = [seg_label]
    if len(laser.ri_return2.range_image_flow_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return2.range_image_flow_compressed, 'ZLIB')
      ri = dataset_pb2.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name].append(ri)

      camera_projection_str_tensor = tf.io.decode_compressed(
          laser.ri_return2.camera_projection_compressed, 'ZLIB')
      cp = dataset_pb2.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name].append(cp)

      if len(laser.ri_return2.segmentation_label_compressed) > 0:  # pylint: disable=g-explicit-length-test
        seg_label_str_tensor = tf.io.decode_compressed(
            laser.ri_return2.segmentation_label_compressed, 'ZLIB')
        seg_label = dataset_pb2.MatrixInt32()
        seg_label.ParseFromString(bytearray(seg_label_str_tensor.numpy()))
        seg_labels[laser.name].append(seg_label)
  return range_images, camera_projections, seg_labels, range_image_top_pose

def convert_range_image_to_point_cloud_with_flow_and_cp(frame,
                                                        range_images,
                                                        flow_images,
                                                        camera_projections,
                                                        range_image_top_pose,
                                                        ri_index=0,
                                                        keep_polar_features=False):
    """Convert range images to point cloud with corresponding flow vectors and camera projections.

    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
    flow_images: A dict of {laser_name, [flow_image_first_return, flow_image_second_return]}.
    camera_projections: A dict of {laser_name, [camera_projection_from_first_return, camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

    Returns:
    points: {[N, 5]} list of 3d lidar points with intensity and elongation.
    flow_vectors: {[N, 3]} list of 3d flow vectors corresponding to each point.
    cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    flow_vectors = []
    cp_points = []

    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index, keep_polar_features)
    
    cartesian_flow_images = frame_utils.convert_range_image_to_cartesian(
      frame, flow_images, range_image_top_pose, ri_index)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0
        
        range_image_cartesian = cartesian_range_images[c.name]
        flow_image_cartesian = cartesian_flow_images[c.name]

        # Extract x, y, z, intensity, elongation
        points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(range_image_mask))
        
        # Extract flow vectors
        flow_tensor = tf.gather_nd(flow_image_cartesian, tf.compat.v1.where(range_image_mask))

        # Extract camera projections
        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.compat.v1.where(range_image_mask))

        points.append(points_tensor.numpy())
        flow_vectors.append(flow_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())

    return points, flow_vectors, cp_points


def create_lidar_flow_and_color(frame):
    # Regular range images
    (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
    
    # Flow range images
    (flow_images, _, _, _) = parse_range_image_and_camera_projection_flow(frame)

    points, flow_vectors, cp_points = convert_range_image_to_point_cloud_with_flow_and_cp(
        frame, range_images, flow_images, camera_projections, range_image_top_pose, ri_index=0, keep_polar_features=True
    )
    points_ri2, flow_vectors_ri2, cp_points_ri2 = convert_range_image_to_point_cloud_with_flow_and_cp(
        frame, range_images, flow_images, camera_projections, range_image_top_pose, ri_index=1, keep_polar_features=True
    )

    # Concatenate points, flow vectors, and camera projections from both returns
    points_all = np.concatenate(points + points_ri2)
    flow_vectors_all = np.concatenate(flow_vectors + flow_vectors_ri2)
    cp_points_all = np.concatenate(cp_points + cp_points_ri2)

    images = sorted(frame.images, key=lambda i: i.name)
    masks = []
    for image in images:
        mask = tf.equal(cp_points_all[..., 0], image.name)
        masks.append(mask)
    
    # Decode all JPEG images
    decoded_images = [tf.image.decode_jpeg(img.image) for img in images]
    
    # Build the color tensor
    color_tensor = tf.zeros((cp_points_all.shape[0], 3))
    
    for mask, image in zip(masks, decoded_images):
        cp_points_all_tensor_image = tf.cast(tf.gather_nd(cp_points_all, tf.where(mask)), dtype=tf.float32)
        # Convert the cp_points_all_tensor_image to integer indices
        pixel_x = tf.cast(cp_points_all_tensor_image[..., 1], tf.int32)
        pixel_y = tf.cast(cp_points_all_tensor_image[..., 2], tf.int32)
        rgb_values = tf.gather_nd(image, tf.stack([pixel_y, pixel_x], axis=-1))
        # Convert RGB values from 0-255 to 0-1 range
        rgb_values = tf.cast(rgb_values, tf.float32) / 255.0
        color_tensor = tf.tensor_scatter_nd_update(color_tensor, tf.where(mask), rgb_values)
    
    velodyne = np.c_[points_all[:, 3:6], points_all[:, 1]]
    velodyne = velodyne.reshape((velodyne.shape[0] * velodyne.shape[1]))

    flow_vectors_all_flattened = flow_vectors_all.reshape((flow_vectors_all.shape[0] * flow_vectors_all.shape[1]))
    color_tensor_flattened = color_tensor.numpy().reshape((color_tensor.shape[0] * color_tensor.shape[1]))

    return velodyne, flow_vectors_all_flattened, color_tensor_flattened


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


def handle_process(file_path, labels_file_path_root, output_root):
    file = os.path.basename(file_path)
    split = os.path.basename(os.path.dirname(file_path))
    print(f"Parsing {split}/{file}")
    save_path = os.path.join(output_root, split, file.split(".")[0])
    os.makedirs(os.path.join(save_path, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "flow"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "color"), exist_ok=True)
    if split != "testing":
        os.makedirs(os.path.join(save_path, "labels"), exist_ok=True)

    data_group = tf.data.TFRecordDataset(file_path, compression_type="")
    labels_file_path = os.path.join(
        labels_file_path_root,
        split,
        os.path.basename(file_path)
    )
    labels_data_group = tf.data.TFRecordDataset(labels_file_path, compression_type="")
    count = 0
    # Delete all files in labels folder if it exists
    # Check if number of processed files matches data group size
    velodyne_dir = os.path.join(save_path, "velodyne")
    flow_dir = os.path.join(save_path, "flow") 
    color_dir = os.path.join(save_path, "color")
    
    # Count files in each directory
    velodyne_count = len([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    flow_count = len([f for f in os.listdir(flow_dir) if f.endswith('.bin')])
    color_count = len([f for f in os.listdir(color_dir) if f.endswith('.bin')])
    
    # Count total frames in data group
    data_count = sum(1 for _ in data_group)
    
    # Reset data group iterator
    data_group = tf.data.TFRecordDataset(file_path, compression_type="")
    # Skip if all counts match and files exist
    if velodyne_count == flow_count == color_count == data_count and data_count > 0:
        print(f"Skipping {split}/{file} - already processed {data_count} frames")
        return
    else:
        print("--------------------------------")
        print("number didn't match")
        print(f"velodyne: {velodyne_count}, flow: {flow_count}, color: {color_count}, data: {data_count}")
    labels_dir = os.path.join(save_path, "labels")
    if os.path.exists(labels_dir):
        for file in os.listdir(labels_dir):
            os.remove(os.path.join(labels_dir, file))
    for data, labels_data in zip(data_group, labels_data_group):
        file_idx = "0" * (6 - len(str(count))) + str(count)
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        labels_frame = open_dataset.Frame()
        labels_frame.ParseFromString(bytearray(labels_data.numpy()))
        point_cloud, flow_vectors, color = create_lidar_flow_and_color(frame)
        # Overwrite existing files by first removing them if they exist
        velodyne_file = os.path.join(save_path, "velodyne", f"{file_idx}.bin")
        flow_file = os.path.join(save_path, "flow", f"{file_idx}.bin") 
        color_file = os.path.join(save_path, "color", f"{file_idx}.bin")

        if os.path.exists(velodyne_file):
            os.remove(velodyne_file)
        if os.path.exists(flow_file):
            os.remove(flow_file)
        if os.path.exists(color_file):
            os.remove(color_file)

        point_cloud.astype(np.float32).tofile(velodyne_file)
        flow_vectors.astype(np.float32).tofile(flow_file)
        color.astype(np.float32).tofile(color_file)
        if labels_frame.lasers[0].ri_return1.segmentation_label_compressed and split != "testing":
            label = create_label(labels_frame)
            label.tofile(os.path.join(save_path, "labels", f"{file_idx}.label"))
        count += 1
    print(f"Finished processing {split}/{file}, count: {count}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--dataset_root_labels",
        required=True,
        help="Path to the waymo data with segmentation labels",
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
        choices=["training", "validation", "testing"],
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
    assert len(file_list) == 1000

    # Create output directories
    for split in config.splits:
        os.makedirs(os.path.join(config.output_root, split), exist_ok=True)

    file_list = [
        file
        for file in file_list
        if os.path.basename(os.path.dirname(file)) in config.splits
    ]

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(tqdm(
        pool.map(handle_process, file_list, repeat(config.dataset_root_labels), repeat(config.output_root)),
        total=len(file_list),
        desc="Overall Progress"
    ))