import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import tensorflow as tf
import os
import argparse
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import glob
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Tell TensorFlow to only use GPU 3
    try:
        # Hide all GPUs from TensorFlow except GPU 3
        # Adjust the index based on which GPU you want TensorFlow to use
        tf.config.set_visible_devices(gpus[3], 'GPU')
        # Allow memory growth to avoid taking all GPU memory
        tf.config.experimental.set_memory_growth(gpus[3], True)
    except RuntimeError as e:
        print(e)

# Configure PyTorch to use GPU 0
torch.cuda.set_device(0)

def create_sam_masks(image, mask_generator):
    """Generate SAM2 masks for a given image.
    
    Args:
        image: RGB image tensor
    
    Returns:
        mask_array: Array of shape (H, W) where each pixel contains its mask ID
    """
    # Convert TensorFlow tensor to NumPy array for PyTorch processing
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    masks = mask_generator.generate(image)
    
    # Convert list of masks to single array where each pixel contains mask ID
    H, W = image.shape[:2]
    mask_array = np.zeros((H, W), dtype=np.int32)
    
    for idx, mask_data in enumerate(masks, start=1):
        mask = mask_data["segmentation"]
        mask_array[mask] = idx
        
    return mask_array

def create_lidar_masks(frame, mask_generator):
    """Modified version of create_lidar_flow_and_color that also handles SAM masks"""
    # Get regular data first
    (
        range_images,
        camera_projections,
        _,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        keep_polar_features=False,
    )
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1,
        keep_polar_features=False,
    )

    # Concatenate points from both returns
    points_all = np.concatenate(points + points_ri2)
    cp_points_all = np.concatenate(cp_points + cp_points_ri2)

    images = sorted(frame.images, key=lambda i: i.name)
    masks = []
    sam_masks = []
    
    # Generate SAM masks for each camera image
    for image in images:
        decoded_image = tf.image.decode_jpeg(image.image).numpy()
        mask = create_sam_masks(decoded_image, mask_generator)
        sam_masks.append(mask)
        masks.append(tf.equal(cp_points_all[..., 0], image.name))

    # Project SAM masks to point cloud
    mask_tensor = tf.zeros((cp_points_all.shape[0],), dtype=tf.int32)
    
    for mask, sam_mask, image in zip(masks, sam_masks, images):
        cp_points_image = tf.cast(tf.gather_nd(cp_points_all, tf.where(mask)), dtype=tf.int32)
        pixel_x = cp_points_image[..., 1]
        pixel_y = cp_points_image[..., 2]
        
        # Get mask values for each projected point
        mask_values = tf.gather_nd(sam_mask, tf.stack([pixel_y, pixel_x], axis=-1))
        mask_tensor = tf.tensor_scatter_nd_update(
            mask_tensor, tf.where(mask), mask_values
        )

    mask_values = mask_tensor.numpy()
    return points_all, mask_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to save the output .npy file",
    )
    parser.add_argument(
        "--tf_gpu",
        type=int,
        default=3,
        help="GPU index for TensorFlow to use (default: 3)",
    )
    parser.add_argument(
        "--torch_gpu",
        type=int,
        default=0,
        help="GPU index for PyTorch to use (default: 0)",
    )
    config = parser.parse_args()

    # Configure GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[config.tf_gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[config.tf_gpu], True)
    
    torch.cuda.set_device(config.torch_gpu)
    
    # SAM2 setup
    checkpoint="/media/staging2/dhwang/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "sam2.1_hiera_s.yaml"
    device = torch.device(f"cuda:{config.torch_gpu}")
    
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(config.torch_gpu).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    
    # Process single frame
    tfrecord_path = '/media/staging1/dhwang/Waymo/archived_files/validation/'
    file_list = glob.glob(
            os.path.join(os.path.abspath(tfrecord_path), "*.tfrecord")
        )
    # Read the TFRecord file
    dataset = tf.data.TFRecordDataset(file_list[0], compression_type='')

    
    # Get first frame only
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        break
    # Process the frame
    points, sam_masks = create_lidar_masks(frame, mask_generator)
    
    # Save both points and masks
    output_base = os.path.splitext(config.output_file)[0]
    np.save(f"{output_base}_masks.npy", sam_masks)
    np.save(f"{output_base}_points.npy", points)
    print(f"Saved masks to {output_base}_masks.npy")
    print(f"Saved points to {output_base}_points.npy")
        