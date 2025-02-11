import torch
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import tensorflow as tf
import os
import glob
import argparse
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Add near the top, before the main code
mp.set_start_method('spawn', force=True)

# Configure TensorFlow to use CPU only and suppress warnings
# List all physical devices
gpus = tf.config.list_physical_devices('GPU')

# Disable all GPUs
if gpus:
    try:
        tf.config.set_visible_devices([], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e) # Disable GPU for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging (0=all, 1=INFO, 2=WARNING, 3=ERROR)

# Add these global variables near the top
sam_models = {}
mask_generators = {}

# Function to initialize and return a SAM model for each process
def initialize_sam_model(gpu_id):
    """Initialize and return a SAM model on a specified GPU."""
    checkpoint = "/media/staging2/dhwang/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "sam2.1_hiera_b+.yaml"
    device = torch.device(f"cuda:{gpu_id}")
    
    sam_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam_model, points_per_side=8,points_per_batch=128)
    return mask_generator

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

    _, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        keep_polar_features=True,
    )
    _, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1,
        keep_polar_features=True,
    )

    # Concatenate points from both returns
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
    return mask_values

def handle_process(file_path, output_root, gpu_id):
    """Process a single file on specified GPU."""
    # Clear GPU memory cache
    torch.cuda.empty_cache()
    
    mask_generator = initialize_sam_model(gpu_id)  # Initialize model for each process
    file = os.path.basename(file_path)
    split = os.path.basename(os.path.dirname(file_path))
    print(f"Parsing {split}/{file} on GPU {gpu_id}")
    save_path = os.path.join(output_root, split, file.split(".")[0])
    os.makedirs(os.path.join(save_path, "sam_masks"), exist_ok=True)
    
    data_group = tf.data.TFRecordDataset(file_path, compression_type="")
    for count, data in enumerate(data_group):
        file_idx = str(count).zfill(6)
        output_file = os.path.join(save_path, "sam_masks", f"{file_idx}.bin")
        if os.path.exists(output_file):
            continue
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        sam_masks = create_lidar_masks(frame, mask_generator)
        sam_masks.astype(np.int32).tofile(output_file)

if __name__ == "__main__":
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA devices available")
    print(f"Found {available_gpus} CUDA devices")
    for i in range(available_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="/media/staging1/dhwang/Waymo/archived_files/",
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        default="/media/staging1/dhwang/Waymo/pointcept_derived_flow_color/",
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
        default=4,
        type=int,
        help="Number of concurrent processes (default: 4)",
    )
    parser.add_argument(
        "--gpu_list",
        default=[0,1,2,3],
        nargs='+',
        type=int,
        help="List of GPU IDs to use (default: [0,1,2,3])",
    )
    config = parser.parse_args()

    # Configure TensorFlow to use CPU only
    tf.config.set_visible_devices([], 'GPU')
    
    # load file list
    file_list = glob.glob(
        os.path.join(os.path.abspath(config.dataset_root), "*", "*.tfrecord")
    )
    print(len(file_list))
    
    # Create output directories
    for split in config.splits:
        os.makedirs(os.path.join(config.output_root, split), exist_ok=True)

    file_list = [
        file
        for file in file_list
        if os.path.basename(os.path.dirname(file)) in config.splits
    ]
    
    # Distribute work across processes with GPU assignments
    print("Processing scenes...")
    with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
        # Create list of GPU assignments cycling through available GPUs
        gpu_assignments = [config.gpu_list[i % len(config.gpu_list)] for i in range(len(file_list))]
        
        # Map work to processes with progress bar
        _ = list(tqdm(
            pool.map(handle_process, file_list, repeat(config.output_root), gpu_assignments),
            total=len(file_list),
            desc="Overall Progress",
            position=1
        ))