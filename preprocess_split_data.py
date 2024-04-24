import os
import numpy as np
from tqdm import tqdm

base_dir = '/media/staging1/dhwang/kitti_dc_flow/'
# Define the paths to the input .npy files
flow3d_path = base_dir + "flow3d_outputs.npy"
pc1_path = base_dir + "pc1_outputs.npy"
pc1_rgb_path = base_dir + "pc1_rgb_outputs.npy"

# Create the output directories if they don't exist
os.makedirs("/media/staging1/dhwang/kitti_dc_flow/pc1", exist_ok=True)
os.makedirs("/media/staging1/dhwang/kitti_dc_flow/pc1_rgb", exist_ok=True)
os.makedirs("/media/staging1/dhwang/kitti_dc_flow/flow3d", exist_ok=True)

# Load the .npy files
flow3d_data = np.load(flow3d_path)
flow3d_data = np.transpose(flow3d_data, (0, 2, 1))
pc1_data = np.load(pc1_path)
pc1_rgb_data = np.load(pc1_rgb_path)

# Get the number of samples
num_samples = flow3d_data.shape[0]

# Iterate over each sample and save as separate .npy files
for i in tqdm(range(num_samples)):
    # Save flow3d data
    flow3d_sample = flow3d_data[i]
    flow3d_filename = os.path.join("flow3d", f"{str(i).zfill(6)}.npy")
    flow3d_filename = os.path.join('/media/staging1/dhwang/kitti_dc_flow', flow3d_filename)
    np.save(flow3d_filename, flow3d_sample)

    # Save pc1 data
    pc1_sample = pc1_data[i]
    pc1_filename = os.path.join("pc1", f"{str(i).zfill(6)}.npy")
    pc1_filename = os.path.join('/media/staging1/dhwang/kitti_dc_flow', pc1_filename)
    np.save(pc1_filename, pc1_sample)

    # Save pc1_rgb data
    pc1_rgb_sample = pc1_rgb_data[i]
    pc1_rgb_filename = os.path.join("pc1_rgb", f"{str(i).zfill(6)}.npy")
    pc1_rgb_filename = os.path.join('/media/staging1/dhwang/kitti_dc_flow', pc1_rgb_filename)
    np.save(pc1_rgb_filename, pc1_rgb_sample)

print("Data stored successfully!")