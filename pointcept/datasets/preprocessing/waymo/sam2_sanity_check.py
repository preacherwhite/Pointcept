import torch
import numpy as np
import cv2
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import glob

# Disable GPU for TensorFlow
tf.config.set_visible_devices([], 'GPU')

New_SAM = True

# use bfloat16 for the entire notebook
if New_SAM:
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Load Waymo frame
tfrecord_path = '/media/staging1/dhwang/Waymo/archived_files/validation/'
file_list = glob.glob(os.path.join(os.path.abspath(tfrecord_path), "*.tfrecord"))
dataset = tf.data.TFRecordDataset(file_list[0], compression_type='')

# Get first frame and first camera image
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    break

# Get first camera image
images = sorted(frame.images, key=lambda i: i.name)
image = tf.image.decode_jpeg(images[0].image).numpy()  # Replace stock image loading with Waymo image

if New_SAM:
    method = "SAM2"
else:
    method = "SAM1"

start_time1 = time.time()

if New_SAM:
    sam2_checkpoint = "/media/staging2/dhwang/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2AutomaticMaskGenerator(sam2_model, points_per_side=4, min_mask_region_area=300)

else:
    model_type = "vit_b"
    sam_checkpoint = "/media/staging2/dhwang/sam2/checkpoints/sam_vit_b_01ec64.pth"          
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to("cuda")
    predictor = SamAutomaticMaskGenerator(sam)

end_time1 = time.time()
load_time = end_time1 - start_time1
print(f"Loading time ({method}): {load_time} seconds")

input_box = np.array([58,107, 213,281])
input_point = np.array([[104, 163]])
input_label = np.array([1])

start_time2 = time.time()
masks = predictor.generate(image)

end_time2 = time.time()
execution_time = end_time2 - start_time2
print(f"Execution time ({method}): {execution_time} seconds")
# Create output directory if it doesn't exist
output_dir = f"mask_outputs_{method}_{images[0].name}"
os.makedirs(output_dir, exist_ok=True)

# Plot and save each mask
plt.figure(figsize=(10, 10))
for i, mask_data in enumerate(masks):
    # Get mask and create colored overlay
    mask = mask_data['segmentation']
    colored_mask = np.zeros_like(image)
    # Make mask more prominent with brighter red and less original image
    colored_mask[mask] = image[mask] * 0.3 + np.array([255, 0, 0]) * 0.7
    
    # Plot original image with mask overlay
    plt.clf()
    plt.imshow(image)
    plt.imshow(colored_mask, alpha=0.7)  # Increased alpha for more opacity
    
    # Add white border around mask
    mask_border = np.zeros_like(image)
    mask_border[mask] = [255, 255, 255]
    plt.imshow(mask_border, alpha=0.3)
    
    plt.axis('off')
    plt.title(f'Mask {i+1}')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'mask_{i+1}.png'), 
                bbox_inches='tight',
                pad_inches=0)

plt.close()
print(f"Saved {len(masks)} masks to {output_dir}/")


mask_array = np.array(masks[0]) 
