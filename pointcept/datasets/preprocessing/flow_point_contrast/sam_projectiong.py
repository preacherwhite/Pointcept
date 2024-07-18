import os
import glob
import numpy as np
import cv2
import copy
from tqdm import tqdm
def get_sam(masks):
    group_ids = np.full((masks[0].shape[0], masks[0].shape[1]), -1, dtype=int)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        group_ids[masks[i] > 0] = group_counter
        group_counter += 1
    return group_ids

def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array

def get_data_list(data_root):
        flow_files = sorted(glob.glob(os.path.join(data_root, 'flow', '*.png')))
        data_list = []
        for flow_file in flow_files:
            file_id = os.path.splitext(os.path.basename(flow_file))[0]
            data_list.append(file_id)
        return data_list

def transform_masks(data_root):
    # Load point cloud coordinates
    pc1_data = np.load(os.path.join(data_root, 'pc1_outputs.npy'))

    file_ids = get_data_list(data_root)
    for idx in tqdm(range(pc1_data.shape[0])):
        file_id = file_ids[idx]
        # Get the list of mask files for the current scene
        mask_files = sorted(glob.glob(os.path.join(data_root, 'image_sam_masks', file_id, '*.png')))

        # Load all masks for the current scene
        masks = []
        for mask_file in mask_files:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)

        # Get SAM group_ids from the masks
        group_ids = get_sam(masks)
        group_ids = num_to_natural(group_ids)

        # Get point cloud coordinates for the current scene
        coord = pc1_data[idx]

        # Load intrinsics for the current scene
        intrinsics_file = os.path.join(data_root, 'intrinsics', f'{file_id}.npy')
        intrinsics = np.load(intrinsics_file)

        # Transform group_ids to correspond to point cloud coordinates

        f = intrinsics[0]
        cx = intrinsics[1]
        cy = intrinsics[2]

        pc_group_ids = np.full(coord.shape[0], -1, dtype=np.int16)

        #TODO: use indexing from N*2 projection
        for j in range(coord.shape[0]):
            x, y, z = coord[j]
            u = int(f * x / z + cx)
            v = int(f * y / z + cy)

            if 0 <= u < group_ids.shape[1] and 0 <= v < group_ids.shape[0]:
                pc_group_ids[j] = group_ids[v, u]

        # Save the transformed group_ids as .npy file
        output_file = os.path.join(data_root, 'pc_sam_masks', f'{file_id}.npy')
        np.save(output_file, pc_group_ids)

    print("Transformation completed.")


# Specify the path to your data directory
data_root = '/media/staging1/dhwang/kitti_dc_flow'

# Run the transformation
transform_masks(data_root)