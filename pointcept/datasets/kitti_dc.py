import os
import glob
import numpy as np
import cv2
from .builder import DATASETS
from .defaults import DefaultDataset

@DATASETS.register_module()
class KITTIdcDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="data/kitti_dc",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        flow_files = sorted(glob.glob(os.path.join(self.data_root, 'flow', '*.png')))
        data_list = []
        for flow_file in flow_files:
            file_id = os.path.splitext(os.path.basename(flow_file))[0]
            data_list.append(file_id)
        return data_list

    def get_data(self, idx):
        file_id = self.data_list[idx % len(self.data_list)]

        # Load point cloud coordinates and RGB values from individual .npy files
        coord_file = os.path.join(self.data_root, 'pc1', f'{file_id}.npy')
        coord = np.load(coord_file)

        color_file = os.path.join(self.data_root, 'pc1_rgb', f'{file_id}.npy')
        color = np.load(color_file)

        # Load 3D flow from individual .npy file
        flow_file = os.path.join(self.data_root, 'flow3d', f'{file_id}.npy')
        flow = np.load(flow_file)

        # Load image
        image_file = os.path.join(self.data_root, 'image', f'{file_id}.png')
        image = cv2.imread(image_file)

        # Load intrinsics
        intrinsics_file = os.path.join(self.data_root, 'intrinsics', f'{file_id}.npy')
        intrinsics = np.load(intrinsics_file)

        data_dict = dict(coord=coord, color=color, flow=flow, image=image, intrinsics=intrinsics)
        return data_dict

    def get_data_name(self, idx):
        return self.data_list[idx % len(self.data_list)]