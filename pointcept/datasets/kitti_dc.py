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
        
        # Load large numpy files during initialization
        self.pc1_data = np.load(os.path.join(self.data_root, 'pc1_outputs.npy'))
        self.pc1_rgb_data = np.load(os.path.join(self.data_root, 'pc1_rgb_outputs.npy'))
        self.flow_3d_data = np.load(os.path.join(self.data_root, 'flow3d_outputs.npy'))
        self.flow_3d_data = np.transpose(self.flow_3d_data, (0,2,1))

    def get_data_list(self):
        flow_files = sorted(glob.glob(os.path.join(self.data_root, 'flow', '*.png')))
        data_list = []
        for flow_file in flow_files:
            file_id = os.path.splitext(os.path.basename(flow_file))[0]
            data_list.append(file_id)
        return data_list

    def get_data(self, idx):
        file_id = self.data_list[idx % len(self.data_list)]
        
        # Get point cloud coordinates and RGB values from loaded data
        coord = self.pc1_data[idx]
        color = self.pc1_rgb_data[idx]
        
        # Get 3D flow from loaded data
        flow = self.flow_3d_data[idx]
        
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