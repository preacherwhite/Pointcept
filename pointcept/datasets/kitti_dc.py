import os
import glob
import numpy as np
import cv2
from .builder import DATASETS
from .defaults import DefaultDataset
from copy import deepcopy
import random
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
            special_test=False,
            loop=loop,
        )
        # Load large numpy files during initialization
        self.pc1_data = np.load(os.path.join(self.data_root, 'pc1_outputs.npy'))
        self.pc1_rgb_data = np.load(os.path.join(self.data_root, 'pc1_rgb_outputs.npy'))
        self.flow_3d_data = np.load(os.path.join(self.data_root, 'flow3d_outputs.npy'))
        self.flow_3d_data = np.transpose(self.flow_3d_data, (0,2,1))
        
        self.index_list = None
        self.data_list = self.get_data_list()
        
        
    def get_data_list(self):
        flow_files = sorted(glob.glob(os.path.join(self.data_root, 'flow', '*.png')))
        data_list = []
        for flow_file in flow_files:
            file_id = os.path.splitext(os.path.basename(flow_file))[0]
            data_list.append(file_id)
        
        if self.split in ['val', 'test']:
            total_samples = len(data_list)
            step = total_samples // 20 
            self.index_list = list(range(0, total_samples, step))[:20] 
            data_list = [data_list[i] for i in self.index_list]
        return data_list
    
    def get_index_list(self):
        
        return None
    
    def get_data(self, idx):

        file_id = self.data_list[idx]
        if self.split in ['val', 'test']:
            idx = self.index_list[idx]
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
        
        # Get SAM 3D labels
        sam_3d_label_file = os.path.join(self.data_root, 'pc_sam_masks', f'{file_id}.npy')
        sam_3d_labels = np.load(sam_3d_label_file)
        
        data_dict = dict(coord=coord, color=color, flow=flow, image=image, intrinsics=intrinsics, sam=sam_3d_labels)
        return data_dict

    def get_data_name(self, idx):
        return self.data_list[idx % len(self.data_list)]

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict