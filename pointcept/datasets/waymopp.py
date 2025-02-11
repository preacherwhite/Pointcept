"""
Waymo dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class WaymoPlusPlusDataset(DefaultDataset):
    def __init__(
        self,
        split="training",
        data_root="/media/staging1/dhwang/Waymo/pointcept_derived_flow_color",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            self.split = [self.split]

        data_list = []
        for split in self.split:
            data_list += glob.glob(
                os.path.join(self.data_root, split, "*", "velodyne", "*.bin")
            )
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = np.tanh(scan[:, -1].reshape([-1, 1]))

        # Read flow data
        flow_file = data_path.replace("velodyne", "flow")
        if os.path.exists(flow_file):
            with open(flow_file, "rb") as f:
                flow = np.fromfile(f, dtype=np.float32).reshape(-1, 3)
        else:
            flow = np.zeros((scan.shape[0], 3), dtype=np.float32)

        # Read color data
        color_file = data_path.replace("velodyne", "color")
        if os.path.exists(color_file):
            with open(color_file, "rb") as c:
                color = np.fromfile(c, dtype=np.float32).reshape(-1, 3)
        else:
            color = np.zeros((scan.shape[0], 3), dtype=np.float32)

        # Read SAM masks
        sam_file = data_path.replace("velodyne", "sam_masks")
        if os.path.exists(sam_file):
            with open(sam_file, "rb") as s:
                sam_labels = np.fromfile(s, dtype=np.int32)
        else:
            sam_labels = np.zeros(scan.shape[0], dtype=np.int32)

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = (
                    np.fromfile(a, dtype=np.int32).reshape(-1, 2)[:, 1] - 1
                )  # ignore_index 0 -> -1
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)

        data_dict = dict(
            coord=coord, 
            strength=strength, 
            flow=flow, 
            color=color, 
            segment=segment,
            sam=sam_labels
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name