_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4 # bs: total bs in all gpus
num_worker = 32
mix_prob = 0
empty_cache = False
enable_amp = False
evaluate = False
find_unused_parameters = True

# model settings
model = dict(
    type="FPC-v1",
    backbone=dict(type="MinkUNet34C", in_channels=3, out_channels=64),
    flow_similarity_threshold=0.8,
    color_similarity_threshold=0.7,
    proximity_threshold=0.5,
    flow_weight=1,
    color_weight=1,
    proximity_weight=1,
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0002, 0.00002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "KITTIdcDataset"
data_root = "/media/staging1/dhwang/kitti_dc_flow"  # Update with your dataset path
ignore_index = -1

data = dict(
    ignore_index=ignore_index,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        transform=[
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size= 0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "flow"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color","flow"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]