_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16 # bs: total bs in all gpus
num_worker = 4
mix_prob = 0
empty_cache = False
enable_amp = False
evaluate = False
find_unused_parameters = False

# model settings
model = dict(
    type="FPC-v1",
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        # pdnorm_bn=False,
        # pdnorm_ln=False,
        # pdnorm_decouple=True,
        # pdnorm_adaptive=False,
        # pdnorm_affine=True,
        # pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    flow_similarity_threshold=0.8,
    color_similarity_threshold=0.7,
    proximity_threshold=0.5,
    flow_weight=0.3,
    color_weight=0.3,
    proximity_weight=0.3,
    sam_weight=0.1,
)

# scheduler settings
epoch = 50
eval_epoch = 50
global_lr = 0.01

optimizer = dict(type="AdamW", lr=global_lr, weight_decay=0.0005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[global_lr, global_lr/10],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=global_lr/10)]

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
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            #dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size= 0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "flow","sam"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color","flow","sam"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size= 0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "flow","sam"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color","flow","sam",'image','intrinsics'),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=True,
        
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

# Tester
test = dict(type="ClusterSegTester", verbose=True)