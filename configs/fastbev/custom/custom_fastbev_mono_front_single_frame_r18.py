# -*- coding: utf-8 -*-
"""N7 单目前视原生单帧 Fast-BEV 配置。

这份配置用于和 ``custom_fastbev_mono_front_r18.py`` 的真实四时序 B0
做公平对比。首版只把 ``n_times=4`` 改为 ``n_times=1``，保持 cam0、
704x256 stretch、前视 ROI、voxel、anchor、distortion 和 GT 口径不变。
"""

_base_ = './custom_fastbev_mono_front_r18.py'

point_cloud_range = [0, -35, -5, 80, 35, 3]
class_names = ['car', 'truck']
file_client_args = dict(backend='disk')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# 每个时间步的 BEV 输入为 64 feature channels * 4 z = 256 channels。
# 四时序 B0 的 fuse 是 1024->256；原生单帧改为 256->256，不能只修改
# pipeline.n_times，否则会在 neck_3d.fuse 发生通道不匹配。
model = dict(
    n_images=1,
    neck_3d=dict(
        in_channels=256,
        fuse=dict(in_channels=256, out_channels=256)))

data_config = dict(
    input_size=(256, 704),
    resize=(0.0, 0.0),
    crop=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    test_input_size=(256, 704),
    test_resize=0.0,
    test_rotate=0.0,
    test_flip=False,
    pad=(0, 0, 0, 0),
    pad_divisor=32,
    pad_color=(0, 0, 0))

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=False, n_images=1, n_times=1, transforms=[
        dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D'),
    dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='RandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0,
        update_img2lidar=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05],
        update_img2lidar=True),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, force_resize=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='FrontCameraVisibleObjectFilter', n_images=1, min_depth=0.1),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='MultiViewPipeline', sequential=False, n_images=1, n_times=1, transforms=[
        dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='RandomAugImageMultiViewImage',
        data_config=data_config,
        is_train=False,
        force_resize=True),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])
]

single_frame_dataset = dict(
    sequential=False,
    n_times=1,
    train_adj_ids=None,
    test_adj_ids=None,
    temporal_compensate=False)

data = dict(
    train=dict(**single_frame_dataset, pipeline=train_pipeline),
    val=dict(**single_frame_dataset, pipeline=test_pipeline),
    test=dict(**single_frame_dataset, pipeline=test_pipeline))
