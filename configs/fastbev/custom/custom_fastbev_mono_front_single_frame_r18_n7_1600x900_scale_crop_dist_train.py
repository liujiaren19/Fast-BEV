# -*- coding: utf-8 -*-
"""EXP-MONO-GEOM1-A：N7 原生 1600x900 单目前视完整训练配置。

这是正式 run-of-record 配置，不使用 ``_base_``。模型、数据、图像/BEV
pipeline、优化器、schedule、初始化和 runtime 均在本文件中显式冻结，避免
6V -> temporal mono -> S0 多层继承造成隐式字段漂移。

图像几何固定为 ``1600x900 -> 704x396 -> crop(0,70,704,326)``；训练和
测试采用同一确定性契约，图像随机增强关闭。完整 15 epoch 训练必须在数据、
F4 可见性和 smoke/吞吐门禁通过并获得用户确认后才能启动。
"""

experiment_id = 'EXP-MONO-GEOM1-A'
work_dir = 'work_dirs/n7_mono_1600_900_scale_crop/EXP-MONO-GEOM1-A/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260729'
seed = 0

# -----------------------------------------------------------------------------
# 模型：与 EXP-MONO-S0 epoch13 保持一致，只更换在线输入几何和数据 PKL。
# -----------------------------------------------------------------------------
model = dict(
    type='FastBEV',
    style='v1',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN', requires_grad=True),
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=[256], out_channels=[64]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=256,
        out_channels=192,
        num_layers=2,
        stride=2,
        is_transpose=False,
        fuse=dict(in_channels=256, out_channels=256),
        norm_cfg=dict(type='BN', requires_grad=True)),
    seg_head=None,
    bbox_head=dict(
        type='FreeAnchor3DHead',
        is_transpose=True,
        num_classes=2,
        in_channels=192,
        feat_channels=192,
        num_convs=0,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -35, -1.8, 80, 35, -1.8]],
            sizes=[
                [0.8660, 2.5981, 1.0],
                [0.5774, 1.7321, 1.0],
                [1.0, 1.0, 1.0],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    multi_scale_id=[0],
    n_voxels=[[160, 140, 4]],
    voxel_size=[[0.5, 0.5, 1.5]],
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500,
        use_scale_nms=True,
        use_tta=False,
        nms_across_levels=False,
        use_rotate_nms=True,
        nms_thr=0.2,
        nms_type_list=[
            'rotate', 'rotate', 'rotate', 'rotate', 'rotate',
            'rotate', 'rotate', 'rotate', 'rotate', 'circle'],
        nms_thr_list=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2],
        nms_radius_thr_list=[4, 12, 10, 10, 12, 0.85, 0.85, 0.175, 0.175, 1],
        nms_rescale_factor=[1.0, 0.7, 0.55, 0.4, 0.7, 1.0, 1.0, 4.5, 9.0, 1.0],
        test_mode='test_pth',
        backbone_onnx='simplified_export_2d_model.onnx',
        head_onnx='simplified_export_3d_model.onnx'),
    use_distortion=True,
    n_images=1,
    feature_resize_mode='nearest')

# -----------------------------------------------------------------------------
# 数据和确定性 GEOM1-A pipeline。
# -----------------------------------------------------------------------------
point_cloud_range = [0, -35, -5, 80, 35, 3]
class_names = ['car', 'truck']
camera_types = ['cam0']
dataset_type = 'CustomMultiViewDataset'
data_root = './data/N7_1600_900/'
ann_dir = data_root + 'mono_front_pkl/'
ann_scope = '20251017-20251030-20251031-20251203'
ann_date = '20260728'
ann_file_prefix = f'custom_fastbev_{ann_scope}'
train_ann_file = ann_dir + f'{ann_file_prefix}_infos_train_{ann_date}.pkl'
val_ann_file = ann_dir + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# src_size 只用于声明 K/原图坐标系。pipeline 仍以真实图像头部尺寸计算
# 0.44 resize 和中心 crop，避免依赖离线 704x256 缓存。
data_config = dict(
    src_size=(900, 1600),
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

# 旧嵌套版的 pipeline 是在子配置 merge 前构造的，因此其 step 内没有
# src_size。这里显式保留该有效运行契约，保证扁平化前后 resolved pipeline
# 完全一致；RandomAugImageMultiViewImage 本身也不读取 src_size。
pipeline_data_config = dict(
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
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=1,
        n_times=1,
        transforms=[dict(
            type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D'),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
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
    dict(
        type='RandomAugImageMultiViewImage',
        data_config=pipeline_data_config,
        force_resize=False,
        enable_random_aug=False,
        n_images=1),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='FrontCameraVisibleObjectFilter', n_images=1, min_depth=0.1),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=1,
        n_times=1,
        transforms=[dict(
            type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='RandomAugImageMultiViewImage',
        data_config=pipeline_data_config,
        is_train=False,
        force_resize=False,
        enable_random_aug=False,
        n_images=1),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

dataset_common = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    box_type_3d='LiDAR',
    camera_types=camera_types,
    sequential=False,
    n_times=1,
    train_adj_ids=None,
    test_adj_ids=None,
    max_interval=10,
    min_interval=0,
    eval_iou_thr=[0.25, 0.5],
    eval_range=point_cloud_range,
    filter_gt_visible_camera='cam0',
    filter_gt_visible_min_depth=0.1,
    temporal_compensate=False)

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        **dataset_common,
        pipeline=train_pipeline,
        test_mode=False,
        ann_file=train_ann_file),
    val=dict(
        **dataset_common,
        pipeline=test_pipeline,
        test_mode=True,
        samples_per_gpu=8,
        ann_file=val_ann_file),
    # 当前没有独立 test split；沿用既有口径，明确记录 test=val。
    test=dict(
        **dataset_common,
        pipeline=test_pipeline,
        test_mode=True,
        samples_per_gpu=8,
        ann_file=val_ann_file))

# -----------------------------------------------------------------------------
# 优化、schedule、初始化与 runtime：全部冻结为 S0 正式训练口径。
# -----------------------------------------------------------------------------
optimizer = dict(
    type='AdamW2',
    lr=0.0001,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35.0, norm_type=2))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False)

total_epochs = 15
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# 四卡只训练并逐 epoch 保存；单卡 eval watcher 独立评估 checkpoint。
evaluation = dict(interval=999999, metric=[0.25, 0.5])

dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
fp16 = dict(loss_scale='dynamic')

load_from = (
    'pretrained_models/'
    'cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_'
    'bbox_mAP_0.5110_segm_mAP_0.4070.pth')
resume_from = None
workflow = [('train', 1)]
