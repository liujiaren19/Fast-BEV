_base_ = './custom_fastbev_6v_r18.py'

front_roi = [0, -35, -5, 80, 35, 3]
front_n_voxels = [160, 140, 4]
front_voxel_size = [0.5, 0.5, 1.5]

model = dict(
    n_images=1,
    feature_resize_mode='nearest',
    n_voxels=[front_n_voxels],
    voxel_size=[front_voxel_size],
    bbox_head=dict(
        anchor_generator=dict(ranges=[[0, -35, -1.8, 80, 35, -1.8]])),
)

dataset_type = 'CustomMultiViewDataset'
data_root = './data/N7_704_256/'
ann_dir = data_root + 'mono_front_pkl/'
ann_prefix = 'custom_fastbev'
# 与 converter 输出命名规则保持一致：{tag}_{scope}_infos_{set}_{YYYYMMDD}.pkl。
# 当前默认按样例单 clip pkl 配置；如果 converter 输出的是 dataset
# 或 sequence scope，只需要同步修改 ann_scope。
ann_scope = '20251017-20251030-20251031-20251203'
ann_date = '20260703'
ann_file_prefix = f'{ann_prefix}_{ann_scope}'
point_cloud_range = front_roi
class_names = ['car', 'truck']
camera_types = ['cam0']
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

data_config = {
    'input_size': (256, 704),
    # mono-front 使用 704x256 离线缓存图，RandomAugImageMultiViewImage
    # 通过 force_resize=True 只做确定性 resize/K 缩放。图像随机增强在
    # 该模式下不会生效；BEV/3D 增强由 RandomFlip3D/GlobalRotScaleTrans 控制。
    'resize': (0.0, 0.0),
    'crop': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'test_input_size': (256, 704),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=1, n_times=4, transforms=[
        dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D'),
    dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='RandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        # 单目前视 ROI 只覆盖 x>=0，前后翻转会把有效目标翻到车后方。
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
    dict(type='MultiViewPipeline', sequential=True, n_images=1, n_times=4, transforms=[
        dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False, force_resize=True),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])
]

custom_dataset_common = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    box_type_3d='LiDAR',
    camera_types=camera_types,
    sequential=True,
    n_times=4,
    train_adj_ids=[0, 1, 2],
    test_adj_ids=[0, 1, 2],
    max_interval=10,
    min_interval=0,
    eval_iou_thr=[0.25, 0.5],
    eval_range=point_cloud_range,
    filter_gt_visible_camera='cam0',
    filter_gt_visible_min_depth=0.1,
)

data = dict(
    _delete_=True,
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        **custom_dataset_common,
        pipeline=train_pipeline,
        test_mode=False,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_train_{ann_date}.pkl'),
    val=dict(
        **custom_dataset_common,
        pipeline=test_pipeline,
        test_mode=True,
        samples_per_gpu=8,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'),
    test=dict(
        **custom_dataset_common,
        pipeline=test_pipeline,
        test_mode=True,
        samples_per_gpu=8,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'))

fp16 = dict(loss_scale='dynamic')
