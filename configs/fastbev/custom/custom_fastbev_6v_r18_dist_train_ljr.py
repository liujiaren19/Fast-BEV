_base_ = '../exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py'

model = dict(
    use_distortion=True,
    n_images=6,
    bbox_head=dict(num_classes=2),
)

dataset_type = 'CustomMultiViewDataset'
data_root = './data/N7_704_256/'
ann_prefix = 'custom_fastbev'
# 与 converter 输出命名规则保持一致：{tag}_{scope}_infos_{set}_{YYYYMMDD}.pkl。
# 当前默认按样例单 clip pkl 配置；如果 converter 输出的是 dataset
# 或 sequence scope，只需要同步修改 ann_scope。
ann_scope = '20251017-20251030-20251031-20251203'
ann_date = '20260625'
ann_file_prefix = f'pkl/{ann_prefix}_{ann_scope}'
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = ['car', 'truck']
camera_types = ['cam0', 'cam11', 'cam9', 'cam3', 'cam8', 'cam10']

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
    # N7 标定内参对应 1600x900 图像；即使训练图片已经离线 resize 到
    # 704x256，投影几何也必须按 1600x900 -> 704x256 缩放一次。
    'force_resize_source_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    'test_input_size': (256, 704),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=6, n_times=4, transforms=[
        dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D'),
    dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='RandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05],
        update_img2lidar=True),
    dict(
        type='RandomAugImageMultiViewImage',
        data_config=data_config,
        force_resize=True,
        force_resize_source_size=data_config['force_resize_source_size']),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=6, n_times=4, transforms=[
        dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='RandomAugImageMultiViewImage',
        data_config=data_config,
        is_train=False,
        force_resize=True,
        force_resize_source_size=data_config['force_resize_source_size']),
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
)

data = dict(
    _delete_=True,
    # samples_per_gpu 是每卡 BEV 样本数；每个样本会加载 6 views * 4 times = 24 张图。
    # 因此这里等价于每卡 24 张图、4 卡全局 96 张图。
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        **custom_dataset_common,
        pipeline=train_pipeline,
        test_mode=False,
        ann_file=data_root + f'{ann_file_prefix}_infos_train_{ann_date}.pkl'),
    val=dict(
        **custom_dataset_common,
        pipeline=test_pipeline,
        test_mode=True,
        ann_file=data_root + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'),
    test=dict(
        **custom_dataset_common,
        pipeline=test_pipeline,
        test_mode=True,
        ann_file=data_root + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'))

evaluation = dict(interval=5, metric=[0.25, 0.5])

# 优化器每个 iter 执行 step；by_epoch=False 表示 warmup/poly 学习率也按 iter 更新。
# 当前 samples_per_gpu=1 等价于每卡 24 张图；先使用 base 的 4e-4 稳定收尾。
optimizer = dict(
    type='AdamW2',
    lr=0.0004,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False)
total_epochs = 20
fp16 = dict(loss_scale='dynamic')
