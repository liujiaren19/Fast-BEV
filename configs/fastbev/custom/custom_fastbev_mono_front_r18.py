_base_ = './custom_fastbev_6v_r18.py'

model = dict(n_images=1)

dataset_type = 'CustomMultiViewDataset'
data_root = './data/nuscenes/'
ann_prefix = 'custom_fastbev'
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
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
    dict(type='MultiViewPipeline', sequential=True, n_images=1, n_times=4, transforms=[
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
    dict(type='RandomAugImageMultiViewImage', data_config=data_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=1, n_times=4, transforms=[
        dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
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
    train_adj_ids=[1, 3, 5],
    test_adj_ids=[1, 3, 5],
    max_interval=10,
    min_interval=0,
    eval_iou_thr=[0.25, 0.5],
)

data = dict(
    _delete_=True,
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        **custom_dataset_common,
        pipeline=train_pipeline,
        test_mode=False,
        ann_file=data_root + ann_prefix + '_infos_train.pkl'),
    val=dict(
        **custom_dataset_common,
        pipeline=test_pipeline,
        test_mode=True,
        ann_file=data_root + ann_prefix + '_infos_val.pkl'),
    test=dict(
        **custom_dataset_common,
        pipeline=test_pipeline,
        test_mode=True,
        ann_file=data_root + ann_prefix + '_infos_val.pkl'))
