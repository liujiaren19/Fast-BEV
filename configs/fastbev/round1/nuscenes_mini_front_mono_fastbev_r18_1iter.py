_base_ = '../exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py'

# Round 1: nuScenes mini, true front monocular, smoke-test training/inference.
model = dict(
    n_images=1,
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=True), init_cfg=None),
    neck=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    neck_3d=dict(
        in_channels=64 * 4,
        fuse=dict(in_channels=64 * 4 * 4, out_channels=64 * 4),
        norm_cfg=dict(type='BN', requires_grad=True)))

data_root = './data/nuscenes/'

point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
dataset_type = 'NuScenesMultiView_Map_Dataset2'
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False,
    use_map=False, use_external=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
data_config = dict(
    src_size=(900, 1600),
    input_size=(256, 704),
    resize=(-0.06, 0.11),
    crop=(-0.05, 0.05),
    rot=(-5.4, 5.4),
    flip=True,
    test_input_size=(256, 704),
    test_resize=0.0,
    test_rotate=0.0,
    test_flip=False,
    pad=(0, 0, 0, 0),
    pad_divisor=32,
    pad_color=(0, 0, 0))

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=1, n_times=4,
         transforms=[dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D', with_bev_seg=True),
    dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='RandomFlip3D', flip_2d=False, sync_2d=False,
         flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5,
         update_img2lidar=True),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05], translation_std=[0.05, 0.05, 0.05],
         update_img2lidar=True),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bev_seg'])]

test_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=1, n_times=4,
         transforms=[dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])]

_common_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    with_box2d=False,
    box_type_3d='LiDAR',
    load_interval=999999,
    sequential=True,
    n_times=4,
    train_adj_ids=[1, 3, 5],
    speed_mode='abs_velo',
    max_interval=10,
    min_interval=0,
    fix_direction=True,
    prev_only=True,
    test_adj='prev',
    test_adj_ids=[1, 3, 5],
    test_time_id=None,
    camera_types=['CAM_FRONT'])

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        _delete_=True,
        **_common_dataset,
        pipeline=train_pipeline,
        test_mode=False,
        ann_file='data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl'),
    val=dict(
        _delete_=True,
        **_common_dataset,
        pipeline=test_pipeline,
        test_mode=True,
        ann_file='data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl'),
    test=dict(
        _delete_=True,
        **_common_dataset,
        pipeline=test_pipeline,
        test_mode=True,
        ann_file='data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl'))

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)
lr_config = dict(policy='poly', warmup=None, power=1.0, min_lr=0, by_epoch=False)
checkpoint_config = dict(by_epoch=False, interval=1, max_keep_ckpts=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=999999)
load_from = None
resume_from = None
fp16 = None
find_unused_parameters = False
work_dir = './work_dirs/round1_nuscenes_mini_front_mono_1iter'

optimizer = dict(type="AdamW", lr=0.0004, weight_decay=0.01, paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
