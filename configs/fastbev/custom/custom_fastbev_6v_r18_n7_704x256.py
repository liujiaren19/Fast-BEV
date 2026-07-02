# -*- coding: utf-8 -*-
"""N7 704x256 离线缓存图像训练配置。

这份配置用于 data/N7_704_256 数据根：
- 图片已经离线从 1600x900 强制 resize 到 704x256。
- info_json / pkl 中的相机内参仍保持 1600x900 标定，不要预先缩放 K。
- 新版 pkl 必须写入 intrinsic_width/height 和 image_width/height；
  RandomAugImageMultiViewImage 会按 intrinsic_* -> input_size 写入 post_rot，
  保证几何只缩放一次。
"""

_base_ = './custom_fastbev_6v_r18.py'

model = dict(feature_resize_mode='nearest')

# 704x256 缓存数据根。converter 生成 pkl 时建议使用：
#   --data-path data/N7_704_256 --output-dir data/N7_704_256/pkl
# 这样 pkl 内图片路径相对 data/N7_704_256，训练时可直接解析。
data_root = './data/N7_704_256/'
ann_dir = data_root + 'pkl/'

# 默认值仅用于示例；正式训练时按 converter 输出文件名修改 ann_scope/ann_date，
# 或在训练脚本里用 --cfg-options 覆盖 ann_file。
ann_prefix = 'custom_fastbev'
ann_scope = '20251031_164821_1'
ann_date = '20260624'
ann_file_prefix = f'{ann_prefix}_{ann_scope}'

# 704x256 缓存图仍使用原始 1600x900 标定 K，因此这里显式覆盖
# pipeline，开启 force_resize 写入 intrinsic_* -> input_size 的 post_rot。
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = ['car', 'truck']
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
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, force_resize=True),
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
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False, force_resize=True),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])
]

# 覆盖数据根、pkl 路径和 704 缓存图专用 pipeline。
data = dict(
    train=dict(
        data_root=data_root,
        pipeline=train_pipeline,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_train_{ann_date}.pkl'),
    val=dict(
        data_root=data_root,
        pipeline=test_pipeline,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'),
    test=dict(
        data_root=data_root,
        pipeline=test_pipeline,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_test_{ann_date}.pkl'))
