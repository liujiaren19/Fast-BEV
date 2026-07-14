# -*- coding: utf-8 -*-
"""N7 单目前视原生单帧 4 卡对比训练配置。"""

_base_ = './custom_fastbev_mono_front_single_frame_r18.py'

model = dict(feature_resize_mode='nearest')

# 与 2026-07-08 四时序 B0 保持相同 sample batch、LR 和 15 epoch 上限。
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    val=dict(samples_per_gpu=8),
    test=dict(samples_per_gpu=8))

optimizer = dict(
    _delete_=True,
    type='AdamW2',
    lr=0.0001,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

total_epochs = 15
checkpoint_config = dict(interval=1)
# 四卡服务器只训练和逐 epoch 保存 checkpoint；validation 由单卡服务器上的
# tools/eval_epoch_checkpoints.py --watch 独立执行。把 interval 放到训练上限之外，
# 避免未传 --no-validate 时四卡训练进程仍执行全量 val。
evaluation = dict(interval=999999, metric=[0.25, 0.5])
fp16 = dict(loss_scale='dynamic')

# 和四时序 B0 使用相同的 2D backbone 初始化口径；单帧 3D neck/head 从头学习。
load_from = 'pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'
