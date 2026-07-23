# -*- coding: utf-8 -*-
"""N7 6V 四时序 704x256 四卡精度基线配置。"""

_base_ = './custom_fastbev_6v_r18_n7_704x256.py'

model = dict(feature_resize_mode='nearest')

# 本轮完整精度基线使用 20260717 版 6V pkl。当前没有独立 test，
# 因此 data.test 显式复用 val，只能用于 val 复评，不能称为独立 test。
data_root = './data/N7_704_256/'
ann_dir = data_root + '6v_pkl/'
ann_scope = '20251017-20251030-20251031-20251203'
ann_date = '20260717'
ann_file_prefix = f'custom_fastbev_{ann_scope}'

# 4 卡 704x256 训练配置：模型、相机顺序、四时序和 pipeline
# 继续继承 704x256 baseline；这里固定本轮数据与吞吐口径。
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=8,
    train=dict(
        data_root=data_root,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_train_{ann_date}.pkl'),
    val=dict(
        data_root=data_root,
        samples_per_gpu=4,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'),
    test=dict(
        data_root=data_root,
        samples_per_gpu=4,
        ann_file=ann_dir + f'{ann_file_prefix}_infos_val_{ann_date}.pkl'))

# 与 2026-07-17 实际 resolved config 对齐。必须删除论文基类中的 Adam，
# 否则只覆盖 lr 会静默继承标准 Adam，与真实 AdamW2 基线不一致。
optimizer = dict(
    _delete_=True,
    type='AdamW2',
    lr=0.0008,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35.0, norm_type=2))

# 优化器每个 iter 执行 step；by_epoch=False 表示 warmup/poly 学习率按 iter 更新。
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False)

total_epochs = 20
checkpoint_config = dict(interval=1)
# 训练机只训练并逐 epoch 保存；epoch checkpoint 由训练外评估机做 val 复评。
# interval=25 高于 20 epoch 上限，与本轮真实 resolved config 一致。
evaluation = dict(interval=25, metric=[0.25, 0.5])
fp16 = dict(loss_scale='dynamic')

# N7 训练从 COCO 2D 权重做部分初始化，不恢复 optimizer/scheduler/epoch。
load_from = 'pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'
resume_from = None

# 20260717 实际 run 由启动命令传入 ``--seed 0``，deterministic=False。
# work_dir 也应由启动命令指定到新目录，避免无意覆盖已有实验资产。
