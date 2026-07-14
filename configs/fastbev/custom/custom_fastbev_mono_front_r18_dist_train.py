_base_ = './custom_fastbev_mono_front_r18.py'

model = dict(feature_resize_mode='nearest')

# 4 卡 704x256 训练配置：只覆盖吞吐和训练超参，其余数据路径、
# 类别、模型、时序和 pipeline 继承 704x256 baseline。
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    val=dict(samples_per_gpu=8),
    test=dict(samples_per_gpu=8))

# 从 COCO 2D 预训练完整重跑 mono-front 稳定基线。
# AdamW2 是仓库自定义的解耦 weight decay optimizer；如老环境签名不兼容，再退回标准 AdamW。
optimizer = dict(
    _delete_=True,
    type='AdamW2',
    lr=0.0001,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
# 与 2026-07-08 已完成的四时序 B0 实际训练上限保持一致，也作为后续
# 原生单帧公平对比的 canonical temporal config。
total_epochs = 15
evaluation = dict(interval=5, metric=[0.25, 0.5])
fp16 = dict(loss_scale='dynamic')

# N7 训练默认从 COCO 2D 预训练权重初始化，显式写出以避免误认为从随机权重训练。
load_from = 'pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'
