_base_ = './custom_fastbev_6v_r18_n7_704x256.py'

model = dict(feature_resize_mode='nearest')

# 4 卡 704x256 训练配置：只覆盖吞吐和训练超参，其余数据路径、
# 类别、模型、时序和 pipeline 继承 704x256 baseline。
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=8,
    val=dict(samples_per_gpu=4),
    test=dict(samples_per_gpu=4))

# 优化器每个 iter 执行 step；by_epoch=False 表示 warmup/poly 学习率也按 iter 更新。
# 单卡 batch=24、4 卡全局 batch=96 时，收尾训练优先使用 8e-4。
optimizer = dict(lr=0.0008)
total_epochs = 20
fp16 = dict(loss_scale='dynamic')

# N7 训练默认从 COCO 2D 预训练权重初始化，显式写出以避免误认为从随机权重训练。
load_from = 'pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'
