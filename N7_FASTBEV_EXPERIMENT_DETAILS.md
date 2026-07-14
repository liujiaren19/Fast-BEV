# N7 Fast-BEV 实验详细记录

> 更新时间：2026-07-13
>
> 汇总入口：[N7_FASTBEV_EXPERIMENT_SUMMARY.md](N7_FASTBEV_EXPERIMENT_SUMMARY.md)
>
> 当前执行清单：[N7_MONO_FRONT_SINGLE_FRAME_TODO.md](N7_MONO_FRONT_SINGLE_FRAME_TODO.md)

## 1. 文档目的和边界

本文把当前能够找到的 N7 Fast-BEV 实验证据按时间顺序整理为一条可追溯链路：

1. N7 6V 四时序可训练基线；
2. 6V 动态畸变/backproject 性能消融；
3. 单目前视四时序模型侧 smoke；
4. 单 sequence 收敛实验；
5. full-data 早期不稳定实验；
6. 2026-07-08 正式四时序 B0；
7. 当前原生单帧 S0 的配置和对比协议。

当前本地环境没有 N7 数据、内网 checkpoint、逐 epoch result pkl，也没有可运行旧版 mmcv/mmdet/mmdet3d 的 legacy 训练栈。因此：

- 本文可以确认日志中已经发生的训练、保存和评估；
- 可以确认当前仓库代码和配置状态；
- 不能在本机确认历史 PTH 实体、SHA256、内网 manifest/pkl/calibration；
- 不能把当前工作树直接等同于任一历史训练时的精确代码快照；
- 需要内网补录的内容会明确标记，不用猜测填充。

## 2. 证据和口径

### 2.1 主证据

| 类型 | 路径 | 主要内容 |
| --- | --- | --- |
| 6V 训练日志 | /workspace/20260626_174629.log | 6V resolved config、环境、epoch1～3 部分训练 |
| 单 sequence 日志 | /workspace/20260703_160930.log | 单目四时序 20 epoch 收敛和 e5/e10/e15/e20 eval |
| full-data 初始日志 | /workspace/20260703_172933.log | full-data epoch1～10、e5 训练内 eval |
| vflip0 clean 日志 | /workspace/20260706_184052.log | full-data clean epoch1～4 |
| epoch2 load-from 日志 | /workspace/20260707_152258.log | 低 LR 权重加载实验 resolved config 和 epoch1～5 部分 |
| 正式 B0 日志 | /workspace/20260708_122556.log | B0 resolved config、epoch1～15、e5/e10/e15 eval |
| B0 逐 epoch 复评 | /workspace/eval_summary.md | epoch1～15 center/BEV/TP error/score/xyz 分桶 |
| 6V 性能笔记 | [N7_FASTBEV_TIMING_ABLATION_20260630.md](N7_FASTBEV_TIMING_ABLATION_20260630.md) | 动态畸变与 backproject 性能消融 |
| 单目适配审查前记录 | [N7_MONO_FRONT_ADAPTATION_REVIEW.md](N7_MONO_FRONT_ADAPTATION_REVIEW.md) | 早期 collapse 指标和原因分析 |
| 代码审查计划 | [N7_MONO_FRONT_AUDIT_FIX_PLAN.md](N7_MONO_FRONT_AUDIT_FIX_PLAN.md) | 结构、数据、部署和指标问题 |
| 当前代办 | [N7_MONO_FRONT_SINGLE_FRAME_TODO.md](N7_MONO_FRONT_SINGLE_FRAME_TODO.md) | 单帧产品路线和优先级 |

### 2.2 指标定义

- mAP：canonical 指标。先在每个类别内对 0.5m、1m、2m、4m center-distance AP 求平均得到 `center_AP`，再对有 GT 的类别求平均。
- `mAP/center_dist`：与 `mAP` 数值相同的历史兼容键；tools/eval_epoch_checkpoints.py 的新默认 best key 是 `mAP`。
- BEV mAP@0.5：BEV IoU 阈值 0.5 的两类平均 AP。
- 历史 schema v1 JSON 中裸 `mAP` 曾等于 BEV mAP@0.5；新代码读取旧记录时优先使用 `mAP/center_dist`，不会把旧裸 `mAP` 当成 canonical mAP。
- ATE/AOE/ASE：只在 center distance ≤2m 的 matched TP 上统计。
- xyz signed mean：有正负号，会发生抵消；不能当作绝对定位误差。
- xyz MAE：mean(abs(pred−gt))，表示各轴平均绝对偏移量，是面向下游的主展示指标。
- xyz p50/p90：当前汇总中的绝对误差分位数，更适合观察误差量级。

### 2.3 跨实验比较限制

以下差异会使早期指标不能和 B0 做严格横向结论：

- 单 sequence val 的 truck GT 为 0；
- 早期 mono 继承过 vertical flip；
- 早期 full-data 和 B0 使用不同 LR/optimizer；
- CustomMultiViewDataset 的 BEV box corner yaw 约定后来修过，修复前后 BEV AP 不应视为完全同口径；
- box bottom-center/gravity-center 修复会改变旧 xyz z 统计；
- 某些历史外部 eval 只保留在审查笔记，本机没有原始 epoch_N_metrics.json；
- 当前代码仍有未提交改动，训练日志没有固化完整 git diff/hash。

## 3. 代码和实验时间线

| 时间 | 里程碑 | 结果 |
| --- | --- | --- |
| 2026-06-26 | EXP-6V-00 启动 | 6V N7 full-data 可训练，至少完成 2 epoch |
| 2026-06-27 | SMK-MONO-00 | nuScenes mini 单目四时序 1 iter/test forward 跑通 |
| 2026-06-30 | ABL-6V-01 | 确认动态畸变/backproject 是 6V 主耗时 |
| 2026-07-02 | 6V 基线整理 | commit 8cc8678 形成 6V trainable baseline；mono 从此基线继续 |
| 2026-07-03 | EXP-MONO-T1 | 单 sequence 20 epoch 收敛 |
| 2026-07-03～07-05 | EXP-MONO-T2 | full-data 高 LR 实验后期退化 |
| 2026-07-06～07-07 | EXP-MONO-T3 | vertical flip 关闭后 clean 高 LR 重跑，仍出现后期 collapse 记录 |
| 2026-07-07～07-08 | EXP-MONO-T4 | 从 T3 epoch2 只加载权重、降低 LR |
| 2026-07-08～07-11 | EXP-MONO-B0 | 从 COCO 初始化训练 15 epoch，epoch5 为最佳 |
| 2026-07-11～07-13 | 审查修复 | box origin、ONNX、eval 可视化和原生单帧配置完成代码侧修复 |
| 2026-07-13 | box-origin 重评 | 用户复用已有 result pkl 重评，确认 car/truck z 偏差恢复正常 |
| 待内网 | EXP-MONO-S0 | 原生单帧训练、逐 epoch eval 和 B0 公平对比 |

主要 git 节点：

- 73b5bcf：适配 N7 自采数据到 Fast-BEV 训练链路；
- 2a64a51：适配 N7 自采数据训练与评估链路；
- 237487b：清理 N7 Fast-BEV 适配训练链路；
- 8cc8678：适配 N7 6V FastBEV 可训练基线；
- 当前分支：feature/n7-mono-front-from-6v-baseline；
- 当前工作树有大量用户已有未提交改动，不能把 HEAD 8cc8678 当作 B0 的完整代码版本。

## 4. 公共环境和模型结构

### 4.1 内网训练环境

完整头部日志确认的主要环境：

- Python 3.8.19；
- PyTorch 1.10.0+cu113；
- CUDA 11.3；
- MMCV 1.4.0；
- MMDetection 2.14.0；
- MMDetection3D 0.16.0+；
- 正式 full-data 训练使用 4×NVIDIA L20；
- dynamic fp16；
- random seed=0，deterministic=False。

EXP-MONO-T1 使用 1×L20。SMK-MONO-00 使用 2080 Ti legacy 环境。

### 4.2 公共模型骨干

6V、四时序 mono 和首版单帧 S0 共同保留：

- model.type=FastBEV；
- style=v1；
- ResNet18 backbone；
- FPN 输出 64 channels；
- neck_fuse 256→64；
- M2BevNeck；
- FreeAnchor3DHead；
- car、truck 两类；
- feature_resize_mode=nearest；
- use_distortion=True；
- 2D backbone 使用同一 COCO 初始化；
- 3D neck/head 并没有完整 Fast-BEV 预训练，主要从头学习。

初始化权重路径：

pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth

### 4.3 6V、四时序 mono、单帧 mono 的结构差异

| 项目 | 6V 四时序 | mono 四时序 B0 | mono 单帧 S0 |
| --- | ---: | ---: | ---: |
| n_images | 6 | 1 | 1 |
| n_times | 4 | 4 | 1 |
| 每样本图片数 | 24 | 4 | 1 |
| 相机 | cam0/cam11/cam9/cam3/cam8/cam10 | cam0 | cam0 |
| ROI | [-50,-50,-5,50,50,3] | [0,-35,-5,80,35,3] | 同 B0 |
| n_voxels | [200,200,4] | [160,140,4] | 同 B0 |
| voxel_size | [0.5,0.5,1.5] | [0.5,0.5,1.5] | 同 B0 |
| 每时间步 BEV channel | 64×4=256 | 64×4=256 | 64×4=256 |
| temporal fuse 输入 | 4×256=1024 | 4×256=1024 | 256 |
| fuse 输出 | 256 | 256 | 256 |
| sequential | True | True | False |
| pose compensation | 有历史帧 dataset 变换 | 有 | 无 |

关键点：相机数决定每个时间步如何聚合视图，时间数决定 3D fusion 输入。6V 改成 mono 不等于 4 时序改成单帧，因此 B0 的 1024→256 不能因为 n_images=1 而缩为 256→256；只有 S0 的 n_times=1 才这样修改。

## 5. SMK-MONO-00：模型侧单目四时序 smoke

### 5.1 目的

在真实 N7 mono pkl 长训练前，验证 active FastBEV 路径不再硬编码六相机，并确认 n_images=1、n_times=4 可以经过 dataset、dataloader、train forward 和 test forward。

### 5.2 已确认结果

- 数据：nuScenes mini，不是 N7；
- 远端历史根目录：/root/Fast-BEV；
- dataloader image shape：torch.Size([4,3,256,704])；
- 单 GPU 训练 1 iter：loss=4.0493；
- 单次 test forward 完成并写出 test_results.pkl；
- 历史 work_dir：
  work_dirs/round1_nuscenes_mini_front_mono_1iter_2080ti_smoke

### 5.3 结论

此实验只证明模型结构和 metadata slicing 能跑通，不提供 N7 精度结论。它支撑了后续从 6V baseline 进入 N7 mono pkl 验证。

## 6. EXP-6V-00：N7 6V 四时序可训练基线

### 6.1 目标

验证 N7 自定义数据、704×256 离线缓存图、六相机四时序、动态畸变和 FreeAnchor 训练链路可以在 4 卡 L20 上运行。

### 6.2 resolved config

对应训练配置：

configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256_dist_train.py

| 项目 | 值 |
| --- | --- |
| 相机 | cam0、cam11、cam9、cam3、cam8、cam10 |
| n_images / n_times | 6 / 4 |
| 输入 | 704×256，force_resize=True |
| distortion | True |
| ROI | [-50,-50,-5,50,50,3] |
| n_voxels | [200,200,4] |
| voxel_size | [0.5,0.5,1.5] |
| train batch | 24/GPU × 4 GPU = 96 |
| workers | 8/GPU |
| optimizer | AdamW2 |
| LR | 8e-4；backbone lr_mult=0.1 |
| schedule | poly，1000 iter linear warmup，by_epoch=False |
| epoch 上限 | 20 |
| checkpoint | 每 epoch |
| in-training eval | 每 5 epoch |
| fp16 | dynamic |
| BEV flip | horizontal=0.5，vertical=0.5 |

### 6.3 数据路径

data_root：

./data/N7_704_256/

train pkl：

./data/N7_704_256/pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_train_20260625.pkl

val/test pkl：

./data/N7_704_256/pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_val_20260625.pkl

### 6.4 work_dir 和权重

历史绝对 work_dir：

/mnt/liujiaren/fastbev-python/work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260626

日志确认：

- epoch_1.pth 已保存；
- epoch_2.pth 已保存；
- epoch3 训练只记录到 630/1858；
- 没有到第 5 epoch，因此本日志没有 6V eval。

### 6.5 训练采样

下表是每个 epoch 最后一个已记录日志点，不是整 epoch 平均：

| epoch | iter | LR | loss | time/iter | max memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1850/1858 | 7.602e-4 | 1.1033 | 34.337s | 30,578MB |
| 2 | 1850/1858 | 7.202e-4 | 0.8506 | 26.725s | 30,578MB |
| 3 | 630/1858 | 7.065e-4 | 0.7731 | 27.307s | 30,578MB |

日志采样中出现过 1 次 grad_norm=inf。dynamic fp16 可能跳过对应 step，但现有日志不足以统计真实 overflow/skip-step 总数。

### 6.6 结论和限制

- 已证明 6V full-data 可以训练至少两个完整 epoch；
- loss 有明显下降；
- 没有 6V 验证指标，不能称为完成的精度 baseline；
- 不能声称存在 epoch20 权重；
- 不能用 loss 跨 6V/mono 直接比较模型能力，因为输入、ROI、GT 数和 batch 都不同；
- 需内网补录 epoch1/2 checkpoint 实体、SHA256 和是否存在后续未同步日志。

## 7. ABL-6V-01：6V 动态畸变和 backproject 性能消融

### 7.1 背景

6V 每 GPU 每 iter：

- samples_per_gpu=24；
- n_times=4；
- 每时间步聚合 6 camera；
- 每 GPU 每 iter 有 24×4=96 次 volume/backproject；
- 每次处理 200×200×4=160,000 个 BEV 点；
- 合计约 15.4M 投影点/GPU/iter。

因此单次投影的几十毫秒会被 96 次调用放大。

### 7.2 主要结果

| 实验 | 条件 | 代表速度 | 结论 |
| --- | --- | ---: | --- |
| 初始动态畸变 | use_distortion=True，旧逐相机实现 | 约 20～22s/iter | 主要瓶颈在 volume/backproject |
| 关闭畸变 | use_distortion=False | 约 6.8～7.2s/iter | 动态畸变是主要耗时，但此配置改变几何，不可作最终精度训练 |
| nearest feature resize | 在无畸变基础上 | 无明显收益 | 2D resize 不是主瓶颈 |
| batched distortion + fused scatter | use_distortion=True | 约 16s/iter | 有效优化，但动态投影仍是剩余大头 |

stage profile 代表值：

| 阶段 | 时间 |
| --- | ---: |
| backbone_neck_fuse | 约 0.36～0.41s |
| volume_total | 约 4.40～5.24s |
| backproject | 约 4.16～4.99s |
| neck3d | 约 0.09～0.14s |

### 7.3 保留到主线的修改

- 训练侧 3D head 输入使用 4D seq-major 布局，与导出/板端约定对齐；
- deploy/head calibration 输入只在需要时生成；
- no-distortion 使用 R@xyz+t；
- fused final-camera scatter；
- batched dynamic distortion projection；
- dynamic self.n_images metadata slicing。

### 7.4 没有保留的实验代码

- 临时 stage/backproject profiler；
- 没有效果的浅拷贝优化；
- 把 fixed-LUT 读写逻辑直接塞进普通训练 forward；
- 旧 fastbev_ljr.py 的硬编码 camera order、绝对路径和 6V 假设。

### 7.5 结论

性能优化没有给出精度指标，不能当作模型效果实验。use_distortion=False 只适合吞吐排查；正式 6V/mono baseline 均继续 use_distortion=True。

## 8. EXP-MONO-T1：单 sequence 四时序收敛实验

### 8.1 目的

在较小的单 sequence 数据上验证 N7 mono-front 训练可以完整收敛，并观察 20 epoch loss 和训练内 eval。

### 8.2 resolved config

| 项目 | 值 |
| --- | --- |
| GPU | 1×L20 |
| n_images / n_times | 1 / 4 |
| 相机 | cam0 |
| ROI | [0,-35,-5,80,35,3] |
| n_voxels | [160,140,4] |
| train batch | 64/GPU |
| optimizer | Adam |
| LR | 4e-4 |
| total epochs | 20 |
| eval interval | 5 |
| vertical flip | 0.5，尚未修复 |
| iteration/epoch | 90 |

train pkl：

./data/N7_704_256/mono_front_pkl/custom_fastbev_20251031_164821_infos_train_20260703.pkl

val/test pkl：

./data/N7_704_256/mono_front_pkl/custom_fastbev_20251031_164821_infos_val_20260703.pkl

历史 work_dir：

/mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251031_164821_10_batch24_work8_20260703

目录名中的 batch24 是旧命名残留；日志 resolved config 明确 samples_per_gpu=64。

### 8.3 loss

| epoch | 末次采样 loss | time/iter | memory |
| ---: | ---: | ---: | ---: |
| 1 | 3.0209 | 2.210s | 22,214MB |
| 5 | 0.8183 | 1.761s | 22,217MB |
| 10 | 0.6049 | 1.734s | 22,217MB |
| 15 | 0.4792 | 1.709s | 22,217MB |
| 20 | 0.3911 | 1.679s | 22,217MB |

日志确认 epoch1～20 checkpoint 均保存。采样日志出现过 1 次 grad_norm=inf。

### 8.4 eval

| epoch | car BEV AP@0.5 | truck BEV AP@0.5 | BEV mAP@0.5 | car GT | truck GT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.3301 | 0 | 0.3301 | 3,416 | 0 |
| 10 | 0.3242 | 0 | 0.3242 | 3,416 | 0 |
| 15 | **0.3415** | 0 | **0.3415** | 3,416 | 0 |
| 20 | 0.2795 | 0 | 0.2795 | 3,416 | 0 |

### 8.5 结论

- 结构和训练链路可以完整跑 20 epoch；
- loss 能持续下降；
- e15 的 BEV AP 在四个评估点中最高，e20 已回落；
- val 中 truck GT=0，因此它本质是单类/单场景 debug，不可与 full-data B0 数值直接比较；
- vertical flip 当时仍为 0.5，这轮不能作为最终 mono baseline。

## 9. EXP-MONO-T2：full-data 初始高 LR 实验

### 9.1 证据完整性

/workspace/20260703_172933.log 是从 runner 启动段之后开始的续写日志，没有完整 resolved config 头。以下分开记录：

- 日志确认：work_dir、epoch1～10、LR 曲线、loss、e5 in-training eval；
- 历史记录确认：该阶段配置仍继承 vertical flip=0.5，初始 LR 约 8e-4；
- 待内网补录：该 work_dir 中保存的 resolved config 和每个 checkpoint hash。

### 9.2 work_dir

/mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260703

日志确认 epoch1～10 均保存。

### 9.3 loss

| epoch | 末次采样 LR | 末次采样 loss | time/iter | memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.923e-4 | 1.1670 | 17.233s | 22,296MB |
| 2 | 7.203e-4 | 1.1873 | 17.636s | 22,296MB |
| 5 | 6.003e-4 | 1.0709 | 17.367s | 22,296MB |
| 10 | 4.003e-4 | 0.9751 | 17.842s | 22,297MB |

采样日志中记录到 7 次 grad_norm=inf；需要注意这只是日志间隔内能看到的次数，不是完整 skip-step 统计。

### 9.4 已确认 eval

epoch5 训练内 eval：

| 指标 | car | truck | overall |
| --- | ---: | ---: | ---: |
| BEV AP@0.25 | 0.3503 | 0.0281 | 0.1892 |
| BEV AP@0.5 | 0.1905 | 0.0042 | 0.0974 |
| center AP@0.5m | 0.0601 | 0.0008 | 0.0305 |
| GT | 69,492 | 15,747 | 85,239 |
| det | 2,737,823 | 5,383,580 | 8,121,403 |

已有交接记录还确认：对 epoch10/latest 做显式外部 eval 时，BEV mAP@0.5 约为 1.44e-06，百万级检测中几乎没有 score>0.2 的有效预测。该外部 eval 原始 JSON 当前不在本机，因此标为历史记录。

### 9.5 初步原因

- front ROI 是 x∈[0,80]；
- vertical BEV flip 会翻转 LiDAR x；
- 继承 6V 的 vertical flip=0.5 会把前视 GT 翻到 x<0；
- 后续 ObjectRangeFilter 会删除或破坏这些目标；
- 高 LR、dynamic fp16 overflow 和 FreeAnchor score calibration 也可能共同放大不稳定。

这轮结果推动了 vertical flip=0 的 clean 重跑。

## 10. EXP-MONO-T3：vflip0 clean 高 LR 重跑

### 10.1 work_dir

/mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260706_vflip0_clean

### 10.2 已确认配置和训练

- 4×L20；
- 64/GPU；
- full-data 785 iter/epoch；
- vertical flip=0；
- 高 LR 路线，日志 LR 曲线与 8e-4 上限一致；
- 本机日志确认 epoch1～4 保存。

| epoch | 末次采样 LR | 末次采样 loss | time/iter | memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.923e-4 | 1.1348 | 17.536s | 22,296MB |
| 2 | 7.203e-4 | 1.1524 | 17.467s | 22,296MB |
| 3 | 6.803e-4 | 1.1108 | 17.827s | 22,296MB |
| 4 | 6.403e-4 | 1.0700 | 18.218s | 22,297MB |

### 10.3 历史逐 epoch 复评

N7_MONO_FRONT_ADAPTATION_REVIEW.md 记录了另一版 full-data clean/checkpoint 逐 epoch结果。结合 vflip0_clean work_dir 和下一轮从其 epoch2 load_from 的关系，本文将其关联到 T3；但本机缺少原始 metrics JSON，内网需要再核对一次。

| epoch | BEV mAP@0.5 | car AP@0.5 | truck AP@0.5 |
| ---: | ---: | ---: | ---: |
| 1 | 0.2709 | 0.4011 | 0.1407 |
| **2** | **0.2897** | **0.4055** | **0.1739** |
| 3 | 0.2296 | 0.3267 | 0.1326 |
| 4 | 0.1510 | 0.2476 | 0.0544 |
| 5 | 0.0000 | 0.0000 | 0.0000 |

历史 epoch5 score：

- total det=14,996；
- truck label 12,178，全部在 0.05～约 0.052，score≥0.1 为 0；
- car label 2,818，全部在 0.05～约 0.056，score≥0.1 为 0。

### 10.4 结论

- 关闭 vertical flip 是必要正确性修复，但不是稳定训练的充分条件；
- 高 LR clean run 仍有后期 score collapse 记录；
- epoch2 被选作下一轮低 LR load-from 起点；
- 这轮的 epoch5=0 绝对不能和 EXP-MONO-B0 的 best epoch5 混称为“epoch5 实验”。

## 11. EXP-MONO-T4：从 T3 epoch2 权重降低 LR

### 11.1 关键语义

配置是：

- load_from=T3/epoch_2.pth；
- resume_from=None。

因此这是**只加载模型权重并重新初始化 optimizer/scheduler**，不是恢复 epoch 计数、optimizer、scheduler 和 fp16 loss scale 的 resume。work_dir 名称虽然包含 resume_epoch2，语义仍以 resolved config 为准。

### 11.2 resolved config

| 项目 | 值 |
| --- | --- |
| GPU / batch | 4×L20；64/GPU |
| optimizer | Adam |
| LR | 2e-4 |
| total epochs | 8 个新 epoch |
| eval interval | 5 |
| vertical flip | 0 |
| image resize/crop/rot/flip | 固定 resize；随机图像增强关闭 |
| fp16 | dynamic |

初始化权重：

work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260706_vflip0_clean/epoch_2.pth

输出 work_dir：

/mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260706_vflip0_clean_resume_epoch2

### 11.3 训练记录

| 新 epoch | iter | 末次采样 loss | time/iter | memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 780/785 | 0.9660 | 18.155s | 22,297MB |
| 2 | 780/785 | 0.9294 | 18.042s | 22,297MB |
| 3 | 780/785 | 0.8899 | 17.682s | 22,297MB |
| 4 | 780/785 | 0.8440 | 18.409s | 22,297MB |
| 5 | 700/785 | 0.8321 | 18.571s | 22,297MB |

本机日志只确认新 epoch1～4 保存，epoch5 尚未跑完/保存，没有完整 eval。它说明低 LR 下 loss 能继续下降，但没有形成可用于产品对比的正式指标。

## 12. EXP-MONO-B0：2026-07-08 正式四时序基线

### 12.1 为什么把它定为 B0

这轮满足：

- vertical flip 已关闭；
- 从统一 COCO 2D 权重重新开始，不继承 collapse checkpoint；
- LR 降为 1e-4；
- 15 epoch 完整训练；
- 每个 epoch checkpoint 可用；
- 已通过独立脚本对 epoch1～15 全部复评；
- 指标和 score 分布未出现早期 collapse。

### 12.2 resolved config

对应训练配置：

configs/fastbev/custom/custom_fastbev_mono_front_r18_dist_train.py

| 项目 | 值 |
| --- | --- |
| GPU | 4×L20 |
| camera | cam0 |
| n_images / n_times | 1 / 4 |
| 每样本输入 | 4 张 704×256 图 |
| ROI | [0,-35,-5,80,35,3] |
| n_voxels | [160,140,4] |
| voxel_size | [0.5,0.5,1.5] |
| anchor range | [0,-35,-1.8,80,35,-1.8] |
| distortion | True |
| feature resize | nearest |
| sequential / adjacent | True；[0,1,2] |
| GT 过滤 | class + front ROI + cam0 pinhole-visible |
| train batch | 64/GPU × 4 = 256 |
| val/test batch | 8/GPU |
| optimizer | AdamW2 |
| LR | 1e-4；backbone lr_mult=0.1 |
| weight decay | 0.01 |
| schedule | poly；1000 iter linear warmup；by_epoch=False |
| total epochs | 15 |
| checkpoint | 每 epoch |
| train hook eval | 每 5 epoch |
| fp16 | dynamic |
| BEV flip | horizontal=0.5；vertical=0 |

### 12.3 数据

train pkl：

./data/N7_704_256/mono_front_pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_train_20260703.pkl

val/test pkl：

./data/N7_704_256/mono_front_pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_val_20260703.pkl

注意：当前 test 仍指向 val pkl，尚未形成独立日期/车辆 test split。

### 12.4 work_dir 和 best 权重

历史 work_dir：

/mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260708

日志确认 epoch_1.pth～epoch_15.pth 均保存。

当前 best：

/mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260708/epoch_5.pth

best 选择依据：

mAP/center_dist=0.344934

待内网补录：

- epoch5 PTH SHA256；
- 实际 resolved config 文件；
- B0 使用的代码 commit/diff；
- pkl、manifest、calibration SHA256；
- epoch_N_val_results.pkl 和 epoch_N_metrics.json 的归档路径。

### 12.5 训练 loss

下表为每个 epoch 最后一个已记录点：

| epoch | LR | loss | time/iter | memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.275e-5 | 1.4288 | 18.088s | 22,296MB |
| 2 | 8.672e-5 | 1.0241 | 18.593s | 22,296MB |
| 3 | 8.005e-5 | 0.8866 | 18.101s | 22,296MB |
| 4 | 7.338e-5 | 0.8131 | 18.263s | 22,296MB |
| 5 | 6.672e-5 | 0.7919 | 18.312s | 22,297MB |
| 6 | 6.005e-5 | 0.7556 | 19.156s | 22,297MB |
| 7 | 5.338e-5 | 0.7297 | 19.071s | 22,297MB |
| 8 | 4.672e-5 | 0.7090 | 17.808s | 22,297MB |
| 9 | 4.005e-5 | 0.7032 | 18.752s | 22,297MB |
| 10 | 3.338e-5 | 0.6916 | 18.661s | 22,297MB |
| 11 | 2.672e-5 | 0.6704 | 19.918s | 22,297MB |
| 12 | 2.005e-5 | 0.6766 | 20.216s | 22,297MB |
| 13 | 1.338e-5 | 0.6627 | 18.237s | 22,297MB |
| 14 | 6.718e-6 | 0.6621 | 19.038s | 22,297MB |
| 15 | 5.096e-8 | 0.6496 | 18.618s | 22,297MB |

采样日志中出现 4 次 grad_norm=inf。训练总体稳定，但下一轮应显式统计 dynamic fp16 overflow/skip-step，而不是只靠日志抽样。

### 12.6 逐 epoch eval

| epoch | mAP（center） | BEV mAP@0.5 | mATE | mAOE | mASE | eval det |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2891 | 0.2638 | 0.9712 | 7.491° | 0.2983 | 3,399,777 |
| 2 | 0.3200 | 0.3222 | 0.9277 | 6.431° | 0.2705 | 2,381,721 |
| 3 | 0.3299 | 0.3415 | 0.9188 | 5.328° | 0.2533 | 2,010,104 |
| 4 | 0.3404 | 0.3504 | 0.8949 | 4.948° | 0.2448 | 1,623,890 |
| **5** | **0.344934** | **0.3562** | **0.8899** | **4.315°** | **0.2369** | **1,507,213** |
| 6 | 0.3422 | 0.3522 | 0.8892 | 4.399° | 0.2369 | 1,437,322 |
| 7 | 0.3433 | 0.3532 | 0.8823 | 4.743° | 0.2326 | 1,416,947 |
| 8 | 0.3375 | 0.3435 | 0.8859 | 4.535° | 0.2263 | 1,362,160 |
| 9 | 0.3417 | 0.3518 | 0.8744 | 4.092° | 0.2220 | 1,205,069 |
| 10 | 0.3366 | 0.3468 | 0.8833 | 4.020° | 0.2249 | 1,181,923 |
| 11 | 0.3380 | 0.3486 | 0.8800 | 4.076° | 0.2156 | 1,172,807 |
| 12 | 0.3399 | 0.3515 | 0.8749 | 3.928° | 0.2176 | 1,084,035 |
| 13 | 0.3405 | 0.3542 | 0.8758 | 3.689° | 0.2166 | 1,102,686 |
| 14 | 0.3400 | 0.3518 | 0.8753 | 3.751° | 0.2150 | 1,072,296 |
| 15 | 0.3387 | 0.3511 | 0.8768 | 3.696° | 0.2154 | 1,107,347 |

eval_summary.md 与训练日志在个别 det 计数上有 1～2 个差异，例如 epoch5 汇总为 1,507,213，而训练日志为 1,507,215。本文总体表以最终 eval_summary.md 为准，训练内原始行保留在日志中。

### 12.7 epoch5 分类别

GT：

- car=69,492；
- truck=15,747；
- total=85,239；
- car:truck≈4.41:1。

| 类别 | center_AP | BEV AP@0.25 | BEV AP@0.5 | AP@0.5m | AP@1m | AP@2m | AP@4m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| car | 0.5135 | 0.6590 | 0.4843 | 0.2448 | 0.4365 | 0.6385 | 0.7341 |
| truck | 0.1764 | 0.3784 | 0.2282 | 0.0254 | 0.0958 | 0.2261 | 0.3583 |

| 类别 | ATE@2m | AOE@2m | ASE@2m |
| --- | ---: | ---: | ---: |
| car | 0.7586m | 3.793° | 0.1594 |
| truck | 1.0212m | 4.836° | 0.3144 |

结论：

- car 明显强于 truck；
- truck 严格 0.5m/1m center AP 很低；
- 总体指标会被 car 主导；
- 后续 CBGS/anchor/NMS 是否需要改，必须建立在真实类别、尺寸和 anchor IoU 分布统计上。

### 12.8 epoch5 与 epoch15

| 项目 | epoch5 | epoch15 | 变化 |
| --- | ---: | ---: | --- |
| train loss | 0.7919 | 0.6496 | 继续下降 |
| mAP（center） | **0.3449** | 0.3387 | 下降 |
| BEV mAP@0.5 | **0.3562** | 0.3511 | 下降 |
| mATE | 0.8899 | **0.8768** | 改善 |
| mAOE | 4.315° | **3.696°** | 改善 |
| mASE | 0.2369 | **0.2154** | 改善 |

这说明 checkpoint 选择不是单指标绝对真理：epoch15 的 matched TP 回归误差更好，但检测 AP 较低。当前产品对照主 key 已明确为 canonical `mAP`，因此仍选 epoch5；如果以后产品更在意已匹配目标的角度，需要在独立生产集上做多目标阈值选择，而不是偷偷改 best 规则。

### 12.9 score 和检测数量

| epoch | car det | truck det | total det | car p50 | truck p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,384,667 | 2,015,110 | 3,399,777 | 0.1643 | 0.0888 |
| 5 | 910,777 | 596,436 | 1,507,213 | 0.1382 | 0.0869 |
| 10 | 785,907 | 396,016 | 1,181,923 | 0.1434 | 0.0933 |
| 15 | 721,063 | 386,284 | 1,107,347 | 0.1447 | 0.0972 |

B0 的 score 并未像 T3 epoch5 那样全部贴在 0.05；这也是判定 B0 为平台波动而不是 collapse 的重要证据。当前精评默认不限制 eval_max_dets_per_sample，百万级 det 会增加 eval 开销，但不应为提速而无记录地改变最终口径。

## 13. xyz mean error 的正确解释

### 13.1 为什么 x/y mean 看起来只有几厘米

eval_summary.md 中的 x_mean、y_mean 是 signed mean。例如一部分目标预测偏前，另一部分偏后，会互相抵消。小 signed mean 表示系统性偏置可能小，不表示单目标误差小。

以 epoch5 car 为例：

| GT x 距离 | x abs p50 | x abs p90 | y abs p50 | y abs p90 | x signed mean | y signed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0～20m | 0.223m | 0.742m | 0.097m | 0.334m | 0.054m | 0.005m |
| 20～40m | 0.623m | 1.476m | 0.118m | 0.535m | 0.052m | -0.019m |
| 40～60m | 0.781m | 1.655m | 0.145m | 0.608m | 0.022m | -0.062m |
| 60～80m | 0.795m | 1.671m | 0.178m | 0.746m | -0.050m | -0.036m |

因此：

- 近距离 y 方向中位绝对误差可接近 10cm；
- x 方向中远距中位绝对误差约 0.62～0.80m；
- p90 可达到约 1.5～1.7m；
- overall mATE@2m=0.8899m，也明确不是几厘米。

### 13.2 旧 z_mean 为什么接近负半车高

epoch5 旧统计大致为：

- car z_mean≈-0.83～-0.86m；
- truck z_mean≈-1.64～-1.68m。

这与 car/truck 高度的一半高度非常接近。代码审查发现：

- pkl GT z 按 gravity center 使用；
- LiDARInstance3DBoxes tensor 的 prediction z 是 bottom center；
- 旧 eval 直接相减。

所以旧 z error 主要是 box origin 语义 bug，不是模型真实垂直定位结论。

当前代码已经：

- prediction box object 优先读取 gravity_center；
- portable numpy 结果声明 box_origin=center 或 bottom；
- 生产输出和可视化统一转成 center；
- 训练 GT 构造保持不变。

用户已在内网直接复用已有 result pkl 重新 evaluate，并确认：

- car/truck 的 z 偏差均已恢复正常；
- 不需要重新训练，也不需要重新 inference。

当前尚未同步精确的新 z signed mean、z MAE/p50/p90 和 result pkl hash，因此本文只记录“重评通过”，不虚构具体数值。后续归档时还应核对 BEV AP、xy center AP 与修复前基本不变。

## 14. 生产测试反馈如何理解

用户当前用 EXP-MONO-B0 epoch5 PTH 测试生产环境图片，观察到：

- 角度容易偏；
- 20～30m 外目标不清楚。

按当前记录，生产脚本对四时序模型默认重复当前帧，板端没有使用训练时等价的历史帧 pose 补偿。这些现象不能只用 B0 val AOE=4.315° 否定，也不能直接归因于 yaw loss：

1. AOE 只统计 ≤2m matched TP，漏检和低分远距目标不进入该均值；
2. 生产车型/标定/日期/光照与 val 可能存在域差异；
3. 四时序重复当前帧与 pose-aware 训练输入分布不同；
4. 生产脚本的 RGB/BGR、resize、K 尺寸、distortion、extrinsic、post transform、yaw 坐标解释仍需同图逐级对齐；
5. 1600×900 被非等比 stretch 到 704×256，远处小目标像素和预训练外观可能受损；
6. 可视化 heading 解释或 box origin 错误也可能造成主观偏角。

因此当前优先顺序是：

- 先做标准 dataset 与生产推理的同图 tensor/feature/logit/box 对齐；
- 建立按距离和 yaw 分桶的生产回归集；
- 再判断是标定/坐标/阈值/召回问题，还是模型方向头本身需要调优。

## 15. 审查问题和当前修复状态

### 15.1 板端时序 pose

审查判断正确：B0 训练会把历史相机变换到当前关键帧 lidar/BEV，而现阶段板端没有等价 pose 输入。六轴 IMU+轮速可以作为未来短时 egomotion 来源，但 pose 模块尚未交付。

当前决策：

- 近期产品：原生单帧 S0；
- B0：离线时序上限和教师候选；
- 重复当前帧：只作诊断，不作最终模型；
- 未来 pose 接口完成后，再评估动态历史 LUT 或历史 BEV warp。

### 15.2 ONNX exporter

已修：

- 2D wrapper 直接调用 model.onnx_export_2d；
- 3D wrapper 直接调用 model.onnx_export_3d；
- 正式支持 n_times=1 和 n_times=4；
- 单帧 3D input 名称为 bev_0；
- metadata 记录 n_images、n_times、temporal_order、channel_layout、shape、dtype、raw logits、config/checkpoint SHA256。

本地合成模型已完成 torch.onnx.export、onnx.checker、ORT，2D/3D max abs diff=0。

待内网：

- 真实 S0/B0 checkpoint；
- 真实图片和 LUT；
- decoded boxes 对齐；
- 确认 sigmoid 只执行一次；
- PTH→ORT→板端 float 对齐。

### 15.3 box origin

代码已修，且已有 result pkl 重评确认 car/truck z 偏差恢复正常。此项不要求重训 S0/B0；剩余工作是归档精确新数值、结果路径/hash，并保证评估、可视化、生产输出和板端接口持续使用相同 box_origin。

### 15.4 数据门禁

未完成：

- train/val/test clip 和 token 零重叠；
- test 不再长期复用 val；
- camera_ids、图像尺寸、K native 尺寸和 distortion 完整率；
- 空帧、负样本、car/truck、x/y/z/l/w/h/yaw；
- manifest、pkl、calibration hash；
- converter 异常分类和 strict 非零退出。

### 15.5 畸变可见性

审查判断正确且仍未修：

- model backproject 使用 distortion；
- FrontCameraVisibleObjectFilter 和 dataset eval visibility 仍以 pinhole 为主；
- 广角边缘可能误删或误保留 GT。

修复应抽取 dataset/pipeline/visualization/LUT 共用投影函数，并在改训练数据口径前先统计 pinhole 与 distortion-aware 的真实差异。

### 15.6 704×256 stretch 和图像增强

当前 1600×900→704×256：

- 水平缩放约 0.44；
- 垂直缩放约 0.284；
- 几何可通过 post transform 自洽；
- 视觉外观相对横向拉宽约 1.55 倍；
- `force_resize=True` 会先采样随机参数、再强制覆盖为固定 resize/crop、`flip=False`、`rotate=0`，因此图像随机几何增强实际被完全关闭。

这不会使当前 S0/B0 首轮比较失效，因为两边使用相同口径；但它是正式训练前需要收口的实现问题，可能限制远距小目标和 COCO 预训练迁移。首轮 A/B 后应先拆分 native K/尺寸到 704×256 的确定性基础变换与可选随机增强，并在随机增强关闭时验证图像、`post_rot/post_tran`、投影和 logits 与旧实现等价。之后才用独立配置 A/B 随机增强；不能简单设置 `force_resize=False`，也不能把该变化与 stretch、704×384、数据修复同时引入。四时序若启用随机几何增强，同一相机四帧必须共享参数。

### 15.7 anchor、CBGS、NMS

当前都属于后置单变量优化：

- car:truck GT≈4.41:1，可以评估 CBGS，但先保留无 CBGS S0；
- anchor 需要真实 l/w/h/z 分布和 max IoU 覆盖；
- truck nms_rescale_factor=0.7 继承 nuScenes，不一定适合 N7；
- box z 语义闭环前不要调 anchor z；
- 不根据几张生产图直接改 focal、FreeAnchor IoU 或 yaw loss。

## 16. EXP-MONO-S0：当前原生单帧实验协议

### 16.1 配置文件

- configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18.py
- configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18_dist_train.py

### 16.2 相对 B0 的唯一主变量

- n_times：4→1；
- dataset/pipeline：sequential=True→False；
- train_adj_ids/test_adj_ids：[0,1,2]→None；
- temporal_compensate：False；
- 3D fuse：1024→256 改成 256→256；
- 不再复制 4 张当前图；
- 不读取历史 pose。

保持不变：

- cam0；
- n_images=1；
- 704×256 stretch；
- ROI、voxel、anchor；
- distortion=True；
- front ROI + cam0 GT 过滤；
- 64/GPU × 4；
- AdamW2 LR=1e-4；
- COCO 2D 初始化；
- dynamic fp16；
- total_epochs=15。

### 16.3 train/eval 分工

四卡训练机：

- tools/dist_train_ljr.sh；
- NO_VALIDATE=1；
- checkpoint 每 epoch；
- config 中 evaluation.interval=999999，避免误触发全量 val。

单卡 eval 机：

~~~bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_epoch_checkpoints.py \
  configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18.py \
  <S0-work-dir> \
  --watch \
  --poll-interval 600 \
  --stable-seconds 120 \
  --best-key mAP
~~~

输出：

- epoch_N_val_results.pkl；
- epoch_N_test.log；
- epoch_N_metrics.json；
- eval_summary.md/csv/json。

### 16.4 停止和比较规则

1. 先完成 legacy 环境 model/dataloader/1-iter train/test smoke；
2. 优先查看 S0 epoch5；
3. 比较 B0 epoch5 与 S0 epoch5；
4. 同时比较双方各自 best；
5. 某 best 后连续 5 个已评估 epoch 未刷新，手动停止；
6. 当前工具没有自动杀训练进程的 early-stop，不能把规则写成“已自动实现”；
7. 只用 val 选 checkpoint，独立 test 不参与调参；
8. 不根据 loss 是否继续下降选 best。

### 16.5 训练前 smoke 验收

- merged config 构建成功；
- dataloader image shape=[B,1,3,256,704]；
- 3D neck 输入=[B,256,160,140]；
- 1 iter train loss 有限；
- test forward 结果数和 dataset 对齐；
- PTH 单图脚本不重复时序；
- 2D/3D ONNX 和单帧 LUT smoke；
- py_compile 和 git diff --check。

### 16.6 正式长训练前的四个收口项

首轮 S0/B0 对比可以尽快启动；进入正式单帧训练/调优前必须完成：

1. pkl split、相机、尺寸、空帧、类别和 hash 门禁；
2. 生产推理与标准 dataset 同图逐级对齐；
3. distortion-aware GT visibility 一致；
4. `force_resize` 确定性基础变换/随机增强拆分，并先完成关闭随机增强的等价性回归。

## 17. eval 和可视化工具

### 17.1 单卡逐 checkpoint eval

tools/eval_epoch_checkpoints.py：

- 监控 epoch_*.pth；
- 等文件稳定后调用 tools/test.py；
- inference 后复用 result pkl 做指标和 score 统计；
- 默认 best key=`mAP`，其定义为 center-distance mAP；旧 `mAP/center_dist` 仍可兼容读取；
- 支持只对已有 result pkl rerun eval，适合 box-origin 修复。
- watch 新 checkpoint 时，类别表不再展开 AP@0.5m/1m/2m/4m；距离分桶运行表只展示 GT、TP@2m、Recall@2m 和 x/y/z MAE。
- eval_summary.md/csv/json 继续保存逐阈值 AP、MAE、绝对误差 p50/p90；Markdown 另存 signed mean bias 表，CSV/JSON 同时保留绝对误差与 signed bias。

正式精评建议：

- eval_max_dets_per_sample=None；
- 不为了速度静默提高 score threshold；
- fast eval 可以单独标记阈值/cap，不能覆盖最终结果。

### 17.2 前视面板尺寸

camera-size WIDTH HEIGHT 只约束相机面板。示例：

~~~bash
--camera-size 1600 900
~~~

如果保留顶部信息栏和 BEV，整体 canvas 会更大。只有同时 no-bev、no-header 时，纯前视视频才可严格为 1600×900。

704×256 缓存图放大到 1600×900 只改变显示尺寸，不恢复原始图像细节。

### 17.3 视频分组和命名

- video-group=clip：
  output/dataset/sequence/clip/clip.mp4
- video-group=sequence：
  output/dataset/sequence/sequence.mp4

当前还支持：

- video-only；
- max-frames-per-video；
- stride/max-frames；
- 多 worker 有序渲染；
- canvas size 变化时报错。

待完善：

- sequence/clip 白名单或正则；
- 跨 clip 标题帧或切换标记；
- 内网真实 B0 epoch5 sequence 视频验证。

## 18. 当前下一步

### P0：立即执行

1. 内网构建 S0 config 并做 1-iter train/test smoke；
2. 固化 B0 epoch5 PTH、resolved config、代码/data/calibration hash；
3. 归档已通过的 B0 box-origin 重评新数值、result pkl 路径和 hash；
4. 启动 S0 四卡训练和单卡逐 epoch eval；
5. 优先产出双方 epoch5 和各自 best 对比。

### P1：首轮对比期间并行

1. 生成 B0 epoch5 clip/sequence 可视化；
2. 建立角度偏和 20～40m 目标生产回归集；
3. 做生产/dataset 同图逐级对齐；
4. 完成 pkl 数据门禁；
5. 统计并修复 distortion visibility 差异；
6. 拆分 `force_resize` 基础变换/随机增强，先完成关闭增强的等价性回归。

### P2：S0 可信后单变量 A/B

1. stretch vs 等比 resize/crop vs 更高输入；
2. 图像增强；
3. CBGS/class-aware sampling；
4. anchors；
5. NMS/score calibration；
6. temporal B0 warm-start 或蒸馏；
7. LR/schedule/epoch。

## 19. 内网资产补录模板

每轮新实验在本节模板复制一份：

~~~text
experiment_id:
date:
purpose:
status:

repo_root:
git_commit:
git_status:
git_diff_sha256:

launch_command:
resolved_config_path:
resolved_config_sha256:
environment_versions:

train_manifest:
train_manifest_sha256:
val_manifest:
val_manifest_sha256:
test_manifest:
test_manifest_sha256:
train_pkl:
train_pkl_sha256:
val_pkl:
val_pkl_sha256:
test_pkl:
test_pkl_sha256:
calibration_files_and_sha256:

init_checkpoint:
init_checkpoint_sha256:
work_dir:
checkpoint_paths_and_sha256:

best_key:
best_epoch:
best_value:
early_stop_reason:

eval_result_dir:
eval_summary:
production_regression_set:
visualization_dir:

known_issues:
conclusion:
next_action:
~~~

## 20. 最终状态摘要

- 6V：可训练证据成立，完整精度实验未归档；
- 6V 性能：动态畸变是主要耗时，主线已保留批量投影和 fused scatter；
- mono 四时序结构：成立；
- 早期 mono：存在 vertical flip、高 LR 和 score collapse 历史；
- 正式 B0：15 epoch 完成，epoch5 canonical mAP=0.344934，为当前时序对照；
- B0 epoch15：loss/TP 回归误差更低，但 AP 未刷新，不延长到 20；
- xyz：x/y signed mean 不能代表绝对精度；box-origin 修复已通过已有 result pkl 重评，car/truck z 偏差恢复正常；
- 产品：近期原生单帧；
- S0：配置和本地合成检查完成，内网真实训练/权重/指标尚未开始；
- 正确的下一步：先形成同口径 S0 vs B0，再做数据、几何、远距和板端闭环优化。
