# N7 Fast-BEV 实验详细记录

> 更新时间：2026-08-01
>
> 汇总入口：[N7_FASTBEV_EXPERIMENT_SUMMARY.md](N7_FASTBEV_EXPERIMENT_SUMMARY.md)
>
> 当前执行清单：[N7_MONO_FRONT_SINGLE_FRAME_TODO.md](N7_MONO_FRONT_SINGLE_FRAME_TODO.md)

## 1. 文档目的和边界

本文把当前能够找到的 N7 Fast-BEV 实验证据按时间顺序整理为一条可追溯链路：

1. N7 6V 四时序旧可训练基线和当前完整精度基线；
2. 6V 动态畸变/backproject 性能消融；
3. 单目前视四时序模型侧 smoke；
4. 单 sequence 收敛实验；
5. full-data 早期不稳定实验；
6. 2026-07-08 正式四时序 B0；
7. 原生单帧 S0 的最终训练、逐 epoch 评估和 PC 侧部署闭环。

当前本地环境没有 N7 原始数据、内网 checkpoint/result pkl 实体，也没有可运行旧版 mmcv/mmdet/mmdet3d 的 legacy 训练栈；但已经同步了当前 6V-B0 训练日志/eval 汇总、早期 B0/S0 证据和 B0 pkl 门禁报告。因此：

- 本文可以确认日志中已经发生的训练、保存和评估；
- 可以确认当前仓库代码和配置状态；
- 可以确认 EXP-6V-B0 epoch1～17 完整训练并保存；正式 checkpoint 选型窗口为已完成 val 复评的 epoch1～16，epoch17 按用户决定排除且不再补评；
- 可以记录用户确认的 S0 15 epoch 完成、最终 best=e13 和后续内网部署验证；
- 不能在本机确认历史 PTH 实体、SHA256、内网 manifest/pkl/calibration；
- 当前代码基线已有明确 GitHub commit，但 S0/6V 实际训练资产仍需补录对应内网 commit 和 hash；
- 需要内网补录的内容会明确标记，不用猜测填充。

## 2. 证据和口径

### 2.1 主证据

| 类型 | 路径 | 主要内容 |
| --- | --- | --- |
| 旧 6V 训练日志 | /workspace/20260626_174629.log | EXP-6V-00 resolved config、环境、epoch1～3 部分训练 |
| 当前 6V-B0 训练日志 | work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717/20260717_173206.log | 真实 resolved config、e1～17 完成并保存 |
| 当前 6V-B0 逐 epoch 复评 | work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717/test_results/eval_summary.md | e1～16 canonical mAP、BEV、TP error、score、正向 x 距离分桶 |
| 单 sequence 日志 | /workspace/20260703_160930.log | 单目四时序 20 epoch 收敛和 e5/e10/e15/e20 eval |
| full-data 初始日志 | /workspace/20260703_172933.log | full-data epoch1～10、e5 训练内 eval |
| vflip0 clean 日志 | /workspace/20260706_184052.log | full-data clean epoch1～4 |
| epoch2 load-from 日志 | /workspace/20260707_152258.log | 低 LR 权重加载实验 resolved config 和 epoch1～5 部分 |
| 正式 B0 日志 | /workspace/20260708_122556.log | B0 resolved config、epoch1～15、e5/e10/e15 eval |
| B0 逐 epoch 复评 | /workspace/eval_summary.md | epoch1～15 center/BEV/TP error/score/xyz 分桶 |
| S0 训练日志 | work_dirs/n7_mono_704_256_single_frame/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260713/20260713_113748.log | S0 resolved config、epoch1～9 checkpoint、epoch10 部分训练 |
| S0 逐 epoch 复评 | work_dirs/n7_mono_704_256_single_frame/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260713/test_results/eval_summary.md | epoch1～9 canonical mAP、BEV、TP error、距离 Recall/MAE |
| B0 pkl 门禁 | work_dirs/n7_pkl_gate_b0/data_gate.md | train/val split、相机/尺寸/标定/GT/pose 审计及旧 converter 失败项 |
| 6V 性能笔记 | [N7_FASTBEV_TIMING_ABLATION_20260630.md](N7_FASTBEV_TIMING_ABLATION_20260630.md) | 动态畸变与 backproject 性能消融 |
| 单目适配审查前记录 | [N7_MONO_FRONT_ADAPTATION_REVIEW.md](N7_MONO_FRONT_ADAPTATION_REVIEW.md) | 早期 collapse 指标和原因分析 |
| 代码审查计划 | [N7_MONO_FRONT_AUDIT_FIX_PLAN.md](N7_MONO_FRONT_AUDIT_FIX_PLAN.md) | 结构、数据、部署和指标问题 |
| 当前代办 | [N7_MONO_FRONT_SINGLE_FRAME_TODO.md](N7_MONO_FRONT_SINGLE_FRAME_TODO.md) | 单帧产品路线和优先级 |

本次分析时的关键本地证据 SHA256：

| 证据 | SHA256 |
| --- | --- |
| B0 `20260708_122556.log` | `59f21980c23ff55c143c521f56e3234664b5be8015123262b2cd93a2c6b63cb3` |
| B0 `eval_summary.md` | `74e782f5506ab237f457928554c1da4552701d1a25152727e65a27062fe074b0` |
| S0 `20260713_113748.log` | `732234cc56225dc69f58ffc47d4dd7c9e96ce07f42a88cc71f78249434be5fc7` |
| S0 `test_results/eval_summary.md` | `03b88d9b1eeb88660a5f1cc49f21695c43d48ba4896ea145b7bd8943c68667d6` |
| B0 `data_gate.md` | `bf73d284adb4916669bfc8d56995d4ccf4de44de690e6bc1e2ef695828df9dda` |

本节中的 B0/S0 SHA256 标识早期同步证据；6V 日志在本次检查期间从 epoch17 2030/2093 更新到完整保存，因此 6V 当前快照 hash 单独记录在 6A 节。当前本地没有训练/eval 进程；若后续再同步 eval 或训练日志，必须继续重算对应 hash。

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
- GitHub 已固化当前 B0/S0 代码基线，但训练日志没有把实际内网 commit、完整 diff 和数据/calibration hash 写入实验资产。

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
| 2026-07-13～07-14 | EXP-MONO-S0 | e1～9 checkpoint/eval 完成，当前观测 best=e8；训练进入 e10 |
| 2026-07-14 | B0 pkl 门禁实跑 | train/val 零泄漏；旧 converter 的 `labels_without_frame=851/90` 触发预期 FAIL |
| 2026-07-14 | B0/S0 代码基线提交 | GitHub 与内网分支均完成两笔代码提交并推送；内网 SHA 待补录 |
| 2026-07-13～07-17 | EXP-MONO-S0 收口 | 15 epoch 完成，最终 best=e13；canonical mAP=0.356425 |
| 2026-07-17～07-23 | EXP-6V-B0 | e1～17 完整训练/日志保存；正式选型窗口 e1～16；final best=e6、challenger=e2、terminal=e16 |
| 2026-07-17～07-23 | S0 PC 部署闭环 | dataset/production PTH、Torch-CUDA LUT、FP ONNX 和独立 CPU 板端参考完成；真实芯片/INT8 仍开放 |

主要 git 节点：

- 73b5bcf：适配 N7 自采数据到 Fast-BEV 训练链路；
- 2a64a51：适配 N7 自采数据训练与评估链路；
- 237487b：清理 N7 Fast-BEV 适配训练链路；
- 8cc8678：适配 N7 6V FastBEV 可训练基线；
- `2d8ab7d9855f787d55da8521b1993c445351037d`：固化 N7 6V、mono 四时序 B0、数据/评估公共链路；
- `c3ec682e78e273fa6142e13825f0cde4857d7883`：新增原生单帧 S0 与 ONNX/LUT 导出支持；
- `8551924be50aa0eb4f88454a36c2ea396955d6af`：补充实验文档和提交约束；
- 当前分支：`feature/n7-mono-front-from-6v-baseline`；GitHub 本地/远端已一致；
- 用户确认内网对应两笔代码提交已检查并推送，但尚未在本文补录内网 commit SHA。

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

## 6A. EXP-6V-B0：2026-07-17 6V 四时序完整 val 精度基线

本节只使用以下本地实际文件，不把旧 EXP-6V-00 的日志或结论混入：

- `work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717/20260717_173206.log`
- `work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717/test_results/eval_summary.md`

首次分析同步快照的证据 hash：

| 文件 | SHA256 |
| --- | --- |
| `20260717_173206.log` | `bf5a98f4908ff1ac5d0d49fec687ae564577a9760dfd0c63492b2c681ba0635b` |
| `test_results/eval_summary.md` | `ad2254ef828efcacf4a5b66cd2f556a037799f6620d4d442b42c91bc6a2ed580` |

上述训练日志 hash 标识 epoch17 已保存后的同步快照，eval hash 标识正式选型窗口 epoch1～16 的汇总；两者都不是 checkpoint/result 资产 hash。用户已决定 epoch17 及以后不再纳入本轮 checkpoint 选型，因此不需要补评 e17，也不等待 e18～20。

### 6A.1 文件、训练、checkpoint 和 eval 完整性

| 项目 | 实际证据 |
| --- | --- |
| 训练日志 | 只有 `20260717_173206.log` 一份 |
| 完整训练 | epoch1～17；每个 epoch 有 209 个 `TextLoggerHook` 点，iter 10～2090/2093 |
| 部分训练 | 无；同步日志停在 epoch17 保存标记，尚无 epoch18 训练证据 |
| 日志证明保存 | epoch1～17 各有一次 `Saving checkpoint at N epochs` |
| 本地实际 PTH | 0 个；`epoch_*.pth` 和 `latest.pth` 均未同步 |
| 已评估 | `eval_summary.md` 覆盖 epoch1～16 |
| 未评且排除出选型 | epoch17；epoch18～20 不再训练 |
| eval 汇总资产 | 只有 `eval_summary.md` |
| 缺失 eval 资产 | `eval_summary.csv/json`、逐 epoch metrics JSON、result pkl、eval log 均不存在 |
| 断点/重启 | `resume_from=None`；单日志连续从 epoch1 到 epoch17，无重复 epoch/iter 对 |
| 本地运行状态 | 本地 `ps` 未发现该训练/eval 进程；同步日志最后一条是 `Saving checkpoint at 17 epochs` |
| 远端运行状态 | 日志来自内网 host 且这里只是同步快照，不能由本地进程表证明远端进程状态；当前证据只支持不再安排 epoch18～20 |

因此“已训练/已保存/已评估”必须分开写：

- 已完整训练：epoch1～17；
- 已开始但未证明完成：无；
- 日志证明曾保存：epoch1～17；
- 本地实际存在的 checkpoint：无；
- 已评估：epoch1～16；
- 正式评估/选型窗口：epoch1～16；
- epoch17 已保存但按用户决策排除，不再要求补评；
- checkpoint 与 eval 的逻辑 epoch 集合在正式窗口 1～16 对齐；PTH、metrics JSON、result pkl 和 eval log 未同步到本地，不能做本地文件级一一对应和 hash 核验，但用户确认相关 PTH/pkl 已在内网保存。

### 6A.2 本轮日志 resolved config

配置入口由日志中的 `Exp name` 确认：

`configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256_dist_train.py`

| 项目 | 本轮 resolved 值 |
| --- | --- |
| work_dir | `/mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717` |
| 相机顺序 | `cam0, cam11, cam9, cam3, cam8, cam10` |
| n_images / n_times | 6 / 4；每样本 24 张图 |
| train pkl | `./data/N7_704_256/6v_pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_train_20260717.pkl` |
| val pkl | `./data/N7_704_256/6v_pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_val_20260717.pkl` |
| test pkl | 与 val 完全相同；本轮没有独立 test 评估 |
| GPU / batch / workers | 4×NVIDIA L20；24/GPU；全局 batch 96；8 workers/GPU |
| optimizer | **AdamW2** |
| LR / weight decay | `8e-4` / `0.01` |
| backbone paramwise | `lr_mult=0.1`、`decay_mult=1.0`；backbone 基础 LR 为 `8e-5` |
| scheduler | poly，power=1，min_lr=0，`by_epoch=False` |
| warmup | linear，1000 iter，ratio `1e-6` |
| grad clip | max_norm=35，norm_type=2 |
| load_from | COCO Cascade Mask R-CNN R18/FPN 2D 预训练权重 |
| resume_from | `None`；从 epoch1 新开，不恢复 optimizer/epoch 状态 |
| 实际初始化语义 | 日志先初始化模型，再从本地 COCO checkpoint 加载兼容参数；FPN 通道不匹配、3D neck/fuse 和 bbox head 缺失键均被明确打印，不是完整 Fast-BEV checkpoint resume |
| fp16 | dynamic loss scale；实际 hook 为 `Fp16OptimizerHook` |
| total epochs | 20 上限 |
| checkpoint / in-train eval | interval 1 / interval 25；20 epoch 内不会触发训练内 eval，逐 epoch 结果来自训练外 val 复评 |
| ROI / eval range | `[-50,-50,-5,50,50,3]` |
| voxel | `n_voxels=[200,200,4]`；`voxel_size=[0.5,0.5,1.5]` |
| anchor | range `[-50,-50,-1.8,50,50,-1.8]`；4 组 size；rotation 0/1.57 |
| distortion | `True` |
| 图像 | native K 尺寸配置 1600×900；输入 704×256；`force_resize=True` |
| 图像增强配置 | resize `[-0.06,0.11]`、crop `[-0.05,0.05]`、rot `[-5.4°,5.4°]`、flip=True 字段存在；实际 pipeline 同时明确 `force_resize=True` |
| BEV 增强 | horizontal/vertical flip 均 0.5；global rot `[-0.3925,0.3925]`、scale `[0.95,1.05]`、translation std 0.05m |

optimizer 版本证据和配置收口：

- 本轮训练日志 resolved config 的唯一结论是 `AdamW2`；
- 旧 EXP-6V-00 也记录为 `AdamW2`，但两轮仍是不同 run；
- 旧仓库配置只写 `optimizer=dict(lr=0.0008)`，会静默继承论文 base 的标准 `Adam`，与真实 run 不一致；
- 分析期间同名配置被外部未提交改动显式使用 `_delete_=True` 重建 AdamW2，并固定 `6v_pkl/*_20260717.pkl`、poly/warmup、grad clip、20 epoch、逐 epoch 保存和 `evaluation.interval=25`；本文没有修改该配置；
- 当前配置口径现与日志一致，但它仍不是训练时 commit/config 快照，本轮真实 optimizer 结论只取日志 resolved config；
- 当前没有独立 test，配置让 `data.test` 明确复用 val，仅用于 val 复评，不能将其表述为 test 指标。

### 6A.3 逐 epoch 核心 val 指标

主选择键是 `eval_summary.md` 明确定义的 canonical center-distance `mAP`，不是 BEV IoU mAP。汇总仅对 best 值保留 6 位精度，其余表格保留 4 位；下表不伪造被汇总舍入的尾数。BEV@0.25 为 car/truck 两类 AP 的等权平均。

| e | canonical mAP | BEV@0.25 | BEV@0.5 | car center AP | truck center AP | mATE | mAOE° | mASE | GT | pred |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3926 | 0.5728 | 0.3921 | 0.5499 | 0.2353 | 0.8821 | 5.9016 | 0.2225 | 64,857 | 1,974,696 |
| 2 | 0.4355 | 0.6197 | 0.4349 | 0.5793 | 0.2917 | 0.8407 | 4.7108 | 0.2187 | 64,857 | 2,226,245 |
| 3 | 0.4445 | 0.6307 | 0.4491 | 0.5901 | 0.2989 | 0.8230 | 4.5577 | 0.1964 | 64,857 | 1,559,025 |
| 4 | 0.4435 | 0.6446 | 0.4592 | 0.5888 | 0.2982 | 0.8179 | 4.0121 | 0.1950 | 64,857 | 1,612,015 |
| 5 | 0.4458 | 0.6376 | 0.4652 | 0.5844 | 0.3071 | 0.7996 | 3.9369 | 0.1860 | 64,857 | 1,219,472 |
| **6** | **0.453768** | **0.6453** | **0.4787** | **0.5956** | 0.3119 | 0.7793 | 3.0615 | 0.1811 | 64,857 | 1,040,837 |
| 7 | 0.4517 | 0.6418 | 0.4714 | 0.5887 | **0.3148** | 0.7818 | 3.2277 | 0.1815 | 64,857 | 976,432 |
| 8 | 0.4462 | 0.6428 | 0.4659 | 0.5886 | 0.3037 | 0.7879 | 4.0217 | 0.1763 | 64,857 | 860,097 |
| 9 | 0.4428 | 0.6312 | 0.4627 | 0.5843 | 0.3014 | 0.7747 | 2.9818 | 0.1763 | 64,857 | 771,736 |
| 10 | 0.4373 | 0.6283 | 0.4545 | 0.5784 | 0.2961 | 0.7763 | **2.9294** | 0.1745 | 64,857 | 830,230 |
| 11 | 0.4355 | 0.6320 | 0.4581 | 0.5703 | 0.3006 | 0.7744 | 3.0044 | 0.1728 | 64,857 | 664,995 |
| 12 | 0.4366 | 0.6257 | 0.4533 | 0.5763 | 0.2969 | **0.7599** | 3.1223 | 0.1712 | 64,857 | 557,368 |
| 13 | 0.4301 | 0.6161 | 0.4452 | 0.5686 | 0.2916 | 0.7654 | 3.1436 | 0.1704 | 64,857 | 603,018 |
| 14 | 0.4265 | 0.6109 | 0.4408 | 0.5680 | 0.2849 | 0.7683 | 3.2640 | 0.1702 | 64,857 | 559,551 |
| 15 | 0.4231 | 0.6111 | 0.4402 | 0.5656 | 0.2805 | 0.7621 | 3.6013 | 0.1697 | 64,857 | 514,376 |
| 16 | 0.4192 | 0.6070 | 0.4354 | 0.5570 | 0.2815 | 0.7636 | 3.1191 | **0.1683** | 64,857 | 492,526 |

### 6A.4 预测数和 score 分布

格式为 p50/p90/p99/score≥0.2；`eval_det=car_det+truck_det`。高分位逐步饱和到 1.0、总预测数持续下降，但这没有转化为 epoch6 之后的 canonical mAP 或 Recall 改善。

| e | car det | car score | car ≥0.2 | truck det | truck score | truck ≥0.2 |
| ---: | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 1,244,848 | .1427/.3767/.9751 | 421,245 | 729,848 | .0853/.2954/.6372 | 149,465 |
| 2 | 1,560,409 | .1699/.4590/.9922 | 665,542 | 665,836 | .0840/.2930/.6621 | 123,683 |
| 3 | 1,177,028 | .1512/.4854/.9995 | 454,090 | 381,997 | .0975/.3630/.9609 | 98,445 |
| 4 | 1,223,811 | .1429/.4609/1.0000 | 432,677 | 388,204 | .0983/.3679/.9810 | 98,881 |
| 5 | 960,373 | .1637/.5718/1.0000 | 391,693 | 259,099 | .1242/.4233/.9995 | 84,786 |
| 6 | 807,781 | .1373/.7139/1.0000 | 285,994 | 233,056 | .1056/.4082/1.0000 | 62,615 |
| 7 | 777,225 | .1459/.7720/1.0000 | 287,946 | 199,207 | .1213/.4873/1.0000 | 64,318 |
| 8 | 702,466 | .1775/.9048/1.0000 | 316,663 | 157,631 | .1281/.5850/1.0000 | 54,225 |
| 9 | 621,452 | .1639/.9673/1.0000 | 263,653 | 150,284 | .1392/.6216/1.0000 | 56,632 |
| 10 | 684,563 | .1603/.9546/1.0000 | 286,501 | 145,667 | .1316/.6343/1.0000 | 52,858 |
| 11 | 514,207 | .1779/.9985/1.0000 | 236,114 | 150,788 | .1186/.6323/1.0000 | 50,089 |
| 12 | 438,770 | .1798/1.0000/1.0000 | 204,875 | 118,598 | .1327/.7188/1.0000 | 44,101 |
| 13 | 470,453 | .1836/1.0000/1.0000 | 222,797 | 132,565 | .1184/.6997/1.0000 | 44,872 |
| 14 | 443,531 | .1919/1.0000/1.0000 | 216,451 | 116,020 | .1364/.7515/1.0000 | 45,002 |
| 15 | 403,226 | .1807/1.0000/1.0000 | 190,793 | 111,150 | .1334/.7852/1.0000 | 42,809 |
| 16 | 380,253 | .1943/1.0000/1.0000 | 187,417 | 112,273 | .1327/.7808/1.0000 | 42,868 |

### 6A.5 距离分桶 Recall@2m 和位置误差

各 epoch 的 GT 固定不变：

| 类别 | 0～20m | 20～40m | 40～60m | 60～80m |
| --- | ---: | ---: | ---: | ---: |
| car | 23,005 | 23,369 | 9,101 | 0 |
| truck | 3,490 | 3,659 | 2,233 | 0 |
| total | 26,495 | 27,028 | 11,334 | 0 |

由于全周 eval ROI 的 x 上限是 50m，`40～60m` 实际只含 40～50m；`60～80m` GT 恒为 0，不能用 0.0000 Recall 评价远距能力。该表还只统计 `gt_x>=0`，没有覆盖 6V ROI 中后向 `x<0` 目标或按径向距离统计纯侧向目标。下表每格为 car/truck Recall@2m：

| e | 0～20m | 20～40m | 40～60m | 60～80m |
| ---: | --- | --- | --- | --- |
| 1 | .9192/.8238 | .8793/.7950 | .8240/.6408 | 0/0（GT=0） |
| 2 | **.9474/.8481** | **.8975**/.7786 | .8227/**.6610** | 0/0（GT=0） |
| 3 | .9392/**.8630** | .8784/.7212 | .8178/.6064 | 0/0（GT=0） |
| 4 | .9434/.8713 | .8572/.7346 | .7797/.5549 | 0/0（GT=0） |
| 5 | .9383/.8599 | .8398/.7166 | .7457/.5007 | 0/0（GT=0） |
| 6 | .9314/.8573 | .8472/.6857 | .7577/.5038 | 0/0（GT=0） |
| 7 | .9282/.8613 | .8482/.6895 | .7495/.5065 | 0/0（GT=0） |
| 8 | .9241/.8304 | .8200/.6726 | .6945/.4478 | 0/0（GT=0） |
| 9 | .9210/.8232 | .8202/.6668 | .6810/.4277 | 0/0（GT=0） |
| 10 | .9218/.8034 | .8080/.6297 | .6536/.3865 | 0/0（GT=0） |
| 11 | .9109/.8172 | .7909/.6480 | .6426/.4026 | 0/0（GT=0） |
| 12 | .8996/.8063 | .7837/.6193 | .6399/.3699 | 0/0（GT=0） |
| 13 | .9027/.7946 | .7705/.6248 | .6048/.3766 | 0/0（GT=0） |
| 14 | .8993/.7874 | .7836/.6215 | .6163/.3735 | 0/0（GT=0） |
| 15 | .8893/.7891 | .7819/.6051 | .6105/.3609 | 0/0（GT=0） |
| 16 | .8859/.7811 | .7625/.6103 | .5773/.3538 | 0/0（GT=0） |

按两类 GT 数加权，40～60m 聚合 Recall 在 epoch2 最佳（0.7908），epoch6 为 0.7077，epoch16 降到 0.5333。因此 epoch2 是有明确证据的中远距 challenger。

位置误差只对 center distance≤2m 的 matched TP 计算。下表每格为 `car x/y/z MAE | truck x/y/z MAE`，单位 m：

| e | 0～20m | 20～40m | 40～60m |
| ---: | --- | --- | --- |
| 1 | .391/.302/.082 \| .718/.496/.169 | .762/.256/.119 \| .873/.451/.211 | .849/.328/.177 \| .871/.550/.354 |
| 2 | .394/.277/.074 \| .652/.484/.167 | .758/.228/.110 \| .828/.436/.210 | .844/.296/.159 \| .879/.516/.327 |
| 3 | .361/.278/.079 \| .667/.481/.157 | .750/.225/.112 \| .809/.412/.183 | .811/.286/.159 \| .865/.486/.298 |
| 4 | .360/.275/.074 \| .619/.435/.179 | .748/.224/.108 \| .821/.408/.198 | .822/.270/.154 \| .858/.475/.275 |
| 5 | .386/.275/.071 \| .617/.435/.145 | .730/.214/.102 \| .837/.414/.179 | .810/.261/.135 \| .866/.476/.249 |
| 6 | .348/.259/.074 \| .585/.442/.148 | .729/.208/.108 \| .774/.398/.169 | .807/.250/.149 \| .860/.469/.248 |
| 7 | .341/.263/.076 \| .601/.414/.157 | .729/.213/.107 \| .828/.403/.177 | .833/.260/.154 \| .831/.462/.231 |
| 8 | .344/.272/.080 \| .607/.444/.141 | .716/.213/.111 \| .812/.408/.153 | .816/.250/.151 \| .848/.449/.213 |
| 9 | .346/.261/.072 \| .607/.439/.147 | .720/.206/.104 \| .792/.393/.163 | .814/.242/.141 \| .828/.464/.217 |
| 10 | .369/.265/.069 \| .601/.457/.138 | .726/.204/.104 \| .801/.372/.153 | .832/.242/.140 \| .837/.473/.212 |
| 11 | .360/.268/.066 \| .592/.435/.140 | .741/.203/.104 \| .827/.401/.150 | .819/.233/.141 \| .864/.476/.217 |
| 12 | .337/.257/.064 \| .600/.414/.137 | .716/.197/.101 \| .783/.376/.157 | .838/.229/.141 \| .848/.468/.208 |
| 13 | .343/.267/.065 \| .580/.437/.135 | .726/.199/.102 \| .795/.384/.146 | .822/.234/.141 \| .837/.484/.218 |
| 14 | .342/.266/.065 \| .583/.434/.141 | .732/.204/.103 \| .789/.388/.149 | .817/.234/.137 \| .815/.495/.220 |
| 15 | .330/.255/.064 \| .588/.433/.140 | .715/.198/.102 \| .809/.383/.147 | .850/.231/.139 \| .801/.466/.206 |
| 16 | .331/.249/.063 \| .568/.427/.139 | .718/.191/.101 \| .789/.385/.152 | .840/.223/.135 \| .853/.466/.228 |

汇总没有逐距离分桶的 size/yaw 误差；只有逐 epoch/类别的全局 AOE/ASE 和每轴位置 MAE/p50/p90。不能补造不存在的分桶尺寸或角度统计。

### 6A.6 训练收敛和异常

统计方法：训练日志每 10 iter 输出一次区间聚合值；对每个 epoch 后 20% 稳定区间（iter≥1680）取中位数，避免只取最后一条。epoch1～17 各有 42 个点。日志只提供 positive/negative bag loss，没有独立 bbox/dir loss 字段。

| e | LR median | positive median | negative median | total loss median | grad_norm median | 状态 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 7.640e-4 | 1.1221 | .1477 | 1.2696 | 2.2319 | 完整 |
| 2 | 7.240e-4 | .8483 | .1405 | .9879 | 1.7357 | 完整 |
| 3 | 6.840e-4 | .7138 | .1351 | .8491 | 1.5296 | 完整 |
| 4 | 6.440e-4 | .6331 | .1280 | .7622 | 1.4821 | 完整 |
| 5 | 6.040e-4 | .5757 | .1214 | .6972 | 1.4243 | 完整 |
| 6 | 5.640e-4 | .5336 | .1166 | .6501 | 1.4214 | 完整、val best |
| 7 | 5.240e-4 | .4999 | .1124 | .6128 | 1.4099 | 完整 |
| 8 | 4.840e-4 | .4693 | .1089 | .5785 | 1.3929 | 完整 |
| 9 | 4.440e-4 | .4476 | .1061 | .5544 | 1.4091 | 完整 |
| 10 | 4.040e-4 | .4252 | .1030 | .5289 | 1.4091 | 完整 |
| 11 | 3.640e-4 | .4093 | .1006 | .5093 | 1.4357 | 完整 |
| 12 | 3.240e-4 | .3916 | .0988 | .4900 | 1.4395 | 完整 |
| 13 | 2.840e-4 | .3739 | .0954 | .4695 | 1.4266 | 完整 |
| 14 | 2.440e-4 | .3588 | .0934 | .4526 | 1.4552 | 完整 |
| 15 | 2.040e-4 | .3478 | .0907 | .4390 | 1.4303 | 完整 |
| 16 | 1.640e-4 | .3316 | .0885 | .4212 | 1.4565 | 完整 |
| 17 | 1.240e-4 | .3222 | .0861 | .4096 | 1.4750 | 完整、待 val |

异常审计：

- 所有记录的 positive/negative/total loss 都是有限值，没有 NaN/Inf loss；
- 共 14 个日志点出现 `grad_norm=inf`：epoch17 有 2 次，epoch1/3/4/5/6/7/8/9/11/12/14/15 各 1 次；
- `grad_norm=inf` 不集中在后段窗口，后 20% 只有 epoch3/4/5/17 各 1 次；
- 日志没有 OOM、CUDA error、Traceback、数据跳过或明确的 fp16 skip-step 记录；
- 唯一 WARNING 是 COCO 初始化 state_dict 与 Fast-BEV 模型不完全匹配，随后训练正常开始；
- loss 从 epoch6 约 0.650 持续降至 epoch17 约 0.410；正式评估窗口的 epoch16 canonical mAP 已从 e6 的 0.453768 降到 0.4192，40～60m 聚合 Recall 也从 0.7077 降到 0.5333。这是训练目标继续拟合、验证召回/排序退化的过拟合式分离，不是数值崩溃。epoch17 已排除出本轮选型，不再补评。

### 6A.7 checkpoint 决策和停止结论

| 角色 | epoch | 证据 | 处理建议 |
| --- | ---: | --- | --- |
| canonical best | **6** | `mAP=0.453768`；BEV@0.25=0.6453、BEV@0.5=0.4787 也同时最佳 | 必须保留 |
| 中远距 challenger | **2** | 40～60m 聚合 Recall@2m=0.7908 最佳；car/truck 分别 .8227/.6610 | 建议保留，用于远距回归 A/B |
| AOE 单项 best | 10 | mAOE=2.9294° | 不单独保留；canonical/Recall 已明显低于 e6/e2 |
| ATE 单项 best | 12 | mATE=0.7599m | 不单独保留；TP 条件误差改善伴随召回下降 |
| ASE 单项 best | 16 | mASE=0.1683 | 不单独作为 challenger |
| terminal | **16** | 正式评估窗口最后一个 checkpoint；mAP=0.4192 | 作为训练终止对照归档 |
| 排除项 | 17 | 日志确认完整训练并保存，但用户决定不再作为本轮参考 | 不补评、不纳入 best/challenger/terminal |

停止判断：

- best 是 epoch6；
- epoch7～11 已构成连续 5 个已评估 epoch 未刷新，首次满足 patience=5；
- 到 epoch16 已连续 10 个已评估 epoch 未刷新；
- 因此不建议继续 epoch18～20，也没有证据支持超过 epoch20；
- epoch17 已完整保存但按用户决定排除，本轮无需再判断其相对 e6 的精度；正式结论冻结为 e6 best、e2 challenger、e16 terminal，不运行 e18～20。

#### 6A.7.1 与 mono single-frame 的收敛现象及 LR 判断

共同点是训练 loss 与 checkpoint 选择指标解耦：优化器持续降低 bag loss，并不等于 canonical mAP 仍在改善。但两者程度不同：

| run | 对比窗口 | 后 20% loss 中位数 | canonical mAP | 预测数 | 解释 |
| --- | --- | --- | --- | --- | --- |
| 6V-B0 | e6→e16 | 0.6501→0.4212（-35.2%） | 0.453768→0.4192（-0.034568） | 1,040,837→492,526（-52.7%） | 持续明显退化，伴随召回/候选数下降 |
| mono S0 | e6→e13 | 0.9460→0.8474（-10.4%） | 0.3541→0.356425（+0.002325） | 1,556,452→1,237,771（-20.5%） | 窄幅平台中仍刷新 best |

因此 6V 与 S0 在“loss 继续下降但验证收益变弱”上相似，但不能视为同一种强度：S0 是正常平台/微增，6V 是明确的后期过拟合或置信度/候选选择极化；它也没有达到旧 mono 高 LR 实验 AP 接近 0 的 collapse 程度。

当前日志没有证明 `8e-4` 导致数值不稳定：loss 有限、grad_norm 中位数稳定、没有 OOM/NaN，且 e6 已得到强 best。仅凭后期 loss 下降不能要求降低 LR 重训。下一轮应先修复 `force_resize`，然后：

1. 以相同数据、seed、schedule 和 `8e-4` 建立单变量修复基线；
2. 若仍出现早期 e5～e7 峰值后持续 Recall/mAP 下降，再以 `4e-4` 做唯一 LR challenger；
3. 可先把两组都跑到 e8 并逐 epoch eval，再只延长领先者；不建议直接照搬 mono S0 的 `1e-4`，因为相机数、全局 batch、ROI 和梯度统计不同。

### 6A.8 数据门禁、test 边界和 mono 比较口径

- 本轮 train/val pkl 都是 20260717 版本，本地没有这些 pkl、converter sidecar 或对应 `data_gate.md/json`，但用户确认 PTH/pkl 已在内网保存；
- 用户确认本轮六目数据门禁失败项的根因是各 clip 起始/末尾帧不齐全，并批准该已知边界条件放行；本轮记录为 `PASS_WITH_ACCEPTED_CLIP_BOUNDARY_EXCEPTION`，不再作为精度基线冻结的阻塞项；
- 该放行结论来自数据所有者确认，本地没有门禁报告可复算；不能将旧 mono 的 `labels_without_frame=851/90` 和 1237 个空 GT label 直接套用到本轮 6V；
- 本轮 resolved `test.ann_file` 与 val 相同，现有指标是 val 精度基线，没有独立 test 结果；
- 6V val 使用 6 camera、全周 ROI `[-50,-50,-5,50,50,3]`，共 64,857 GT；mono B0/S0 使用 cam0、前视 ROI `[0,-35,-5,80,35,3]` 和 cam0 可见性过滤；
- 因 GT 可见性、ROI、相机输入和评估集合不同，6V `0.453768` 与 mono B0 `0.344934`、S0 `0.356425` 只能作有限背景，不能直接宣称 6V 提升或下降。EXP-6V-B0 内部逐 epoch 才是 checkpoint 选择依据。

### 6A.9 最终资产归档缺口

| 资产 | 日志/汇总对应路径 | 本地状态 |
| --- | --- | --- |
| canonical best | `<work_dir>/epoch_6.pth` | 用户确认内网已保存；本地未同步，大小/SHA256 未知 |
| far challenger | `<work_dir>/epoch_2.pth` | 用户确认内网已保存；本地未同步，大小/SHA256 未知 |
| terminal | `<work_dir>/epoch_16.pth` | 用户确认内网已保存；本地未同步，大小/SHA256 未知 |
| 排除的 continuation | `<work_dir>/epoch_17.pth` | 日志确认保存，但不纳入本轮 checkpoint 角色 |
| 对应 metrics/result/eval log | 应按 epoch2/6/16 归档 | 用户确认相关资产在内网；本地不能核验实际文件名或 hash |
| resolved config snapshot | 日志内有打印文本 | 没有训练时独立 config 副本；当前仓库同名 config 已按日志显式收口，仍需在 legacy 环境解析复核 |
| code commit | 应写入 checkpoint meta/资产 manifest | 训练日志未打印；当前本地 HEAD 不能倒推训练 commit |
| data/calibration | 20260717 train/val pkl 及其标定 | 用户确认内网已保存；本地未同步版本 sidecar/hash |

本轮精度结论已闭环，不再以补评 epoch17、同步本地资产或重跑 gate 为前置条件。建议后续在内网 manifest 中补齐 e2/e6/e16 的大小/SHA256、训练时 config/commit/data/calibration，仅用于资产治理与复现，不改变当前 checkpoint 决策。

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

下表为每个 epoch 的 780/785 采样点。旧记录曾把 `positive_bag_loss` 简写为总 loss；这里按日志原字段拆开，`loss=positive_bag_loss+negative_bag_loss`：

| epoch | LR | positive bag | negative bag | 总 loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.275e-5 | 1.4288 | 0.1493 | 1.5780 |
| 2 | 8.672e-5 | 1.0241 | 0.1477 | 1.1717 |
| 3 | 8.005e-5 | 0.8866 | 0.1470 | 1.0335 |
| 4 | 7.338e-5 | 0.8131 | 0.1472 | 0.9603 |
| 5 | 6.672e-5 | 0.7919 | 0.1452 | 0.9371 |
| 6 | 6.005e-5 | 0.7556 | 0.1422 | 0.8978 |
| 7 | 5.338e-5 | 0.7297 | 0.1434 | 0.8731 |
| 8 | 4.672e-5 | 0.7090 | 0.1411 | 0.8502 |
| 9 | 4.005e-5 | 0.7032 | 0.1447 | 0.8479 |
| 10 | 3.338e-5 | 0.6916 | 0.1420 | 0.8336 |
| 11 | 2.672e-5 | 0.6704 | 0.1433 | 0.8138 |
| 12 | 2.005e-5 | 0.6766 | 0.1393 | 0.8159 |
| 13 | 1.338e-5 | 0.6627 | 0.1429 | 0.8055 |
| 14 | 6.718e-6 | 0.6621 | 0.1395 | 0.8015 |
| 15 | 5.096e-8 | 0.6496 | 0.1433 | 0.7928 |

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
| positive bag loss | 0.7919 | 0.6496 | 继续下降 |
| 总 loss | 0.9371 | 0.7928 | 继续下降 |
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

用户先用 EXP-MONO-B0 epoch5 PTH 测试生产环境图片，观察到：

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

随后用户使用 EXP-MONO-S0 epoch8 对同类无真值生产车辆数据做 PTH 推理和可视化，主观结果为：

- 相比 B0 四时序模型重复当前帧，整体效果明显更好；
- 直道约 40m 内大部分目标可以检出，预测位置在可视化上较准确；
- 道路尽头转弯、目标车与自车约 90°、车门朝向自车等横向姿态仍较差；
- 40～80m 中远距仍较差。

这与 S0 val 的量化趋势并不矛盾：S0 全局 canonical mAP 只比 B0 小幅提高，而 epoch8 truck Recall@2m 在 40～60m/60～80m 仅为 0.7107/0.7124。生产结论属于无 GT 的定性证据，足以支持“原生单帧继续作为产品 base”，但不能替代标准 val/test，也不能单凭可视化决定 yaw loss 或输入分辨率修改。

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

门禁代码和旧 B0 train/val pkl 实跑已经完成。报告路径：

work_dirs/n7_pkl_gate_b0/data_gate.md

已确认：

- train=200,874 infos/364 clips，val=24,909 infos/45 clips；
- train/val token overlap=0、clip overlap=0；
- 每个 info 都是 cam0，缓存图为 704×256，K native 尺寸为 1600×900，5 项 distortion 字段完整；
- key pose 覆盖率 100%，每个 clip 首帧 history shortage 符合时序边界；
- GT、类别、x/y/z/l/w/h/yaw 和当前 pinhole-visible 统计可审计。

当前报告仍为 `FAIL`，原因不是脚本崩溃，而是旧 converter sidecar 明确记录：

- train/val `labels_without_frame=851/90`；
- train 另有 1237 个 `labels_without_gt` 被旧 converter 丢弃，当前 pkl 无法恢复这些负样本；
- legacy pkl 没有 manifest provenance 和显式 `gt_box_origin`；
- 当前没有独立 test pkl，config 暂时继续以 val 充当 test。

因此“门禁实现”已经完成，“正式数据门禁 PASS”尚未完成。下一步应先查看 `labels_without_frame` 样例和空 GT 丢弃对负样本比例的影响，再决定对旧 B0 风险书面接受，或使用新 strict converter+manifest 重生正式 train/val/test。

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

2026-07-28 状态更新：上述 `force_resize` 收口已经完成。实现拆分了基础几何和
可选随机阶段，增加 `view_layout` 契约、1×1/1×4/6×4 camera-slot 共享回归、
真实旧实现 fixture 和 LUT 一致性检查；本地完整 suite 为 `Ran 50 tests, OK`。
内网用相同 epoch13 checkpoint、val pkl 和 `atol=0, rtol=0` 完成 pre/post T7，
input、2D feature、BEV input、raw cls/bbox/dir logits 和 decoded boxes 全部
bit-exact。代码已提交并推送为
`134d5f3124e0d11f40dbcbade9e5bfe07c28ff96`。这只关闭旧 704×256 基线的
兼容性问题，不代表 AUG1 已完成；F4 GT 可见性、F5 resize/rotate 上游亚像素
语义以及 `force_resize=False` 时序随机共享仍需在启用相关路径前独立处理。

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

### 16.7 当前训练进度和路径

日志记录的 work_dir：

work_dirs/n7_mono_704_256_single_frame/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260713

本地早期证据：

- 训练日志：`20260713_113748.log`；
- eval 汇总：`test_results/eval_summary.md`；
- 4×NVIDIA L20，64/GPU，全局 batch 256，AdamW2 LR=1e-4，15 epoch 上限；
- 从与 B0 相同的 COCO 2D 权重初始化；
- `evaluation.interval=999999`，训练机不做逐 epoch validation；
- 当前本地文件只确认 epoch1～9 checkpoint，并记录 epoch10 到 190/785；
- 早期 eval 汇总只覆盖 epoch1～9，best key 虽显示兼容名称 `mAP/center_dist`，数值定义与 canonical `mAP` 相同。

后续内网收口结果：

- 用户确认 epoch1～15 已全部完成并评估；
- 最终 canonical best 为 epoch13：mAP=`0.356425`、BEV mAP@0.5=`0.3773`、mATE=`0.8588m`、mAOE=`4.0545°`、mASE=`0.2132`；
- epoch10 保留为正向中远距 challenger，epoch15 只作终止归档；
- epoch13 已完成 dataset/production PTH 同图逐级对齐，并进入后续 ONNX/LUT/板端参考验证；
- 本地仍缺 e10～15 完整 eval summary、PTH/result pkl、resolved config 和代码/数据/标定 hash；因此指标结论可以更新，资产闭环仍需补录。

epoch 末 780/785 采样 loss：

| epoch | positive bag | negative bag | 总 loss |
| ---: | ---: | ---: | ---: |
| 1 | 1.4643 | 0.1520 | 1.6163 |
| 2 | 1.0624 | 0.1501 | 1.2125 |
| 3 | 0.9366 | 0.1509 | 1.0875 |
| 4 | 0.8729 | 0.1466 | 1.0195 |
| 5 | 0.8336 | 0.1455 | 0.9791 |
| 6 | 0.7944 | 0.1428 | 0.9371 |
| 7 | 0.7674 | 0.1421 | 0.9095 |
| 8 | 0.7582 | 0.1441 | 0.9023 |
| 9 | 0.7449 | 0.1437 | 0.8885 |

epoch10 的 190/785 采样总 loss=0.8775 是早期同步快照，只能说明当时训练仍在继续，不能与各 epoch 的 780/785 采样点直接当成同位置比较；最终 checkpoint 选择以后续完整 val 指标为准。

### 16.8 S0 逐 epoch eval

| epoch | mAP（center） | BEV mAP@0.5 | mATE | mAOE | mASE | eval det |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2894 | 0.2796 | 0.9501 | 6.212° | 0.2928 | 3,110,680 |
| 2 | 0.3305 | 0.3388 | 0.9160 | 5.516° | 0.2522 | 2,597,226 |
| 3 | 0.3308 | 0.3422 | 0.9230 | 6.472° | 0.2487 | 2,328,429 |
| 4 | 0.3349 | 0.3496 | 0.9029 | 4.581° | 0.2514 | 1,888,694 |
| 5 | 0.3518 | 0.3625 | 0.8818 | 4.771° | 0.2437 | 1,622,288 |
| 6 | 0.3541 | 0.3693 | 0.8742 | 4.264° | 0.2333 | 1,556,452 |
| 7 | 0.3527 | 0.3680 | 0.8816 | 4.428° | 0.2257 | 1,583,307 |
| 8 | 0.355078 | 0.3650 | 0.8681 | 4.592° | 0.2297 | 1,584,150 |
| 9 | 0.3548 | **0.3713** | **0.8667** | **3.971°** | **0.2256** | 1,349,207 |
| **13** | **0.356425** | **0.3773** | **0.8588** | **4.055°** | **0.2132** | 待补录 |

早期截至 epoch9：

- canonical mAP 选 epoch8；
- BEV mAP、ATE、AOE、ASE 单项在 epoch9 更好；
- epoch6～9 已进入 0.3527～0.3551 的平台区间，没有 collapse；
- 当时 epoch8 后只有 epoch9 一个已评估 checkpoint 未刷新。

最终 15 epoch 结果由 epoch13 刷新 canonical mAP，因此当前主模型必须写作 e13，不能继续沿用早期 e8 wording。e10～12/e14～15 的精确逐行值待同步完整 eval summary 后补表。

### 16.9 S0 阶段/最终 best 距离诊断

早期 epoch8 分类别结果：

| 类别 | center_AP | BEV AP@0.5 | ATE | AOE | ASE |
| --- | ---: | ---: | ---: | ---: | ---: |
| car | 0.5258 | 0.5023 | 0.7361m | 3.558° | 0.1564 |
| truck | 0.1844 | 0.2277 | 1.0000m | 5.625° | 0.3030 |

epoch8 Recall@2m/MAE：

| 类别 | GT x | Recall@2m | x MAE | y MAE | z MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| car | 0～20m | 0.9894 | 0.3205m | 0.1585m | 0.0586m |
| car | 20～40m | 0.9420 | 0.6817m | 0.2021m | 0.0998m |
| car | 40～60m | 0.8497 | 0.8297m | 0.2180m | 0.1360m |
| car | 60～80m | 0.7976 | 0.8591m | 0.2747m | 0.1931m |
| truck | 0～20m | 0.8803 | 0.6248m | 0.3438m | 0.1432m |
| truck | 20～40m | 0.7916 | 0.7960m | 0.4497m | 0.1749m |
| truck | 40～60m | 0.7107 | 0.8702m | 0.3937m | 0.2339m |
| truck | 60～80m | 0.7124 | 0.9201m | 0.3467m | 0.2389m |

MAE 只描述 center distance≤2m 的已匹配 TP；距离增大时 Recall 同时下降，因此不能用 MAE 单独声称 40～80m 精度已经解决。最终 e13 的 truck Recall@2m 在 40～60m/60～80m 为 `0.6498/0.6435`，低于 e10 的 `0.6853/0.6693`。因此 e13 保持 canonical 主模型，e10 只作为中远距 challenger；不能按单个切片替换主 best。

### 16.10 与 B0 的阶段性结论

| 口径 | B0 | S0 | 阶段性结论 |
| --- | ---: | ---: | --- |
| epoch5 canonical mAP | 0.344934 | 0.3518 | S0 约 +0.0069 |
| epoch5 BEV mAP@0.5 | 0.3562 | 0.3625 | S0 +0.0063 |
| epoch5 mATE | 0.8899m | 0.8818m | S0 略好 |
| epoch5 mAOE | 4.315° | 4.771° | S0 较差约 0.456° |
| 各自最终 best mAP | B0 e5 0.344934 | S0 e13 0.356425 | S0 +0.011491 |
| 各自最终 best BEV@0.5 | 0.3562 | 0.3773 | S0 +0.0211 |
| 各自最终 best mATE | 0.8899m | 0.8588m | S0 -0.0311m |
| 各自最终 best mAOE | 4.315° | 4.055° | S0 -0.260° |
| 各自最终 best mASE | 0.2369 | 0.2132 | S0 -0.0237 |

这轮首要结论不是“时序一定无效”，而是：在板端没有 pose、生产侧只能给 B0 重复当前帧的现实约束下，原生 S0 更匹配部署输入，现有 val 指标也没有因去掉时序而下降。B0 仍是 pose-aware 离线对照和未来教师候选。

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

## 17A. EXP-MONO-GEOM1-A：1600×900 等比缩放中心裁剪

### 17A.1 Run-of-record 契约

- 唯一配置：
  `configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18_n7_1600x900_scale_crop_dist_train.py`。
  该文件为完整独立 config，不再依赖多层 `_base_`。
- 本地配置 SHA256：
  `d134b0bdbc4af0ec3aea391b884c20f46e792248eceecefa742f0639992b89fc`；
  冻结的 resolved contract SHA256：
  `dcfd9882e1e8d2f75348d8d693d0bc411adbb7930e47be306da18b8c67d961d4`。
  旧记录 `8b4ecfdd...439e` 是正式 work_dir 写入配置前的训练前版本；当前文件的
  修改时间早于正式日志启动时间，且日志中的 resolved config 与当前关键字段一致。
  训练日志本身未嵌入配置文件 byte hash，因此不能把这一交叉核对表述为 checkpoint
  已记录了配置 SHA256。
- `force_resize=False`、`enable_random_aug=False`、`n_images=1`、`n_times=1`。
  train/test 使用同一确定性图像契约：`1600×900 -> resize 704×396 ->
  crop(0,70,704,326) -> 704×256`。对应
  `post_rot=diag(0.44,0.44)`、`post_tran=[0,-70,0]`。
  GEOM1-A 的配置、回归测试和正式训练日志均证明这是原生单帧 `1×1`；项目中的
  `n_images=1,n_times=4` 是 temporal B0 契约，不能用于改写本实验历史。
- ROI、voxel、anchor、类别、GT 口径、BEV 增强、AdamW2、`lr=1e-4`、
  seed=0、15 epoch、COCO 初始化和 dynamic FP16 均与 S0 保持一致。
- 初始化权重来源：
  `pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth`。
  日志确认从该路径加载兼容的 COCO Cascade Mask R-CNN R18/FPN 参数；它不是
  Fast-BEV 完整模型 resume。当前工作区没有该权重实体，SHA256 待内网补录。
- 正式 work_dir：
  `work_dirs/n7_mono_1600_900_scale_crop/EXP-MONO-GEOM1-A/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260729`。

### 17A.2 PKL 迁移和数据门禁

内网按旧 704×256 train/val clip manifest 生成了一套指向原生图片的 PKL：

- train PKL：
  `data/N7_1600_900/mono_front_pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_train_20260728.pkl`；
- val PKL：
  `data/N7_1600_900/mono_front_pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_val_20260728.pkl`；
- train：200,874 infos、364 clips、3,086,700 GT；
- val：24,909 infos、45 clips、300,701 GT；
- train/val token 和 clip overlap 均为 0；全部 camera 为 cam0，图片和 K metadata
  均为 1600×900；`test=val`，没有虚构独立 test。

新旧严格迁移对比输出
`N7_PKL_MIGRATION_COMPARE=PASS failures=0`，确认 token/order、clip、timestamp、
GT box/name、K、distortion、extrinsic 和 split 保持不变；只允许的图片路径、尺寸和
相应 metadata 发生变化。PKL/manifest 精确 SHA256 已由内网报告生成，但实体和数值尚未
同步到当前工作区，不能在本文虚构补写。

全量图片检查覆盖 train 200,874 和 val 24,909 张，缺失文件、JPEG header 尺寸不符和
解码错误均为 0。train/val 各抽 200 个 info 的 geometry check 失败数和最大误差均为 0。

原始 strict data gate 仍保留 `FAIL`：converter provenance 记录 train/val 分别有
851/90 个 clip 首尾标签无对应图像，train 另有 1,237 个空 GT 标签被旧 converter
丢弃。该缺帧数量与旧 704×256 manifest 的已知边界条件一致，用户明确批准本实验放行，
因此实验级结论为 `PASS_WITH_ACCEPTED_CLIP_BOUNDARY_EXCEPTION`。不得修改原始报告来
制造无条件 PASS。

已知内网报告入口为
`work_dirs/validate_n7_1600_900_candidate/data_gate.md` / `data_gate.json` 及其
train/val 子目录中的 `geometry_check.json`；全量图片扫描和 migration 报告/hash 应随
最终资产一起归档。当前本地未同步这些实体，因此本文只记录用户回传结果，不声称本地复算。

### 17A.3 F4、crop offset 和真实 pipeline

F4 对 225,783 个 info 做了全量 keep/yaw 统计，并以固定 stride=100 对 2,259 帧、
104,660 个记录做像素指标。中心裁剪 top=70 的结果为：

| split | eval keep | train keep | eval-only | 比例 |
| --- | ---: | ---: | ---: | ---: |
| train | 958,707 | 958,606 | 101 | 0.0105% |
| val | 85,239 | 85,229 | 10 | 0.0117% |

合计差异 111/1,043,946=`0.01063%`，即 keep-mask 一致率 99.98937%。111 个目标全部
位于 0～20m：car/truck=65/46；yaw 中 107 个位于 [-45°,45°]，另有
[-90°,-45°] 和 [135°,180°] 各 2 个。20～80m 损失为 0，也没有侧向 yaw 聚集。
两个被像素指标抽中的 crop-dropped 框位于 x=0.45/1.54m、y≈4.5m，其 box edge
穿过相机近裁面，形成异常大的负 distortion margin。

direct stretch 的 train/eval visibility 完全一致；已测 crop offset 均不为精确零差异。
top=140 只损失 55 个目标，但会把输出主点 y 从 128 移到 58，不应为少 56 个边缘目标
破坏冻结的中心裁剪契约。最终保留 top=70；原始零容忍状态仍是
`F4_REVIEW_REQUIRED`，实验级记录为
`PASS_WITH_ACCEPTED_NEAR_FIELD_CROP_EXCEPTION`，不修改 train/eval GT 口径。

真实 `RandomAugImageMultiViewImage` pipeline 可视化 train/val 均输出
`N7_SCALE_CROP_PIPELINE_VIS=PASS`，分别渲染 34/32 张，覆盖远距、上下边缘、
非零 yaw 和 train 的两个采样 crop-dropped 目标。这一证据验证了真实训练 pipeline 的
图像结果、post transform、lidar2img 与标记点同步；LANCZOS 定性工具不作为数值替代。
F4 和样本选择的内网入口为 `work_dirs/n7_1600_900_scale_crop_analysis`；保留其中
统计报告、`sample_selection_train.json`、`sample_selection_val.json` 和最终渲染图。

### 17A.4 测试和吞吐门禁

优化后的 F4 analyzer focused 回归 10/10、完整 `tools/tests` 62/62、相关
`py_compile` 和 whitespace 检查通过。单卡 20GB 测试机以 batch64/workers8 运行
10 warmup + 50 measure，两个候选均完成 dataloader、forward、backward 和 optimizer：

| 配置 | iter mean | data mean | peak allocated | wall time |
| --- | ---: | ---: | ---: | ---: |
| S0 | 11.810291s | 0.094683s | 9926.1 MiB | 13m08.878s |
| GEOM1-A | 13.276409s | 0.199413s | 9929.3 MiB | 15m04.678s |

GEOM1-A 的 iter+data 增加 13.19%，样本吞吐下降 11.66%，显存仅增加 3.2 MiB。
这是同机单变量正确性/开销证据，不能把绝对耗时外推到 4×L20。
对应机器可读结果为 `work_dirs/n7_1600_900_training_gate/s0_50iter.json` 和
`work_dirs/n7_1600_900_training_gate/geom1a_50iter.json`。

### 17A.5 正式训练和最终精度结论

4×L20、每卡 batch64、workers8、dynamic FP16 的 15 epoch run 已于
2026-07-30 完成，e1～15 全部独立 val。canonical best=e11：

历史启动记录与正式日志 resolved config 交叉确认的训练命令为：

~~~bash
cd /mnt/liujiaren/fastbev-python-custom-fastbev-adapter

CONFIG="configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18_n7_1600x900_scale_crop_dist_train.py" \
WORK_DIR="work_dirs/n7_mono_1600_900_scale_crop/EXP-MONO-GEOM1-A/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260729" \
TRAIN_BATCH=64 \
EVAL_BATCH=8 \
WORKERS=8 \
NO_VALIDATE=1 \
bash tools/dist_train_ljr.sh
~~~

`tools/dist_train_ljr.sh` 使用 `torch.distributed.launch --nproc_per_node=4`，并把上述
batch/worker 参数通过 `--cfg-options` 写入运行时配置。正式 train/val PKL 分别是：

- `./data/N7_1600_900/mono_front_pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_train_20260728.pkl`；
- `./data/N7_1600_900/mono_front_pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_val_20260728.pkl`。

当前工作区的可核验运行资产为：

| 资产 | 当前路径 | SHA256 / 状态 |
| --- | --- | --- |
| 完整训练日志 | `<work_dir>/20260729_110603.log` | `1abbd30779fe65b6ff5e255e5eb3751c61427f29ee86f45753395f8b3ee9079b` |
| 逐轮评估汇总 | `<work_dir>/test_results/eval_summary.md` | `8e75a795c571f8e59ca468300232be2184ba20cb542d2b789f7943cf5ceb30f0` |
| 本地 data gate Markdown | `work_dirs/validate_n7_1600_900_candidate/data_gate.md` | `aa012f55c7315b98dc55aa4fe9c92b5501fabaf0c590734aba846eb15d4aeef6` |

日志确认 epoch11、epoch13、epoch15 均执行过 checkpoint 保存。按正式 work_dir 的
运行契约，三者文件名为 `epoch_11.pth`、`epoch_13.pth`、`epoch_15.pth`，逐轮评估
约定输出为 `test_results/epoch_N_val_results.pkl`、`epoch_N_metrics.json` 和
`epoch_N_test.log`。这些文件当前均未同步，不能做实体存在性、大小或 SHA256 核验；
train/val PKL、预训练权重和缓存图也未同步，均明确列为待内网补录。

| 指标 | S0 e13 | GEOM1-A e11 | 差值 |
| --- | ---: | ---: | ---: |
| mAP | 0.356425 | 0.381993 | +0.025568 |
| BEV mAP@0.5 | 0.3773 | 0.4016 | +0.0243 |
| mATE@2m | 0.8588m | 0.8615m | +0.0027m |
| mAOE@2m | 4.0545° | 4.6376° | +0.5831° |
| mASE@2m | 0.2132 | 0.2113 | -0.0019 |

GEOM e11 相对 S0 e13 的 40～60/60～80m Recall@2m：

- car：`0.8313/0.7864 -> 0.8738/0.8178`，提升 `+0.0425/+0.0314`；
- truck：`0.6498/0.6435 -> 0.7029/0.7112`，提升 `+0.0531/+0.0677`。

方向误差没有同步改善：car AOE `3.0892° -> 3.5533°`，truck
`5.0199° -> 5.7219°`。因此 GEOM1-A 证明等比 resize+center crop 对检测和
中远距召回有效，但不能解释为 yaw 已解决。e11 冻结为下一轮候选基线；e13 的
overall AOE=`4.4920°`、mASE=`0.2045` 为 GEOM 内最优，保留为方向/尺度
challenger；e15 为 terminal。e15 没有刷新，不设计 EXT5。

e11 作为后续微调起点的依据是预先声明的 canonical mAP 主键在 e1～15 中最高，
同时相对 S0 e13 提升总体 mAP、BEV@0.5 和 car/truck 中远距 Recall。e13 仅在
GEOM 内 AOE/ASE 更优，生产转弯场景未显示相对 e11 的明确 yaw 收益；e15 只是完整
schedule 的终点。因此选 e11 不代表方向问题已经解决，后续应由真实城区路口和倾斜
环道 GT 覆盖来补齐数据分布。

## 18. 当前下一步

GEOM1-A 的门禁、15 epoch 训练和逐 epoch val 均已完成。当前阶段转为冻结资产、
生产/yaw 分桶回归及后续真实数据微调设计。

### P0：立即执行

1. 冻结 e11/e13/e15 三个角色并补 PTH/result/config/data/calibration SHA256；
2. 对 S0 e13、GEOM e11/e13 跑同一有 GT 生产回归，覆盖转弯、非共线 yaw、
   40～80m 和倾斜弯道；
3. 新增 yaw 分桶评估和 180°翻转率；现有 mAOE 只覆盖匹配 TP 且受高快共线分布
   主导，不能直接解释生产斜向车辆；
4. 对“x 方向一个车身偏差”增加更宽匹配阈值/未匹配统计；当前 x MAE 只统计
   center distance<=2m 的 TP，会排除大误差样本；
5. 不做 EXT5，不清理门禁、日志、评估或 checkpoint 证据。

### P1：下一轮数据准备

1. `DATA1-CITY-YAW`：筛选城区路口/转弯 N7 真实 GT，统计 scene/frame/instance、
   yaw/距离/类别分布并按 scene 防泄漏；
2. `DATA3-BANKED-RING`：准备倾斜环道 N7 真实 GT，先核对标定、地面姿态和
   yaw-only box 口径；两类真实数据可并行准备并保留来源标签；
3. `DATA2-ENDURANCE-PSEUDO`：准备试车场耐久路 BEVFusion 伪标注，但后置于真实
   GT；冻结 teacher/hash/阈值并人工抽检，验证集保持纯真实 GT；
4. 用同步 lidar/image 独立验证生产车型物理标定，并继续 distortion visibility；
5. 继续真实芯片 tensor dump、实际板端 runtime 和真实 INT8 数值验收。

### P2：GEOM1-A 后单变量 A/B

1. 优先从 GEOM e11 做真实 yaw/弯道数据微调；算力有限时可在质量审计后合并
   DATA1+DATA3，但不能同时混入 AUG1 或伪标注；
2. 在真实数据 winner 或 GEOM e11 上独立做 AUG1；
3. 再评估 DATA2 伪标注生产域微调；
4. 其后才考虑更高输入、CBGS、anchor、NMS、蒸馏和 LR/schedule。

### P3：低优先级 6V 重训

1. 当前不重开 EXP-6V-B0；该基线继续冻结为 e6/e2/e16；
2. `force_resize` 修复和等价性回归前置条件已满足，但没有新的排期或算力授权时仍不启动；
3. 第一轮保持原 `8e-4`、数据、seed、schedule、ROI/voxel/anchor 不变，只验证 `force_resize` 修复；
4. 若修复后仍出现 e5～e7 达峰并持续退化，再做独立 `4e-4` challenger；该项不阻塞 P0/P1。

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

- 旧 EXP-6V-00：只保留可训练性历史事实，没有完整 eval；
- EXP-6V-B0：e1～17 完整训练并保存，正式选型窗口为逐 epoch val 的 e1～16；final canonical best=e6、mAP=0.453768、BEV@0.5=0.4787，challenger=e2，terminal=e16；
- EXP-6V-B0 停止：e6 后连续 10 个已评估 epoch 未刷新，已满足 patience=5；e17 及以后排除出本轮参考，不补评、不继续训练；
- EXP-6V-B0 资产：用户确认 PTH/pkl 已在内网保存；本地没有实体、metrics JSON、result pkl 或 eval log，因此不能核验大小、SHA256、训练 commit 和 20260717 data/calibration hash；
- 6V 性能：动态畸变是主要耗时，主线已保留批量投影和 fused scatter；
- mono 四时序结构：成立；
- 早期 mono：存在 vertical flip、高 LR 和 score collapse 历史；
- 正式 B0：15 epoch 完成，epoch5 canonical mAP=0.344934，为当前时序对照；
- B0 epoch15：positive bag loss=0.6496、总 loss=0.7928，TP 回归误差更低，但 AP 未刷新，不延长到 20；
- xyz：x/y signed mean 不能代表绝对精度；box-origin 修复已通过已有 result pkl 重评，car/truck z 偏差恢复正常；
- 产品：近期原生单帧；
- S0：用户确认 15 epoch 完成，最终 best=e13，canonical mAP=0.356425、BEV@0.5=0.3773；精确最终资产 hash 待补录；
- S0 vs B0：同口径 mono 内部为 S0 小幅领先；6V 与 mono 的相机、ROI、GT 过滤和 eval 集合不同，禁止直接用 mAP 数值宣称 6V 升降；
- 数据门禁：用户确认 EXP-6V-B0 的失败项来自六目 clip 起始/末尾不齐，并批准 `PASS_WITH_ACCEPTED_CLIP_BOUNDARY_EXCEPTION`；本地未同步报告，本轮仍只有 val、无独立 test 结果；
- `force_resize`：两阶段重构、8 场景 fixture、完整 50 项测试和真实 epoch13 T7 已通过；commit `134d5f3` 已推送；
- 收敛/LR：6V 与 S0 都有 loss/验证指标解耦，但 6V 是 e6 后持续明显退化，S0 是平台后仍在 e13 刷新；若未来启动 EXP-6V-B1，先用原 `8e-4` 控制，仅在退化复现时 A/B `4e-4`；
- Git：当前主分支基线为 `134d5f3124e0d11f40dbcbade9e5bfe07c28ff96`；历史 B0/S0 代码基线为 `2d8ab7d`、`c3ec682`；
- GEOM1-A：15 epoch 和逐 epoch val 已完成；e11 canonical mAP=0.381993、
  BEV=0.4016，中远距 Recall 明显提升；mAOE 比 S0 e13 差 0.5831°，yaw 未解决；
- 当前主任务：冻结 GEOM e11/e13/e15 资产，建立 yaw/生产回归，并优先准备
  `DATA1-CITY-YAW` 与 `DATA3-BANKED-RING` 真实 GT；AUG1 独立后续，BEVFusion
  耐久路伪标注再后置。独立 test、物理标定和真实板卡/INT8 继续并行。
