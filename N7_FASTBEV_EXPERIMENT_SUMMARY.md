# N7 Fast-BEV 实验汇总

> 更新时间：2026-07-13；范围：N7 6V 可训练基线、6V 性能消融、单目前视四时序实验、当前原生单帧 S0 计划。
>
> 详细记录见：[N7_FASTBEV_EXPERIMENT_DETAILS.md](N7_FASTBEV_EXPERIMENT_DETAILS.md)
>
> 后续代办见：[N7_MONO_FRONT_SINGLE_FRAME_TODO.md](N7_MONO_FRONT_SINGLE_FRAME_TODO.md)

## 1. 先看结论

1. **N7 6V 基线已经证明可以训练，但现有日志不能证明完成了 20 epoch，也没有可确认的 6V eval 指标。**
   2026-06-26 日志确认保存了 epoch1、epoch2，日志停在 epoch3 的 630/1858；epoch2 末采样 loss 为 0.8506。

2. **单目前视四时序的模型结构适配成立。**
   四时序基线是每个样本 1 个相机 × 4 个时间步，共 4 张图；从 6V 改成单目时没有错误缩减四时序 3D fusion 的 1024 输入通道。

3. **早期单目 full-data 实验和 2026-07-08 B0 必须分开。**
   早期高学习率实验出现过 AP 和 score collapse；其中一版还继承了不适用于前视 ROI 的 BEV vertical flip。2026-07-08 B0 没有发生这种 collapse。

4. **当前可信的四时序对照是 EXP-MONO-B0，最佳权重为 epoch5。**
   canonical `mAP`（center-distance mAP；历史键 `mAP/center_dist`）在 epoch5 为 0.344934；对应 BEV IoU mAP@0.5 为 0.3562。epoch15 的训练 loss 已降到约 0.65，但 AP 没有刷新，因此没有证据支持继续训练到 20 epoch。

5. **epoch5 之后是平台波动，不是崩溃。**
   epoch6～15 的 `mAP` 约在 0.3366～0.3433 间波动；ATE、AOE、ASE 仍有部分改善，但主 AP 未超过 epoch5。

6. **旧 eval 中 xyz 的小 signed mean 不能解释为“只有几厘米定位误差”。**
   x/y signed mean 会正负抵消；epoch5 的总体 mATE@2m 仍为 0.8899 m。旧 z_mean 约为 car -0.83 m、truck -1.67 m，符合 prediction bottom-center 与 GT gravity-center 混算产生的负半车高偏差。代码修复后，用户已用已有 result pkl 直接重评并确认 car/truck 的 z 偏差恢复正常；精确新数值和结果文件 hash 待归档。

7. **当前产品路线已经切换为原生单帧。**
   板端虽有六轴 IMU 和轮速，但 pose 接口尚未完成，因此近期不把四时序重复当前帧作为最终方案。EXP-MONO-S0 配置已准备好，尚无训练权重和指标。

8. **下一轮先做公平的 S0 与 B0 对比，不同时调其他变量。**
   单帧保持 cam0、704×256、ROI、voxel、anchor、distortion、GT 过滤、全局 batch、LR 和 COCO 初始化与 B0 一致；上限 15 epoch，独立单卡逐 epoch eval，优先比较 epoch5，并采用 best 后连续 5 个 epoch 未刷新则停止的规则。

## 2. 证据等级

| 标记 | 含义 |
| --- | --- |
| 日志确认 | 可由 /workspace 下原始训练日志直接核对 |
| 评估汇总确认 | 可由 /workspace/eval_summary.md 直接核对 |
| 代码确认 | 当前仓库配置或实现已存在，但不代表真实内网数据/权重已跑通 |
| 历史记录 | 来自已有审查、交接或实验笔记；原始指标 JSON/PTH 当前不在本机 |
| 推导 | 根据日志、work_dir 和相邻实验关系得到，已明确标注，不等同于原始证据 |
| 待内网补录 | 当前环境无数据、权重或 legacy MMDetection 运行栈，需在内网补齐 |

## 3. 统一实验编号

| 实验 ID | 日期 | 实验目的 | 关键配置 | 训练/权重状态 | 主要结果 |
| --- | --- | --- | --- | --- | --- |
| SMK-MONO-00 | 2026-06-27 | 验证动态单目四时序模型路径 | nuScenes mini；1 camera × 4 times | 2080 Ti 完成 1 iter 和 test forward | train loss 4.0493；仅结构 smoke，不是 N7 精度 |
| EXP-6V-00 | 2026-06-26 | N7 6V 四时序可训练基线 | 6 camera × 4 times；4×L20；每卡 24；LR 8e-4；20e 上限 | epoch1/2 已保存；epoch3 仅到 630/1858 | epoch2 末 loss 0.8506；无可信 eval |
| ABL-6V-01 | 2026-06-30 | 6V backproject/畸变耗时消融 | 同类 6V full-data 链路 | 性能实验，不选精度权重 | 动态畸变旧路径约 20～22s/iter；优化后约 16s；关闭畸变约 6.8～7.2s |
| EXP-MONO-T1 | 2026-07-03 | 单 sequence 单目四时序收敛验证 | 1×L20；每卡 64；Adam 4e-4；20e；vertical flip 0.5 | epoch1～20 已保存 | 单类有效 val；BEV mAP@0.5 在 e15 为 0.3415，truck GT=0，不可代表全量 |
| EXP-MONO-T2 | 2026-07-03 | full-data 初始单目四时序 | 4×L20；每卡 64；约 8e-4；仍继承 vertical flip 0.5 | 日志确认 epoch1～10 已保存 | e5 训练内 eval BEV mAP@0.5=0.0974；历史外部 e10 eval 近 0 |
| EXP-MONO-T3 | 2026-07-06 | 关闭 vertical flip 后的 clean 高 LR 重跑 | 4×L20；每卡 64；vertical flip 0；高 LR | 本机日志确认 e1～4；历史记录还有 e5 复评 | 历史逐 epoch BEV mAP@0.5：e2 最好 0.2897，e5=0；不能与 B0 混用 |
| EXP-MONO-T4 | 2026-07-07 | 从 T3 epoch2 权重低 LR 恢复试验 | load_from T3 e2；不是 resume；Adam 2e-4；8e 上限 | e1～4 保存，e5 到 700/785 | loss 继续降至约 0.83；本机无完整 e5/eval，未形成正式基线 |
| EXP-MONO-B0 | 2026-07-08 | 正式四时序单目前视对照 | 4×L20；每卡 64；AdamW2 1e-4；15e；vertical flip 0 | epoch1～15 均保存；best=e5 | canonical mAP 0.344934；BEV mAP@0.5 0.3562 |
| EXP-MONO-S0 | 待内网 | 原生单帧产品基线 | 1 camera × 1 time；其余尽量与 B0 相同 | 配置已就绪，尚无 PTH | 待训练；先对比双方 e5 和各自 best |

## 4. 当前权重与路径总表

以下均为日志记录的**历史内网路径**，不表示当前本机存在文件。同步到新内网环境后应保存实际绝对路径和 SHA256。

### 4.1 6V 基线

- 初始化权重：
  pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth
- 历史 work_dir：
  /mnt/liujiaren/fastbev-python/work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260626
- 日志确认存在过：
  epoch_1.pth、epoch_2.pth
- 配置对应入口：
  configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256_dist_train.py
- 证据日志：
  /workspace/20260626_174629.log

### 4.2 单 sequence 收敛验证

- 历史 work_dir：
  /mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251031_164821_10_batch24_work8_20260703
- 日志确认存在过：
  epoch_1.pth ～ epoch_20.pth
- 注意：目录名包含 batch24，但 resolved config 明确是 samples_per_gpu=64；以后以 resolved config 为准。
- 证据日志：
  /workspace/20260703_160930.log

### 4.3 full-data 初始/clean/recovery

- EXP-MONO-T2：
  /mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260703
- EXP-MONO-T3：
  /mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260706_vflip0_clean
- EXP-MONO-T4：
  /mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260706_vflip0_clean_resume_epoch2
- T4 实际初始化权重：
  work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260706_vflip0_clean/epoch_2.pth
- 证据日志：
  /workspace/20260703_172933.log、/workspace/20260706_184052.log、/workspace/20260707_152258.log

### 4.4 当前四时序 B0

- 历史 work_dir：
  /mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260708
- 当前 best 权重：
  /mnt/liujiaren/fastbev-python-custom-fastbev-adapter/work_dirs/n7_mono_704_256/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260708/epoch_5.pth
- 日志确认存在过：
  epoch_1.pth ～ epoch_15.pth
- 训练日志：
  /workspace/20260708_122556.log
- 逐 epoch eval 汇总：
  /workspace/eval_summary.md
- 配置对应入口：
  configs/fastbev/custom/custom_fastbev_mono_front_r18_dist_train.py
- 待内网补录：
  checkpoint SHA256、resolved config 副本、代码 commit/diff、train/val manifest 与 pkl hash、车型标定 hash。

### 4.5 原生单帧 S0

- 模型配置：
  configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18.py
- 训练配置：
  configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18_dist_train.py
- work_dir、PTH、eval 结果：
  尚未产生，禁止预填虚构路径。

## 5. B0 核心指标

### 5.1 逐 epoch 总体趋势

| epoch | mAP（center） | BEV mAP@0.5 | mATE@2m | mAOE | mASE | 结论 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.2891 | 0.2638 | 0.9712 | 7.491° | 0.2983 | 初始可用 |
| 2 | 0.3200 | 0.3222 | 0.9277 | 6.431° | 0.2705 | 明显提升 |
| 3 | 0.3299 | 0.3415 | 0.9188 | 5.328° | 0.2533 | 继续提升 |
| 4 | 0.3404 | 0.3504 | 0.8949 | 4.948° | 0.2448 | 接近最佳 |
| **5** | **0.344934** | **0.3562** | **0.8899** | **4.315°** | **0.2369** | **当前 best** |
| 10 | 0.3366 | 0.3468 | 0.8833 | 4.020° | 0.2249 | AP 未刷新 |
| 15 | 0.3387 | 0.3511 | 0.8768 | 3.696° | 0.2154 | 误差改善，AP 仍低于 e5 |

说明：

- canonical `mAP`：先对每类 0.5m、1m、2m、4m center-distance AP 求平均得到 `center_AP`，再对有 GT 类别求平均；`mAP/center_dist` 仅作为兼容别名。
- BEV 指标只使用显式名称 `mAP/bev_iou@0.5` / `BEV_mAP@0.5`。历史 schema v1 的裸 `mAP` 曾表示 BEV，新汇总会优先读取旧记录中的 `mAP/center_dist`。
- ATE/AOE/ASE 只统计 center distance ≤2m 的 matched TP，不能覆盖漏检、远距未匹配目标或生产域差异。
- watch 距离分桶表展示 GT、TP@2m、Recall@2m 和各轴 MAE=mean(abs(pred−gt))；p50/p90 与 signed mean 分别保留在持久化精度表和偏置诊断表。

### 5.2 epoch5 分类别结果

| 类别 | center_AP | BEV AP@0.5 | AP@0.5m | AP@1m | AP@2m | AP@4m | ATE | AOE | ASE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| car | 0.5135 | 0.4843 | 0.2448 | 0.4365 | 0.6385 | 0.7341 | 0.7586m | 3.793° | 0.1594 |
| truck | 0.1764 | 0.2282 | 0.0254 | 0.0958 | 0.2261 | 0.3583 | 1.0212m | 4.836° | 0.3144 |

验证集 GT：

- car：69,492
- truck：15,747
- total：85,239
- car:truck 约为 4.41:1

当前最明显的精度短板是 truck 和严格 center-distance AP；不能只看总体 BEV AP。

## 6. 为什么 loss 降到 0.65 仍不建议继续到 20 epoch

EXP-MONO-B0 的采样总 loss：

| epoch | 末次采样 loss | mAP（center） |
| ---: | ---: | ---: |
| 1 | 1.4288 | 0.2891 |
| 5 | 0.7919 | **0.3449** |
| 10 | 0.6916 | 0.3366 |
| 15 | 0.6496 | 0.3387 |

loss 是训练目标，canonical `mAP` 是模型选择目标。epoch5 之后 loss 继续下降，但验证 AP 没有同步提升，表现更符合平台期或轻度过拟合。当前合理做法是保留 15 epoch 上限和 patience=5，而不是把单轮直接延长到 20。

## 7. 单帧 S0 与四时序 B0 的公平对比

| 项目 | EXP-MONO-B0 | EXP-MONO-S0 |
| --- | --- | --- |
| 相机 | cam0 | cam0 |
| n_images | 1 | 1 |
| n_times | 4 | 1 |
| 每样本图片数 | 4 | 1 |
| sequential | True | False |
| 历史帧/pose | 3 个历史帧，训练数据 pose-aware | 无 |
| 3D fuse | 1024→256 | 256→256 |
| 输入 | 704×256 stretch | 相同 |
| ROI | [0,-35,-5,80,35,3] | 相同 |
| voxel | [160,140,4]；[0.5,0.5,1.5] | 相同 |
| distortion | True | 相同 |
| GT 过滤 | cam0 + front ROI | 相同 |
| batch | 64/GPU × 4 | 相同 |
| optimizer | AdamW2，LR 1e-4 | 相同 |
| 初始化 | COCO 2D pretrained | 相同 |
| 上限 | 15 epoch | 15 epoch |
| eval | B0 已逐 epoch复评 | 单卡服务器独立监控 |

第一轮只改变 n_times 和对应 3D fuse 通道。输入几何、anchor、CBGS、NMS、增强、LR 等优化放到 S0 基线形成后逐项 A/B。

## 8. 当前代码审查项状态

| 项目 | 当前状态 | 对下一轮训练的影响 |
| --- | --- | --- |
| 板端无 pose 的时序不等价 | 已做产品决策：近期原生单帧 | S0 路线正确；B0 只作离线对照 |
| prediction box origin / z | 代码已修；已有 result pkl 重评确认 car/truck z 偏差正常 | 不需重训；补归档新数值、结果路径和 hash |
| ONNX wrapper 分支 | 代码已修；合成 ONNX/ORT diff=0 | 真实 S0 checkpoint 仍需内网导出 |
| 原生单帧配置 | 已新增并通过本地静态/合成检查 | 需 legacy 环境 model/dataloader/train/test smoke |
| pkl strict 门禁 | 未完成 | 正式长训练前必须完成 split、相机、尺寸、类别、空帧审计 |
| distortion-aware GT 可见性 | 未完成 | 当前模型 backproject 与 GT 过滤口径仍不完全一致 |
| 生产推理 vs dataset 同图对齐 | 未完成 | 角度偏、20～30m 不清楚前先排除预处理/标定/坐标差异 |
| `force_resize` 关闭随机增强 | 首轮 A/B 后作为第 4 个正确性收口项 | 先拆分基础变换并验证无增强等价，再单独 A/B 随机增强 |
| 704×256 stretch / 输入几何 | 待 S0 后 A/B | 可能影响远距小目标和预训练迁移上限 |
| CBGS、anchor、NMS | 待数据统计后单变量 A/B | 当前不应盲调 |

## 9. eval 可视化现状

当前工具支持只指定**前视相机面板**大小，而不是强制整个视频画布：

~~~bash
python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
  --gt-pkl <val_gt.pkl> \
  --pred-pkl <epoch_5_val_results_portable.pkl> \
  --data-root <N7_704_256> \
  --output-dir <vis_output> \
  --camera-ids cam0 \
  --camera-size 1600 900 \
  --video-only \
  --video-group sequence \
  --max-frames-per-video 2000 \
  --score-thr 0.2 \
  --workers 4
~~~

- camera-size 1600 900 只约束前视相机面板。
- 保留 BEV 或顶部信息栏时，完整 canvas 会大于 1600×900。
- 如只要严格 1600×900 的纯前视画布，再加 no-bev 和 no-header。
- clip 模式输出：
  output/dataset/sequence/clip/clip.mp4
- sequence 模式输出：
  output/dataset/sequence/sequence.mp4

## 10. 下一步执行口径

1. 在内网先构建 EXP-MONO-S0 merged config，检查 dataloader 为 [B,1,3,256,704]、3D neck 输入为 [B,256,160,140]，完成 1 iter train/test smoke。
2. 四卡只训练并逐 epoch 保存；使用 NO_VALIDATE=1。单卡服务器运行 tools/eval_epoch_checkpoints.py --watch。
3. 优先得到 S0 epoch5，与 B0 epoch5 同口径比较；同时保留双方 best。
4. best 连续 5 个已评估 epoch 未刷新时停止，不用训练 loss 代替 early-stop 判断。
5. 归档 box-origin 重评后的 car/truck z signed mean、绝对误差、result pkl 路径和 hash；该修复已确认不需要重新训练。
6. 初步 S0/B0 对比后，正式长训练前完成四项收口：pkl 门禁、生产/dataset 同图逐级对齐、distortion-aware GT 可见性一致，以及 `force_resize` 基础变换/随机增强拆分的无增强等价性回归。
7. 正式 S0 基线建立后，再依次做输入几何/分辨率、增强、CBGS、anchor、NMS、初始化/蒸馏等单变量实验。

## 11. 每轮实验必须补录的资产

- 实验 ID、目的和唯一 work_dir。
- 实际启动命令和 resolved config。
- 代码 commit；若 dirty，保存 git diff 和 diff SHA256。
- Python/CUDA/PyTorch/mmcv/mmdet/mmdet3d 版本。
- train/val/test manifest、pkl、车型 calibration 的路径与 SHA256。
- 初始化权重、每个 checkpoint 的路径与 SHA256。
- 每 epoch loss、LR、fp16 overflow/skip-step 次数。
- 每 epoch canonical mAP、BEV AP、car/truck、距离分桶 Recall@2m/MAE 和 score 分布。
- best key、best epoch、early-stop 原因。
- 生产回归集版本、推理配置、可视化和人工结论。
- PTH/ONNX/ORT/板端使用的预处理、box_origin、坐标系和后处理版本。
