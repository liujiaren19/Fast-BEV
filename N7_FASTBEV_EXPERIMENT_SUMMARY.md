# N7 Fast-BEV 实验汇总

> 更新时间：2026-08-01；范围：N7 6V 四时序精度基线、单目前视四时序 B0、原生单帧 S0、PC 侧 ONNX/LUT 与板端参考链路、`force_resize` 收口、1600×900 GEOM1-A 结果及后续数据/AUG 路线。
>
> 详细记录见：[N7_FASTBEV_EXPERIMENT_DETAILS.md](N7_FASTBEV_EXPERIMENT_DETAILS.md)
>
> 后续代办见：[N7_MONO_FRONT_SINGLE_FRAME_TODO.md](N7_MONO_FRONT_SINGLE_FRAME_TODO.md)

## 1. 先看结论

1. **旧 EXP-6V-00 仍只是可训练性证据；新的 EXP-6V-B0 已形成首个可信 6V 精度基线。**
   2026-07-17 新 run 使用 6 camera × 4 times、4×L20、24/GPU、AdamW2 `8e-4` 和 20 epoch 上限。唯一日志已完整覆盖并保存 epoch1～17；训练外 `eval_summary.md` 对 epoch1～16 做了 val 复评。用户决定 epoch17 及以后不再作为本轮 checkpoint 选型参考，因此正式评估窗口冻结为 e1～16：canonical `mAP` 最佳为 epoch6 `0.453768`，BEV mAP@0.5 同样在 epoch6 最佳（`0.4787`）。

2. **EXP-6V-B0 已正式闭环：e6=best、e2=中远距 challenger、e16=terminal。**
   epoch6 后，epoch7～16 已连续 10 个已评估 epoch 未刷新 canonical mAP，远超 patience=5；epoch16 已降至 `0.4192`。训练后段 loss 继续下降，但 val mAP、BEV、中远距 Recall 和预测数同步退化。epoch17 虽已保存，但按用户决策排除出正式选型窗口，不再要求补评，也不继续 epoch18～20。

3. **单目前视四时序的模型结构适配成立。**
   四时序基线是每个样本 1 个相机 × 4 个时间步，共 4 张图；从 6V 改成单目时没有错误缩减四时序 3D fusion 的 1024 输入通道。

4. **早期单目 full-data 实验和 2026-07-08 B0 必须分开。**
   早期高学习率实验出现过 AP 和 score collapse；其中一版还继承了不适用于前视 ROI 的 BEV vertical flip。2026-07-08 B0 没有发生这种 collapse。

5. **当前可信的四时序对照是 EXP-MONO-B0，最佳权重为 epoch5。**
   canonical `mAP`（center-distance mAP；历史键 `mAP/center_dist`）在 epoch5 为 0.344934；对应 BEV IoU mAP@0.5 为 0.3562。epoch15 末采样的 `positive_bag_loss=0.6496`，但总 loss 是 0.7928；AP 没有刷新，因此仍没有证据支持继续训练到 20 epoch。

6. **epoch5 之后是平台波动，不是崩溃。**
   epoch6～15 的 `mAP` 约在 0.3366～0.3433 间波动；ATE、AOE、ASE 仍有部分改善，但主 AP 未超过 epoch5。

7. **旧 eval 中 xyz 的小 signed mean 不能解释为“只有几厘米定位误差”。**
   x/y signed mean 会正负抵消；epoch5 的总体 mATE@2m 仍为 0.8899 m。旧 z_mean 约为 car -0.83 m、truck -1.67 m，符合 prediction bottom-center 与 GT gravity-center 混算产生的负半车高偏差。代码修复后，用户已用已有 result pkl 直接重评并确认 car/truck 的 z 偏差恢复正常；精确新数值和结果文件 hash 待归档。

8. **原生单帧 EXP-MONO-S0 已完成 15 epoch，并最终冻结 epoch13。**
   最终 canonical `mAP=0.356425`、BEV mAP@0.5=`0.3773`、mATE=`0.8588m`、mAOE=`4.0545°`、mASE=`0.2132`。epoch10 保留为中远距 challenger，epoch15 只作终止归档；最终 checkpoint/config/data/calibration hash 仍需补录。

9. **S0 仍是近期产品 base，但优势是温和、且 6V/mono 指标不能直接横比。**
   S0 e13 比 B0 e5 canonical mAP 高 `0.011491`，ATE/AOE/ASE 也更好；但 e13 truck Recall@2m 在 40～60m/60～80m 为 `0.6498/0.6435`，低于 e10 的 `0.6853/0.6693`。6V 使用全环视 ROI 和不同 GT 口径，不能用 `0.453768` 直接宣称相对 mono 的产品收益。

10. **EXP-6V-B0 数据门禁按已知 clip 边界例外放行。**
    用户确认 20260717 六目数据的门禁失败根因是各 clip 起始/末尾帧不齐全，并批准该已知边界条件放行；本轮记录为 `PASS_WITH_ACCEPTED_CLIP_BOUNDARY_EXCEPTION`。本地没有同步对应 gate/pkl，不能独立复算，但该问题不再阻塞本轮基线冻结。resolved test 仍复用 val pkl，因此当前指标只能称为 val 精度基线，不是独立 test。

11. **PC 侧单帧部署链路已闭环到浮点参考，下一步分成资产归档和真实板卡验收两条线。**
    dataset/production epoch13 PTH 对齐、Torch-CUDA fixed LUT、FP ONNX 功能对齐、独立 CPU 板端参考和 38 项 mono-front 回归已经完成；`force_resize` 提交门禁对应完整 `tools/tests` 50 项，叠加尚未提交的 GEOM1-A 工具和回归后当前工作区完整 discovery 为 62 项。真实芯片 tensor dump、实际板端 runtime 和真实 INT8 数值验证仍未完成。用户确认 6V 的 PTH、pkl 和相关资产已在内网保存；当前工作区未同步实体/hash，但不再作为本轮精度结论的阻塞项。

12. **`force_resize` 已完成代码、回归、内网 epoch13 T7 和提交收口。**
    两阶段基础几何/随机增强已拆分，8 组真实旧实现 fixture 和完整 50 项测试通过；内网 pre/post 的 input、2D feature、BEV input、raw logits、decoded boxes 在 `atol=0, rtol=0` 下 bit-exact。代码已提交并推送为 `134d5f3124e0d11f40dbcbade9e5bfe07c28ff96`。未来 AUG1 仍必须使用独立配置；F4 GT 可见性、F5 上游亚像素/旋转语义和普通路径时序共享仍是启用增强前的独立事项。

13. **EXP-MONO-GEOM1-A 已完成并证明输入几何升级有效，但没有解决 yaw。**
    训练前 migration/F4/真实 pipeline/50 iter 门禁及两个约定例外保持不变；4×L20、每卡 64、dynamic FP16、15 epoch 和全部逐 epoch val 已完成。canonical best=e11：`mAP=0.381993`、BEV@0.5=`0.4016`、mATE=`0.8615m`、mAOE=`4.6376°`、mASE=`0.2113`。相对 S0 e13，mAP `+0.025568`、BEV `+0.0243`，car 40～60/60～80m Recall@2m 提升 `+0.0425/+0.0314`，truck 提升 `+0.0531/+0.0677`；但整体 mAOE 变差 `+0.5831°`，car/truck AOE 分别变差 `+0.4641°/+0.7020°`。因此 e11 是下一轮候选基线，S0 e13 仍保留为生产回退；e13 保留为 GEOM 内方向/尺度 challenger，e15 为 terminal，不启动 EXT5。

14. **下一阶段优先补真实 yaw/弯道数据，AUG1 和伪标注后置。**
    当前 val 以高快共线车辆为主，缺少对城区路口非 0/180° yaw 的充分约束。先建立 yaw 分桶和生产有 GT 回归，再准备 `DATA1-CITY-YAW` 与 `DATA3-BANKED-RING` 两类 N7 真实 GT；随后在 winner 上独立做 `AUG1`。试车场耐久路 BEVFusion 数据登记为 `DATA2-ENDURANCE-PSEUDO`，必须按伪标注管理、保持纯真实 GT 验证集，优先级低于真实 GT。

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
| EXP-6V-B0 | 2026-07-17～07-23 | N7 6V 四时序完整 val 精度基线 | 6 camera × 4 times；4×L20；每卡 24；AdamW2 8e-4；20e 上限 | e1～17 完成/保存；正式选型窗口冻结为 e1～16 | final best=e6；challenger=e2；terminal=e16；patience=10 |
| ABL-6V-01 | 2026-06-30 | 6V backproject/畸变耗时消融 | 同类 6V full-data 链路 | 性能实验，不选精度权重 | 动态畸变旧路径约 20～22s/iter；优化后约 16s；关闭畸变约 6.8～7.2s |
| EXP-MONO-T1 | 2026-07-03 | 单 sequence 单目四时序收敛验证 | 1×L20；每卡 64；Adam 4e-4；20e；vertical flip 0.5 | epoch1～20 已保存 | 单类有效 val；BEV mAP@0.5 在 e15 为 0.3415，truck GT=0，不可代表全量 |
| EXP-MONO-T2 | 2026-07-03 | full-data 初始单目四时序 | 4×L20；每卡 64；约 8e-4；仍继承 vertical flip 0.5 | 日志确认 epoch1～10 已保存 | e5 训练内 eval BEV mAP@0.5=0.0974；历史外部 e10 eval 近 0 |
| EXP-MONO-T3 | 2026-07-06 | 关闭 vertical flip 后的 clean 高 LR 重跑 | 4×L20；每卡 64；vertical flip 0；高 LR | 本机日志确认 e1～4；历史记录还有 e5 复评 | 历史逐 epoch BEV mAP@0.5：e2 最好 0.2897，e5=0；不能与 B0 混用 |
| EXP-MONO-T4 | 2026-07-07 | 从 T3 epoch2 权重低 LR 恢复试验 | load_from T3 e2；不是 resume；Adam 2e-4；8e 上限 | e1～4 保存，e5 到 700/785 | loss 继续降至约 0.83；本机无完整 e5/eval，未形成正式基线 |
| EXP-MONO-B0 | 2026-07-08 | 正式四时序单目前视对照 | 4×L20；每卡 64；AdamW2 1e-4；15e；vertical flip 0 | epoch1～15 均保存；best=e5 | canonical mAP 0.344934；BEV mAP@0.5 0.3562 |
| EXP-MONO-S0 | 2026-07-13 | 原生单帧产品基线 | 1 camera × 1 time；4×L20；每卡 64；AdamW2 1e-4；15e | e1～15 完成；最终 best=e13 | canonical mAP 0.356425；BEV mAP@0.5 0.3773 |
| EXP-MONO-GEOM1-A | 2026-07-29～07-30 | 原生 1600×900 等比缩放+中心裁剪单变量对照 | 1 camera × 1 time；4×L20；每卡 64；AdamW2 1e-4；dynamic FP16；15e | e1～15 训练/评估完成；best=e11 | mAP 0.381993；BEV 0.4016；中远距 Recall 提升，AOE 退化 |

## 4. 当前权重与路径总表

以下均为日志记录的**历史内网路径**，不表示当前本机存在文件。同步到新内网环境后应保存实际绝对路径和 SHA256。

### 4.1 6V 基线

- 初始化权重：
  pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth
- 旧 EXP-6V-00 work_dir：
  /mnt/liujiaren/fastbev-python/work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260626
- 旧日志确认存在过：
  epoch_1.pth、epoch_2.pth
- 当前 EXP-6V-B0 work_dir：
  work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717
- 当前训练日志：
  work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717/20260717_173206.log
- 当前逐 epoch eval 汇总：
  work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717/test_results/eval_summary.md
- 当前状态：
  epoch1～17 checkpoint 已由日志确认保存；训练外 val 复评到 epoch16。用户决定 epoch17 及以后不纳入本轮选型，当前本机没有训练/eval 进程；PTH/pkl 由用户确认已在内网保存，本地只同步日志和汇总。
- 最终 best：
  epoch_6.pth，canonical `mAP=0.453768`，BEV mAP@0.5=`0.4787`。
- 配置对应入口：
  configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256_dist_train.py
- 旧实验对照日志：
  /workspace/20260626_174629.log

### 4.1A 当前 6V 精度基线 EXP-6V-B0

- 本地 work_dir：
  `work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717`
- 训练日志：
  `20260717_173206.log`，epoch17 保存后同步快照 SHA256 `bf5a98f4908ff1ac5d0d49fec687ae564577a9760dfd0c63492b2c681ba0635b`
- eval 汇总：
  `test_results/eval_summary.md`，SHA256 `ad2254ef828efcacf4a5b66cd2f556a037799f6620d4d442b42c91bc6a2ed580`
- 训练完整性：
  epoch1～17 各有 209 个日志点并出现保存标记。
- eval 完整性：
  汇总覆盖正式选型窗口 epoch1～16；epoch17 未评且按用户决策不再要求补评。目录内没有 `epoch_*.pth`、`eval_summary.csv/json`、逐 epoch metrics JSON、result pkl 或 eval log，不能在本地计算 checkpoint hash，也不能实体核验 checkpoint/result 一一对应。
- 当前选择：
  canonical best=`epoch_6.pth`；40～60m Recall challenger=`epoch_2.pth`；正式 terminal=`epoch_16.pth`。epoch17 仅保留为内网原始训练延续资产，不纳入本轮选型和结论。
- 终止结论：
  epoch6 后已有连续 10 个已评估 epoch 未刷新；本轮已经停止并闭环，不补评 e17，不继续 epoch18～20。
- 复现缺口：
  训练日志没有代码 commit；20260717 train/val pkl、标定版本和门禁报告未同步到本地，但用户确认 PTH/pkl 已在内网保存，并确认门禁仅为可接受的 clip 边界不齐。分析开始时仓库同名 dist config 只覆盖 `optimizer.lr`，沿论文 base 会继承标准 `Adam`，与本轮日志 resolved `AdamW2` 不一致；分析期间该配置被外部未提交改动显式收口为 `_delete_=True` 的 `AdamW2` 和本轮数据/schedule，本文没有修改该配置。真实训练结论继续以日志为准。

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
- 日志记录的 work_dir（相对内网仓库）：
  work_dirs/n7_mono_704_256_single_frame/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260713
- 早期本地留存训练日志/逐 epoch 汇总只覆盖到 e9；后续完成状态来自用户确认和内网实跑记录。
- 最终训练状态：
  epoch_1.pth～epoch_15.pth 均完成，epoch13 为 canonical best，epoch10 保留为中远距 challenger，epoch15 为终止归档。
- 最终 best：
  epoch_13.pth，canonical `mAP=0.356425`，BEV mAP@0.5=`0.3773`，mATE=`0.8588m`，mAOE=`4.0545°`，mASE=`0.2132`。
- 待内网补录：
  epoch13/epoch10/epoch15 PTH、result pkl、resolved config、代码/数据/标定 SHA256。

### 4.5A 当前 GEOM1-A

- 唯一 run-of-record 配置：
  `configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18_n7_1600x900_scale_crop_dist_train.py`
- 配置为完整独立文件，不再依赖多层 `_base_`；本地文件 SHA256：
  `d134b0bdbc4af0ec3aea391b884c20f46e792248eceecefa742f0639992b89fc`。
  旧记录 `8b4ecf...439e` 对应正式 work_dir 写入前的训练前版本，不再作为
  run-of-record 配置 hash。
- 已冻结 resolved contract SHA256：
  `dcfd9882e1e8d2f75348d8d693d0bc411adbb7930e47be306da18b8c67d961d4`。
- 图像契约：`force_resize=False`、`enable_random_aug=False`、`n_images=1`、
  `n_times=1`；train/test 均为 `1600×900 -> 704×396 -> crop(0,70,704,326)`，
  `post_rot=diag(0.44,0.44)`、`post_tran=[0,-70,0]`。
  这里的 `1×1` 是 GEOM1-A 已训练配置；`n_images=1,n_times=4` 属于 temporal B0，
  不能倒写成 GEOM1-A 的复现契约。
- 数据规模：train 200,874 infos / 364 clips / 3,086,700 GT；val 24,909
  infos / 45 clips / 300,701 GT；train/val token 和 clip 零重叠，`test=val`，
  当前没有虚构独立 test。
- 严格迁移：`N7_PKL_MIGRATION_COMPARE=PASS failures=0`；全量图片检查为
  225,783/225,783 存在，JPEG header 尺寸和解码错误均为 0。原始 data gate
  因旧 manifest 的 clip 边界缺帧保持 `FAIL`，实验级为
  `PASS_WITH_ACCEPTED_CLIP_BOUNDARY_EXCEPTION`。
- F4：全量 keep-mask 一致率 99.98937%；111 个差异全部在 0～20m 近场，
  不涉及 20～80m 或侧向 yaw 聚集。保留原始 `F4_REVIEW_REQUIRED`，实验级为
  `PASS_WITH_ACCEPTED_NEAR_FIELD_CROP_EXCEPTION`。
- 真实 pipeline 可视化：train/val 分别通过并渲染 34/32 张；涵盖远距、上下边缘、
  非零 yaw 和采样 crop-dropped 目标。
- 单卡 20GB 机器的 10 warmup + 50 iter：S0/GEOM1-A 的 iter mean 为
  `11.810291s/13.276409s`，data mean 为 `0.094683s/0.199413s`，峰值显存为
  `9926.1/9929.3 MiB`。GEOM1-A 端到端慢 13.19%，但该绝对时间不能外推到 L20。
- 正式 work_dir：
  `work_dirs/n7_mono_1600_900_scale_crop/EXP-MONO-GEOM1-A/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260729`。
- 最终状态：4×L20 15 epoch 于 2026-07-30 完成，e1～15 全部评估。
  canonical best=e11（mAP `0.381993`）；GEOM 内 e13 的 overall AOE/ASE 最优，
  作为方向/尺度 challenger，e15 为 terminal。训练完整日志和评估汇总已同步，
  SHA256 分别为 `1abbd307...079b` 和 `8e75a795...0f0`。当前工作区没有
  e11/e13/e15 PTH、逐轮 result/metrics、原始 PKL 和预训练权重实体；这些资产的
  路径与 SHA256 仍待内网归档，不得用日志/汇总 hash 代替。

### 4.6 当前 6V-B0 阶段精度

以下结果来自当前 `eval_summary.md`，只覆盖已经完成复评的 epoch1～16：

| epoch | canonical mAP | BEV mAP@0.5 | mATE | mAOE | mASE | 结论 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.3926 | 0.3921 | 0.8821m | 5.902° | 0.2225 | 初始可用 |
| 2 | 0.4355 | 0.4349 | 0.8407m | 4.711° | 0.2187 | 快速提升 |
| 3 | 0.4445 | 0.4491 | 0.8230m | 4.558° | 0.1964 | 继续提升 |
| 4 | 0.4435 | 0.4592 | 0.8179m | 4.012° | 0.1950 | mAP 小幅回落 |
| 5 | 0.4458 | 0.4652 | 0.7996m | 3.937° | 0.1860 | 接近最佳 |
| **6** | **0.453768** | **0.4787** | **0.7793m** | **3.062°** | **0.1811** | **当前 canonical best** |
| 7 | 0.4517 | 0.4714 | 0.7818m | 3.228° | 0.1815 | 接近最佳 |
| 8 | 0.4462 | 0.4659 | 0.7879m | 4.022° | 0.1763 | 开始回落 |
| 10 | 0.4373 | 0.4545 | 0.7763m | 2.929° | 0.1745 | AP 继续回落 |
| 12 | 0.4366 | 0.4533 | 0.7599m | 3.122° | 0.1712 | 回归误差改善 |
| 14 | 0.4265 | 0.4408 | 0.7683m | 3.264° | 0.1702 | AP 未恢复 |
| 16 | 0.4192 | 0.4354 | 0.7636m | 3.119° | 0.1683 | AP 明显低于 e6 |

截至 e16，epoch6 后已有 10 个已评估 checkpoint 未刷新 canonical mAP。loss 和 mASE 继续下降，但检测 AP 持续回落，表现为典型的后期过拟合/分类排序退化，而不是训练发散。patience=5 在 e11 已满足；e17 已保存，补评后即可停止，不能仅因训练 loss 更低就改用后期 checkpoint。

epoch6 分类别为 car/truck center AP=`0.5956/0.3119`，BEV AP@0.5=`0.5627/0.3947`。正向 x 距离分桶的 Recall@2m 为：

- car：0～20m `0.9314`、20～40m `0.8472`、40～60m `0.7577`；
- truck：0～20m `0.8573`、20～40m `0.6857`、40～60m `0.5038`。

该分桶只覆盖正向 x，不能代表 6V 全环视 ROI 的后向和纯侧向目标；因此 6V 与 mono 的距离表也不能直接横比。

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

## 6. 为什么 positive bag loss 降到 0.65 仍不建议继续到 20 epoch

EXP-MONO-B0 的 epoch 末采样项如下。旧笔记中的 0.6496 是 `positive_bag_loss`，不是总 loss：

| epoch | positive bag | negative bag | 总 loss | mAP（center） |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.4288 | 0.1493 | 1.5780 | 0.2891 |
| 5 | 0.7919 | 0.1452 | 0.9371 | **0.3449** |
| 10 | 0.6916 | 0.1420 | 0.8336 | 0.3366 |
| 15 | 0.6496 | 0.1433 | 0.7928 | 0.3387 |

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
| eval | epoch1～15 已逐 epoch 复评 | epoch1～15 已完成；最终按 canonical mAP 选 e13 |

第一轮只改变 n_times 和对应 3D fuse 通道。输入几何、anchor、CBGS、NMS、增强、LR 等优化放到 S0 基线形成后逐项 A/B。

### 7.1 S0 当前逐 epoch 指标

| epoch | mAP（center） | BEV mAP@0.5 | mATE@2m | mAOE | mASE | 结论 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.2894 | 0.2796 | 0.9501 | 6.212° | 0.2928 | 初始可用 |
| 2 | 0.3305 | 0.3388 | 0.9160 | 5.516° | 0.2522 | 快速提升 |
| 3 | 0.3308 | 0.3422 | 0.9230 | 6.472° | 0.2487 | AP 持平、角度波动 |
| 4 | 0.3349 | 0.3496 | 0.9029 | 4.581° | 0.2514 | 继续提升 |
| 5 | 0.3518 | 0.3625 | 0.8818 | 4.771° | 0.2437 | 公平 epoch5 对比点 |
| 6 | 0.3541 | 0.3693 | 0.8742 | 4.264° | 0.2333 | 接近当前最佳 |
| 7 | 0.3527 | 0.3680 | 0.8816 | 4.428° | 0.2257 | 平台波动 |
| 8 | 0.355078 | 0.3650 | 0.8681 | 4.592° | 0.2297 | 截至 e9 的历史 best |
| 9 | 0.3548 | **0.3713** | **0.8667** | **3.971°** | **0.2256** | BEV/回归更好，主 mAP 略低 |
| **13** | **0.356425** | **0.3773** | **0.8588** | **4.055°** | **0.2132** | **最终 canonical best** |

表中保留 e1～9 作为早期过程证据；e10～12/e14～15 的精确逐 epoch 数值尚未同步到本地文档资产，但内网已完成 15 epoch 评估并最终选择 e13，不能再把 e8 写成当前 best。

### 7.2 B0 与 S0 的同口径对比

| 对比口径 | B0 | S0 | S0 相对变化 |
| --- | ---: | ---: | ---: |
| epoch5 canonical mAP | 0.344934 | 0.3518 | 约 +0.0069 |
| epoch5 BEV mAP@0.5 | 0.3562 | 0.3625 | +0.0063 |
| epoch5 mATE | 0.8899m | 0.8818m | -0.0081m |
| epoch5 mAOE | 4.315° | 4.771° | +0.456°，较差 |
| 各自最终 best canonical mAP | e5 0.344934 | e13 0.356425 | +0.011491 |
| 各自最终 best BEV mAP@0.5 | e5 0.3562 | e13 0.3773 | +0.0211 |
| 各自最终 best mATE | 0.8899m | 0.8588m | -0.0311m |
| 各自最终 best mAOE | 4.315° | 4.055° | -0.260° |
| 各自最终 best mASE | 0.2369 | 0.2132 | -0.0237 |

S0 e13 是总体主模型，但不是每个产品切片都最优：e13 truck Recall@2m 在 40～60m/60～80m 为 `0.6498/0.6435`，低于 e10 的 `0.6853/0.6693`。因此保留 e10 作为中远距 challenger 是合理的；生产发布仍以 e13 为 canonical checkpoint，不能按单个距离切片偷偷替换主 best。

## 8. 当前代码审查项状态

| 项目 | 当前状态 | 对下一轮训练的影响 |
| --- | --- | --- |
| 板端无 pose 的时序不等价 | 已做产品决策：近期原生单帧 | S0 路线正确；B0 只作离线对照 |
| prediction box origin / z | 代码已修；已有 result pkl 重评确认 car/truck z 偏差正常 | 不需重训；补归档新数值、结果路径和 hash |
| ONNX/LUT 浮点链路 | 真实 e13 已完成 PTH、FP ONNX、Torch-CUDA fixed LUT 和 canonical decode 功能对齐 | raw tensor 仍有 ORT/PTH 数值差；真实芯片/INT8 仍开放 |
| 原生单帧配置 | 已完成 15 epoch、最终选择 e13，并完成生产图片 PTH/ONNX-FP 推理 | 最终 PTH/result/config/data/calibration hash 待归档 |
| pkl strict 门禁 | 新旧 migration PASS、全量图片/几何检查通过；原始报告因已知 clip 边界 851/90 缺帧保持 FAIL | GEOM1-A 按 `PASS_WITH_ACCEPTED_CLIP_BOUNDARY_EXCEPTION` 放行；保留 1,237 个 train 空 GT 丢弃警告，仍无独立 test |
| distortion-aware GT 可见性 | GEOM1-A F4 全量完成；中心裁剪 111/1,043,946 个 eval-only GT，全部为 0～20m 近场 | 原始 `F4_REVIEW_REQUIRED` 保留；按 `PASS_WITH_ACCEPTED_NEAR_FIELD_CROP_EXCEPTION` 放行，不改 GT 口径 |
| 生产推理 vs dataset 同图对齐 | e13 PTH 在 N7 黄金样本和 5 个固定样本已 PASS；生产 info-json 数值路径也 PASS | 代码路径已闭环；物理标定仍需独立 lidar/image 验证 |
| `force_resize` 关闭随机增强 | 已完成；commit `134d5f3`，内网 T7 bit-exact PASS | 旧 704×256 baseline 已冻结；AUG1 仍需独立实验 |
| 704×256 stretch / 输入几何 | GEOM1-A e1～15 完成；e11 mAP=0.381993、BEV=0.4016，中远距 Recall 明显提升但 AOE 退化 | 采用 e11 作为下一轮候选基线；先做 yaw 分桶/生产回归与真实 yaw/弯道数据，不做 EXT5 |
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

1. EXP-6V-B0 已闭环：保留 e6 为 canonical best、e2 为正向 x 中远距 challenger、e16 为 terminal；e17 及以后不再纳入选型，不补评、不继续训练。
2. 保留旧 704×256 数据、原生 1600×900 PKL/manifest/gate、epoch13/9797/板端黄金证据；当前不执行测试资产清理。实验结束后另列 retain/archive/delete 精确清单并等待批准。
3. 冻结 GEOM1-A：e11=canonical best，e13=方向/尺度 challenger，e15=terminal；补齐 PTH/result/config/data/calibration hash，不做 EXT5。
4. 先对 S0 e13、GEOM e11/e13 跑同一生产有 GT 回归集，并新增按“到 0°/180° 最近角度”的 yaw 分桶指标；当前 overall mAOE 不能代表城区路口斜向车辆。
5. 并行筛选 `DATA1-CITY-YAW` 与 `DATA3-BANKED-RING` N7 真实 GT，按 scene 防泄漏并保留来源/yaw/距离标签。算力有限时可在质量审计后合并为一次真实数据微调，但不同时启用 AUG1。
6. 在真实数据 winner 或 GEOM e11 上单独做 `AUG1`；固定数据、几何、初始化和 schedule，重点看 yaw 分桶、远距 recall 和生产回归。
7. `DATA2-ENDURANCE-PSEUDO` 后置：固定 BEVFusion teacher/hash/阈值，人工抽检并控制权重，验证集保持纯真实 GT。
8. 并行补独立 test、S0/6V/GEOM 资产 hash、同步 lidar/image 物理标定，以及真实芯片 tensor/runtime/INT8 闭环。
9. `EXP-6V-B1` 保持 P3 低优先级；没有新排期或算力授权时不主动启动。

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
