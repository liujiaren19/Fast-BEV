# Fast-BEV N7 单目前视单帧产品化代办清单

> 更新时间：2026-07-23
> 当前仓库：`/workspace/Fast-BEV_test_custom-fastbev-adapter`
> 当前产品决策：板端近期使用原生单帧单目推理；六轴 IMU + 轮速 pose 尚在开发，四时序模型保留为离线上限和未来候选，不作为当前产品主路径。

## 0. 使用规则

- `[ ]` 未完成，`[x]` 已完成，`[-]` 暂缓或等待外部条件。
- P0：阻断下一版单帧训练、可信评估或生产推理正确性。
- P1：单帧 baseline 完成后应尽快收口，影响数据、评估、导出或生产验证质量。
- P2：模型能力和精度优化，必须基于稳定 baseline 做单变量 A/B。
- P3：板端量化、工程维护和未来时序能力。
- 每次实验只修改一个主变量；配置、代码、pkl、标定、checkpoint 和评估结果必须可追溯。
- 不根据训练 loss 的绝对值选择 checkpoint，以独立 val/test 指标为准。
- 不把旧的 epoch5/epoch10 collapse 实验与 2026-07-08 B0 混在一起。

## 1. 当前已确认基线

### 1.1 四时序单目 B0

- [x] 输入为 `n_images=1, n_times=4`，每个样本 4 张 `cam0` 图像。
- [x] 历史帧在 dataset 路径中使用 pose 补偿。
- [x] 前视 ROI：`[0, -35, -5, 80, 35, 3]`。
- [x] voxel：`n_voxels=[160, 140, 4]`，`voxel_size=[0.5, 0.5, 1.5]`。
- [x] 训练关闭 `flip_ratio_bev_vertical`，避免把前视 GT 翻到车后。
- [x] 2026-07-08 内网训练实际为 4 卡 L20、每卡 64、`lr=1e-4`、15 epoch。
- [x] B0 最佳 checkpoint 为 epoch5：
  - `mAP=0.344934`（center-distance mAP；历史兼容键为 `mAP/center_dist`）
  - `mAP/bev_iou@0.5≈0.3562`
  - car/truck GT 为 `69492/15747`
- [x] epoch5 后为平台波动，不是训练崩溃；epoch15 回归误差继续改善，但 AP 未刷新。
- [x] 当前生产用 epoch5 四时序模型重复当前帧，仅作为过渡诊断，不是最终产品方案。

### 1.2 当前产品单帧定义

```text
S0 = cam0 + n_images=1 + n_times=1 + 无历史 pose
     + 当前 704x256 stretch + 当前 ROI/voxel/anchor
     + use_distortion=True + 与 B0 相同数据和初始化口径
```

首版 S0 只改变“去掉时序”，不同时修改分辨率、anchor、CBGS、NMS 或 loss。

### 1.3 已确认的近期执行顺序

1. 当前先修 box origin 和 ONNX 导出，并新增原生 `n_times=1` 配置。
2. 启动第一轮单帧/时序公平对比实验：
   - `total_epochs=15`；
   - 优先查看 epoch5；
   - 四卡训练服务器每 epoch 保存 checkpoint，单卡评估服务器逐 epoch 独立评估；
   - best epoch 后连续 5 个 epoch 未刷新 best 即停止；
   - 本轮只比较“真实四时序”和“原生单帧”，不混入数据/分辨率/anchor 调优。
3. 对比实验期间并行完善 eval 可视化，重点检查四时序 epoch5 的角度和中远距效果。
4. 对比实验后、正式产品单帧训练前完成四个正确性收口项：
   - pkl 训练前数据门禁；
   - 生产推理与标准 dataset 路径逐级对齐；
   - distortion-aware GT 可见性统一，修复当前模型畸变投影与 train/eval pinhole 过滤不一致。
   - 拆分 `force_resize` 的确定性基础变换与随机图像增强，先做关闭随机增强的等价重构验证。
5. 上述正确性问题收口后，再启动正式单帧训练和后续单变量能力优化。

### 1.4 2026-07-23 当前同步与验证状态

- [x] 当前仓库代码与内网代码已由用户确认同步一致；Markdown、Claude/Codex 本地辅助文件未同步到内网。
- [x] S0 已完成 15 epoch；最终 canonical best 为 epoch13：mAP=`0.356425`、BEV@0.5=`0.3773`、mATE=`0.8588m`、mAOE=`4.0545°`、mASE=`0.2132`。
- [x] S0 epoch13 为产品主 checkpoint；epoch10 保留为中远距 challenger，epoch15 只作终止归档。
- [x] S0 e13 已完成 N7 dataset/production PTH 黄金样本与 5 个固定样本逐级对齐；生产 info-json 数值路径也通过。
- [x] Torch-CUDA fixed LUT、FP ONNX 功能链和独立 CPU 板端参考已完成 PC 侧实测；38 项回归通过。
- [ ] S0 e13/e10/e15 PTH、result、resolved config、代码/数据/标定 SHA256 仍待最终归档；现有生产集没有独立 GT。
- [ ] 真实芯片 tensor dump、实际板端 runtime 和真实 INT8 数值验证仍未完成；PC 侧参考 PASS 不等于 M4 板端验收。
- [x] 已对现有 B0 train/val pkl 运行严格门禁：token/clip 零重叠，cam0、图片/K/畸变、GT 分布和 pose 可审计。
- [ ] 现有 B0 pkl 门禁仍为 `FAIL`：旧 converter 统计中 train/val 分别有 `labels_without_frame=851/90`，且尚无独立 test pkl；应保留报告并在重新转换或风险确认后关闭该项。该 FAIL 是数据问题被正常检出，不是门禁脚本异常崩溃。
- [x] EXP-6V-B0 已收口：canonical best=e6、40～60m Recall challenger=e2、terminal=e16；e17 及以后不再纳入本轮选型。
- [-] `force_resize` 修复后重训一次 6V 基线已登记为 P3 低优先级待办；当前不启动，也不阻塞单帧产品、板端或其他 P0/P1 工作。

## 2. P0：正式单帧基线和板端闭环前必须完成

### 2.1 固化 B0 实验资产

- [x] 仓库 temporal dist config 已同步 `total_epochs=15`；内网仍需保存 2026-07-08 run 的完整 resolved config。
- [x] 单帧对比配置固定 `total_epochs=15`，与 B0 使用相同上限和调度口径。
- [x] 当前 GitHub 代码已拆分固化：B0/公共链路为 `2d8ab7d`，S0/ONNX/LUT 增量为 `c3ec682`；两笔 author/committer 均为 `liujiaren19 <1334282612@qq.com>`。
- [ ] 继续归档内网对应 commit 和提交前 dirty diff；环境版本、4×L20 和 seed=0 已从日志留证。
- [ ] 保存 train/val manifest、pkl、每车型标定和 epoch5 checkpoint 的 SHA256。
- [x] 已在实验汇总/详细记录中保存 `/workspace/20260708_122556.log` 和 `/workspace/eval_summary.md` 的归档位置。
- [x] 旧 collapse run 已单独编号为 T2/T3，B0 使用独立实验 ID 和 work_dir；后续不得覆盖。
- [ ] 记录预训练实际加载口径：ResNet backbone 可加载；FPN64、neck_fuse、3D neck 和 head 因 shape/结构不同主要为新初始化。

验收：任何人可以从记录中唯一确定 B0 的代码、数据、标定、配置、checkpoint 和指标。

### 2.2 修复 box origin 和输出坐标语义

- [ ] 确认 N7 标注协议中的 `location.z` 是 box 重心还是底中心；用点云/标注规范留证据。
- [x] 在 `tools/n7_box_origin.py` 统一实现 center/bottom/native 转换，dataset eval、canonical 可视化、生产 mono JSON/pkl 和 portable eval exporter 已复用。
- [ ] 芯片厂家旧 `tools/utils.py`/板端固件后处理尚未接入公共函数；该路径仍有固定 6V、绝对 anchor 路径和旧类别/阈值假设，不能视为 canonical mono 后处理。
- [x] 对 `LiDARInstance3DBoxes` 预测优先读取 `gravity_center`，避免直接把底中心 tensor 与 GT 重心相减。
- [x] portable prediction 增加 `box_origin=center|bottom` metadata，禁止无语义裸数组在不同工具间传递。
- [x] 保持训练 GT 口径不变；本项首先是评估/输出修复，不应因此修改训练标签。
- [x] 用已有 `epoch_N_val_results.pkl` 直接重评，不重新跑 inference；已确认 car/truck z 偏差恢复正常，精确新数值和 result hash 待归档。
- [x] 添加已知高度合成框单元测试：center↔bottom 往返误差为 0。

验收：

- BEV AP、xy center AP 基本不变；
- car `z_mean` 从约 `-0.83m` 回到接近 0；
- truck `z_mean` 从约 `-1.67m` 回到接近 0；
- 可视化、评估和生产输出对同一个框给出相同 z。

### 2.3 修复 ONNX 导出入口并支持正式单帧

- [x] `FastBEV2DExportWrapper.forward()` 直接调用 `model.onnx_export_2d()`。
- [x] `FastBEV3DExportWrapper.forward()` 直接调用 `model.onnx_export_3d()`。
- [x] eager reference、`torch.onnx.export` 和 ORT 使用同一个明确分支。
- [x] 正式支持 `n_times=1`，不再依赖通用的 `--allow-non-4-times` 绕过产品约束。
- [x] 单帧 2D ONNX 输入为一张 `[1,3,256,704]` 图像。
- [x] 单帧 3D ONNX 输入为一个 `[1,256,160,140]` BEV tensor。
- [x] `export_metadata.json` 写入：
  - `n_images=1`
  - `n_times=1`
  - `channel_layout=ZC`
  - 输入输出名称、shape、dtype、layout
  - checkpoint/config hash
  - raw logits 标记
- [ ] 检查 sigmoid 只在 canonical postprocess 中执行一次。

验收：真实单帧 checkpoint 完成 PTH eager、ONNX checker、ORT 2D/3D 和 decoded boxes 对齐。

### 2.4 新增原生单帧配置，不覆盖四时序配置

- [x] 新增独立配置，例如 `custom_fastbev_mono_front_single_frame_r18.py`。
- [x] 保持 `model.n_images=1`。
- [x] train/test `MultiViewPipeline.n_times=1`。
- [x] dataset `n_times=1`、`sequential=False`。
- [x] `train_adj_ids/test_adj_ids=None`。
- [x] 3D neck 每帧通道保持 `64 feature × 4 z = 256`。
- [x] `neck_3d.in_channels=256`。
- [x] `neck_3d.fuse.in_channels=256`、`out_channels=256`，保留单帧 channel mixing。
- [x] 保持首版 ROI、voxel、anchor、distortion、图像尺寸和 GT 过滤不变。
- [x] 单帧 inference 不再创建或重复 4 份当前图。
- [x] 现有 pkl+config LUT 路径在 `n_times=1,sequential=False` 时只选择当前帧/cam0，只导出一个时序片段，不读取 `prev` 历史帧或历史 pose。
- [ ] 新增板端固定路径标定参数直接生成单帧 LUT 的入口，不能要求生产侧先构造训练 pkl。
- [ ] 对同一车型标定分别生成 pkl+config LUT 和固定标定 LUT，比较 projection、valid mask、gather/scatter index 与 metadata；一致后再确定板端唯一生产入口。

验收：模型/dataloader 实际输入 shape 为 `[B,1,3,256,704]`，3D neck 收到 `[B,256,160,140]`。

### 2.5 当前 pkl 的训练前数据门禁

- [x] 已确认当前 B0 train/val clip 和 token 零重叠；独立 test 尚未提供，因此 train/val/test 全量零泄漏仍待最终验收。
- [-] config 的 `test` 暂时继续指向 val pkl；按当前决策先不改，后续准备独立日期/车辆 test split。
- [x] 已检查当前 B0 train/val 每个 info 的 `camera_ids == ['cam0']`。
- [x] 已统计当前 B0 raw label、空 GT、car/truck frame、cam0 pinhole-visible GT；旧 pkl 转换时已删除 1237 个 train 空 GT label。
- [ ] 统计 converter 丢弃空帧对单帧训练负样本比例的影响。
- [x] 已检查 pkl 记录的图片尺寸、K native 尺寸和 distortion 字段完整率；真实图片 header 全量检查仍可按需开启。
- [x] 已输出 car/truck 数量和 `x/y/z/l/w/h/yaw` 分布。
- [x] S0 不依赖 pose；门禁仍保留 key/history pose、offset 和 history shortage 审计供未来时序恢复使用。
- [ ] 处理或书面接受旧 pkl 的 `labels_without_frame=851/90`，补独立 test 后重新运行 `--strict-data`，归档最终 PASS 报告。

验收：数据审计报告无 split 泄漏、相机/尺寸契约错误；发现严重问题时先修数据再启动长训练。

### 2.6 生产推理与标准 dataset 路径逐级对齐

- [ ] 从 val 选同一张图，分别走 `tools/test.py` 和生产推理脚本。
- [ ] 对比原图路径、resize/normalize 后 tensor、RGB/BGR、dtype 和 layout。
- [ ] 对比 K、`intrinsic_width/height`、实际图片尺寸、distortion、extrinsic、`post_rot/post_tran`。
- [ ] 对比 2D feature、BEV feature、raw cls/bbox/dir logits 和 decoded boxes。
- [ ] 明确生产输出 yaw 使用顶部主 lidar 坐标，不误当相机 yaw 或后轴 ego yaw。
- [ ] 每车型做 lidar 点/地面网格/已知 3D 框投影检查，重点看左右边缘和 20～40m。
- [ ] 为生产样本保存推理 metadata，禁止只保留渲染图而无法追溯标定和预处理。

验收：相同 val 图两条 PTH 路径在浮点容差内一致；否则不得用重新训练掩盖推理链路问题。

### 2.7 修复 force_resize 强制关闭图像增强

- [ ] 保持当前首轮 S0/B0 A/B 不变：两边继续使用相同的 `force_resize=True`，避免引入第二个实验变量。
- [ ] 把“native K/尺寸到 704×256 的确定性基础变换”和“训练随机 resize/crop/flip/rotate”拆成两个显式阶段。
- [ ] 第一阶段关闭随机增强，对同一输入逐项对比当前实现与重构实现的图像、`post_rot/post_tran`、投影矩阵和模型 logits；必须在约定容差内一致。
- [ ] 不直接把 `force_resize` 改成 `False`：704×256 缓存图与 `intrinsic_width/height` 对应的 native K 尺寸不同，必须保证 K/post transform 只缩放一次。
- [ ] 单帧路径验证通过后，再为随机增强增加独立 config 开关和单变量 A/B，不能和数据门禁、畸变可见性或输入分辨率一起改。
- [ ] 四时序未来重新启用随机几何增强时，同一相机的四帧必须共享同一组参数；不能逐图片独立采样导致时序几何不一致。
- [ ] 增加无随机增强、固定随机种子、边界 crop/flip/rotate 的几何回归测试。

验收：关闭随机增强时重构前后输入和投影等价；开启增强时图像变换与投影严格同步，且 S0/B0 各自通过独立 A/B 后才进入默认训练配置。

## 3. P1：单帧 S0 训练、评估与诊断

### 3.1 单帧 smoke validation

- [x] legacy MMDetection 内网环境已成功构建并长时间训练 S0，证明 config/model/train 主链可运行。
- [ ] train dataloader shape 检查通过。
- [x] S0 已完成 epoch1～15，loss 可正常反向传播和保存 checkpoint；最终 canonical best=e13。
- [x] epoch13 已完成生产图片 PTH 推理、可视化和标准 dataset/production 同图逐级对齐。
- [x] 原生 `n_times=1` config 下 PTH 单图生产脚本只创建当前帧，不再重复四份输入。
- [x] 真实 epoch13 的 2D/3D FP ONNX、Torch-CUDA fixed LUT、canonical decode 和独立 CPU 板端参考已完成 PC 侧功能对齐。
- [x] 基线提交前相关 `py_compile`、合成检查和 `git diff --check` 已通过；真实芯片/INT8 仍按 M4 独立验收。

### 3.2 第一轮单帧/时序对比训练计划

- [x] S0 从与 B0 相同的 2D backbone 预训练起步，建立干净单帧 baseline。
- [x] 保持 4 卡、每卡 64、全局 batch 256，`lr=1e-4`。
- [x] `total_epochs=15` 作为上限，与 B0 保持一致；S0 已实际跑满。
- [x] epoch1～15 checkpoint 已完成；最终保留 e13/e10/e15 三个角色。
- [x] 四卡启动脚本支持 `NO_VALIDATE=1`，训练服务器不构建/执行 validation。
- [x] epoch1～15 已由单卡服务器完成评估；本地文档仍待同步 e10～15 完整逐行 summary 资产。
- [x] best key 以 canonical `mAP` 为主（兼容旧键 `mAP/center_dist`），同时监控 BEV mAP、car/truck AP、召回和 score 分布。
- [x] 已比较双方 epoch5 和最终 best；S0 e13 比 B0 e5 canonical mAP 高 `0.011491`。
- [x] S0 已按 15 epoch 上限完成，不再延长训练。
- [ ] 记录 dynamic fp16 loss scale、overflow/skip-step 次数，确认零星 `grad_norm=inf` 是否可忽略。
- [x] 当前 best 按 canonical `mAP` 选择，不根据 loss 是否继续下降判断。

### 3.3 checkpoint 和阈值复评

- [ ] 默认评估保留当前精确口径，不设置 `eval_max_dets_per_sample` cap。
- [ ] 对 best 和两个代表 checkpoint 使用 `model.test_cfg.score_thr=0.01` 复评，检查 0.05 是否截断低分 TP。
- [ ] 产品 score threshold 单独标定，不与学术 AP 阈值混为一谈。
- [ ] 记录每帧 det 数量和 score p50/p90/p99。
- [ ] 只在 val 选 checkpoint；独立 test 不参与调参。

### 3.4 扩展评估维度

- [x] 增加 0～20、20～40、40～60、60～80m 的 GT、唯一 TP@2m 和 Recall@2m。
- [ ] 如需分析分段误检，再增加各距离段 det/FP/AP；Recall@2m 本身不表达 false positive。
- [x] watch 运行表主展示每距离段 GT、TP@2m、Recall@2m 和 x/y/z MAE，不显示逐阈值 AP、p50/p90 或 signed mean。
- [x] eval Markdown/CSV/JSON 保留每距离段 x/y/z MAE 和绝对误差 p50/p90。
- [x] 在 eval Markdown/CSV/JSON 中另存每距离段 x/y/z signed mean，供系统偏置诊断。
- [ ] 增加 yaw 误差分布：signed mean、abs p50/p90、180°方向翻转率。
- [ ] 增加 size error、类别混淆和重复框统计。
- [ ] 按日期、sequence、车辆、白天/夜晚、天气和场景分别报告。
- [ ] 建立 100～300 帧生产回归集，覆盖角度偏、20～40m小目标、拥挤和边缘目标。
- [ ] 对生产回归集保存原图、标定版本、预测和人工结论。

### 3.5 eval 可视化和视频输出

- [x] 当前支持 `--camera-width 1600 --display-aspect 16:9`，前视相机面板可显示为 1600×900。
- [x] 当前支持 `--no-bev`、`--video-only`、`--stride`、`--max-frames` 和多线程渲染。
- [x] 明确文档说明：704×256 缓存图放大到 1600×900只改变显示，不恢复原始像素细节。
- [x] 增加显式 `--camera-size WIDTH HEIGHT`，该尺寸只约束相机面板，不包含顶部信息栏或 BEV。
- [x] 增加 `--no-header`；配合 `--camera-width 1600 --display-aspect 16:9 --no-bev` 时最终视频画布严格为 1600×900。
- [x] 新增 `--video-group clip|sequence`，默认保持 `clip` 兼容现状。
- [x] 视频按分组命名：clip 模式输出 `<output>/<dataset>/<sequence>/<clip>/<clip>.mp4`，sequence 模式输出 `<output>/<dataset>/<sequence>/<sequence>.mp4`。
- [x] clip/sequence 视频帧先按输出目标再按 timestamp 排序，保证 sequence 跨 clip 的时间顺序稳定。
- [ ] sequence 跨 clip 时可选插入标题帧或显式记录 clip 切换。
- [x] 增加 `--max-frames-per-video`，避免全量 sequence 视频过长。
- [x] 保持 `--video-only` 不落数万张中间帧。
- [x] 视频 writer 对变化的 canvas size 给出明确错误，不静默跳帧。
- [ ] 大规模可视化支持按 sequence 白名单/正则筛选。

当前仅看前视预测的建议命令口径：

```bash
python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
  --gt-pkl <val_gt.pkl> \
  --pred-pkl <epoch_result.pkl> \
  --data-root <N7_704_256> \
  --output-dir <vis_output> \
  --camera-ids cam0 \
  --camera-size 1600 900 \
  --no-bev \
  --video-only \
  --score-thr 0.2 \
  --workers 4
```

说明：默认仍按 clip 生成视频；需要合并 sequence 时增加
`--video-group sequence --max-frames-per-video 2000`。

## 4. P1：数据转换和投影口径收口

### 4.1 converter strict 模式

- [x] 正式转换默认要求 manifest；自动发现只允许显式 `--allow-auto-discovery` debug 开启。
- [x] 同次转换在输出前检查 split manifest clip 重叠和跨 split token 重叠。
- [x] 支持 `--keep-empty` 保留完整时间轴；key-frame 是否采样空帧仍由后续 dataset/实验策略决定。
- [x] 关键 camera/label/pose/图片字段缺失时 strict 模式只写失败 summary、非零退出且不写 pkl。
- [x] 转换失败按类别计数并保存可追溯样本；`--allow-partial` 仅用于显式 debug。
- [x] sidecar summary 写出空帧、类别、相机、pose、历史覆盖、失败原因、尺寸和 data-gate 状态。
- [ ] 使用新 strict converter 和 manifest 在内网重新生成正式 train/val/test pkl，并让外部 `--strict-data` 门禁最终 PASS。

### 4.2 distortion-aware GT 可见性

- [ ] 抽取 dataset/pipeline/可视化/LUT 共用的 N7 投影函数。
- [ ] 统一 K、distortion、sensor2lidar、post transform 和图片尺寸口径。
- [ ] 在修改训练过滤前，先统计 pinhole 与 distortion-aware 的 GT 差异。
- [ ] 按类别、距离段和图像边缘位置统计误删/误保留。
- [ ] 对 box 边缘采样或轮廓求交，避免仅靠中心和八角点。
- [ ] 人工检查广角边缘样本并和 OpenCV/标定工具对齐。

验收：train/eval/可视化对同一 GT 给出一致可见性结论。

### 4.3 未来时序 pose 数据质量（当前产品暂缓）

- [-] pose 插值到相机曝光时间，而不是只取最近 pose。
- [-] 统计 pose 时间误差 p50/p90/p99 和异常比例。
- [-] 验证 0.5/1.0/1.5s 历史 offset 实际分布。
- [-] 静态点跨时序 BEV 对齐可视化。

## 5. P2：单帧模型能力优化

所有项目先完成 S0，再按单变量 A/B；禁止同时改变数据、分辨率、anchor、采样和 LR。

### 5.1 输入几何和中远距能力

- [ ] A：当前 704×256 非等比 stretch，作为 S0。
- [ ] B：1600×900 等比缩放到约 704×396，再裁剪到 704×256。
- [ ] C：芯片允许时测试 704×384 或其他接近 16:9 的输入。
- [ ] 根据真实 GT 投影统计选择 vertical crop offset，不默认中心裁剪。
- [ ] 统计 20～40m、40～60m目标的原图/输入图像素宽高和 stride-4 feature 尺寸。
- [ ] 比较总体、远距 car/truck AP/recall、吞吐、显存和板端延迟。
- [ ] 所有基础 resize/crop 只通过一次 K/post transform 生效。

### 5.2 图像增强和域泛化

- [ ] 按 2.7 的等价性门禁，把确定性基础 resize/crop 与训练随机增强拆分。
- [ ] 单帧先测试轻量 photometric/resize/crop/flip 增强。
- [ ] 未来恢复时序时，四帧必须共享同一组几何增强参数。
- [ ] 增加生产车型、日期、光照、天气和相机成像差异相关增强。
- [ ] 检查增强对远处小目标是否有负面裁剪。

### 5.3 yaw/方向专项诊断

- [ ] 先排除生产标定、坐标系和可视化 heading 解释问题。
- [ ] 分类别、距离、车辆和图像区域统计 yaw signed/absolute error。
- [ ] 统计 180°方向分类翻转率，检查 `dir_offset/dir_limit_offset`。
- [ ] 对明显偏角样本叠加 GT、预测、车辆前向和相机光轴。
- [ ] 只有证据显示训练头本身有系统偏差时，才 A/B direction loss 或 yaw 编码。
- [ ] 不因少量生产图片主观观感直接调高 yaw loss。

### 5.4 class balance / CBGS

- [ ] 固化无 CBGS 的 S0。
- [ ] 统计 car/truck frame 和 instance 比例，以及不同距离段比例。
- [ ] A/B CBGS 或 class-aware sampler。
- [ ] 按 optimizer step 而非 epoch 数对齐比较。
- [ ] 关注 truck AP、car AP 是否被过采样反向损害。

### 5.5 anchor 优化

- [ ] 统计每类 `l/w/h/z` 的 p10/p50/p90。
- [ ] 计算当前每个 anchor 对 GT 的 max IoU。
- [ ] 报告 IoU≥0.3/0.5/0.6 覆盖率，并按类别和距离分桶。
- [ ] 基于真实尺寸聚类候选 anchor。
- [ ] box z center/bottom 语义确认后再调整 anchor z。
- [ ] 单独训练/评估 anchor A/B，不混入 CBGS 或分辨率变化。

### 5.6 NMS 和 score calibration

- [ ] 统计 duplicate FP、拥挤场景 recall 和每帧 det 数。
- [ ] 检查 truck 继承的 `nms_rescale_factor=0.7` 是否合适。
- [ ] 小范围测试 score threshold、NMS threshold/rescale。
- [ ] PTH、ORT 和板端使用同一类别顺序和 NMS 配置。
- [ ] 产品 threshold 以独立生产/测试集标定，不影响最终 AP 精评口径。

### 5.7 初始化、蒸馏和结构优化

- [ ] 记录当前预训练只充分覆盖 ResNet backbone 的事实。
- [ ] 可选 A/B 与 FPN64 兼容的初始化。
- [ ] 可选把 temporal epoch5 的 fuse 四个时间块权重求和，生成单帧 warm-start S1。
- [ ] S1 必须先验证其输出与“四时序重复当前帧”在同输入下近似一致。
- [ ] 可选使用 pose-aware 四时序 epoch5 作为教师，蒸馏单帧学生。
- [ ] 保留干净 COCO 初始化的 S0，避免只有 warm-start 结果而无法判断结构收益。

### 5.8 ROI、voxel、head 和训练超参（最后处理）

- [ ] 根据真实距离分桶和召回判断是否需要修改 ROI。
- [ ] 根据误差/速度预算 A/B BEV xy 分辨率，不盲目减小 voxel。
- [ ] 只有 anchor/data 修复后仍有证据时，再调 FreeAnchor IoU、focal、loss weight。
- [ ] LR、schedule、weight decay 和训练 epoch 做独立实验。
- [ ] loss 继续下降但 AP 不升时，以 AP 为准，不延长训练掩盖过拟合。

## 6. P3：单帧板端闭环

### 6.1 浮点数值闭环

- [-] 单帧固定 LUT 已能按车型标定生成；车型/标定版本化和发布目录规范仍待固化。
- [-] 新生成 LUT 已保存 info.json hash、输入变换、distortion、voxel、stride 和 channel layout；已有历史 bin 缺少 hash 时只能明确标记 `SKIP`，不能猜测资产关系。
- [x] 已建立并用 N7 固定五样本、生产车固定十样本验证 `预处理 -> 2D ONNX -> LUT -> 3D ONNX -> canonical postprocess` PC 浮点模拟主路径；严格 raw tensor 差异仍按报告保留。
- [x] 已逐级对齐 input tensor、2D feature、BEV、raw logits 和 decoded boxes；功能口径通过，PTH/ONNX 严格 tensor 口径不通过，差异边界已有记录。
- [x] 已由 `tools/run_mono_front_board_inference.py` 固化只读 fixed-LUT 板端参考，`tools/analyze_mono_front_pipeline.py` 统一逐级诊断；旧 simulator/runtime/compare 已在 golden fixture 接管等价守卫后删除。真实板端 tensor dump 与真实 INT8 数值闭环仍单独跟踪。
- [ ] 删除 `tools/utils.py` 中绝对 anchor 路径、十类 reshape 和固定阈值依赖。
- [x] 以 `bbox_head.get_bboxes()` 为后处理权威参考；板端展开实现必须旁路对照它，不能反向用旧 `tools/utils.py` 覆盖 canonical 语义。

### 6.2 量化校准

- [ ] 同时保存原图片和真实送入 2D ONNX 的 normalized tensor。
- [ ] 明确 RGB/BGR、mean/std、dtype、NCHW/NHWC。
- [ ] 保存真实单帧 3D head BEV 输入。
- [ ] 校准集按车型、距离、光照、天气和场景分层抽样。
- [ ] 对比编译器预处理和保存 tensor。
- [ ] 完成 FP32/FP16/INT8 精度、延迟、内存和功耗报告。

### 6.3 输出接口和坐标系

- [ ] 模型内部继续使用顶部主 lidar 坐标。
- [ ] 输出 API 明确轴方向、坐标原点、yaw 定义和 `box_origin`。
- [ ] 当前生产接口只需要 `car`，但本轮浮点数值对齐仍保留已训练的 `car/truck` 两类 head、raw logits 和 canonical decode；待数值链路闭环后，再在统一输出层显式过滤 `truck`，并与板端 car-only 后处理逐项对齐，不能通过临时修改类别数掩盖差异。
- [ ] 如下游需要后轴 ego，在统一输出层完成刚体变换。
- [ ] 不单独修改 label、外参、ROI、anchor 或 LUT 中某一个坐标环节。

### 6.4 本轮部署验证收尾

- [x] 已完成当前 PTH/ORT/固定 LUT 验证阶段的代码梳理：正式入口已收敛为资产构建、生产推理、standalone 板端参考和统一 analyzer；旧 runtime/simulator/compare、一次性相机去畸变与 yaw 诊断、重复可视化模块、临时验证产物和缓存均已清理。旧/新等价结果已固化为 golden fixtures，38 项回归、删除模块 import audit 和任务范围 `git diff --check` 均通过。
- [ ] 真实芯片 tensor dump、板端 runtime 和真实 INT8 数值验证后续单独完成；当前代码整理完成不等同于 M4 板端验收，模型侧可以继续进入输入几何、增强、采样/anchor/NMS 和优化策略实验。

## 7. P3：未来恢复时序的前置工作

- [-] 定义六轴 IMU + 轮速融合 egomotion 接口和时间戳协议。
- [-] 明确相机曝光时刻 pose 插值、坐标系版本和质量状态。
- [-] 评估当前帧固定 LUT + 历史 BEV warp，或历史动态 LUT。
- [-] pose 异常时清空历史/重复当前帧，并在训练中加入对应 history dropout。
- [-] PTH pose-aware、板端模拟和真实板端对齐后，才重新启用四时序产品模型。

### 7.1 低优先级：`force_resize` 修复后重训 6V 基线

- [-] 当前 EXP-6V-B0 保持冻结，不补评 e17、不继续 e18～20；保留 e6/e2/e16 三个正式角色。
- [ ] 只有在 2.7 的 `force_resize` 拆分、关闭增强等价性和四时序共享增强参数回归全部通过后，才创建新的 6V 实验；暂定新实验 ID 为 `EXP-6V-B1`，必须使用新 work_dir，不覆盖 B0。
- [ ] 第一轮只改变 `force_resize` 修复，保持 20260717 数据、6 camera × 4 times、ROI/voxel/anchor、AdamW2、`lr=8e-4`、seed 和 schedule 不变。
- [ ] 从 epoch1 开始逐 epoch val，主键仍为 canonical center-distance mAP；同时监控 BEV、40～60m Recall、预测数和 score 分布，patience=5。
- [ ] 若修复后的 `8e-4` 仍复现 e5～e7 达峰后持续退化，再单独建立 `4e-4` challenger；不在同一轮同时改变输入分辨率、数据、anchor、CBGS、NMS 或 loss。
- [ ] 该项优先级低于当前单帧产品正确性、独立 test、真实板端/INT8 和物理标定闭环；没有新的排期或算力授权时不主动启动。

验收：形成一轮可与 EXP-6V-B0 同口径比较的新 6V 基线，并能把变化归因于 `force_resize` 修复；是否降低 LR 由修复后轨迹决定。

## 8. 建议执行里程碑

### M0：可信 B0

- [ ] B0 资产固化。
- [x] box origin 修复并用已有 result pkl 重评；car/truck z 偏差已确认恢复正常。
- [ ] 独立 test split 和数据零泄漏确认。

### M1：单帧链路跑通

- [x] 原生 `n_times=1` config 已完成真实训练和 PTH 生产图片推理。
- [x] 真实 S0 checkpoint FP ONNX/ORT、Torch-CUDA fixed LUT、decoded boxes 和独立 CPU 板端参考已在 PC 侧跑通。
- [x] 标准 dataset 与 production e13 PTH 同图逐级一致；生产车型物理标定仍需 lidar/image 独立验证。

### M2：第一轮单帧/时序对比结果

- [x] S0 15 epoch 上限训练完成，最终 best=e13。
- [x] 已完成 epoch5 公平对比和最终 best 对比；S0 e13 比 B0 e5 canonical mAP 高 0.011491。
- [ ] val best 和距离分桶已形成；独立 test、最终资产 hash 和有 GT 生产回归集仍缺。
- [x] 已完成与当前四时序重复帧生产方案的无 GT 可视化比较；正式有 GT/回归集结论仍归入上一项。

### M3：单变量能力优化

- [ ] 输入几何/分辨率。
- [ ] 图像增强。
- [ ] CBGS、anchor、NMS。
- [ ] warm-start/蒸馏。

### M4：板端可验收

- [ ] PTH/ORT/PC 板端参考 float 对齐已完成；真实芯片 tensor/runtime 对齐仍未完成。
- [ ] 量化闭环。
- [ ] 每车型标定/LUT/版本管理和输出坐标协议完成。

## 9. 暂时不要做

- [ ] 不继续把四时序模型重复当前帧作为最终产品模型。
- [ ] 不从 temporal epoch15 直接“续到20”并期待 loss 下降带来 AP 提升。
- [ ] 不在同一轮同时改 `n_times`、输入分辨率、anchor、CBGS、NMS 和 LR。
- [ ] 不在 production pipeline/标定未对齐前，用重新训练解释角度偏差。
- [ ] 不在 z 语义未确认前调整 anchor z 或评价垂直定位精度。
- [ ] 不在数据分布统计前盲目修改 FreeAnchor/focal/loss 权重。
