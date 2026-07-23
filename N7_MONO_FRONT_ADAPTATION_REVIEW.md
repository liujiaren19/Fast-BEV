# N7 Mono-Front Adaptation Review Brief

本文档用于后续代码审核。目标是把当前仓库从 N7 6V baseline 到 mono-front
适配、效率/板端改动、训练现状和待审核问题压缩到一个入口文档中。

## 1. 项目目标

- 主任务：在 Fast-BEV N7 自定义数据上做单目前视时序检测适配。
- 当前 mono 不是单帧模型，而是 temporal mono：
  - `n_images=1` 表示每个时间步只用 `cam0`。
  - `n_times=4` 表示当前帧加 3 个历史帧。
  - 每个样本实际输入 `1 camera * 4 time steps = 4` 张图。
- 当前类别口径：`car`、`truck`。
- 当前 mono-front ROI：`[0, -35, -5, 80, 35, 3]`。
- 当前重点不是板端最终精度结论，而是确认 6V baseline、mono 适配和训练口径没有隐性错误。

## 2. 基于 6V 的 Baseline

### 2.1 6V 训练 baseline

核心配置：

- `configs/fastbev/custom/custom_fastbev_6v_r18.py`
- 继承 `configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py`
- 模型：
  - `model.type='FastBEV'`
  - `model.n_images=6`
  - `model.use_distortion=True`
  - `model.feature_resize_mode='nearest'`
  - `bbox_head.num_classes=2`
- 数据：
  - `dataset_type='CustomMultiViewDataset'`
  - `camera_types=['cam0', 'cam11', 'cam9', 'cam3', 'cam8', 'cam10']`
  - `n_times=4`
  - `train_adj_ids/test_adj_ids=[0, 1, 2]`
  - `point_cloud_range=[-50, -50, -5, 50, 50, 3]`
  - `class_names=['car', 'truck']`

### 2.2 704x256 离线缓存图 baseline

核心配置：

- `configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256.py`
- 用于 `data/N7_704_256/` 离线 resize 图像。
- pkl 中的 `cam_intrinsic` 仍保持原始标定坐标系，不能预先缩放 K。
- converter 需要写入：
  - `intrinsic_width`
  - `intrinsic_height`
  - `image_width`
  - `image_height`
- `RandomAugImageMultiViewImage(force_resize=True)` 根据
  `intrinsic_* -> input_size` 写入 `post_rot/post_tran`，避免几何重复缩放。
- 6V 离线缓存模式仍保留 BEV 层增强：
  - `RandomFlip3D`
  - `GlobalRotScaleTrans`

### 2.3 6V baseline 与 mono 的关系

mono-front 配置没有从论文原始 config 重新复制一套，而是继承当前 6V N7
baseline：

- `configs/fastbev/custom/custom_fastbev_mono_front_r18.py`
  继承 `./custom_fastbev_6v_r18.py`。
- mono-front 只覆盖相机数、相机列表、ROI、voxel grid、anchor range、
  pipeline 和数据路径。
- 这样做的目的：保留 N7 6V baseline 已验证过的模型结构、部署约束和
  board/export 相关行为，减少分叉。

## 3. 效率与板端相关改动

### 3.1 FastBEV 模型侧动态相机数

核心文件：

- `mmdet3d/models/detectors/fastbev.py`

关键点：

- 使用 `self.n_images` 控制时序切片，不再按 6 个相机硬编码。
- `extract_feat()` 中按 `self.n_images` split 每个时间步的图像特征。
- `_slice_view_meta()` 会同步切片：
  - `extrinsic`
  - `lidar2img_aug`
  - `lidar2img_extra`
  - `img_shape`
  - `ori_shape`
  - `pad_shape`
  - `img_info`
  - `filename`
- `n_images=1, n_times=4` 时，原 temporal fusion 路径仍保留，不会把模型误改成单帧。
- active path 应继续使用 `mmdet3d/models/detectors/fastbev.py`，不要混用
  `fastbev_ljr.py`、`fastbev_bst.py`、`m2bevnet_seq.py` 等旧硬编码 6V 文件。

### 3.2 2D/3D head 输入与 feature resize

核心约束：

- 当前训练/导出约定 `feature_resize_mode='nearest'`。
- 2D FPN feature 对齐板端导出路径，避免训练和 ONNX/板端 feature resize 行为不一致。
- `onnx_export_3d()` 返回 raw bbox-head logits，sigmoid/topk/NMS 保持在图外。
- `test_onnx` 和 `test_custom` 共享当前主线 metadata slicing/backproject 逻辑。

相关文件：

- `mmdet3d/models/detectors/fastbev.py`
- `tools/export_onnx.py`
- `tools/build_mono_front_board_assets.py`
- `tools/run_mono_front_board_inference.py`

### 3.3 动态畸变和 backproject

当前主线支持 N7 pkl 中的 distortion 信息：

- `model.use_distortion=True` 时，backproject 使用动态畸变投影路径。
- `lidar2img_aug/lidar2img_extra` 中的标定和尺寸字段会参与投影。
- no-distortion 和 distortion 路径都在 `fastbev.py` 内部保留。

历史上做过 backproject 性能优化和 profile，但当前需要审核的是：

- active path 是否仍然保持几何一致。
- 是否存在旧 6V hard-code 泄漏到 mono-front active path。
- board/export path 和 pth train/test path 是否存在 resize 或 logits 后处理不一致。

### 3.4 Eval 和结果追踪效率

核心文件：

- `mmdet3d/datasets/custom_multiview_dataset.py`
- `tools/eval_epoch_checkpoints.py`

当前 eval 改动：

- N7 自定义 eval 不依赖 nuScenes devkit。
- 支持 BEV IoU AP 和 center distance AP。
- `_eval_single_class()` 对多阈值复用排序和匹配结果。
- BEV IoU 使用 polygon/AABB 缓存与预过滤。
- `_parse_det_result()` 支持：
  - `eval_score_thr`
  - `eval_range`
  - `eval_max_dets_per_sample`
- `eval_max_dets_per_sample` 默认 `None`，最终精评默认不改变口径。

新增独立追踪脚本：

- `tools/eval_epoch_checkpoints.py`
- 用于 4 卡训练机产出 `epoch_*.pth`，单卡 eval 机器持续扫描评估。
- 输出：
  - `epoch_N_val_results.pkl`
  - `epoch_N_test.log`
  - `epoch_N_metrics.json`
  - `eval_summary.md`
  - `eval_summary.csv`
  - `eval_summary.json`
- 汇总内容包括 mAP、car/truck AP、GT/det 数量和 score 分布。

## 4. Mono-Front 适配

### 4.1 Mono-front 核心配置

核心文件：

- `configs/fastbev/custom/custom_fastbev_mono_front_r18.py`
- `configs/fastbev/custom/custom_fastbev_mono_front_r18_dist_train.py`

核心覆盖：

- `model.n_images=1`
- `camera_types=['cam0']`
- `n_times=4`
- `train_adj_ids/test_adj_ids=[0, 1, 2]`
- `point_cloud_range=[0, -35, -5, 80, 35, 3]`
- `n_voxels=[160, 140, 4]`
- `voxel_size=[0.5, 0.5, 1.5]`
- anchor range `[[0, -35, -1.8, 80, 35, -1.8]]`

注意：`neck_3d` temporal 输入通道没有因为 6V -> mono 而缩小。当前 baseline
保留原 temporal fusion 设计，改变的是每个时间步的相机数。

### 4.2 Mono-front train pipeline

当前训练 pipeline：

- `MultiViewPipeline(sequential=True, n_images=1, n_times=4)`
- `LoadAnnotations3D`
- dummy `LoadPointsFromFile`
- `RandomFlip3D`
  - `flip_ratio_bev_horizontal=0.5`
  - `flip_ratio_bev_vertical=0.0`
- `GlobalRotScaleTrans`
- `RandomAugImageMultiViewImage(force_resize=True)`
- `ObjectRangeFilter(point_cloud_range=front_roi)`
- `FrontCameraVisibleObjectFilter(n_images=1, min_depth=0.1)`
- `KittiSetOrigin`
- normalize / bundle / collect

为什么关闭 vertical flip：

- LiDAR BEV vertical flip 会翻转 x 方向。
- mono-front ROI 只覆盖 `x >= 0`。
- 前后翻转会把有效前视目标翻到车后，随后被 `ObjectRangeFilter` 删除或导致训练目标异常。

### 4.3 Train/eval GT 过滤口径

目标：训练、验证、测试使用同一套 GT 口径。

当前口径：

- 训练 pipeline 中用 `FrontCameraVisibleObjectFilter` 保留 cam0 几何可见目标。
- dataset 初始化参数中用：
  - `filter_gt_visible_camera='cam0'`
  - `filter_gt_visible_min_depth=0.1`
  - `eval_range=front_roi`
- `CustomMultiViewDataset._parse_gt_info()` 在 eval/test 时按类别、ROI、cam0 可见性过滤 GT。
- 该过滤不修改原始 pkl，只在 dataset/eval 层生效。

需要重点审核：

- `mmdet3d/datasets/pipelines/__init__.py` 是否正确注册
  `FrontCameraVisibleObjectFilter`。
- 训练和 eval 是否在 key frame 上使用一致的可见性判断。
- `FrontCameraVisibleObjectFilter` 与 dataset-level `filter_gt_visible_camera`
  是否有重复或边界不一致。

### 4.4 Converter 与可视化

核心文件：

- `tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py`
- `tools/data_converter/n7/visualize_n7_fastbev_pkl.py`
- `tools/data_converter/n7/n7_pointcloud_gt_diagnostics.py`

converter 当前能力：

- 支持选择 `--camera-ids cam0` 生成 mono-front pkl。
- 保留原始 `intrinsic_width/height` 和实际图片 `image_width/height`。
- 生成 metadata 中的 `camera_ids`。
- 生成 `metadata['output_summary']`，检查：
  - camera id/count mismatch
  - temporal prev 覆盖数量

可视化当前能力：

- mono-front pkl 默认识别 `camera_ids=['cam0']`。
- 单目前视可视化图会放大显示，线条按图像尺寸缩放。
- `--gt-view-mode split|used` 可以查看 raw GT 中哪些被 mono-front train/eval 口径使用。
- BEV 默认使用 front ROI。

点云辅助诊断：

- `n7_pointcloud_gt_diagnostics.py` 读取 pkl、cam0 图片和原始点云。
- 输出每帧 BEV/前视图和每个 GT 的 CSV/JSON 诊断字段。
- 当前只做诊断，不自动删除 GT。

## 5. 当前训练现状

### 5.1 已跑通项

- 单个 sequence 上曾跑通 mono-front 训练并能收敛。
- 当前 full-val 评估链路可以跑通，GT 统计稳定：
  - `car gt = 69492`
  - `truck gt = 15747`
  - `eval/gt_num = 85239`
- epoch 级单卡 eval 和 score 分布统计已经可用。

### 5.2 当前 4 卡大数据训练现象

一次 clean mono-front full-data 训练中观察到：

| epoch | mAP/bev_iou@0.5 | car bev_AP@0.5 | truck bev_AP@0.5 | 备注 |
| --- | ---: | ---: | ---: | --- |
| 1 | 0.2709 | 0.4011 | 0.1407 | 初始可用 |
| 2 | 0.2897 | 0.4055 | 0.1739 | 当前已知最佳 |
| 3 | 0.2296 | 0.3267 | 0.1326 | 开始下降 |
| 4 | 0.1510 | 0.2476 | 0.0544 | 明显退化 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 单卡显式 eval 也为 0 |

epoch5 的 score 分布：

```text
total det: 14996
label 1 num 12178
  quantile: [0.05004883 0.05032349 0.05081177 0.05099487 0.05126953 0.05154419 0.05175781]
  >=0.05/0.1/0.2: 12178 0 0
label 0 num 2818
  quantile: [0.05004883 0.05203247 0.05358887 0.05430908 0.05511475 0.05560303 0.05581665]
  >=0.05/0.1/0.2: 2818 0 0
```

解释：

- `model.test_cfg.score_thr=0.05`。
- epoch5 的预测只剩少量刚擦过 0.05 阈值的框。
- 没有任何预测超过 0.1。
- 这说明不是 AP 计算本身出错，而是模型分类置信度已经塌到接近背景。

### 5.3 当前训练风险判断

主要怀疑点：

- 当前 4 卡配置里 `optimizer.lr=0.0008`，全局 batch 为 `64 * 4 = 256`。
- 近期日志显示第 2 个 epoch 尾部 lr 约 `7.2e-4`，epoch4 仍在 `6.4e-4` 附近。
- 对 mono-front 当前 FreeAnchor 训练来说，这个 lr 可能偏高。
- 训练 loss 继续下降不等于 AP 继续上升，当前现象更像 score/分类头校准崩溃。
- 日志中曾出现少量 `grad_norm: inf/nan`，需要审核是否与 fp16 dynamic loss scale 或梯度稳定性有关。

当前建议：

- 不建议继续使用 epoch5，也不建议从 epoch5 `resume_from`。
- 当前最佳 checkpoint 是 epoch2。
- 如果继续训练，优先使用 `--load-from epoch_2.pth`，不要 `--resume-from epoch_2.pth`：
  - `load_from` 只加载模型权重，重新初始化 optimizer/scheduler。
  - `resume_from` 会恢复旧 optimizer/scheduler/loss scale，可能继续沿着原不稳定轨迹走。
- 续训 lr 建议从 `1e-4` 或 `2e-4` 开始验证。
- 后续训练建议每个 epoch 都用 `tools/eval_epoch_checkpoints.py` 做单卡 eval 和 score 分布记录。

## 6. 需要 Claude Code 重点审核的问题

1. active train/eval/export path 是否仍存在硬编码 6V 假设。
2. `custom_fastbev_mono_front_r18.py` 继承 6V baseline 是否有隐藏副作用。
3. `n_images=1, n_times=4` 是否贯穿：
   - config
   - dataset
   - pipeline
   - `fastbev.py`
   - eval/test
   - ONNX/export
4. mono-front ROI、anchor range、voxel grid、eval range 是否一致。
5. train/eval/test GT 过滤是否使用同一套 cam0 + ROI 口径。
6. `FrontCameraVisibleObjectFilter` 和 dataset-level `filter_gt_visible_camera`
   是否一致、是否存在边界误删 GT。
7. `force_resize=True`、`intrinsic_width/height`、`image_width/height`、
   `post_rot/post_tran` 是否避免了重复缩放或漏缩放。
8. 当前 eval 加速是否可能改变最终精评口径，尤其是：
   - `eval_score_thr`
   - `eval_max_dets_per_sample`
   - `eval_range`
9. 当前训练崩溃更可能是代码 bug、配置副作用还是超参不稳定。
10. 如果建议修改，请优先给出最小改动，避免大范围重构。

## 7. Claude Code 审核 Prompt

可以直接把下面这段发给 Claude Code：

```text
我们在 /workspace/Fast-BEV_test_custom-fastbev-adapter 做 Fast-BEV N7 单目前视时序模型适配。请先阅读 AGENTS.md、CODEX_HANDOFF.md 和 N7_MONO_FRONT_ADAPTATION_REVIEW.md，然后执行 git status --short。使用普通 shell 命令，不要回退任何已有改动。

审核目标：请以代码审查方式检查当前 6V baseline、效率/板端优化、mono-front 适配和训练异常分析是否存在漏洞。重点不是重写代码，而是找 bug、风险、配置不一致和缺失测试。

背景摘要：
1. 6V baseline 入口是 configs/fastbev/custom/custom_fastbev_6v_r18.py，离线 704x256 baseline 是 custom_fastbev_6v_r18_n7_704x256.py。
2. mono-front 入口是 configs/fastbev/custom/custom_fastbev_mono_front_r18.py，分布式训练配置是 custom_fastbev_mono_front_r18_dist_train.py。
3. mono-front 是 temporal mono：n_images=1、n_times=4，不是单帧模型。
4. mono-front 使用 cam0，ROI 为 [0, -35, -5, 80, 35, 3]，n_voxels=[160,140,4]，voxel_size=[0.5,0.5,1.5]，anchor range 为 [[0,-35,-1.8,80,35,-1.8]]。
5. train/eval/test 应统一使用 cam0 几何可见 + front ROI 的 GT 口径：训练 pipeline 有 FrontCameraVisibleObjectFilter，dataset eval 有 filter_gt_visible_camera='cam0' 和 eval_range。
6. 当前 full-data 训练 epoch2 最好，epoch5 单卡 eval 也为 0，score 全部贴近 0.05 阈值，怀疑 lr 偏大或 FreeAnchor 分类头不稳定，但也需要排查代码/配置漏洞。

请重点检查：
- active path 中是否还有 hard-coded 6 view 假设，比如 seq_id * 6、24 * tta_id、固定相机列表、固定 metadata 长度。
- mmdet3d/models/detectors/fastbev.py 中 self.n_images 的 metadata slicing、temporal split、backproject、test_onnx/test_custom 是否对 n_images=1 正确。
- configs/fastbev/custom/custom_fastbev_mono_front_r18.py 继承 custom_fastbev_6v_r18.py 是否遗留了不适合 mono 的设置，例如 augmentation、test_cfg、anchor/class/NMS、distortion、feature_resize_mode。
- train pipeline 和 eval/test dataset 中 GT 过滤是否一致，FrontCameraVisibleObjectFilter 是否正确注册在 mmdet3d/datasets/pipelines/__init__.py。
- force_resize=True 与 pkl 中 intrinsic_width/height、image_width/height 的几何缩放是否正确，是否会重复缩放 K 或漏写 post_rot/post_tran。
- CustomMultiViewDataset 的 eval 优化是否改变了最终精评口径，尤其 eval_score_thr、eval_max_dets_per_sample、eval_range。
- tools/eval_epoch_checkpoints.py 的 score 分布是否和 dataset.evaluate 使用同一预测过滤口径。
- 当前训练崩溃是否可能由代码 bug 引起；如果更像超参问题，请给出最小配置建议。

输出要求：
1. 先列 Findings，按严重程度排序，每条给出文件路径和行号。
2. 如果没有明确 bug，也请列出 residual risks 和建议的验证命令。
3. 不要大范围重构；如需修改，请先说明最小补丁范围。
4. 本地如果缺少 mmcv/mmdet 环境，静态审查即可；能跑的检查包括 python -m py_compile 和 git diff --check。
```
