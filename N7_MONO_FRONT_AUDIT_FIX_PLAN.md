# Fast-BEV N7 单目前视时序适配审查总结与修复计划

> 更新时间：2026-07-11
> 适用仓库：`/workspace/Fast-BEV_test_custom-fastbev-adapter`
> 当前目标：尽量保留 Fast-BEV 论文模型能力，同时适配 N7 单目前视、四时序输入和芯片板端部署。

## 1. 总体结论

当前 `n_images=1, n_times=4` 的模型侧适配在结构上成立：

- `n_images=1` 表示每个时间步只有一个前视相机；
- `n_times=4` 表示当前帧加三个历史时间步；
- 3D neck 的时序融合通道没有因为六目改成单目而被错误缩减；
- train/test 主路径已经按 `self.n_images` 动态切分相机 metadata；
- 当前模型可以作为 N7 单目前视时序训练 baseline。

但当前仓库还不能直接认定为“时序板端产品闭环”，主要缺口是：

1. 训练使用了历史帧 pose 补偿，生产推理和固定 LUT 尚未具备等价的实时 pose 路径；
2. ONNX、固定 LUT、后处理、量化校准尚未完成逐级数值闭环；
3. 图像强制拉伸虽然投影几何可自洽，但可能损害预训练视觉能力，且关闭了图像随机增强；
4. 数据转换、畸变可见性、坐标原点和 z 语义仍有若干需要收口的正确性问题；
5. anchor、CBGS、NMS 等性能调优应放在几何和部署链路正确之后。

优先级定义：

- **P0**：阻断“时序板端方案”验收；
- **P1**：影响几何、数据、评估或导出正确性，应在下一轮正式训练/部署前修复；
- **P2**：性能优化问题，需要通过 A/B 实验决定；
- **P3**：复现、环境和工程可维护性问题。

## 2. 九个主要问题总结

| 编号 | 问题 | 当前定性 | 优先级 | 修改方向 |
| --- | --- | --- | --- | --- |
| 1 | 板端时序 pose 与历史帧补偿缺失 | 训练和生产推理不等价 | P0 | 接入实时 egomotion，动态历史 LUT 或历史 BEV warp；否则改原生单帧 |
| 2 | ONNX exporter eager/export 分支不一致 | 确定性导出入口问题 | P1 | wrapper 直接调用 `onnx_export_2d/3d`，再做真实 checkpoint 对齐 |
| 3 | `AdamW2` 兼容性 | 内网已完成训练，不是当前阻断 | P3 | 固化内网环境和源码版本，保留可移植回退方案 |
| 4 | manifest、空帧、pose 和异常处理 | 可能影响 split、时序完整性和数据可靠性 | P1 | strict converter、split 零重叠、完整时间轴、pose 插值和质量门禁 |
| 5 | GT 可见性过滤忽略畸变 | train/eval 与模型投影口径不一致 | P1 | 共用 distortion-aware 投影函数，重新统计真实 GT 差异 |
| 6 | 704×256 强制拉伸且关闭图像增强 | 几何可运行，但可能影响模型上限 | P2，增强缺失为 P1 | stretch/crop/高分辨率 A/B；拆分基础变换和随机增强 |
| 7 | baseline 计划 | 当前配置可作为 B0 | 基线任务 | 固化配置、数据、环境、checkpoint 和逐 epoch 指标 |
| 8 | epoch5 最好、之后轻微波动 | 正常早期达峰，不是训练崩溃 | P2 | 使用 epoch5；独立 test、重复 seed、按 best checkpoint 管理 |
| 9 | CBGS、anchor、NMS 沿用问题 | 尚未按 N7 分布优化 | P2 | 先统计 GT/anchor 覆盖，再逐项单变量 A/B |

## 3. 各问题的修改方向与验收标准

### 3.1 问题 1：板端 pose、时序和 LUT

#### 问题表现

训练数据集会将历史相机变换到当前关键帧 lidar/BEV 坐标系：

```text
key_from_history = inverse(global_from_key) @ global_from_history
key_lidar_from_history_camera = key_from_history @ history_lidar_from_history_camera
```

相关代码：

- `mmdet3d/datasets/custom_multiview_dataset.py`
- `_sensor2reference_lidar()`

当前生产推理脚本则：

- 默认把当前图像重复四次；或
- 接收真实历史图像，但不输入 pose、不做 ego-motion compensation。

固定 LUT 工具又是从单个 pkl 样本的投影 metadata 生成，因此历史 LUT 可能固化了该样本的运动状态。

#### 为什么是 P0

它不阻断训练，也不阻断板端程序运行；它阻断的是“时序模型板端效果与训练一致”的验收：

- 重复当前帧会使时序分支输入分布发生变化；
- 使用真实历史帧但不补偿，会产生 BEV 错位、拖影和重复响应；
- 固定某一个历史 pose 的 LUT 无法覆盖正常转弯、加速和直行运动。

#### 量产 pose 来源

推荐由车辆 egomotion/定位模块提供带时间戳的 pose history，而不是由检测模型求 pose。典型融合输入：

- 三轴陀螺仪、三轴加速度计；
- 轮速/车速、转角等 CAN 信号；
- GNSS/RTK、视觉或激光里程计用于校正漂移。

只有裸六轴 IMU 通常不足以稳定提供平移 pose；六轴 IMU 加轮速/转角并经过状态估计，可以支持短时历史对齐。

板端只需要历史相机曝光时刻到当前关键帧时刻的相对位姿，不强制需要绝对全球坐标。

#### 推荐实现路线

优先路线 A：保留时序模型。

1. 定义板端 pose 接口：时间戳、旋转、平移、质量状态、坐标系版本；
2. 在相机曝光时刻插值 pose，而不是取最近 pose；
3. 当前帧使用每车型固定 LUT；
4. 历史帧选择以下一种方案：
   - 根据相对 pose 动态生成历史 LUT；
   - 每帧先固定 LUT 投到自身 BEV，再用相对 pose warp 到当前 BEV；
5. pose 异常时清空历史或重复当前帧，并在训练中加入相同的 history-dropout 策略；
6. 板端历史时间间隔必须和训练一致。当前 converter 默认目标约为 0.5 s、1.0 s、1.5 s，不能在板端随意改成三个连续视频帧。

备选路线 B：平台不能提供可靠 pose。

1. 建立真正的 `n_times=1` 原生单帧配置；
2. 直接训练单帧 baseline；
3. 使用有 pose 的四时序模型作为教师，蒸馏到单帧学生；
4. 明确单帧学生无法完全恢复遮挡历史信息和真实速度观测能力。

#### 验收标准

- 静态路标/车辆在四时序 BEV 中对齐，无系统性拖影；
- pose 时间差和插值误差有 p50/p90/p99 统计；
- PTH 动态投影与板端模拟的各时序 BEV、raw logits 和 decoded boxes 可对齐；
- 转弯、坡道、加减速和 pose 异常场景均有回退测试；
- 时序板端精度明显优于重复帧方案，并接近离线 pose-aware 测试结果。

### 3.2 问题 2：ONNX 导出入口

#### 问题表现

`tools/export_onnx.py` 的 wrapper 通过普通 `model.forward()` 传入 `export_2d/export_3d`，但 `FastBEV.forward()` 只在 `torch.onnx.is_in_onnx_export()` 为真时识别这些参数。导出前计算 eager expected output 时会误入普通测试路径。

#### 修改方向

最小修改：

```text
FastBEV2DExportWrapper.forward -> model.onnx_export_2d(img, None)
FastBEV3DExportWrapper.forward -> model.onnx_export_3d(tuple(inputs), None)
```

同时补充：

- 真实 checkpoint 的 2D/3D ONNX 导出 smoke test；
- `onnx.checker` 和 ONNXRuntime 验证；
- NCHW/NHWC 两种 2D 输出布局检查；
- 确认 3D ONNX 输出 raw logits，sigmoid 仅在后处理执行一次；
- 保存输入/输出名称、shape、layout 和 channel layout 到 `export_metadata.json`。

#### 验收标准

- eager reference、`torch.onnx.export`、ONNXRuntime 均可运行；
- 2D feature 和 3D logits 与 PTH 的误差满足浮点容差；
- 使用真实 N7 checkpoint 和真实样本完成一次端到端导出/推理。

### 3.3 问题 3：`AdamW2`

#### 修订结论

用户内网环境已经使用 `AdamW2` 完成多个 epoch 训练，因此：

- 不再视为当前训练 bug；
- 内网运行环境是该问题的权威结论；
- 本地现代 PyTorch 的函数签名不兼容只属于可移植性风险。

#### 修改方向

- 固化 PyTorch、CUDA、mmcv、mmdet、mmdet3d 版本；
- 记录 `mmdet3d/models/opt/adamw.py` 的源码 hash；
- 保存 runner 最终 resolved config；
- 验证 checkpoint 能正常 `resume` optimizer state；
- 如迁移到新 PyTorch，再增加版本适配或标准 `AdamW` 回退，不要影响内网已验证路径。

### 3.4 问题 4：数据转换完整性

#### 风险点

1. 缺失 manifest 时可能自动发现全部 clip，造成 train/val/test 混用；
2. converter 默认删除无有效 GT 的帧，破坏真实时间轴和负样本；
3. pose 缺失时仍可能写出 info，错误延迟到 dataset 取历史帧时才暴露；
4. broad `except` 会静默跳过异常帧；
5. pose 目前以最近时间戳匹配为主，未完成严格的曝光时刻插值；
6. label、camera 和 pose 时间戳即使都存在，也可能不是同一个物理时刻。

#### 修改方向

- 正式转换默认要求 manifest 存在，自动发现仅用于显式 debug 模式；
- 输出前检查 train/val/test 的 clip 和 token 零重叠；
- 完整时间轴保留空帧，key-frame 是否采样空帧由 dataset 配置决定；
- temporal 模式默认 `require_pose=True`；
- pose 插值到相机曝光时刻，并保存插值来源和时间误差；
- 错误按类别统计，strict 模式遇到关键字段缺失直接非零退出；
- sidecar summary 写出相机、pose、历史覆盖、类别、空帧和失败原因统计。

#### 验收标准

- split 零重叠；
- temporal 样本 pose coverage 100%；
- 四个时间步实际 offset 分布符合配置；
- 随机抽样投影历史静态点可对齐；
- converter 不再静默吞掉关键错误。

### 3.5 问题 5：畸变可见性过滤

#### 问题表现

模型 backproject 支持 N7 distortion，但训练 pipeline 和 dataset eval 的 GT 可见性过滤仍以 pinhole 投影为主，可能在广角边缘误删或误保留目标。

#### 修改方向

- 抽取共享的 N7 投影模块；
- dataset、pipeline、可视化、LUT 构建共同使用同一畸变公式；
- 统一使用 `K + distortion + sensor2lidar + post_rot/post_tran + image size`；
- 对 box 边缘采样或做投影轮廓与图像边界相交，避免只判断中心和八角点；
- 修复前后在真实 pkl 上输出每类、每距离段 GT 保留差异。

#### 验收标准

- train/eval 对同一个 GT 给出相同可见性结论；
- 投影结果与 OpenCV/标定工具或人工可视化一致；
- 广角边缘目标不再出现明显的系统性误删。

### 3.6 问题 6：强制拉伸和图像增强

#### 当前结论

`1600×900 -> 704×256` 非等比拉伸：

- 水平缩放约为 `0.44`；
- 垂直缩放约为 `0.284`；
- 物体外观相对被横向拉宽约 `1.55` 倍。

如果 K/post transform 同步按 x/y 缩放，投影几何可以正确；但图像外观与 COCO 预训练、论文输入分布差异较大，可能限制模型能力。

`force_resize=True` 还会关闭图像随机 resize/crop/rotate/flip，因此当前只有 BEV 级增强，图像级泛化能力可能不足。

#### 修改方向

至少进行三组同资源 A/B：

1. A：当前 704×256 非等比拉伸；
2. B：1600×900 等比缩放到 704×396，再裁剪到 704×256；
3. C：芯片允许时使用 704×384 或其他接近 16:9 的尺寸。

裁剪方案需要按真实 N7 GT 选择 vertical crop offset：

- 统计 0～20、20～40、40～60、60～80 m 的 GT 投影保留率；
- 重点检查远处小目标是否被裁掉；
- 中心裁剪只是候选，不应未经统计直接固定。

工程实现建议：

- 将确定性的基础 resize/crop 与训练随机增强拆分；
- 四个时序帧共享同一组几何增强参数；
- 所有变换只通过一次 `post_rot/post_tran` 生效；
- 可离线缓存 704×396，再在线做轻量 crop/flip，兼顾 IO 和增强；
- 训练、测试、ONNX 和板端必须使用同一确定性基础预处理。

#### 验收标准

- 投影可视化无重复缩放或主点偏移；
- B/C 相对 A 的总体和远距指标有明确结论；
- 恢复图像增强后训练吞吐下降可测量且可接受；
- 板端输入与 test pipeline 逐像素一致。

### 3.7 问题 7：建立 baseline

当前配置可定义为：

```text
B0 = temporal mono + n_times=4 + pose-aware train/test
     + 704×256 stretch + force_resize + 无图像随机增强
```

需要固化：

- 内网实际 `total_epochs=15` 的 resolved config；
- 每 epoch eval 的实际 hook/config；
- 代码 commit、环境版本；
- train/val/test manifest 和 pkl hash；
- 每车型标定 hash；
- 每个 epoch checkpoint、指标、score 分布；
- 最佳 checkpoint 和选择依据。

B0 用于比较后续修改，不代表最终图像几何或最终板端精度。

### 3.8 问题 8：epoch5 最优

#### 修订结论

本次内网训练表现为：

- epoch1～5 各项指标逐步提升；
- epoch5 当前最好；
- epoch5 后略微下降并持续波动，截至 epoch12 没有刷新最好结果。

这属于正常的早期达峰、轻度过拟合或调度后半段收益不足，不是训练崩溃。

仓库旧记录中“epoch5 score 塌到 0.05 附近”的实验属于另一版 run/config，不应与本次结果混合。

#### 修改方向

- 当前使用 epoch5 作为 B0 best checkpoint；
- 在独立日期/车辆 test split 上验证 epoch5；
- 输出 car/truck、距离分桶、score 分布和 z/yaw/size 误差；
- 再跑一个 seed，确认最佳 epoch 是否稳定落在 4～6；
- 按 mAP 自动保存 best，early-stopping patience 可先设为 4～5；
- 未看到 loss/score 异常前，不要仅凭 epoch5 达峰盲目降低 LR。

### 3.9 问题 9：CBGS、anchor 和 NMS

#### 修改原则

必须在数据、坐标、z 语义和评估修复之后调优，并坚持单变量实验。

#### CBGS

- 当前 car/truck 数量约为 4.4:1；
- 先保留无 CBGS 的 B0；
- 再比较 CBGS 或 class-aware sampler；
- 比较时按 optimizer step 对齐，因为 CBGS 会改变每 epoch 长度和训练速度。

#### Anchor

- 统计每类 `l/w/h` 的 p10/p50/p90；
- 计算当前 anchors 对 GT 的最大 IoU；
- 分别报告 IoU≥0.3、0.5、0.6 的覆盖率；
- 再通过真实尺寸分布聚类候选 anchors；
- anchor z 必须等 z center/bottom 语义确认后再调整。

#### NMS

- 当前两类只使用十类配置中的前两项；
- truck 的 `nms_rescale_factor=0.7` 是 nuScenes 经验值；
- 根据 duplicate FP、recall 和拥挤场景做小范围网格测试；
- 板端后处理必须使用和 PTH 相同的 NMS 参数与类别顺序。

## 4. 七个板端兼容性补充问题

### 4.1 Channel 排列变化

论文源码的大致布局：

```text
[B, T*C, X, Y, Z] -> channel order [Z, T, C]
```

厂家和当前部署布局：

```text
每个时间 [B, C, X, Y, Z] -> [B, Z*C, X, Y]
四时序 concat -> channel order [T, Z, C]
```

原因是厂家需要把四个时序 BEV 暴露成四个独立 3D ONNX 输入。该排列与 `fastbev_bst.py` 的行为一致，部署上合理，且只是通道置换，不丢信息。

影响：

- 当前仅加载 COCO 2D 权重，3D neck 从头学习，因此 B0 不会因该排列本身损失表达能力；
- 如果加载论文完整 Fast-BEV checkpoint，需要把 `neck_3d.fuse.weight` 的输入通道从 `[Z,T,C]` 置换到 `[T,Z,C]`；
- PTH、导出、LUT 和板端必须始终保持同一排列。

修改方向：

- 在 export metadata 中记录 `channel_layout=TZC`；
- 记录 `temporal_order=[key, history_1, history_2, history_3]` 和每帧时间差；
- 加入用唯一 `t/z/c` 编码的 layout 单元测试。

### 4.2 固定 LUT 推理闭环缺失

新增板端模拟主路径：

```text
预处理 -> 2D ONNX -> LUT gather/scatter -> 四时序 BEV
       -> 3D ONNX -> canonical postprocess
```

逐级保存并比较：

- 输入 tensor；
- 2D feature；
- 每个时序 BEV；
- raw cls/bbox/dir logits；
- decoded boxes/scores/labels。

每份 LUT metadata 必须包含车型标定 hash、图像变换、畸变、voxel grid、stride、channel layout 和 time slot。

### 4.3 厂家遗留后处理硬编码

当前 `tools/utils.py` 存在绝对 anchor 文件路径、十类 reshape、固定阈值等遗留行为。

修改方向：

- 从 config/export metadata 获取 `num_classes=2`、code size、anchor、feature shape 和 NMS 参数；
- 离线生成并版本化 anchor 文件；
- 删除绝对路径；
- ONNX 输出 raw logits，后处理只 sigmoid 一次；
- 以 `bbox_head.get_bboxes()` 为权威参考做逐框一致性测试。

### 4.4 量化校准数据不是明确的 ONNX 输入 tensor

当前 backbone 校准主要拷贝原始/缓存图片，是否正确取决于编译器是否执行完全一致的预处理。

修改方向：

- 同时保存原图片和真正送入 2D ONNX 的 tensor；
- 明确 RGB/BGR、mean/std、dtype、NCHW/NHWC；
- 文件名包含 token、camera、time slot，避免覆盖；
- 3D head 保存四个真实 BEV 输入；
- 校准集按车型、光照、天气、场景分层抽样；
- 对比编译器预处理输出和保存 tensor。

### 4.5 标定内参与图像尺寸口径

推荐原则：原始标定是唯一 source of truth，有效投影是派生结果。

必须保存：

- `K_native`；
- K 对应的 `intrinsic_width/height`；
- distortion；
- camera-to-lidar/vehicle extrinsic；
- 实际 `image_width/height`；
- resize/crop/pad 的 `post_rot/post_tran`。

只允许以下一种表达方式：

1. 原始 K + 明确 post transform；或
2. 已派生的 `K_eff` + identity post transform。

禁止提前缩放 K 后又重复应用 post transform。

每辆车使用自己的标定和 LUT；不能把 sensor 标称分辨率当成 K 的坐标尺寸，必须用主点、标定协议和投影可视化验证。

### 4.6 顶部 lidar 原点与后轴 ego 原点

当前训练坐标是：

```text
x 向前，y 向左，z 向上，原点为 N7 顶部主 lidar
```

尚未迁移到后轴地面投影点。

推荐先选接口方案：

- 优先保留模型内部顶部 lidar 坐标，在输出 API 统一变换到后轴 ego；
- 只有芯片或下游硬性要求时，才整体迁移 label、相机外参、pose、ROI、anchor、LUT 和评估。

不能只修改其中一个环节。

### 4.7 z 方向约 0.8 m 偏差

当前高概率原因：

- pkl GT 的 z 被当成 box 重心；
- `LiDARInstance3DBoxes(origin=(0.5,0.5,0.5))` 将训练内部 z 转成底中心；
- prediction tensor 也是底中心；
- eval 却直接用 prediction 底中心减 pkl GT 重心。

因此：

```text
z_error ≈ -h/2
car height ≈ 1.6 m -> z_error ≈ -0.8 m
```

修改方向：

1. 用标注协议和点云确认原始 `location.z` 是重心还是底中心；
2. 若原始是重心，eval 将 prediction 转成 `gravity_center` 后再计算 xyz；
3. 可视化、评估、JSON 输出和板端 API 共用一份 origin 转换函数；
4. 输出 metadata 明确 `box_origin`；
5. 修复后直接重评已有 prediction pkl，无需重新训练；
6. 若 `z_mean` 不是负约 0.8 m，再检查原始标注是否本来就是底中心。

## 5. 推荐修改顺序

### 阶段 0：固化当前实验，不改模型

1. 保存内网实际 15 epoch resolved config；
2. 保存 epoch5 best checkpoint 和所有逐 epoch 评估；
3. 给旧“epoch5 collapse”实验单独命名，避免和新 B0 混淆；
4. 固化代码、环境、pkl、manifest 和 calibration hash。

产物：可复现的 `B0`。

### 阶段 1：先修不需要重训的正确性问题

1. 确认并统一 box z/origin 语义；
2. 修复 eval、可视化和输出 API 的 z 转换；
3. 修复 ONNX export wrapper；
4. 参数化板端后处理，删除绝对路径和十类硬编码；
5. 重评 epoch1～15 已有 prediction/checkpoint。

产物：可信指标和可用的浮点 ONNX 导出链路。

### 阶段 2：修数据和投影口径

1. converter 增加 strict manifest/split/pose/error gate；
2. 保留完整时间轴和空帧；
3. pose 插值到相机曝光时间；
4. 统一 distortion-aware GT 可见性投影；
5. 生成真实数据审计报告和投影可视化。

产物：可证明正确的 N7 train/val/test pkl。

### 阶段 3：决定最终时序产品路线

1. 和芯片/整车系统确认实时 egomotion 接口；
2. 定义 pose 坐标系、时间同步、质量状态和回退策略；
3. 有可靠 pose：实现动态历史 LUT 或历史 BEV warp；
4. 无可靠 pose：建立 `n_times=1` 原生单帧 baseline；
5. 需要时再做时序教师到单帧学生蒸馏。

产物：不再依赖“真实历史无补偿”或“长期重复当前帧”的产品方案。

### 阶段 4：图像几何和增强 A/B

1. 保留当前 stretch 作为 B0；
2. 测试等比 resize+crop；
3. 芯片允许时测试接近 16:9 的输入；
4. 恢复时序一致的图像随机增强；
5. 比较精度、远距召回、训练吞吐和板端时延。

产物：确定最终输入几何和训练增强策略。

### 阶段 5：完成板端数值闭环

1. 固定/动态 LUT 推理流程；
2. TZC channel layout 和 temporal order metadata；
3. PTH、ONNXRuntime、板端 float 逐级对齐；
4. 保存真实 2D/3D 量化校准 tensor；
5. INT8/FP16 精度和性能评估；
6. 每车型 calibration/LUT 管理。

产物：可验收的板端部署包。

### 阶段 6：最后做模型性能调优

推荐依次只改一个变量：

1. CBGS/class-aware sampling；
2. anchor sizes 和 anchor z；
3. NMS threshold/rescale；
4. LR、schedule、总 epoch；
5. 蒸馏或其他模型增强。

产物：建立在正确数据、正确几何和正确部署链路之上的最终模型。

## 6. 最终验收清单

### 数据与训练

- [ ] train/val/test clip 和 token 零重叠；
- [ ] pose coverage、时间误差和历史 offset 有完整统计；
- [ ] z center/bottom 语义有标注协议和点云证据；
- [ ] train/eval 使用相同的 ROI、可见性和畸变投影；
- [ ] resolved config、环境和数据版本可复现；
- [ ] best checkpoint 由独立 val/test 指标选择。

### 模型能力

- [ ] stretch 与等比 crop 已做同资源 A/B；
- [ ] 远距、车型、日期、光照分别报告指标；
- [ ] CBGS、anchor、NMS 均基于数据统计调优；
- [ ] 单帧和时序方案使用相同口径比较；
- [ ] 若做蒸馏，报告同结构直接单帧学生作为对照。

### 板端部署

- [ ] 实时 pose 接口或明确的单帧产品决策；
- [ ] 当前帧固定 LUT 和历史帧动态补偿闭环；
- [ ] PTH、ORT、板端 float 的 feature/logit/box 逐级对齐；
- [ ] `channel_layout=TZC` 和 temporal order 固化；
- [ ] 后处理无绝对路径、无十类硬编码、无双重 sigmoid；
- [ ] 量化校准输入与真实 ONNX 输入一致；
- [ ] 每车型使用匹配的标定和 LUT；
- [ ] 输出坐标原点、轴方向和 `box_origin` 明确；
- [ ] pose 丢失、时间戳异常和历史不足时有安全回退；
- [ ] 精度、延迟、内存和功耗满足最终板端指标。

## 7. 当前最优先的五件事

1. **固化这次 15 epoch 训练及 epoch5 best，避免实验信息继续漂移。**
2. **确认并修复 z 重心/底中心评估口径，重新评估已有结果。**
3. **确认量产平台是否能提供带时间戳的融合 egomotion，决定时序还是原生单帧。**
4. **修复 ONNX exporter，并建立固定/动态 LUT 的板端模拟闭环。**
5. **完成 stretch 与等比 crop A/B，再调 CBGS、anchor、NMS 和训练超参。**
