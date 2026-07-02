# N7 Fast-BEV 6V 耗时消融实验记录

日期：2026-06-30
分支：`test/custom-fastbev-adapter`
相关提交：`237487b 清理 N7 Fast-BEV 适配训练链路`
用途：给后续对话继续分析训练耗时问题，记录每组耗时实验对应的代码/配置修改点和结论。

## 统一背景

- 用户确认：这里提到的 batch size 均为 `samples_per_gpu`。
- 旧 8 卡参考训练最终确认也是每卡 24；不要再用当前迁移文件里的 `fix_lut=True` 解释旧训练速度，因为旧训练时 `fix_lut=False`，且用户曾测试 `fix_lut=True/False` 对训练耗时影响不大。
- 当前 4 卡 N7 全量训练主要配置：`samples_per_gpu=24`、`workers_per_gpu=8`，每个 BEV 样本为 `6 views * 4 times = 24 images`。
- 当前 4 卡每张 GPU 每 iter 约有 24 个 BEV 样本；每个样本 4 个时序，backproject 调用量很高，因此单次 `backproject_inplace` 的小耗时会被放大。
- 旧 8 卡参考读取原图 `1600x900`；当前 4 卡为了减轻磁盘压力读取 `704x256`，所以当前 `data_time≈0.3s` 时主要不是磁盘瓶颈。

### 单 iter backproject 计算量量化

- 对当前 v1/v2 style、`n_voxels=[200, 200, 4]` 而言，每次 backproject 处理 `200 * 200 * 4 = 160,000` 个 BEV/lidar 网格点。
- 当前每 GPU 每 iter 是 `samples_per_gpu(24) * n_times(4) = 96` 次 backproject 调用。
- 因此每 GPU 每 iter 总投影点数约为 `96 * 160,000 = 15,360,000`，即约 `15.4M` 个点。
- 动态畸变开启时，每个点至少经过 `lidar2cam transform -> distortion polynomial(k1-k6, p1-p2) -> intrinsic projection -> post_rot/post_tran`。
- 如果使用 v3/v4 style 的多 BEV scale，还需要按 BEV scale 数量继续乘上对应倍数。因此单次 backproject 看起来只有 `0.05-0.13s`，累计到 96 次后仍会变成数秒级训练耗时。

## 关键对比表

| 编号 | 实验/修改点 | 条件 | 代表日志 | 结论 |
| --- | --- | --- | --- | --- |
| A0 | 旧 8 卡参考基线，`fastbev_ljr.py` | 8 GPU，每卡 24，`work=8`，旧数据链路，动态畸变，`fix_lut=False`，读 `1600x900` | `Epoch [3][860/1060] time=4.081, data_time=0.548` | 这是速度参考，但代码、数据链路、GPU 数、几何链路都与当前不同，不能直接等价比较。 |
| A1 | 当前 4 卡初始慢速现象 | 4 GPU，每卡 24，当前 `fastbev.py`，读 `704x256` | `Epoch [5][350/1858] time=27.047, data_time=0.301` | `data_time` 很低，主要是模型/几何计算慢，不是读图慢。 |
| A2 | 单卡 L20 单 sequence 参考 | 单卡，约 5000 帧，`samples_per_gpu=16`，`workers=4`，当前 `fastbev.py` | `Epoch [3][230/357] time=5.570, data_time=0.619` | 当前 `fastbev.py` 不是所有场景都异常慢；全量 4 卡问题需要看 full-data 几何/backproject 路径。 |
| A3 | 仅关闭动态畸变 | 4 GPU，每卡 24，`model.use_distortion=False` | 后续稳定约 `6.8-7.2s/iter`，`data_time≈0.3s` | 动态畸变是重要耗时来源之一；无畸变时当前训练可回到较合理区间。 |
| A4 | `feature_resize_mode=nearest` | 在 `use_distortion=False` 基础上加 `model.feature_resize_mode=nearest` | 用户反馈日志无明显变化 | resize 插值不是主瓶颈。 |
| A5 | `_slice_view_meta` 浅拷贝切片 | 替换 `copy.deepcopy` 为浅拷贝后测试 | 用户反馈日志无明显变化 | 元数据深拷贝不是主瓶颈；最终已恢复 `copy.deepcopy`，避免无收益改动。 |
| A6 | stage profile 诊断 | `FASTBEV_STAGE_PROFILE=1`，`use_distortion=False` | 稳定后 `extract≈5.0-5.7s`，`backproject≈4.2-5.0s`，`train≈6.8-7.4s` | 在无畸变时，`extract_feat` 内的 volume/backproject 是主要耗时。profile 有同步开销，只看比例。 |
| A7 | backproject profile，未 fused scatter | `FASTBEV_BACKPROJECT_PROFILE=1`，`use_distortion=False`，旧逐相机 scatter | 单次 backproject 常见 `total≈0.07-0.10s`，`scatter≈0.04-0.07s`；stage 中 `backproject≈4.4-7.8s` | 每 GPU 每 iter 有 96 次 backproject、约 15.4M 点投影；逐相机 scatter 单次只有几十毫秒，也会被调用次数放大成数秒级瓶颈。 |
| A8 | fused final-camera scatter | 计算每个 BEV 点最终由哪个相机覆盖，然后一次性 gather/scatter | fused 后 scatter 约 `0.008-0.017s/backproject` | 这是有效优化。语义保持为“当前主线自然相机顺序逐相机覆盖”的最终结果，不等同于恢复旧 `fastbev_ljr.py` 的自定义 `camera_list` 顺序。 |
| A9 | 旧动态畸变路径，关 profile | `use_distortion=True`，未加批量畸变优化 | `[10] time=22.020`，后续多在 `20-22s/iter`，`data_time≈0.3s` | 动态畸变路径非常重，且不是数据读取造成。 |
| A10 | 批量动态畸变 + fused scatter，开 backproject profile | `use_distortion=True`，batched distortion，`fused_scatter=True` | 多数 backproject `project≈0.05-0.13s`，`scatter≈0.008-0.017s`，偶发 spike `project≈0.540s`；第 10 iter `time=17.476` | scatter 已不是主要瓶颈；但 `0.05-0.13s/backproject` 的畸变投影乘以 96 次调用后，理论上就是约 `5-12s/iter` 的量级。profile 会引入同步开销。 |
| A11 | 批量动态畸变 + fused scatter，关 profile | `use_distortion=True`，最终优化训练路径 | `[20] time=16.716`、`[30] 16.308`、`[40] 16.786`、`[50] 15.991`，`memory≈26818 MB` | 相比旧动态畸变 `20-22s` 降到约 `16s`，有效但仍明显慢于 `use_distortion=False` 的约 `7s`。 |
| A12 | batch size 调大到 28 | 4 GPU，每卡 28，后续调参测试 | `Epoch [10] eta=8 days, 12:35:22, time=23.125, data_time=2.015, memory=31225` | 比每卡 24 更慢，显存接近压力区，吞吐没有随 batch 增大改善。 |
| A13 | batch size 调大到 30 | 4 GPU，每卡 30，后续调参测试 | `[10] time=23.922, data_time=3.875, memory=33430`；`[20] time=20.686, data_time=0.363, memory=33430` | warmup 后仍比每卡 24 慢，显存约 33.4GB；当前更像计算/显存压力增大，不是简单 worker 或读图问题。 |
| A14 | 同机同 batch/work/1600x900 几何 profile 对照 | 分别在当前 `fastbev.py` 仓库和旧 `fastbev_ljr.py` 仓库跑各自 pkl；两边都读 `1600x900`；`fastbev_ljr.py fix_lut=False` | current `valid_sum≈333k, assigned_count≈159.7k, post_scale=(0.44,0.284444)`；ljr `valid_sum≈255k, any_valid_count≈159.3k, post_scale=(1,1)` | 能证明两边几何投影/有效 mask 不等价；不能单独证明 4 倍级耗时差。current 的逐相机 valid 总数约多 30%，但至少被一个相机覆盖的 BEV 点数量基本相同。 |

## 具体日志摘录

### 1. 当前 4 卡初始慢速

```text
Epoch [5][350/1858] eta: 10 days, 5:01:35, time: 27.047, data_time: 0.301
```

含义：`data_time` 很低，说明当时不应优先怀疑磁盘读取。

### 2. 无畸变 profile 后稳定耗时

```text
Epoch [1][20/1858] time: 7.155, data_time: 0.301, memory: 26810
Epoch [1][30/1858] time: 7.199, data_time: 0.312, memory: 26810
Epoch [1][40/1858] time: 6.817, data_time: 0.325, memory: 26810
Epoch [1][50/1858] time: 7.095, data_time: 0.334, memory: 26810
```

结论：`use_distortion=False` 时，当前训练链路可以稳定到约 `7s/iter`。

### 3. 旧动态畸变路径，未加批量优化

```text
Epoch [1][20/1858] time: 21.244, data_time: 0.320, memory: 26809
Epoch [1][30/1858] time: 20.494, data_time: 0.301, memory: 26809
Epoch [1][40/1858] time: 20.433, data_time: 0.293, memory: 26809
Epoch [1][50/1858] time: 20.636, data_time: 0.336, memory: 26809
```

结论：动态畸变旧路径约 `20-22s/iter`，明显是计算侧问题。

### 4. 批量畸变 + fused scatter 后，关 profile

```text
Epoch [1][20/1858] time: 16.716, data_time: 0.256, memory: 26818
Epoch [1][30/1858] time: 16.308, data_time: 0.278, memory: 26818
Epoch [1][40/1858] time: 16.786, data_time: 0.279, memory: 26818
Epoch [1][50/1858] time: 15.991, data_time: 0.331, memory: 26818
```

结论：优化有效，约从 `20-22s` 降到 `16s`，但动态畸变仍是剩余大头。

### 5. stage profile 显示 backproject 是主耗时

```text
[FASTBEV_STAGE extract #002] total=5.693s backbone_neck_fuse=0.360s volume_total=5.239s backproject=4.992s neck3d=0.086s
[FASTBEV_STAGE train #002] total=7.355s extract=5.694s bbox_head=0.005s bbox_loss=1.656s

[FASTBEV_STAGE extract #004] total=4.962s backbone_neck_fuse=0.409s volume_total=4.402s backproject=4.155s neck3d=0.142s
[FASTBEV_STAGE train #004] total=6.806s extract=4.963s bbox_head=0.005s bbox_loss=1.837s
```

结论：2D backbone/FPN 不是主要耗时，volume/backproject 是主要耗时。

### 6. backproject profile 对比

未 fused scatter，`use_distortion=False`：

```text
[FASTBEV_BACKPROJECT #003] total=0.0776s project=0.0019s valid=0.0078s scatter=0.0652s
[FASTBEV_BACKPROJECT #007] total=0.0804s project=0.0024s valid=0.0058s scatter=0.0695s
```

批量畸变 + fused scatter，`use_distortion=True`：

```text
[FASTBEV_BACKPROJECT #001] total=0.0738s project=0.0584s valid=0.0045s scatter=0.0086s fused_scatter=True
[FASTBEV_BACKPROJECT #004] total=0.0806s project=0.0623s valid=0.0049s scatter=0.0103s fused_scatter=True
[FASTBEV_BACKPROJECT #011] total=0.1651s project=0.1350s valid=0.0085s scatter=0.0173s fused_scatter=True
```

结论：fused scatter 后 scatter 不再是主要瓶颈；动态畸变时主要耗时转移到 `project`。单次 `project≈0.05-0.13s` 不能孤立看，因为每 GPU 每 iter 有 96 次 backproject，约 15.4M 个点要走完整几何链路，所以累计后自然会变成 `5-12s/iter` 量级。

### 7. 同机 1600x900 几何 profile 对照

测试条件：

- 两边都使用同一台机器、相同 batch/work、读取 `1600x900` 图。
- 当前仓库跑 `fastbev.py` 和当前 N7 pkl；旧仓库跑 `fastbev_ljr.py` 和旧链路 pkl。
- `fastbev_ljr.py` 明确使用 `fix_lut=False`。
- 本组日志打开了 counts 统计；如果没有同时打开 `FASTBEV_GEOM_PROFILE_SYNC=1`，日志中的细分耗时不作为严谨耗时结论，只使用 valid/count/meta 字段做几何差异判断。

current `fastbev.py` 摘要：

```text
points=160000 stride=4 use_distortion=True camera_rule=natural_last
valid_sum≈332935-336264
assigned_count≈159670-159695
assigned_ratio≈0.998
dist_lens=[5, 5, 5, 5, 5, 5]
src_hw/img_hw=[(900,1600) ...]
post_scale=[(0.44,0.284444) ...]
```

旧 `fastbev_ljr.py` 摘要：

```text
points=160000 fix_lut=False camera_list=[1, 2, 0, 4, 5, 3]
valid_sum≈252639-262777
any_valid_count≈159162-159493
any_valid_ratio≈0.995-0.997
dist_lens=[5, 5, 5, 5, 5, 5]
src_hw/img_hw=[(-1,-1) ...]
post_scale=[(1.0,1.0) ...]
```

客观结论：

- 两边的 per-camera valid 总数明显不同：current 约 `333k`，ljr 约 `255k`，current 多约 `30%`。
- 两边至少被一个相机覆盖的 BEV 点数量接近：current `assigned_count≈159.7k`，ljr `any_valid_count≈159.3k`，都接近每次 backproject 的 `160k` 个 BEV 点。
- 两边元数据/投影链路不同：current 明确带 `post_scale=(704/1600, 256/900)`；ljr profile 中 `post_scale=(1,1)`，且 `src_hw/img_hw` 未从旧 meta 中取到。
- 这组数据能证明两份代码不是等价几何链路；不能单独证明 current 慢 4 倍。原因是投影阶段仍会对全部 `160k * 6 camera` 点计算，valid 数量主要影响后续 gather/scatter；而 current 已使用 fused scatter 时实际写入量约为 `assigned_count≈160k`，不是 `valid_sum≈333k`。

## 已保留到最终代码的有效改动

1. 训练侧 3D head 输入对齐导出/部署路径：
   - `style in ['v1', 'v2']` 下，每个时序 BEV volume 先 fold 成 `[bs, z*c, x, y]`。
   - 时序再 concat 成 `[bs, t*z*c, x, y]`。
   - 这让训练侧 3D neck/head 的输入布局与 `export_3d`/板端部署使用的 4D seq-major 形式一致。

2. deploy/head calibration 输入只在需要时生成：
   - 仅 `test_onnx`、`test_custom` 或保存 head calibration 数据时 materialize。
   - 避免训练时额外保留无用中间输入。

3. no-distortion 投影直接使用 `R @ xyz + t`：
   - 避免每次构造 homogeneous `[x, y, z, 1]`。
   - 属于小优化，不是主瓶颈。

4. fused final-camera scatter：
   - 按当前主线“自然相机顺序逐相机覆盖”的最终胜出相机计算 assigned camera。
   - 每个 BEV 点只 gather/scatter 一次。
   - 不改变当前主线覆盖语义，但不等价于旧 `fastbev_ljr.py` 的自定义相机顺序。

5. batched dynamic distortion projection：
   - 将 6 个相机的 `rot/tran/intrin/post_rot/post_tran/distortion` stack 后批量计算。
   - 曾用独立 torch 对比脚本验证新旧公式 `max_diff 0.0`。
   - 这是动态畸变路径的主要有效优化。

## 已撤回或清理的改动

- 临时 `FASTBEV_STAGE_PROFILE`、`FASTBEV_BACKPROJECT_PROFILE`、`FASTBEV_BACKPROJECT_FUSED` 等环境变量和日志打印已从最终训练代码移除。
- `_slice_view_meta` 浅拷贝优化无明显收益，已恢复 `copy.deepcopy`。
- `feature_resize_mode=nearest` 没有明显加速，不作为最终优化方向。
- 旧 `fastbev_ljr.py` 中通过 `backproject_inplace` 保存 `x.npy`、`y.npy`、
  `valid.npy`、`projection.npy`、`features.npy`、`volume.npy` 的 fixed-LUT
  debug/export 逻辑没有合入当前可训练主线。该功能对板端固定 LUT 效果验证和
  固定 LUT 索引表生成有价值，但它属于部署工具链；当前旧实现还带有 hardcoded
  输出路径、camera order 和 `fix_lut` 分支，直接进入训练代码会污染普通
  train/test 语义。后续如果要恢复，应做成显式离线导出工具或受配置保护的
  debug 模式，并固定输入 pkl、相机顺序、图像尺寸、`post_rot/post_tran` 和
  畸变解释后再生成索引表。

## 已验证差异与不能推出的结论

### 已验证差异

- 旧 `fastbev_ljr.py` 8 卡历史训练使用每卡 24、`fix_lut=False`。不能再用当前迁移文件里的 `fix_lut=True` 解释旧训练速度。
- 当前 4 卡 full-data 慢速日志中 `data_time≈0.3s`，不能把主要瓶颈归因到磁盘读取。
- `use_distortion=False` 时当前链路稳定约 `6.8-7.2s/iter`；`use_distortion=True` 旧动态畸变路径约 `20-22s/iter`；batched distortion + fused scatter 后约 `16s/iter`。
- 同机 1600x900 几何 profile 证明 current 与 ljr 的 valid mask 分布不同：current per-camera `valid_sum≈333k`，ljr `valid_sum≈255k`；但 unique 覆盖数量接近，分别约 `159.7k` 和 `159.3k`。

### 不能推出的结论

- 不能仅凭 `valid_sum` 差异说明 4 倍级总耗时差。`valid_sum` 更多主要影响逐相机 scatter，而当前主线已使用 fused scatter，实际写入量接近 unique 覆盖点数。
- 不能把旧 `fastbev_ljr.py` 速度直接视为等价优化结果。最新日志显示两边几何链路不等价，至少 `post_scale`/尺寸元数据路径不同。
- 不能在没有精确 A/B 的情况下切换到旧投影公式或旧畸变参数解释。那会改变训练几何语义，可能影响模型效果。

### 后续只建议做的客观验证

- 若继续查旧/新差异，应固定同一份 `img_meta`、同一份 `points`、同一组 feature shape，分别执行两边投影函数。
- 必须记录 `projection time`、projected point max/mean diff、valid mask diff count/ratio、per-camera valid count、unique valid count、实际 scatter 写入点数。
- 如果测耗时，必须用 `torch.cuda.synchronize()` 包住被测段；否则 profile 的阶段耗时会受 CUDA 异步执行影响，不适合作为结论。
- 如果验证 5 参数畸变解释差异，应作为单独 A/B：只改变“第 5 个畸变参数是否作为 k3 使用”，不要同时改变 resize/post_scale 链路。
- 如果推进固定 LUT 板端索引表，应作为独立任务处理：先用当前主线几何链路导出
  `x/y/valid/projection`，再和旧 `fastbev_ljr.py` 生成结果做同输入 A/B，
  不要在 N7 6V 可训练 baseline commit 中直接打开 fixed-LUT 读写分支。

## 实时训练日志验证方案

本轮临时方案是在两个代码仓库分别跑各自 pkl 和训练框架，并在 detector 内输出同一组字段。该 profile 代码只用于本轮采样，当前代码已回退到可训练提交，不再保留这部分日志逻辑。

如后续需要复现，典型用法曾是：

```bash
FASTBEV_GEOM_PROFILE=1 \
FASTBEV_GEOM_PROFILE_CALLS=24 \
FASTBEV_GEOM_PROFILE_SEQ=all \
PYTHONPATH=$PWD:$PYTHONPATH \
python tools/train.py <config> --work-dir <work_dir> --no-validate
```

日志重点字段：

- `valid_counts`：每相机 valid 点数量。
- `valid_sum`：逐相机 valid 总和，可近似代表旧 sequential scatter 写入次数。
- `assigned_count` / `any_valid_count`：至少被一个相机覆盖的 BEV 点数。
- `project` / `project_or_lut`：投影或 LUT 读取耗时。
- `scatter`：实际写 feature 耗时。
- `dist_lens`、`src_hw`、`img_hw`、`post_scale`：检查两边 pkl 尺寸、畸变参数和 post transform 是否一致。

对比判断：

- 如果旧 `valid_sum` / `any_valid_count` 明显低于当前，速度差异更可能来自有效点/写入量不同。
- 如果 valid 数量接近但 `project` 耗时差距很大，再继续分析投影实现效率。
- 如果 `post_scale`、`dist_lens` 或 camera order 不一致，应先按几何链路差异处理，不能直接把旧速度视为等价优化。

## 训练继续策略

- 如果 epoch-4 checkpoint 是在“3D head 输入 4D seq-major 对齐”之前训练出来的，不建议继续 resume；应从预训练/init checkpoint 重新训练，因为 3D neck/head 看到的 channel/temporal 布局不同。
- 如果 checkpoint 已经包含 4D seq-major 对齐，只是缺少 fused scatter、批量畸变、profile 清理等优化，则原则上可以继续。

## 当前建议

- 当前全量训练优先使用每卡 24；每卡 28/30 的早期测试显示 iter time 和显存压力都变差。
- 如果目标是尽快完成全量训练验证：
  - 使用最终提交 `237487b` 后的代码。
  - 保持 `use_distortion=True` 时预期约 `16s/iter`，这是当前保留模型几何语义前提下的已验证优化结果。
  - 如果只做吞吐/链路排查，可临时 `model.use_distortion=False`，预期约 `7s/iter`，但这会改变训练几何效果，不应用作最终训练配置。
