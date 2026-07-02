# Fast-BEV Codex Project

Stable project rules for Codex sessions in this repository.

## Startup

- Canonical root: `/workspace/Fast-BEV_test_custom-fastbev-adapter`.
- Start new project sessions from this root, read `AGENTS.md` and `CODEX_HANDOFF.md`, then inspect `git status --short` before editing.
- Do not resume long historical Codex sessions unless the user explicitly asks. Treat old JSONL sessions as provenance only.
- Use normal shell commands available in the environment. Do not rely on Headroom or `rtk`; they are not project requirements.

## Communication

- 默认使用中文与用户沟通。
- 新增或修改代码注释时默认使用中文。
- 命令、路径、配置项、API 名称、错误信息、日志原文和第三方库标识保持原样。
- 不要批量翻译无关既有英文注释；只在触及相关代码或用户明确要求时调整。

## Environment

- Local default: current GPU container at `/workspace/Fast-BEV_test_custom-fastbev-adapter`.
- Local GPU: RTX 5070 Ti, CUDA 12.8, modern PyTorch. Use it for data conversion, dataset analysis, visualization, config editing, and GPU tasks that do not need the legacy MMDetection stack.
- AutoDL/Seeta: use only when training or model-side validation needs old `mmcv`/`mmdet`/`mmdet3d`. The canonical remote repo path is now `/root/Fast-BEV_test_custom-fastbev-adapter` because the system disk is preserved in saved images.
- On AutoDL/Seeta, use `/root/Fast-BEV_test_custom-fastbev-adapter` as the working directory by default. Treat historical `/root/autodl-tmp/Fast-BEV` as a data-disk copy only; do not rely on it surviving image migration.
- Keep only small smoke-test data such as nuScenes mini plus required Fast-BEV info pkl files under `/root/Fast-BEV_test_custom-fastbev-adapter/data`. For full nuScenes or other large datasets, use the AutoDL public mount and extract/copy to `/root/autodl-tmp`.

## Paths

- Migrated source archive: `/workspace/vps_migration_20260626-102832`.
- Useful old roots:
  - `/workspace/vps_migration_20260626-102832/Fast-BEV`
  - `/workspace/vps_migration_20260626-102832/Fast-BEV_autodl`
  - `/workspace/vps_migration_20260626-102832/fastbev_new_tool`
- Map historical `/root/autodl-tmp/Fast-BEV` and `/workspace/Fast-BEV_autodl` notes to the current root unless the task is explicitly a comparison or recovery task. For current AutoDL/Seeta work, translate repo paths to `/root/Fast-BEV`.

## Search And Reads

- Treat these as large files: `new_tool/unified_processor_raw.py`, `tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py`, `tools/data_converter/n7/visualize_n7_fastbev_pkl.py`, `mmdet3d/models/detectors/fastbev.py`.
- Prefer `rg -n`, `rg --files`, and bounded `sed -n` ranges before reading large files.
- Exclude `data`, `work_dirs`, `.git`, `build`, `node_modules`, and migrated archive roots from broad searches unless needed.

## N7 And Camera Rules

- N7 conversion tools should natural-sort numbered clip/frame names.
- Preserve source `image_width` and `image_height` in generated Fast-BEV metadata.
- Store camera calibration fields in generated infos once, then reuse them in dataset and visualization tools.
- Keep N7 converters and diagnostics under `tools/data_converter/` or a clear subdirectory there.
- For mono/front-camera or non-6-view experiments, update config and detector assumptions together: `n_images`, `camera_ids`, pipeline view counts, `num_views`, hard-coded `*6`, `seq_id * 6`, and TTA paths.
- `LoadAnnotations3D` lives in `mmdet3d/datasets/pipelines/loading.py`; do not invent `loading_3d.py`.

## Mono-Front Temporal Model Adaptation

- Current mono-front workstream name: Fast-BEV N7 单目前视时序模型侧适配.
- In Fast-BEV configs, `n_images=1` means one camera per time step, not one temporal frame. Current mono-front baseline remains temporal: `n_images=1`, `n_times=4`, so each sample uses `1 camera * 4 time steps = 4 images`.
- Model-side structural adaptation should keep the original temporal fusion path unless the user explicitly asks for single-frame mono. Do not reduce `neck_3d` temporal input channels just because camera count changes from 6 to 1.
- The model-side initial adaptation is considered sufficient for data-side validation when:
  - view metadata slicing uses `self.n_images` for `extrinsic`, `lidar2img_aug`, `lidar2img_extra`, and image-shape metadata;
  - voxel grid, anchor range, object range, and eval range are all aligned to the front-camera ROI;
  - train/test smoke checks pass on the legacy MMDetection environment.
- Do not tune loss weights, focal parameters, assigner IoU thresholds, or anchor sizes blindly. For N7 mono-front, first inspect the generated pkl and data distribution: camera count, temporal prev coverage, visible object counts, car/truck balance, box size distribution, ROI coverage, and anchor-to-GT max IoU distribution.
- Treat current front ROI and anchors as a runnable baseline, not as final training-optimal settings.

## Git And Verification

- The worktree may contain pre-existing dirty files. Do not revert changes you did not make.
- Show `git diff --stat` before focused diffs.
- Stage or discuss only task-relevant files.
- Before treating changes as complete, run targeted `py_compile` or smoke checks plus `git diff --check` when practical.
- Update `CODEX_HANDOFF.md` after major debugging, conversion, or validation milestones.
