# Fast-BEV Codex Project

Stable project rules for Codex sessions in this repository.

## Startup

- Canonical root: `/workspace/Fast-BEV_test_custom-fastbev-adapter`.
- Start new project sessions from this root, read `AGENTS.md` and `CODEX_HANDOFF.md`, then inspect `git status --short` before editing.
- Do not resume long historical Codex sessions unless the user explicitly asks. Treat old JSONL sessions as provenance only.
- Use normal shell commands available in the environment.

## Session And Terminal Naming

- 当前名为“总任务”的会话是项目总控会话，固定承担整体进度把控、任务拆分与派发、跨会话结果汇总、优先级调整和风险收口。
- “总任务”会话应使用 `gpt-5.6-sol` 模型和 `max` 推理强度。若实际运行环境不符合该配置，应明确告知用户，不得默认为已经满足。
- 每个新任务会话启动后，应根据该会话的实际任务内容自动概括一个简短、明确、可区分的中文任务名称；不要沿用含糊名称，也不要覆盖“总任务”这个保留名称。
- 将概括出的任务名称同时用于会话 rename 和该会话所在终端的标题；如果任务主线发生实质变化，应同步更新两者。
- 会话名称与终端标题应保持一致，并优先采用“对象 + 动作/目标”的形式，例如“force_resize 几何验证”“S0 AUG1 训练”“板端 INT8 对齐”。
- 执行 rename 或终端命名时，使用当前客户端和终端提供的原生能力。若当前环境没有相应接口，应立即向用户说明限制并给出建议名称，不得声称已经完成重命名。

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
- 所有提交（包括代码、配置、工具、测试和文档）统一采用“内网优先”流程：先把精确候选范围同步到内网 canonical repo，在要求的 legacy/真实数据环境完成验证，审查最终 diff，并取得用户明确确认后先完成内网提交；再把内网验证并提交的最终内容回同步到当前本地工作区，复核文件 hash/diff 和本地可运行门禁，最后才允许本地提交。
- 内网与本地历史不同时，不要求 commit SHA 相同，但必须保持提交范围、提交信息和文件内容一致，并用文件 SHA256 或 patch diff 核对。若当前会话无法访问内网仓库或缺少内网验证结果，必须停在 `git add`/`git commit` 之前，明确报告阻塞；不得用本地现代环境测试替代 legacy/真实数据门禁，也不得先做本地提交。
- 内网同步和内网提交默认排除 Codex/Claude 会话型 Markdown 与本地助手配置，例如 `CODEX_HANDOFF.md`、`CODEX_PROMPTS.md`、`CLAUDE.md` 和 `.claude/`；这些文件只在本地按需维护，不得为了内网代码提交一并复制或暂存。项目正式实验文档不因扩展名为 `.md` 自动排除，是否同步由用户逐任务确认。
- All commits must use the user's GitHub identity `liujiaren19 <1334282612@qq.com>` for both author and committer. Never create a commit as `Codex`, `codex@openai.local`, or another assistant identity.
- Before every commit, show the exact file scope and commit message to the user and wait for explicit approval. Then verify `git var GIT_AUTHOR_IDENT` and `git var GIT_COMMITTER_IDENT` before running `git commit`.
- Before treating changes as complete, run targeted `py_compile` or smoke checks plus `git diff --check` when practical.
- Update `CODEX_HANDOFF.md` after major debugging, conversion, or validation milestones.
