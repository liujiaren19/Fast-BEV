# Fast-BEV Codex Handoff

Compressed state for new Codex sessions. Read with `AGENTS.md`; do not load raw JSONL history unless a specific missing detail requires a narrow search.

## Snapshot

- Date: 2026-07-31
- Root: `/workspace/Fast-BEV_test_custom-fastbev-adapter`
- Branch: `feature/n7-mono-s0-optimization-v1`
- HEAD: `134d5f3124e0d11f40dbcbade9e5bfe07c28ff96`，已与 `origin/feature/n7-mono-s0-optimization-v1` 同步。
- Main workstream: N7 Fast-BEV 6V→单目前视四时序 B0→原生单帧 S0→1600×900 GEOM1-A 的训练、评估和板端闭环。
- Current product decision: 板端近期使用原生单帧；四时序 B0 保留为 pose-aware 离线对照/教师候选，不把重复当前帧作为最终产品方案。
- Local container: RTX 5070 Ti, CUDA 12.8, `torch 2.7.1+cu128`.
- Legacy training/model validation should run on the latest AutoDL/Seeta environment after path verification.
- Current remote canonical repo path: `/root/Fast-BEV_test_custom-fastbev-adapter`. Use this as the working directory on future AutoDL/Seeta servers because it lives on the system disk and is preserved in saved images.
- Treat `/root/autodl-tmp` as data disk space for large extracted datasets and transient outputs. It is not image-safe. Only use `/root/autodl-tmp/Fast-BEV` as a temporary data-disk mirror if explicitly needed.
- Keep nuScenes mini plus required Fast-BEV smoke-test info pkl files under `/root/Fast-BEV_test_custom-fastbev-adapter/data/nuscenes` when preparing a reusable image. Full nuScenes should stay on the public mount or be extracted to `/root/autodl-tmp`.

## Current Task State

- Canonical experiment notes: `N7_FASTBEV_EXPERIMENT_SUMMARY.md` for the short view, `N7_FASTBEV_EXPERIMENT_DETAILS.md` for evidence, and `N7_MONO_FRONT_SINGLE_FRAME_TODO.md` for priorities.
- `force_resize` closeout is complete. Commit `134d5f3` contains the two-stage
  geometry/augmentation contract, legacy fixture and tests; the internal
  epoch13 T7 pre/post comparison passed bit-exact for input, 2D feature, BEV
  input, raw logits and decoded boxes at `atol=0, rtol=0`.
- EXP-MONO-GEOM1-A migration/gates, 4×L20 15-epoch training and all epoch
  evaluations are complete. Canonical best is epoch11: mAP `0.381993`,
  BEV@0.5 `0.4016`, mATE `0.8615m`, mAOE `4.6376°`, mASE `0.2113`.
  Relative to S0 e13, mAP/BEV and medium/far Recall improve materially, while
  overall and per-class AOE regress. Keep e11 as the next fine-tune candidate,
  e13 as the GEOM orientation/scale challenger and e15 as terminal; do not run
  EXT5. S0 e13 remains the production fallback until targeted regression.
- Next model-optimization task is real-GT yaw/curve coverage. Prepare N7 city
  intersection/turning data (`DATA1-CITY-YAW`) and N7 banked-ring data
  (`DATA3-BANKED-RING`) with source/yaw/range tags. Run image augmentation
  (`AUG1`) separately after selecting the data winner. Proving-ground
  endurance-road BEVFusion labels (`DATA2-ENDURANCE-PSEUDO`) are lower-trust
  pseudo labels and remain behind real-GT work.
- EXP-6V-B0 is the final full 6V four-temporal val baseline and must remain separate from the old trainability-only EXP-6V-00. The synchronized log proves epoch1～17 complete/save, while `eval_summary.md` covers epoch1～16. The user excludes epoch17 and later from formal selection, so the final roles are epoch6 canonical best (`mAP=0.453768`, BEV@0.5=`0.4787`), epoch2 40～60m Recall challenger and epoch16 terminal.
- EXP-6V-B0 missed its epoch6 best for 10 evaluated epochs (epoch7～16), satisfying patience=5. The run is closed: do not evaluate epoch17 for selection and do not run epoch18～20. Any post-`force_resize` rerun is a separate P3 low-priority experiment, not a reopening of B0.
- Only the training log and `eval_summary.md` are synchronized locally. The user confirms PTH/pkl assets are preserved internally; their local size/SHA256 remain unavailable but are not a blocker for the finalized decision. The user also confirms the 20260717 6V gate failure is caused by incomplete clip start/end boundaries and accepts that exception (`PASS_WITH_ACCEPTED_CLIP_BOUNDARY_EXCEPTION`). Resolved test still reuses val, so this remains a val baseline rather than independent test.
- The EXP-6V-B0 training log resolves `AdamW2`, LR `8e-4`, weight decay `0.01`, backbone `lr_mult=0.1`, global batch 96, poly/linear-warmup and dynamic fp16. At analysis start the repository dist config only overrode LR and would inherit standard Adam; it was externally changed during analysis to `_delete_=True` AdamW2 plus the run's data/schedule. This documentation closeout did not modify that config. Treat the log as the run-of-record and the current uncommitted config only as a later alignment, not the original training snapshot.
- B0 is mono-front temporal: `cam0`, `n_images=1`, `n_times=4`, pose-aware history, 1024→256 temporal fuse. It completed 15 epochs; canonical mAP best is epoch5 `0.344934`, BEV mAP@0.5 `0.3562`.
- B0 loss correction: epoch15 `positive_bag_loss=0.6496`, `negative_bag_loss=0.1433`, total `loss=0.7928`. Earlier notes that called 0.6496 total loss were wrong; the checkpoint decision remains unchanged because AP did not refresh.
- S0 is native single-frame: `cam0`, `n_images=1`, `n_times=1`, `sequential=False`, no history pose, 256→256 fuse. All other first-round variables stay aligned with B0.
- S0 completed 15 epochs and is frozen at epoch13: canonical mAP `0.356425`, BEV mAP@0.5 `0.3773`, mATE `0.8588m`, mAOE `4.0545°`, mASE `0.2132`. Keep epoch10 as the medium/far-distance challenger and epoch15 only as terminal archive; final PTH/result/config/data/calibration hashes still need archival.
- Same-epoch comparison: S0/B0 epoch5 canonical mAP is `0.3518/0.344934`; final-best comparison is S0 e13 `0.356425` vs B0 e5 `0.344934`. This is a same-ROI mono comparison; do not directly compare either value with EXP-6V-B0 because camera coverage, ROI, GT visibility/filtering and eval sets differ.
- Historical no-GT production visualization already favored native single-frame over B0 repeated-current-frame inference. S0 e13 has since passed dataset/production PTH alignment on the N7 golden sample and five fixed samples; Torch-CUDA fixed LUT, FP ONNX functional-chain comparison and the independent CPU board-reference path are complete on PC, with 38 focused regressions passing. Real chip/runtime/INT8 and independent physical-calibration validation remain open.
- Existing B0 pkl gate is a deliberate `FAIL`, not a crash: train/val token and clip overlap are zero, but legacy converter stats contain `labels_without_frame=851/90`; train also dropped 1237 empty-GT labels, and there is no independent test pkl.
- Box-origin publicization and re-evaluation are complete for canonical dataset eval/visualization/production JSON+pkl; the vendor fixed-6V `tools/utils.py` path is still not canonical mono postprocess. Exact corrected z metrics/result hash still need archival.
- GitHub code baseline commits are `2d8ab7d9855f787d55da8521b1993c445351037d`, `c3ec682e78e273fa6142e13825f0cde4857d7883` and current `134d5f3124e0d11f40dbcbade9e5bfe07c28ff96`; the current branch matches origin.
- Low-priority model-training backlog: the `force_resize` prerequisite for
  EXP-6V-B1 is now satisfied, but the rerun remains P3. If explicitly
  scheduled, create a new experiment/work_dir and rerun the original `8e-4`
  as a single-variable control; test `4e-4` only if the post-e5～e7 decline
  repeats. It does not reopen EXP-6V-B0 or outrank GEOM1-A, independent test,
  physical calibration or real chip/runtime/INT8 work.

## Mono-Front Model-Side Status

- Scope: two baselines now coexist. B0 is temporal mono (`n_images=1,n_times=4`); S0 is native single-frame (`n_images=1,n_times=1`) and is the near-term product base.
- Structural model adaptation is in place for first-pass N7 data validation:
  - `mmdet3d/models/detectors/fastbev.py` slices per-view metadata with `self.n_images` instead of assuming six cameras.
  - The slicing covers `extrinsic`, `lidar2img_aug`, `lidar2img_extra`, `img_shape`, `ori_shape`, `pad_shape`, `img_info`, and `filename` when they are list-valued.
  - Sequence feature splitting uses `self.n_images`, so `n_images=1, n_times=4` becomes four single-camera temporal chunks.
  - The 3D voxel grid is still generated by Fast-BEV's existing `get_points(n_voxels, voxel_size, origin)` path; the adaptation is through front-camera `point_cloud_range`, `n_voxels`, and `voxel_size`, not by replacing the backprojection algorithm.
  - Current front baseline uses `point_cloud_range=[0, -35, -5, 80, 35, 3]`, `n_voxels=[160, 140, 4]`, `voxel_size=[0.5, 0.5, 1.5]`, and anchor range `[[0, -35, -1.8, 80, 35, -1.8]]`.
  - This creates an 80m x 70m front ROI at the original 0.5m x/y BEV resolution and removes the rear half of the original full-surround grid.
- Validation already completed on the 2080 Ti legacy stack with nuScenes mini from `/root/Fast-BEV`:
  - dataset/dataloader produced `img_shape=torch.Size([4, 3, 256, 704])`;
  - one GPU training iteration completed with loss `4.0493`;
  - one test forward completed and wrote `test_results.pkl`.
- Current conclusion: both B0 and S0 model/train paths have completed real N7 long-running training. Remaining uncertainty is deployment/data/projection closure and final S0 asset archival, not basic structural trainability.
- Do not tune loss or anchors before N7 distribution analysis. The next optimization decisions should be based on car/truck counts, visible target counts per frame, x/y/z and l/w/h distributions, anchor-to-GT max IoU, and early `positive_bag_loss` / `negative_bag_loss` curves.

## N7 Front-Camera Notes

- Immediate target remains original `cam0`, mounted at about 1.874 m.
- Extra front-facing cameras exist for later experiments:
  - `cam0_l`, about 1.68 m, filename suffix `_8`
  - `cam_m`, about 1.755 m, filename suffix `_10`
  - `cam_h`, about 1.9 m, filename suffix `_11`
- These suffixes are file suffixes, not existing six-view camera ids. Do not map them to `cam8`, `cam10`, or `cam11`.
- Future support should add explicit logical ids such as `cam0_l`, `cam_m`, `cam_h` and map them to suffixes/calibration entries.

## Current Worktree Scope

This repository has pre-existing tracked and untracked changes from multiple workstreams. The 2026-07-29 documentation synchronization owns only:

```text
CODEX_HANDOFF.md
N7_FASTBEV_EXPERIMENT_DETAILS.md
N7_FASTBEV_EXPERIMENT_SUMMARY.md
N7_MONO_FRONT_SINGLE_FRAME_TODO.md
```

- `AGENTS.md` already contains the user-requested session/terminal naming policy and is not part of this documentation synchronization.
- The untracked GEOM1-A configs, diagnostics and tests documented below are
  pre-existing output from a separate local-preparation workstream; this
  documentation synchronization does not claim or stage them.
- No source/config/test file was edited, no file was deleted, and no commit was created by this documentation synchronization.
- Other modified/deleted source files and all untracked paths shown by `git status --short` are pre-existing or belong to other workstreams; do not revert, delete, stage or submit them without explicit scope confirmation.
- `tools/utils.py` remains a chip-vendor legacy fixed-6V postprocess path, not the canonical mono board postprocess. `.claude/`, `CLAUDE.md`, and `scripts/` are local workflow aids.
- Do not add generated `mmdet3d.egg-info/` or cache changes to commits.

## Suggested Next Action

1. Keep EXP-6V-B0 frozen at epoch6 best, epoch2 challenger and epoch16 terminal; do not evaluate epoch17 for selection or run later epochs.
2. Freeze GEOM e11/e13/e15 and archive checkpoint/result/config/data/calibration hashes; no EXT5.
3. Compare S0 e13 and GEOM e11/e13 on a real-GT production set, add yaw-angle bins and unmatched/looser-match depth-error statistics.
4. Prepare `DATA1-CITY-YAW` and `DATA3-BANKED-RING` in parallel; prioritize real GT and prevent scene leakage. A product-oriented combined real-GT fine-tune is acceptable after source-specific QA, but keep source tags and do not mix AUG1/pseudo labels in that first run.
5. Run `AUG1` separately on the selected real-data winner or GEOM e11, measuring yaw bins, far recall and production slices.
6. Evaluate `DATA2-ENDURANCE-PSEUDO` only after teacher/hash/threshold/manual-QA and sampling-weight contracts are frozen; validation/test stay real-GT.
7. Independent test, final asset hashes, physical calibration and real chip/runtime/INT8 validation remain open parallel work. EXP-6V-B1 stays P3.

## Data-Cleaning Conversation Prompt

Use this prompt when starting a new Codex CLI conversation dedicated to N7 data cleaning and point-cloud-assisted GT visualization:

```text
我们现在从 /workspace/Fast-BEV_test_custom-fastbev-adapter 开始一个新任务，只做 N7 数据清洗/数据处理，不改当前 mono-front 训练模型逻辑。请先阅读 AGENTS.md 和 CODEX_HANDOFF.md，执行 git status --short，确认当前分支和已有改动，不要回退任何已有改动。

项目背景：当前 Fast-BEV N7 单目前视时序训练已经适配到 cam0，关键配置是 configs/fastbev/custom/custom_fastbev_mono_front_r18.py 和 custom_fastbev_mono_front_r18_dist_train.py。mono-front 使用 n_images=1、n_times=4，前视 ROI 为 [0, -35, -5, 80, 35, 3]。训练/评估 GT 已按 cam0 前视几何可见和 ROI 统一过滤，但这个过滤还没有判断真实遮挡。现在想从 N7 原始点云入手，先做离线可视化和统计，辅助后续决定是否清洗 GT 或增加可见性字段。

本轮目标：实现或扩展一个 N7 点云辅助 GT 可视化/诊断工具。优先放在 tools/data_converter/n7/ 下，不要破坏现有 pkl，不要直接改训练/eval 过滤逻辑。第一版只做诊断输出：读取 Fast-BEV N7 pkl、读取对应原始点云、读取 cam0 图片和 pkl 标定，输出每帧可视化图和每个 GT 的统计 CSV/JSON。

第一版功能要求：
1. 支持命令行输入 --pkl、--data-root 或 --pointcloud-root、--output-dir、--camera-id cam0、--frame-indices/--max-frames、--bev-range 0 -35 80 35。
2. BEV 可视化叠加：原始点云 XY、raw GT、当前 mono-front used GT、filtered GT、前视 ROI、距离刻度；颜色区分 car/truck 和 used/filtered。
3. 前视图可视化叠加：cam0 图片、投影 GT 3D 框、可选投影点云深度点；线条不要太粗，单目前视图需要足够大。
4. 每个 GT 输出诊断字段：token、class、box、center distance、是否在 ROI、是否 cam0 几何可见、box 内点数、点云深度/高度范围、投影 2D bbox 面积。遮挡先不要做复杂结论，最多给出可解释的候选指标。
5. 第一版不要删除 GT，只输出 used/filtered/point-count 等诊断结果，供人工看图和后续 A/B。
6. 使用 pkl 中已有 calibration 字段：cam_intrinsic、sensor2lidar_rotation/translation、intrinsic_width/height、image_width/height、distortion 等。704x256 缓存图要避免重复缩放 K。
7. 复用现有 tools/data_converter/n7/visualize_n7_fastbev_pkl.py 中已有的投影、GT 过滤和绘图逻辑，能复用就复用，不要重新写一套容易不一致的几何。
8. 不要使用旧的 fastbev_ljr.py、fastbev_bst.py、m2bevnet_seq.py 等硬编码 6V 的路径。

交付要求：给出实现文件、示例命令、输出文件说明；至少运行 python -m py_compile 和 --help。如果内网点云数据当前不可访问，就先把 CLI 和解析逻辑写好，并用 pkl/合成点云做最小 smoke test。
```

## Historical Provenance

- Main historical Codex session: `/root/.codex/sessions/2026/06/15/rollout-2026-06-15T07-16-09-019eca23-436e-7613-ae85-bb11aed2f79e.jsonl`
- Earlier setup sessions:
  - `/root/.codex/sessions/2026/06/12/rollout-2026-06-12T02-51-14-019eb9bd-a5dd-7251-9ce9-a19277db623a.jsonl`
  - `/root/.codex/sessions/2026/06/11/rollout-2026-06-11T00-54-09-019eb42c-18ed-7933-a850-fdcc06d2602d.jsonl`
- Search these files narrowly with `rg`, `tail`, or bounded `sed -n` only when needed.

## Maintenance

Append a short dated note here after major debugging, conversion, or validation milestones:

- What changed.
- Which files changed.
- Which command verified it.
- What remains uncertain.

### 2026-06-27 Mono-Front Model-Side Pass

- What changed: Added shared FastBEV view-metadata slicing so per-sequence and TTA paths slice `extrinsic`, `lidar2img_aug`, and `lidar2img_extra` consistently by `self.n_images`; updated the N7 mono-front config to use a front-only ROI/voxel grid/anchor range and front-camera visible object filtering.
- Which files changed: `mmdet3d/models/detectors/fastbev.py`, `configs/fastbev/custom/custom_fastbev_mono_front_r18.py`, and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile mmdet3d/models/detectors/fastbev.py configs/fastbev/custom/custom_fastbev_mono_front_r18.py configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_1iter.py configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_debug.py`; `python -m py_compile mmdet3d/datasets/pipelines/multi_view.py mmdet3d/datasets/custom_multiview_dataset.py mmdet3d/datasets/nuscenes_dataset.py`; `git diff --check`.
- What remains uncertain: The local RTX 5070 Ti container lacks `mmcv`/`mmdet`, so real config build and forward/training smoke validation still need the legacy AutoDL/MMDetection environment.

### 2026-06-27 2080 Ti Mono-Front Smoke Validation

- What changed: Synced the current working tree to a new 2080 Ti AutoDL/Seeta container, extracted nuScenes mini from the public mount, generated Fast-BEV 4D mini info files, and prepared a system-disk copy at `/root/Fast-BEV` so saved images retain the repo and editable `mmdet3d` install.
- Which files changed: No local source change besides this handoff note. Remote-only generated artifacts included `data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl`, `data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl`, and `work_dirs/round1_nuscenes_mini_front_mono_1iter_2080ti_smoke/`.
- Which command verified it: On the 2080 Ti server, `python tools/train.py configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_1iter.py --work-dir work_dirs/round1_nuscenes_mini_front_mono_1iter_2080ti_smoke --no-validate --seed 0 --deterministic` completed one GPU training iteration with loss `4.0493`; `python tools/test.py configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_1iter.py work_dirs/round1_nuscenes_mini_front_mono_1iter_2080ti_smoke/epoch_1.pth --out work_dirs/round1_nuscenes_mini_front_mono_1iter_2080ti_smoke/test_results.pkl` completed one test forward.
- What remains uncertain: This validates the nuScenes mini mono-front model path, not the N7 pkl data path. The image-safe repo/install lives under `/root/Fast-BEV`; future remote work should use that path as the repo root.

### 2026-06-27 System-Disk Remote Root

- What changed: Promoted `/root/Fast-BEV` to the remote working root and copied nuScenes mini smoke-test data into that system-disk repo so it survives image migration. The copied data includes `maps/`, `samples/`, `sweeps/`, `v1.0-mini/`, `nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl`, `nuscenes_infos_train_4d_interval3_max60.pkl`, and `nuscenes_infos_val_4d_interval3_max60.pkl`; it excludes GT database and COCO export files.
- Which files changed: `AGENTS.md` and `CODEX_HANDOFF.md` locally and on the 2080 Ti server. Remote data now lives under `/root/Fast-BEV/data/nuscenes`; `/root/autodl-tmp/Fast-BEV` was removed to avoid relying on the non-image-safe data disk.
- Which command verified it: From `/root/Fast-BEV` with `PYTHONPATH` unset, dataset build produced `img_shape torch.Size([4, 3, 256, 704])`; `python tools/train.py configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_1iter.py --work-dir work_dirs/root_nuscenes_mini_front_mono_1iter_smoke --no-validate --seed 0 --deterministic` completed one GPU iteration with loss `4.0493`; `python tools/test.py configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_1iter.py work_dirs/root_nuscenes_mini_front_mono_1iter_smoke/epoch_1.pth --out work_dirs/root_nuscenes_mini_front_mono_1iter_smoke/test_results.pkl` completed one test forward.
- What remains uncertain: Future full nuScenes work still needs the public mount and extraction/copy to `/root/autodl-tmp`; only mini smoke-test data is image-safe under `/root/Fast-BEV`.

### 2026-06-29 N7 6V Adapter Cleanup

- What changed: On `test/custom-fastbev-adapter`, removed temporary test/model profiling hooks and removed old pkl size-field compatibility from the N7 data path. `RandomAugImageMultiViewImage`, `CustomMultiViewDataset`, and N7 geometry helpers now require explicit `intrinsic_width/height` and `image_width/height` fields from the current converter output.
- Which files changed: `mmdet3d/apis/test.py`, `tools/test.py`, `tools/test_n7_6v_eval.sh`, `mmdet3d/models/detectors/fastbev.py`, `mmdet3d/datasets/custom_multiview_dataset.py`, `mmdet3d/datasets/pipelines/transforms_3d.py`, N7 custom configs, and N7 converter/geometry/visualization helpers.
- Which command verified it: `python -m py_compile mmdet3d/apis/test.py tools/test.py mmdet3d/models/detectors/fastbev.py mmdet3d/datasets/custom_multiview_dataset.py mmdet3d/datasets/pipelines/transforms_3d.py tools/data_converter/n7/fastbev_geometry.py tools/data_converter/n7/visualize_n7_fastbev_pkl.py tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py configs/fastbev/custom/custom_fastbev_6v_r18.py configs/fastbev/custom/custom_fastbev_6v_r18_dist_train_ljr.py configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256.py`; `git diff --check`.
- What remains uncertain: Runtime train/eval was not rerun in the legacy MMDetection environment after cleanup; run a short N7 dataset/test smoke before final internal submission.

### 2026-06-29 FastBEV 4D Head Input Alignment

- What changed: After comparing the old 8-card `fastbev_ljr.py` training log/code path with the current 4-card full-data run, changed `FastBEV.extract_feat` for `style in ['v1', 'v2']` to fold each temporal BEV volume to `[bs, z*c, x, y]` before temporal concatenation. Normal training now feeds the 3D neck with the same 4D seq-major tensor order used by `export_3d`; `deploy_head_inputs` are only materialized for `test_onnx`, `test_custom`, or calibration-data export.
- Which files changed: `mmdet3d/models/detectors/fastbev.py` and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile mmdet3d/models/detectors/fastbev.py mmdet3d/models/necks/m2bev_neck.py`; `git diff --check`.
- What remains uncertain: The performance fix still needs an actual 4-card full-data rerun in the legacy training environment to confirm whether iteration time returns near the expected range.

### 2026-06-29 FastBEV Stage Profiling Pass

- What changed: Replaced `FastBEV._slice_view_meta` deep-copy slicing with shallow top-level/lidar2img dict copies, and added temporary `FASTBEV_STAGE_PROFILE` instrumentation for `extract_feat` and `forward_train`. The profile prints rank-0 stage timings for backbone/FPN/fuse, metadata slicing, projection, point cache, backprojection, stack/cat, 3D neck, bbox head, and bbox loss.
- Which files changed: `mmdet3d/models/detectors/fastbev.py` and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile mmdet3d/models/detectors/fastbev.py`; `git diff --check`.
- What remains uncertain: The new shallow-copy optimization and profile output still need to be tested on the 4-card N7 training environment. Because the profile synchronizes CUDA at each stage, use it only for a few iterations.

### 2026-06-29 FastBEV Backproject Profiling And Scatter Fusion

- What changed: Added `FASTBEV_BACKPROJECT_PROFILE` instrumentation inside `backproject_inplace` to split flatten, project, valid-mask, allocation, scatter, and view costs and print per-camera valid counts. The no-distortion path now computes `R @ xyz + t` directly instead of constructing `[x, y, z, 1]` homogeneous points for every call. Feature gathering now defaults to fused final-camera scatter: it computes the final camera that would win under the existing sequential overwrite order, then gathers/writes each assigned BEV point once. Set `FASTBEV_BACKPROJECT_FUSED=0` to fall back to the previous per-camera scatter path for A/B.
- Which files changed: `mmdet3d/models/detectors/fastbev.py` and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile mmdet3d/models/detectors/fastbev.py`; `git diff --check`.
- What remains uncertain: Need a short 4-card profile run with `FASTBEV_STAGE_PROFILE=1 FASTBEV_BACKPROJECT_PROFILE=1` to verify that fused scatter is numerically accepted by the training path and whether it reduces `backproject` time. For regression A/B, rerun once with `FASTBEV_BACKPROJECT_FUSED=0`.

### 2026-06-29 Batched Distortion Projection

- What changed: Added a batched `_project_points_with_distortion` path that stacks per-camera `rot/tran/intrin/post_rot/post_tran/distortion` and computes all camera distortion projections with batched torch operations. The previous per-camera loop is kept as `_project_points_with_distortion_loop` and is used as a fallback if any camera metadata or distortion field is missing.
- Which files changed: `mmdet3d/models/detectors/fastbev.py` and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile mmdet3d/models/detectors/fastbev.py`; `git diff --check`; an isolated torch equivalence script comparing the old per-camera formula with the new batched formula reported `max_diff 0.0`.
- What remains uncertain: Needs a short 4-card profile run with `model.use_distortion=True` to confirm the `FASTBEV_BACKPROJECT project` component drops in the real N7 training environment.

### 2026-06-29 N7 6V Training-Time Conclusion And Cleanup

- What changed: Consolidated the temporary timing investigation into clean training code. Removed `FASTBEV_STAGE_PROFILE`, `FASTBEV_BACKPROJECT_PROFILE`, their env toggles, counters, and log printing from `mmdet3d/models/detectors/fastbev.py`. Also removed the fused-scatter A/B env switch so normal training directly uses fused final-camera scatter. Reverted `_slice_view_meta` to the original `copy.deepcopy` path because shallow-copy slicing was measured as negligible. Kept only the effective/required model-path changes: 4D seq-major 3D-head input alignment, direct no-distortion `R @ xyz + t`, fused scatter, and batched dynamic distortion projection.
- Which files changed: `mmdet3d/models/detectors/fastbev.py` and `CODEX_HANDOFF.md`.
- Timing facts to preserve:
  - User confirmed the old `fastbev_ljr.py` 8-card run used `samples_per_gpu=24` and `fix_lut=False`; do not explain the old speed by assuming the current migrated file's `fix_lut=True`.
  - Current 4-card N7 full-data run uses 24 BEV samples per GPU; each sample has `6 views * 4 times`, so one GPU performs 96 backprojects per iteration.
  - With profile off and `use_distortion=False`, fused scatter stabilized around `6.8-7.2s/iter`, `data_time≈0.3s`, memory about `26810 MB`.
  - With profile off and old dynamic-distortion path, `use_distortion=True` stabilized around `20-22s/iter`.
  - With batched distortion plus fused scatter, profile-off `use_distortion=True` stabilized around `16s/iter`, memory about `26818 MB`; this is an improvement but still slower than no-distortion because dynamic distortion projection remains real compute.
  - Profile logs showed fused scatter itself was reduced to roughly `0.008-0.017s/backproject`; remaining cost was mostly distortion projection.
- Interpretation to preserve: The remaining mismatch versus old `fastbev_ljr.py` is not fully explained. A plausible geometry-chain difference is that current mainline applies the newer pkl `post_rot/post_tran` path for `intrinsic_width/height -> input_size`, while old `get_points3d_from_pinhole_camera` used its hand-written `stride_intr @ intrinsic` path. Do not change to the old formula for speed unless a same-metadata A/B proves projected points and valid masks are equivalent.
- Which command verified it: `python -m py_compile mmdet3d/models/detectors/fastbev.py`; `git diff --check`.
- What remains uncertain: To explain old/new runtime exactly, run a future same-environment comparison that computes old `fastbev_ljr.py` dynamic projection and current batched projection on the same `img_meta/points`, logging projection time, point max difference, and valid-mask difference. For training continuation, if an epoch-4 checkpoint was produced before the 4D seq-major head-input alignment, restart training from the intended pretrained/init checkpoint instead of resuming epoch 4 because the 3D neck/head saw a different channel ordering.

### 2026-06-30 N7 Timing Ablation Record And Rollback

- What changed: Updated `N7_FASTBEV_TIMING_ABLATION_20260630.md` with the full timing-ablation record from this round, including the final same-machine `1600x900` geometry profile comparison between current `fastbev.py` and old `fastbev_ljr.py`. The record now separates observed facts from non-conclusions: current per-camera `valid_sum≈333k`, ljr `valid_sum≈255k`, while unique covered BEV points are close (`≈159.7k` vs `≈159.3k`). This proves geometry/mask differences but does not by itself explain a 4x wall-clock gap.
- Which files changed: Documentation only after rollback: `N7_FASTBEV_TIMING_ABLATION_20260630.md` and `CODEX_HANDOFF.md`. Temporary profiler edits in `mmdet3d/models/detectors/fastbev.py` and `mmdet3d/models/detectors/fastbev_ljr.py` were reverted to commit `237487b`.
- Which command verified it: `git restore --source=HEAD -- mmdet3d/models/detectors/fastbev.py mmdet3d/models/detectors/fastbev_ljr.py`; `rg -n "FASTBEV_GEOM_PROFILE|FASTBEV_STAGE_PROFILE|FASTBEV_BACKPROJECT_PROFILE" mmdet3d/models/detectors/fastbev.py mmdet3d/models/detectors/fastbev_ljr.py` returned no matches; `python -m py_compile mmdet3d/models/detectors/fastbev.py mmdet3d/models/detectors/fastbev_ljr.py`; `git diff --check`.
- What remains uncertain: The exact old/new runtime gap is still not fully explained. Future work should avoid changing training geometry for speed without a synchronized function-level A/B on the same `img_meta`, `points`, and feature shape. If investigating further, test 5-parameter distortion interpretation and `post_rot/post_tran` versus old `stride_intr @ intrinsic` as separate variables.

### 2026-07-01 N7 6V Config/Calibration Review

- What changed: Reviewed the N7 6V trainable path before internal submission. `custom_fastbev_6v_r18.py` remains the 1600x900/paper-style image augmentation config. `custom_fastbev_6v_r18_n7_704x256.py` is the 704x256 offline-cache config and uses `force_resize=True`; this is a geometry-correct baseline for 1600x900 K + 704x256 images but intentionally disables image-level random resize/crop/flip/rotate inside `RandomAugImageMultiViewImage`. BEV-level `RandomFlip3D` and `GlobalRotScaleTrans` remain active. Treat this as a runnable baseline, not a final augmentation-optimal policy.
- What changed: `tools/generate_calibrate_data.py` is now a calibration-only script. It forces `samples_per_gpu=1`, disables format/eval/output collection, uses `test_pth`, saves backbone image inputs and 3D head BEV inputs, and stops after `--max-calibration-samples`. Do not mix the vendor `test_bst.py` interval-based calibration flag logic into `mmdet3d/apis/test.py`.
- What changed: Added code comments in `mmdet3d/models/detectors/fastbev.py` and `mmdet3d/models/necks/m2bev_neck.py` explaining the current v1/R18 board path, 4D head-input alignment, calibration data assumptions, and the difference from the hard-coded `fastbev_bst.py`/`m2bev_neck_bst.py` demo path.
- What changed: Aligned the board ONNX export path with the vendor demo where it matters for deployment: `onnx_export_2d` now uses nearest resize for FPN feature resizing, and `onnx_export_3d` returns raw bbox-head logits instead of applying sigmoid inside the ONNX graph. This keeps board outputs compatible with postprocess code that performs sigmoid/topk/NMS outside the exported model. Normal train/test pth behavior still uses `feature_resize_mode` for feature fusion.
- What changed: Deliberately did not merge the old `fastbev_ljr.py` fixed-LUT debug/export path that saves `x.npy`, `y.npy`, `valid.npy`, `projection.npy`, `features.npy`, and `volume.npy` from `backproject_inplace`. That code is useful for fixed-LUT A/B and board index-table generation, but it is deployment tooling rather than the N7 6V trainable baseline. It also contains hard-coded output paths, camera order, and `fix_lut` behavior, so it should be reintroduced later as an explicit offline export tool or guarded debug mode, not hidden inside normal train/test forward.
- Which files changed: `configs/fastbev/custom/custom_fastbev_6v_r18.py`, `configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256.py`, `configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256_dist_train.py`, `tools/generate_calibrate_data.py`, `mmdet3d/models/detectors/fastbev.py`, `mmdet3d/models/necks/m2bev_neck.py`, and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile tools/generate_calibrate_data.py mmdet3d/models/detectors/fastbev.py mmdet3d/models/necks/m2bev_neck.py`; `git diff --check`.
- What remains uncertain: Whether to submit the `CBGSDataset(disable_cbgs=...)` source change is still a review decision. Current custom N7 configs use `data = dict(_delete_=True, ...)` and do not need CBGS to be disabled through the wrapper. Fixed-LUT index generation remains a future board-deployment task after the trainable baseline is committed.

### 2026-07-02 N7 Fixed-LUT Export And Visualization Tools

- What changed: Added `tools/data_converter/n7/build_fastbev_lut.py` as an offline fixed-LUT/index export tool. It reads a Fast-BEV config plus N7 pkl, reconstructs the same test-mode camera metadata path used by `CustomMultiViewDataset` and `RandomAugImageMultiViewImage`, then exports `x.npy`, `y.npy`, `valid.npy`, `projection.npy`, `points.npy`, dense/per-camera gather-scatter indices, fused final-camera indices, optional deterministic `features.npy`/`volume.npy`, and board-compatible `LUT/gather_new_i.bin`, `LUT/scatter_nd_new_i.bin`, `LUT/featurePointLength.bin`. This keeps deployment LUT generation out of `mmdet3d/models/detectors/fastbev.py`.
- What changed: Added `tools/data_converter/n7/visualize_fastbev_lut.py` to visualize LUT correctness. It renders per-Z BEV camera-choice maps, per-camera valid masks, board-bin-vs-npy diff maps, per-camera feature-plane hit maps from bin gather indices, and reconstructs `volume.npy` from board bin when debug `features.npy` exists.
- Which files changed: `tools/data_converter/n7/build_fastbev_lut.py`, `tools/data_converter/n7/visualize_fastbev_lut.py`, and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile tools/data_converter/n7/build_fastbev_lut.py tools/data_converter/n7/visualize_fastbev_lut.py`; `python tools/data_converter/n7/build_fastbev_lut.py --help`; `python tools/data_converter/n7/visualize_fastbev_lut.py --help`; a synthetic 2-camera pkl/config smoke run with `--dump-debug-volume` followed by visualization. Smoke summary reported `bin_vs_npy_choice_mismatch=0`, `bin_vs_npy_gather_mismatch=0`, and `bin_volume_nonzero_diff=0`; `git diff --check`.
- What remains uncertain: The script follows the current mainline overwrite rule by default: natural camera order, later valid camera wins. To reproduce the old `fastbev_ljr.py` debug order, pass `--camera-overwrite-order 1,2,0,4,5,3`. Real board integration still needs one actual N7 pkl export and a PC-vs-board gather/scatter comparison.

### 2026-07-02 N7 6V Internal Submission Commit Scope

- What changed: Prepared the local submission commit for the N7 6V trainable baseline. The scope covers two explicit N7 6V configs: `custom_fastbev_6v_r18.py` for raw `1600x900` images and `custom_fastbev_6v_r18_n7_704x256.py` for offline `704x256` image cache with `force_resize=True`. The distributed training config is renamed to `custom_fastbev_6v_r18_n7_704x256_dist_train.py` and now inherits the offline-cache config directly.
- What changed: The model path keeps the chip-vendor v1/R18 deployment assumptions that matter for training/export consistency: dynamic `n_images` metadata slicing, seq-major 4D BEV head input layout, nearest resize for board-aligned feature export, raw head logits in 3D ONNX export, calibration data saving hooks, and comments explaining the difference from the hard-coded vendor demo path.
- What changed: Tooling is split by purpose. `tools/generate_calibrate_data.py` is calibration-only and forces `samples_per_gpu=1`; fixed-LUT board index generation is implemented as offline scripts under `tools/data_converter/n7/`, not hidden in normal train/test forward.
- What changed: This commit intentionally does not stage local/reference/debug files such as `.claude/`, `CLAUDE.md`, `mmdet3d/apis/test_bst.py`, `mmdet3d/models/necks/m2bev_neck_bst.py`, `tools/utils.py`, `tools/vis.py`, and `tools/dist_train_ljr_debug.sh`. They are useful as reference or local workflow files but should not be part of the clean N7 6V trainable baseline unless a later review decides otherwise.
- Which files changed: N7 custom configs, FastBEV detector/neck code, calibration/LUT tools, N7 train/test shell entrypoints, `N7_FASTBEV_TIMING_ABLATION_20260630.md`, and `CODEX_HANDOFF.md`.
- Which command verified it: Before commit, run targeted `py_compile` on the changed Python/config files plus `git diff --check` and `git diff --cached --check`.
- What remains uncertain: The current baseline is known to train for at least the user's two-epoch internal run, but full 20-epoch results, final CBGS policy, and board-side LUT integration comparison are still follow-up validation items.

### 2026-07-02 Merge Mono-Front Work Into N7 6V Baseline

- What changed: Created `feature/n7-mono-front-from-6v-baseline` from the committed N7 6V trainable baseline `8cc8678`. The merge strategy is current-repo-first: do not overlay `/workspace/Fast-bEV`, and do not merge its older `fastbev.py`. Only carry forward mono-front specific config and converter diagnostics.
- What changed: Merged `/workspace/Fast-bEV` mono-front config intent into `configs/fastbev/custom/custom_fastbev_mono_front_r18.py`: front ROI/voxel grid, front anchor range, `FrontCameraVisibleObjectFilter`, and dated pkl file names under `ann_dir`. Kept the current repo's `feature_resize_mode='nearest'` and current N7 6V baseline inheritance.
- What changed: Added `summarize_output_infos()` back to `tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py`, so generated mono or 6V pkl files record camera-count, camera-id, and temporal-prev coverage in `metadata['output_summary']` and the sidecar summary JSON.
- What changed: Kept current `mmdet3d/models/detectors/fastbev.py` unchanged because it already has the newer dynamic `self.n_images` metadata slicing, 4D head-input layout, calibration hooks, ONNX export alignment, N7 distortion path, and comments. The older `/workspace/Fast-bEV` detector diff is a subset/older version and should not overwrite the current baseline.
- Which files changed: `configs/fastbev/custom/custom_fastbev_mono_front_r18.py`, `tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py`, and `CODEX_HANDOFF.md`.
- Which command verified it: Run targeted `py_compile` on the edited config/converter and `git diff --check`.
- What remains uncertain: The mono-front custom config still needs an actual N7 mono pkl generated from the target camera id and a dataset/dataloader smoke check in the legacy MMDetection environment before training.

### 2026-07-02 Conversation Closeout Notes

- Feature resize mode: current N7 custom configs explicitly set `model.feature_resize_mode='nearest'`. In normal train/test pth, `FastBEV._resize_feature()` uses this config when resizing FPN `c2/c3/c4` to `c1` before `neck_fuse`; this affects 2D feature fusion values, not camera calibration or BEV projection geometry. `bilinear` is smoother and may be slightly better for pure model quality, but `nearest` is simpler, easier to quantize/deploy, and matches the chip-vendor demo path.
- Deployment caution: `onnx_export_2d()` currently hard-codes `nearest` for 2D ONNX export. Therefore training with `bilinear` while using the current exporter would create a train/export mismatch. If a future experiment wants `bilinear`, update both training config and `onnx_export_2d()` to bilinear, then compare PyTorch feature, ONNXRuntime feature, and board feature. If the board compiler implements the same bilinear semantics, train/export/board can be consistent; the mismatch is not inherent to bilinear, it comes from unsupported or changed export/board operator semantics.
- Old repository directory: `/workspace/Fast-BEV` and `/workspace/Fast-bEV` are the same old directory and are no longer the main work root. The useful mono-front pieces have been migrated into the current repo. The old directory is about 3.2G, mostly `vendor/mmcv-full-1.4.0`, `build/`, and `work_dirs_docker/`. Keep it only temporarily until the current mono branch changes are reviewed/committed; then it can be deleted or at least cleaned of those large build/vendor artifacts.
- Next conversation entrypoint: start from `/workspace/Fast-BEV_test_custom-fastbev-adapter` on `feature/n7-mono-front-from-6v-baseline`. First read `AGENTS.md` and this handoff, inspect `git status --short`, then review the three current modified files before generating or validating mono pkl. Do not resume old `/workspace/Fast-BEV` work or copy its `fastbev.py` over this branch.
- Verification already run for the mono merge: `python -m py_compile configs/fastbev/custom/custom_fastbev_mono_front_r18.py tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py mmdet3d/datasets/pipelines/multi_view.py`; `git diff --check`.

### 2026-07-03 Mono-Front Eval GT Visibility Filter

- What changed: Added dataset-level GT visibility filtering controlled by `filter_gt_visible_camera`, so train annotations and N7 custom eval/test GT parsing can use the same front-camera visible target口径 without destructively deleting GT in the converter pkl. The mono-front config enables `filter_gt_visible_camera='cam0'` and `filter_gt_visible_min_depth=0.1`.
- What changed: Updated `tools/data_converter/n7/visualize_n7_fastbev_pkl.py` for mono-front inspection. Single-camera views now default to `camera_width=1280`, box projection line/text thickness scales with image size, and `--gt-view-mode split|used` can show which raw pkl GT are used by the mono-front train/eval口径 versus filtered out.
- What changed: `FrontCameraVisibleObjectFilter` remains in `mmdet3d/datasets/pipelines/multi_view.py`; make sure `mmdet3d/datasets/pipelines/__init__.py` imports it from `multi_view` and lists it in `__all__`. Missing this import causes `FrontCameraVisibleObjectFilter is not in the pipeline registry` during config build.
- Which files changed: `mmdet3d/datasets/custom_multiview_dataset.py`, `mmdet3d/datasets/pipelines/multi_view.py`, `configs/fastbev/custom/custom_fastbev_mono_front_r18.py`, `tools/data_converter/n7/visualize_n7_fastbev_pkl.py`, and `CODEX_HANDOFF.md`. If an internal branch is missing the registry import, also sync `mmdet3d/datasets/pipelines/__init__.py`.
- Which command verified it: `python -m py_compile mmdet3d/datasets/custom_multiview_dataset.py mmdet3d/datasets/pipelines/multi_view.py mmdet3d/datasets/pipelines/__init__.py configs/fastbev/custom/custom_fastbev_mono_front_r18.py tools/data_converter/n7/visualize_n7_fastbev_pkl.py`; `git diff --check`.
- What remains uncertain: Needs a real N7 mono pkl dataset/eval smoke in the legacy MMDetection environment to confirm filtered GT counts and metric口径 match expectation.

### 2026-07-06 Mono-Front Full-Val Eval Speedup

- What changed: Optimized `CustomMultiViewDataset` N7 custom eval for large full-val runs without changing the default metric口径. `_eval_single_class()` now scans sorted predictions once for all thresholds of one metric, reuses prediction-to-GT match values across thresholds, vectorizes center-distance matching, and caches GT BEV polygon/AABB geometry for BEV IoU. BEV IoU also uses exact AABB prefiltering before polygon clipping.
- What changed: Added optional `eval_max_dets_per_sample` to cap low-score detections per frame after `eval_score_thr` and ROI filtering. Default is `None`, so exact full evaluation remains unchanged. For fast online training eval, consider setting `eval_distance_thr=[]` if only BEV AP is needed, and optionally `eval_score_thr=0.1` or `eval_max_dets_per_sample=100/200`; keep final reported metrics on the exact口径.
- Which files changed: `mmdet3d/datasets/custom_multiview_dataset.py` and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile mmdet3d/datasets/custom_multiview_dataset.py`; `git diff --check -- mmdet3d/datasets/custom_multiview_dataset.py`.
- What remains uncertain: Needs timing on the internal full-val result set. The main remaining cost can still be millions of low-score detections; exact eval will be faster but may still be too slow for frequent in-training validation unless fast-eval caps/thresholds are enabled.

### 2026-07-06 Mono-Front Epoch10 Collapse Investigation

- Observed: User's 4-card mono-front run produced reasonable epoch5 in-training val metrics (`car bev_AP@0.5≈0.1905`, `mAP≈0.0974`) but explicit epoch10/latest eval was nearly zero (`mAP/bev_iou@0.5≈1.44e-06`) with millions of detections and almost no useful predictions above score 0.2.
- Diagnosis: `configs/fastbev/custom/custom_fastbev_mono_front_r18.py` still inherited the six-view training augmentation `flip_ratio_bev_vertical=0.5`. In MMDetection3D LiDAR boxes, BEV vertical flip flips the x axis. For front-only ROI `x=[0,80]`, this sends valid front GT to `x<0`; the following `ObjectRangeFilter` then removes or corrupts front-only training targets. This is a strong candidate for unstable training and the epoch10 collapse.
- What changed: Set mono-front `flip_ratio_bev_vertical=0.0` and kept `flip_ratio_bev_horizontal=0.5`. Do not resume treating the existing epoch10/latest checkpoint as a good model. Re-test explicit `epoch_5.pth` under the same current eval command to verify whether epoch5 still reproduces the earlier AP, then restart a clean training run with vertical flip disabled.
- What changed: Updated `configs/fastbev/custom/custom_fastbev_mono_front_r18_dist_train.py` back to a clean new-run schedule: `optimizer.lr=8e-4`, `total_epochs=15`, and `evaluation.interval=5`. This is not random-init from scratch: N7 runs intentionally inherit/use the COCO 2D pretrained `load_from='pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'`.
- Which files changed: `configs/fastbev/custom/custom_fastbev_mono_front_r18.py`, `configs/fastbev/custom/custom_fastbev_mono_front_r18_dist_train.py`, and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile configs/fastbev/custom/custom_fastbev_mono_front_r18.py`; `git diff --check`.
- What remains uncertain: If current-code eval of explicit `epoch_5.pth` is also near zero, then the earlier epoch5 metric came from a different config/code path or checkpoint mismatch and needs separate inspection of `latest.pth` symlink, checkpoint `meta.epoch`, and the exact test command.

### 2026-07-06 Mono-Front 6V Assumption Re-Audit

- Rechecked current mono-front train/eval path for 6V leftovers. The active config builds `model.type='FastBEV'`; `FastBEV.extract_feat()` uses `self.n_images` for temporal feature splits and metadata slicing, and `aug_test()` slices by total views per TTA rather than `24 * tta_id`.
- Rechecked data path. `CustomMultiViewDataset` uses `camera_types=['cam0']`, sequential `n_times=4`, and the same raw front-camera visible GT filter for train/eval parsing. `MultiViewPipeline` uses `n_images=1`; `FrontCameraVisibleObjectFilter` filters only the first time step (`cam0` current frame), which is the intended key-frame GT口径.
- Remaining hard-coded 6V paths are old/reference/deployment files such as `mmdet3d/models/detectors/fastbev_ljr.py`, `fastbev_bst.py`, `m2bevnet_seq.py`, and old paper configs. Do not use those files for mono-front training or evaluation without a separate view-count audit.
- `RandomAugImageMultiViewImage` has a debug-only `% 6` camera label block under `is_debug`; normal train/test does not enter it.
- Local container lacks `mmcv`, so final merged-config construction still needs an internal environment check with `Config.fromfile(...)`. Static audit and `py_compile` passed locally.

### 2026-07-06 N7 Pointcloud-Assisted GT Diagnostics

- What changed: Added a standalone offline diagnostic script `tools/data_converter/n7/n7_pointcloud_gt_diagnostics.py`. It reads Fast-BEV N7 pkl, cam0 images, and optional raw pointcloud files; reuses existing `visualize_n7_fastbev_pkl.py` geometry/filter helpers for calibration scaling, GT used/filtered口径, projection, and BEV drawing; and outputs per-frame BEV/cam0/combined images plus per-GT CSV/JSON diagnostics. The script does not modify pkl files and does not touch train/eval filtering logic.
- What changed: The CLI supports `--pkl`, `--data-root`, `--pointcloud-root`, `--pointcloud-glob`, `--output-dir`, `--camera-id cam0`, `--frame-indices`, `--max-frames`, `--bev-range 0 -35 80 35`, class/ROI overrides, raw-vs-FastBEV pointcloud coordinate selection, `.pkl/.pcd/.bin/.npy/.npz/.txt/.csv` pointcloud input, optional projected depth points, and `--no-render` stats-only mode. Internal N7 pointclouds such as `.../<sequence>/parsed_data/<clip>/lidar_main/<timestamp>.pkl` can be read directly; if the local root differs from pkl `dataset/sequence` fields, pass `--pointcloud-glob '{sequence}/parsed_data/{clip}/lidar_main/{timestamp}.pkl'`.
- Output files: `gt_diagnostics.csv`, `gt_diagnostics.json`, `summary.json`, and `frames/<dataset>/<sequence>/<clip>/*_{bev,cam0,combined}.jpg|png`. Per-GT fields include frame/GT token, class, box, center distance, ROI flag, cam0 geometry visibility, used/filtered reason, box point count, point depth/height ranges, camera-depth range, and projected 2D bbox area.
- Which command verified it: `python -m py_compile tools/data_converter/n7/n7_pointcloud_gt_diagnostics.py`; `python tools/data_converter/n7/n7_pointcloud_gt_diagnostics.py --help`; a synthetic one-frame smoke test under `/tmp/n7_pc_diag_smoke` with a generated pkl, cam0 image, `.npy` pointcloud, and dict-style `.pkl` pointcloud wrote 2 GT rows and BEV/cam0/combined PNGs; `git diff --check`.
- What remains uncertain: Real N7 pointcloud directory layout and pointcloud coordinate convention need validation on the internal network data. If default path guesses do not find pointcloud files, pass `--pointcloud-glob` with the actual internal pattern and set `--pointcloud-coordinate raw` or `fastbev` explicitly after checking the source coordinate.

### 2026-07-07 Fast-BEV ONNX Export And 2D-to-3D LUT Tooling

- What changed: Reworked `tools/export_onnx.py` from a test-script derivative into a dedicated pth-to-ONNX export tool. It no longer builds dataset/dataloader for export, infers input size, `n_images`, `n_times`, `n_voxels`, and 3D head BEV input shape from the config, writes `export_metadata.json`, and can export 2D, 3D, or both. It checks the current board assumptions before export: v1 style, 4 temporal inputs, `feature_resize_mode='nearest'`, and 3D head concat/fuse channel consistency.
- What changed: `tools/export_onnx.py` preserves PC `test_onnx` compatibility by defaulting 2D output layout to NCHW. If the chip compiler requires the deployment NHWC output of `FastBEV.onnx_export_2d()`, pass `--2d-output-layout nhwc`. In NCHW mode the script explicitly unsets `DEPLOY` during 2D export to avoid shell environment leakage. 3D export still returns raw bbox-head logits; sigmoid/topk/NMS remain outside the ONNX graph.
- What changed: Replaced `tools/export_2d_to_3d_model.py` with a parameterized fixed-LUT ONNX exporter. It reads `build_fastbev_lut.py` outputs, preferring board-compatible `LUT/gather_new_i.bin`, `LUT/scatter_nd_new_i.bin`, and `LUT/featurePointLength.bin`; it also supports `LUT_arr`, dense final-camera arrays, and legacy `x/y/valid.npy` fallback. It exports a single-sample 2D feature to 3D BEV gather/scatter ONNX with NCHW or NHWC input layout and outputs `[1, channels * z, x, y]`.
- Which files changed: `tools/export_onnx.py`, `tools/export_2d_to_3d_model.py`, and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile tools/export_onnx.py tools/export_2d_to_3d_model.py`; `python tools/export_onnx.py --help`; `python tools/export_2d_to_3d_model.py --help`; `python tools/export_2d_to_3d_model.py --lut-dir /tmp/fastbev_2d_to_3d_lut_smoke --out /tmp/fastbev_2d_to_3d_lut_smoke/2d_to_3d.onnx --no-simplify --verify`; `python tools/export_2d_to_3d_model.py --lut-dir /tmp/fastbev_2d_to_3d_lut_smoke --out /tmp/fastbev_2d_to_3d_lut_smoke/2d_to_3d_nhwc.onnx --input-layout nhwc --no-simplify --verify`; both synthetic 2D-to-3D ORT checks reported `verify_max_abs_diff=0.0`; `git diff --check -- tools/export_onnx.py tools/export_2d_to_3d_model.py`.
- What remains uncertain: Local container still lacks the legacy `mmcv`/`mmdet` stack, so real pth load, 2D/3D ONNX export from a Fast-BEV checkpoint, and end-to-end `test_onnx`/board-simulation comparison must run in the internal legacy environment with the real checkpoint, pkl/LUT, and N7 data. `onnxsim` is not installed locally, so local 2D-to-3D smoke used `--no-simplify`; internal export can run simplify if the package is available.

### 2026-07-07 Mono-Front Epoch Checkpoint Eval Tracker

- What changed: Added standalone `tools/eval_epoch_checkpoints.py` for the current 4-card-train / single-card-eval workflow. It scans `epoch_*.pth`, calls `tools/test.py` to write `epoch_N_val_results.pkl`, then reuses that pkl to compute `dataset.evaluate()` metrics and per-class score distributions without changing training code.
- Output files: per epoch `epoch_N_val_results.pkl`, `epoch_N_test.log`, and `epoch_N_metrics.json`; rolling summaries `eval_summary.csv`, `eval_summary.md`, and `eval_summary.json` under `<work_dir>/test_results` by default. The summary marks the best checkpoint by `mAP` and includes car/truck AP, GT/det counts, and p50/p90/p99/score>=0.2 counts.
- Suggested use on the single-GPU eval machine: `CUDA_VISIBLE_DEVICES=0 python tools/eval_epoch_checkpoints.py configs/fastbev/custom/custom_fastbev_mono_front_r18.py work_dirs/<run_name> --watch --poll-interval 600 --stable-seconds 120`. To evaluate only existing result pkl files, add `--skip-inference --rerun-eval`.
- Which command verified it: `python -m py_compile tools/eval_epoch_checkpoints.py`; `python tools/eval_epoch_checkpoints.py --help`.
- What remains uncertain: Local container lacks the legacy `mmcv`/`mmdet` stack, so real inference/eval execution must be validated on the internal evaluation machine with the synced work_dir and N7 pkl data.

### 2026-07-07 Mono-Front Adaptation Review Brief

- What changed: Added `N7_MONO_FRONT_ADAPTATION_REVIEW.md` as a single review entrypoint for Claude Code / Codex review. It summarizes the N7 6V baseline, 704x256 cache path, efficiency and board-side export optimizations, mono-front adaptation, current training metrics, suspected epoch5 collapse cause, and an audit prompt.
- Which files changed: `N7_MONO_FRONT_ADAPTATION_REVIEW.md` and `CODEX_HANDOFF.md`.
- Which command verified it: `git diff --check -- N7_MONO_FRONT_ADAPTATION_REVIEW.md CODEX_HANDOFF.md`.
- What remains uncertain: The review brief records current analysis only; code-level findings still need an independent audit of active train/eval/export paths.

### 2026-07-08 Mono-Front Review Fixes

- What changed: Made the inherited COCO 2D pretrained initialization explicit in both N7 distributed train configs. `configs/fastbev/custom/custom_fastbev_mono_front_r18_dist_train.py` and `configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256_dist_train.py` now both set `load_from='pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'` with comments, matching the intended N7 training setup.
- What changed: Fixed `CustomMultiViewDataset._box_bev_corners()` yaw convention to match `LiDARInstance3DBoxes.corners` / `rotation_3d_in_axis(axis=2)`. BEV AP values after this fix are not strictly comparable with earlier epoch logs because the BEV IoU polygon orientation was mirrored before this correction.
- What changed: Added a train-side defensive `label >= 0` filter in `CustomMultiViewDataset.get_ann_info()` before the visibility mask, mirroring eval-side `_parse_gt_info()` and preventing unknown classes from reaching `FreeAnchor3DHead`. Also added a comment in `_select_adjacent()` clarifying that `max_interval` is unused and `min_interval` only applies when explicit `adj_ids` are absent; behavior was not changed.
- Follow-up: After N7 distribution analysis, revisit `test_cfg` scale-NMS settings. With 2 classes, only the first two entries from the 10-class nuScenes-tuned lists are used; `truck nms_rescale_factor=0.7` is inherited from nuScenes and may not be optimal for N7. Also revisit the four inherited nuScenes anchor sizes against the N7 car/truck size distribution.
- Which files changed: `configs/fastbev/custom/custom_fastbev_mono_front_r18_dist_train.py`, `configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256_dist_train.py`, `mmdet3d/datasets/custom_multiview_dataset.py`, and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile configs/fastbev/custom/custom_fastbev_mono_front_r18_dist_train.py configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256_dist_train.py mmdet3d/datasets/custom_multiview_dataset.py`; yaw convention check script comparing old/fixed BEV corners against the `rotation_3d_in_axis(axis=2)` convention; `git diff --check`.
- What remains uncertain: The local workspace does not contain the internal mono-front N7 pkl, so the current `gt_names` set could not be re-read locally. The converter is expected to filter to configured classes, and the new train-side filter is defensive.

### 2026-07-08 Mono-Front Production Image Inference Entry

- What changed: Added `tools/infer_mono_front_image.py`, a standalone production-style mono-front inference script. It reads N7-format `info.json`, selects `front_wide`, builds a temporary `cam0` Fast-BEV sample, repeats a single image to satisfy temporal mono `n_times=4` by default, and supports PTH checkpoint inference or the current split ONNX convention (`2D backbone ONNX + 3D head ONNX` with project-local `bbox_head.get_bboxes` postprocess). It writes `infos.pkl`, `pred_results.pkl`, `prediction_summary.json`, and optional `frames/*.jpg|png` visualization using the existing N7 visualizer.
- What changed: The script intentionally follows the converter/dataset camera contract: `to_lidar_main` is treated as raw N7 lidar axes by default and converted with `RAW_TO_FASTBEV`; use `--info-extrinsic-coordinate fastbev` only if an input `info.json` already stores camera-to-lidar in Fast-BEV/MMDet3D axes. `--intrinsic-size WIDTH HEIGHT` declares the image coordinate size of `cam_intrinsic`, while actual loaded image dimensions are stored as `image_width/image_height`.
- What changed: Fixed a JSON syntax typo in `data/info_json/2025_04_18_2k_byd_info_9797_UKEF.json` (`-3.u072...` -> `-3.072...`) so it can be parsed. Its `front_wide` K is effectively `1600x900` while the sensor `width/height` fields still say `3840x2160`; production commands should pass `--intrinsic-size 1600 900` explicitly for clarity. Auto inference can also detect `1600x900` from the principal point.
- Calibration guidance: Runtime inference should consume a converted N7-format `info.json` whose K and `to_lidar_main` are already in the contract expected by the converter/dataset. Keep original vehicle calibration JSON as the source of truth for regenerating that `info.json`, but do not feed raw `camera_0.json` directly into Fast-BEV because its coordinate origin/axes and manually modified Euler fields differ; if regeneration is needed, reuse the recovery logic in `new_tool/CoordinateTransformer.py`.
- Which files changed: `tools/infer_mono_front_image.py`, `data/info_json/2025_04_18_2k_byd_info_9797_UKEF.json`, and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile tools/infer_mono_front_image.py`; `python tools/infer_mono_front_image.py --help`; `python -m json.tool data/info_json/2025_04_18_2k_byd_info_9797_UKEF.json`; function-level smoke checks parsed both baseline and 9797 `front_wide` calibration and confirmed 9797 auto/CLI intrinsic size resolves to `1600x900`; `git diff --check -- tools/infer_mono_front_image.py data/info_json/2025_04_18_2k_byd_info_9797_UKEF.json`.
- What remains uncertain: Local container lacks the legacy `mmcv`/`mmdet` runtime, so real model build/checkpoint load/ONNXRuntime inference still need to run in the internal legacy Fast-BEV environment with the actual mono-front checkpoint or exported ONNX pair and production images.

### 2026-07-08 Mono-Front Production Inference Review Fixes

- What changed: Updated `tools/infer_mono_front_image.py` after review. The temporary sample now injects `box_type_3d`/`box_mode_3d` from `get_box_type('LiDAR')` after deferred `mmdet3d` imports, matching `Custom3DDataset.pre_pipeline()` so `Anchor3DHead.get_bboxes_single()` can read `input_meta['box_type_3d']`.
- What changed: Added ONNX 2D output-layout validation when reading `export_metadata.json`: `nhwc` now raises because this script assumes the NCHW `test_onnx` convention, while missing/unknown layout or glob fallback logs a warning and continues as NCHW.
- What changed: `pred_results.pkl` is now written as numpy-native prediction dicts (`boxes_3d`, `scores_3d`, `labels_3d`) so it can be unpickled without `mmdet3d/mmcv`; in-process visualization still uses the original model result.
- What changed: Tightened `auto_info` intrinsic-size selection with a principal-point consistency guard. Sensor width/height are accepted only when `abs(2*cx-width)/width <= 0.15` and the same check passes for height; otherwise the existing principal-point inference path and warning are used.
- What changed: PTH CPU inference now skips `wrap_fp16_model` with a warning because `mmcv auto_fp16` would cast `img` to fp16 and CPU half conv is unsupported. `--temporal-images` help text and output metadata now state that real history frames are used without ego-motion compensation or pose input; repeat fallback matches training clip-start samples.
- Calibration clarification: Earlier production-inference wording should not be read as saying 9797_UKEF sensor `width/height` are `2560x1440`. The actual `data/info_json/2025_04_18_2k_byd_info_9797_UKEF.json` stores `front_wide` sensor fields as `3840x2160`; its K principal point resolves to `1600x900` under auto inference, and explicit `--intrinsic-size 1600 900` remains recommended for production commands.
- Which files changed: `tools/infer_mono_front_image.py` and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile tools/infer_mono_front_image.py`; `python tools/infer_mono_front_image.py --help`; `git diff --check`; `git diff --no-index --check /dev/null tools/infer_mono_front_image.py` produced no whitespace-error output (expected nonzero exit because no-index compares `/dev/null` with a new file).
- What remains uncertain: Real end-to-end PTH/ONNX runs still require the internal legacy environment with `mmcv/mmdet/mmdet3d`, the actual mono-front checkpoint or split ONNX pair, and production images.

### 2026-07-08 6V Front-Image Black-Fill Inference Tool

- What changed: Added `tools/infer_6v_front_image_blackfill.py` as a diagnostic script for loading a 6V PTH checkpoint while feeding a real image only to `cam0/front_wide`; the other configured 6V cameras are filled with generated black images. It keeps the 6V camera order from `config.data.test.camera_types` by default and supports directory, list, or repeated `--image` inputs.
- What changed: The script builds N7/Fast-BEV sample metadata for all six cameras from `info.json`, including per-sensor K size inference, raw-N7-to-FastBEV extrinsic conversion, `box_type_3d/box_mode_3d`, and portable numpy-native `pred_results.pkl`. It repeats the same 6V black-fill set for all temporal steps and records that no ego-motion compensation or pose input is used.
- Suggested use: `python tools/infer_6v_front_image_blackfill.py --config configs/fastbev/custom/custom_fastbev_6v_r18.py --checkpoint work_dirs/6v/epoch_20.pth --info-json data/info_json/2025_04_18_2k_byd_info.json --image-dir /path/to/front_images --image-glob '*.jpg' --output-dir outputs/6v_blackfill_front`. If the info K is original 4K, pass `--intrinsic-size 3840 2160`; if it is converted 1600x900, pass `--intrinsic-size 1600 900` or rely on auto principal-point inference when the JSON is consistent.
- Which files changed: `tools/infer_6v_front_image_blackfill.py` and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile tools/infer_6v_front_image_blackfill.py`; `python tools/infer_6v_front_image_blackfill.py --help`; `git diff --check`; `git diff --no-index --check /dev/null tools/infer_6v_front_image_blackfill.py` produced no whitespace-error output (expected nonzero exit because no-index compares `/dev/null` with a new file).
- What remains uncertain: Real 6V checkpoint loading and inference still require the internal legacy `mmcv/mmdet/mmdet3d` runtime. This black-fill path is intentionally a diagnostic hack, not a geometry-complete production substitute for real 6V images.

### 2026-07-08 Mono-Front Inference Visualization/Sampling Update

- What changed: Updated `tools/infer_mono_front_image.py` so visualization files default to the original input image filename (`--visualization-name image`); `--visualization-name token` keeps the previous token-based naming and uses `--image-ext`.
- What changed: Mono-front visualization now exposes `--display-aspect 16:9|native` and defaults to `16:9` for the camera panel. `--no-bev` can skip the BEV panel for faster render and a more image-focused output.
- What changed: Added directory/list sampling controls for faster runs: `--start-index`, `--stride`, and `--max-frames` select which images are actually inferred. Added `--visualization-stride` and `--max-visualizations` to reduce render output while still running inference on the selected frames.
- What changed: Replaced the inference wrapper's `torch.no_grad()` with `torch.inference_mode()` when available for a small PyTorch inference overhead reduction.
- Which files changed: `tools/infer_mono_front_image.py` and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile tools/infer_mono_front_image.py`; `python tools/infer_mono_front_image.py --help`; `git diff --check`; `git diff --no-index --check /dev/null tools/infer_mono_front_image.py` produced no whitespace-error output (expected nonzero exit because no-index compares `/dev/null` with a new file).
- What remains uncertain: Real PTH/ONNX throughput still needs measurement in the internal legacy runtime with actual checkpoint/export files. The largest speedups available from this script side are fewer frames (`--stride/--max-frames`), fewer visualizations (`--visualization-stride/--max-visualizations`), lower `--camera-width`, `--raw-distorted`, and `--no-bev`.

### 2026-07-08 Mono-Front Multi-Vehicle Directory Inference

- What changed: Extended `tools/infer_mono_front_image.py` with a vehicle batch mode. Passing `--vehicle-id <id...>` now auto-resolves calibration from `--info-json-dir` plus `--info-json-template` and recursively finds images under `--image-root/<vehicle-id>` using `--vehicle-image-glob`.
- Default path convention: `--info-json-dir data/info_json/2k`, `--info-json-template '2025_04_18_2k_byd_info_{vehicle_id}.json'`, `--image-root data/qingping/2k`, and image globs `**/*.jpg **/*.jpeg **/*.png`. The template also accepts `{vehicle}` as an alias.
- Output layout: Each vehicle writes to `--output-dir/<vehicle-id>/` with its own `infos.pkl`, portable `pred_results.pkl`, `prediction_summary.json`, and `frames/`. In `--visualization-name image` mode, frame outputs preserve the path relative to that vehicle directory, so multiple clips can contain the same filename without overwriting each other. The root output directory also gets `batch_summary.json`.
- Suggested use: `python tools/infer_mono_front_image.py --checkpoint work_dirs/mono_front/epoch_5.pth --vehicle-id 9797_UKEF 1234_ABC --info-json-dir data/info_json/2k --image-root data/qingping/2k --intrinsic-size 1600 900 --stride 5 --max-frames 200 --visualization-stride 5 --output-dir outputs/mono_front_batch`.
- Which files changed: `tools/infer_mono_front_image.py` and `CODEX_HANDOFF.md`.
- Which command verified it: `python -m py_compile tools/infer_mono_front_image.py`; `python tools/infer_mono_front_image.py --help`; `git diff --check`.
- What remains uncertain: Real batch inference still needs the internal legacy runtime and actual checkpoint/export files. The local workspace does not contain `data/qingping/2k`, so image discovery was verified statically via CLI/help rather than by reading the real directory tree.

### 2026-07-13 Native Single-Frame Comparison Preparation

- Product decision: The near-term board path is native mono-front single-frame inference. Six-axis IMU and wheel speed are available, but the pose interface is not ready; the four-frame temporal model remains the offline comparison/reference and is not emulated in production by repeating the current frame as the final solution.
- Experiment contract: Added `configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18.py` and `custom_fastbev_mono_front_single_frame_r18_dist_train.py`. The only main structural variable versus temporal B0 is `n_times=1`: `n_images=1`, dataset/pipeline `sequential=False`, no adjacent IDs or temporal compensation, and 3D fusion changes from temporal `1024->256` to native single-frame `256->256`. ROI, voxel, anchor, distortion, 704x256 input, GT filtering, per-GPU batch 64, AdamW2 LR `1e-4`, and COCO 2D initialization remain unchanged. Both comparison configs now use a 15-epoch upper bound. The four-card server only trains and saves every epoch: the config uses `evaluation.interval=999999`, and `tools/dist_train_ljr.sh` accepts `NO_VALIDATE=1` to skip even constructing the validation workflow. A separate one-card server runs `tools/eval_epoch_checkpoints.py --watch`, inspects epoch5 first, and stops training manually if `mAP/center_dist` best is not refreshed for five consecutive evaluated epochs.
- Box-origin fix: `CustomMultiViewDataset` now converts prediction `LiDARInstance3DBoxes` to `gravity_center` before eval-range filtering and xyz error statistics; explicit portable numpy arrays may declare `box_origin=center|bottom`. `tools/infer_mono_front_image.py` writes center-origin portable boxes and metadata, and the N7 visualizer converts object/bottom-origin predictions to center z before drawing. Training GT construction remains unchanged. Existing `epoch_N_val_results.pkl` should be re-evaluated internally without rerunning inference; expected behavior is unchanged BEV/xy AP and corrected car/truck z signed means.
- ONNX fix: `FastBEV2DExportWrapper` and `FastBEV3DExportWrapper` call `model.onnx_export_2d/3d` directly, avoiding the eager-reference fall-through into normal `forward_test`. Board validation formally accepts `n_times=1` and `n_times=4`; 3D inputs use stable names `bev_0...bev_N` because numeric-only names were renamed by PyTorch ONNX and broke ORT feeds. Single-frame metadata records `temporal_order=['key']`, `channel_layout='ZC'`, input/output names, shapes and dtypes, raw-logit status, and config/checkpoint SHA256. Real checkpoint export, decoded-box and sigmoid-once checks remain internal-runtime work.
- Eval visualization: Added `--video-group clip|sequence`, sequence timestamp sorting, `--max-frames-per-video`, ordered multi-worker rendering, and explicit canvas-size mismatch errors. Video files use their group names instead of the fixed `visualization.mp4`: clip mode writes `<clip>/<clip>.mp4`, and sequence mode writes `<sequence>/<sequence>.mp4`. `--camera-size WIDTH HEIGHT` controls only each camera panel, so `--camera-size 1600 900` keeps the front-view portion at exactly 1600x900 while the optional top header and BEV remain outside that size. `--no-header` remains optional but is not required for the requested panel sizing. `--video-only` still avoids frame-file explosion. A local synthetic cross-clip smoke produced one video per sequence with the expected frame counts.
- Planning: `N7_MONO_FRONT_SINGLE_FRAME_TODO.md` is the prioritized source of truth. After the first S0-vs-B0 comparison, but before formal single-frame training, the next P0 items are pkl data gates, production-vs-standard-dataset same-image tensor/logit alignment, and distortion-aware GT visibility consistency. The current model backprojection uses distortion while `FrontCameraVisibleObjectFilter` and eval visibility still use pinhole projection; this mismatch is explicitly not considered resolved.
- Local verification: targeted synthetic checks passed for box object `gravity_center`, explicit bottom arrays, center↔bottom round trip, production portable metadata, direct ONNX wrapper branches, `n_times=1` board-spec validation, config/checkpoint SHA256, fully resolved single-frame config invariants, sequence video grouping/limits, video-only behavior, and exact 1600x900 canvas. A minimal native-single-frame model also completed real `torch.onnx.export`, ONNX checker and 2D/3D ONNXRuntime comparison with max absolute diff `0.0`; this test exposed and verified the `bev_0` input-name fix. `python -m py_compile` passed for the touched Python/config files and `git diff --check` passed.
- What remains uncertain: This workspace has no N7 data and no legacy `mmcv/mmdet/mmdet3d` runtime. Internal validation must build the merged config, inspect `[B,1,3,256,704]` dataloader and `[B,256,160,140]` 3D-neck inputs, run one train/test smoke, export a real single-frame checkpoint to ONNX/ORT, directly re-evaluate old result pkl files, and render the temporal epoch5 real-data sequence videos.

### 2026-07-13 Win10 Ozone Eval Visualization Transfer

- What changed: Added `tools/data_converter/n7/pull_n7_eval_images_from_ozone.py` for Windows/Linux. It reads the eval GT/info pkl, applies the same start/stride/max-frame selection as the N7 visualizer, writes an exact rclone `--files-from` manifest, downloads only current-frame camera images, preserves pkl-relative paths, resizes Ozone source images to pkl `image_width/image_height`, skips verified existing files, and writes a JSON transfer summary. It intentionally does not download temporal `prev` frames because the target is offline visualization rather than Windows model inference.
- What changed: Added `tools/data_converter/n7/export_n7_eval_predictions_portable.py` for the legacy eval server. It converts `LiDARInstance3DBoxes`/torch prediction results into center-origin numpy-only results so Windows visualization does not need the old MMDetection3D/CUDA stack.
- Video diagnosis: `visualize_n7_fastbev_pkl.py` currently writes `mp4v`. The local OpenCV build can open `mp4v` but cannot open `avc1`/`H264`, and this container has no standalone `ffmpeg`/`ffprobe`; therefore simply changing the fourcc is not a reliable fix. Remote VS Code playback should install ffmpeg on the Remote-SSH host and transcode to H.264/yuv420p or use a remote-aware extension with an ffmpeg compatibility cache.
- Which command verified it: `python -m py_compile tools/data_converter/n7/pull_n7_eval_images_from_ozone.py tools/data_converter/n7/export_n7_eval_predictions_portable.py`; both `--help` commands; a synthetic eval pkl selection/manifest smoke; a synthetic 1600x900-to-704x256 download-resize/verify smoke; a bottom-origin-to-center portable prediction conversion smoke; `git diff --check` on the two new scripts.
- What remains uncertain: This workspace has no rclone/Ozone configuration and no real internal eval pkl, so the actual Ozone transfer must be tested on the Win10 intranet host. The pkl `data_path` must be relative to the supplied Ozone root; older absolute-path pkl files need `--path-prefix`.

### 2026-07-13 N7 Experiment Notes Consolidation

- What changed: Added `N7_FASTBEV_EXPERIMENT_SUMMARY.md` and `N7_FASTBEV_EXPERIMENT_DETAILS.md` as the unified experiment record from the first N7 6V run through the current temporal-mono B0 and pending native-single-frame S0. The notes assign stable experiment IDs, distinguish raw-log evidence from historical/inferred records, list resolved training settings, pkl/work/checkpoint paths, and preserve the old-collapse-versus-B0 distinction.
- Results captured: The 6V log confirms two completed checkpoints but no 6V eval; the single-sequence mono run completed 20 epochs but has no truck GT; the early full-data/high-LR runs are recorded separately; the 2026-07-08 B0 completed 15 epochs and selects epoch5 by `mAP/center_dist=0.344934` with BEV mAP@0.5 `0.3562`. The notes also explain why small signed xyz means are not centimeter-level absolute errors and why the old z error needs box-origin re-evaluation.
- Current experiment contract: Native single-frame S0 keeps the B0 camera, image geometry, ROI, voxel, anchor, distortion, GT filtering, batch, LR, 15-epoch cap, and COCO initialization while changing only the temporal structure. Four-card training skips validation, single-card checkpoint evaluation watches every epoch, epoch5 is compared first, and patience is five evaluated epochs without a new best.
- Verification: Cross-checked the six `/workspace/*.log` files and `/workspace/eval_summary.md`; verified Markdown table column counts and referenced local files; `git diff --no-index --check /dev/null N7_FASTBEV_EXPERIMENT_SUMMARY.md` and the corresponding detailed-note command passed.
- What remains uncertain: Historical PTH/result files are not present locally. Internal follow-up must archive checkpoint/config/data/calibration hashes, verify the T3 historical metric-to-work-dir association, re-evaluate B0 result pkl files after the box-origin fix, and fill in the S0 work_dir, weights, and metrics after training.

### 2026-07-13 Box-Origin Re-Eval And XYZ Accuracy Reporting

- Internal result: The user re-evaluated the existing result pkl after the box-origin fix and confirmed that both car and truck z bias returned to a normal level. No retraining or repeated inference was needed. Exact corrected values, result path, and SHA256 still need archival.
- Metric decision: Downstream-facing distance-bin xyz accuracy now uses per-axis MAE, defined as `mean(abs(pred-gt))`, plus absolute-error p50/p90. Do not use `abs(mean(pred-gt))`, because cancellation has already occurred before the absolute value.
- Runtime display: `CustomMultiViewDataset` now computes `x/y/z_abs_mean` globally and by GT x range. Its eval log table, including output seen while `tools/eval_epoch_checkpoints.py --watch` evaluates a new checkpoint, shows only MAE and absolute p50/p90 for each class/range; signed means are not shown in this runtime table.
- Persisted diagnostics: `tools/eval_epoch_checkpoints.py` writes a primary `TP XYZ Absolute Error By GT X Range` table and a separate `TP XYZ Signed Bias By GT X Range` table to `eval_summary.md`. `eval_range_summary.csv` and `eval_summary.json` retain both absolute and signed fields.
- Interpretation: These xyz tables still describe matched TPs with center distance <=2m and include TP counts; they do not include misses or unmatched distant objects. MAE expresses average displacement, while p90 remains the better tail-quality indicator for downstream acceptance.
- What remains uncertain: Old metrics JSON files do not contain the newly added `*_abs_mean` keys. Re-run evaluation from existing result pkl files with `--skip-inference --rerun-eval` to populate MAE; inference does not need to be repeated.

### 2026-07-13 N7 FFmpeg H.264 Visualizer Candidate

- What changed: Added the test-only replacement candidate `tools/data_converter/n7/visualize_n7_fastbev_pkl_ffmpeg.py`; the current `visualize_n7_fastbev_pkl.py` was intentionally left unchanged. The candidate writes raw BGR frames directly to FFmpeg and defaults to `libx264` (`veryfast`, CRF 20, `yuv420p`, faststart). `--video-encoder h264_nvenc` is an explicit optional GPU path and never silently falls back to CPU.
- Reliability/refactor: FFmpeg capability is checked by actually encoding one 64x64 frame before rendering. Each output is written to a same-directory temporary MP4 and atomically published only after FFmpeg exits successfully; failure preserves any previous output and retains the FFmpeg log. Both clip and sequence modes are sorted by target video plus timestamp, only one encoder process stays active, odd dimensions are padded to even, and the former three serial/parallel/video loops are consolidated into one bounded ordered render pipeline. Compact-mode videos use the full possible legend set only to reserve a fixed Header layout; the rendered legend still contains only classes/sources present in the current frame. At 1600px mono-front width the worst six-item car/truck split legend fits beside the title in a fixed 40px single row.
- Suggested internal CPU test: `python tools/data_converter/n7/visualize_n7_fastbev_pkl_ffmpeg.py --gt-pkl <val.pkl> --pred-pkl <results.pkl> --data-root <data_root> --output-dir <output_dir> --camera-ids cam0 --camera-size 1600 900 --no-bev --video-only --video-group sequence --max-frames-per-video 2000 --workers 4`.
- Suggested internal GPU test: append `--video-encoder h264_nvenc`; confirm `ffmpeg -hide_banner -encoders | grep h264_nvenc`, `nvidia-smi`, playback in the Remote-SSH VS Code video extension, and `ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate -of default=nw=1 <video.mp4>`.
- Local verification: `py_compile`, `--help`, `--no-render`, and no-index whitespace checks passed. Synthetic N7 data exercised two-worker frame rendering, clip and sequence grouping, timestamp order, per-video limits, fixed header dimensions, odd-to-even padding, and a three-worker direct-pipe run. Follow-up Header tests covered empty/subset/full current-frame legends and short/long titles at 1600px and 321px; all 1600px frames stayed `1600x940`, used a 40px single-row Header, and empty frames did not draw absent class icons. A simulated encoder failure verified temporary-file cleanup, retained error logs, and preservation of an existing final MP4. The original visualizer SHA256 remained `ec216c03b13b43b723695ead39d2dfb30acef16b0713f57083d5700222153258`.
- What remains uncertain: This local container has no standalone `ffmpeg`/`ffprobe`, so the candidate has not produced a real H.264 bitstream here. Run the CPU test first on the internal Ubuntu 20.04 server, then compare NVENC throughput and visual quality; replace the original script only after those checks pass.

### 2026-07-13 Canonical mAP, Range Recall And force_resize Plan

- Metric schema: N7 custom eval schema v2 now defines bare `mAP` as center-distance mAP: average AP@0.5m/1m/2m/4m within each class, then average classes with GT. `mAP/center_dist` remains a same-value compatibility alias. BEV is only reported under explicit keys such as `mAP/bev_iou@0.5`; it no longer overwrites `mAP`. Per-class canonical names are `car/center_AP` and `truck/center_AP`, with the old `center_dist_mAP` keys retained as aliases.
- Runtime display: The class table no longer expands every center-distance threshold. It shows class GT/det, center_AP, Recall@2m, ATE/AOE/ASE and BEV_AP@0.5. The x-range table now shows GT, unique TP@2m, Recall@2m and x/y/z MAE only. MAE is still computed only over unique center-distance matches <=2m, while Recall@2m exposes the missed-GT fraction.
- Persisted reports: `tools/eval_epoch_checkpoints.py` defaults to `--best-key mAP`. For historical schema-v1 records it always prefers `mAP/center_dist`, so old BEV-valued bare `mAP` cannot silently select the best epoch. Markdown/CSV/JSON retain per-threshold AP, absolute-error p50/p90 and a separate signed-bias table; range rows now also contain GT and Recall@2m.
- force_resize decision: Keep `force_resize=True` unchanged for the current S0-vs-B0 first comparison because both baselines currently use the same deterministic resize and no random image geometry augmentation. After that A/B, treat it as the fourth correctness closeout item alongside pkl gates, production-vs-dataset alignment and distortion-aware visibility. Refactor native-K/cache-to-704x256 deterministic geometry separately from optional stochastic augmentation, first prove the no-augmentation path equivalent in image/post transform/projection/logits, then enable augmentation in a separate A/B. Do not simply set `force_resize=False`; temporal frames must share geometry augmentation parameters.
- Which files changed: `mmdet3d/datasets/custom_multiview_dataset.py`, `tools/eval_epoch_checkpoints.py`, `N7_FASTBEV_EXPERIMENT_SUMMARY.md`, `N7_FASTBEV_EXPERIMENT_DETAILS.md`, `N7_MONO_FRONT_SINGLE_FRAME_TODO.md`, and `CODEX_HANDOFF.md`.
- Local verification: `py_compile` passed for the two Python files; an AST-isolated dataset smoke verified canonical mAP aliases, per-range GT/TP/Recall@2m and MAE; a tracker compatibility smoke verified schema-v1/v2 best selection and the new Markdown tables; a terminal-table smoke verified threshold AP/p50/p90/signed bias are absent from runtime output; tracker `--help`, Markdown table checks, tracked `git diff --check` and untracked no-index whitespace checks passed.
- What remains uncertain: Real metrics must be regenerated from internal result pkl files to populate schema-v2 range GT/Recall/MAE fields. The force_resize refactor is deliberately planned but not implemented before the first S0/B0 A/B.

### 2026-07-13 N7 FFmpeg Visualizer Low-Risk Render Optimizations

- What changed: Optimized only the validated candidate `tools/data_converter/n7/visualize_n7_fastbev_pkl_ffmpeg.py`; the current original visualizer remains unchanged. A one-camera mosaic now returns its panel directly, equal-size padding returns the source array, single-row mosaics avoid redundant concatenation, and `--camera-size` skips `cv2.resize` when the rendered panel already has the requested dimensions.
- Usage examples: Replaced the mixed example block with four copyable primary commands: 6V GT-only, 6V GT+Pred, mono-front GT-only, and mono-front GT+Pred. All four explicitly use `--gt-view-mode split`, so used GT keeps its class color and filtered GT is gray; all four use the matching eval GT ROI, mono-front also explicitly enables `cam0` visibility filtering, and both prediction examples expose `--score-thr`, `--max-preds`, and compact labels. `used` remains documented as the optional mode when filtered GT should be hidden completely.
- Undistort cache: Replaced the unbounded global map dictionary with a thread-safe LRU using both a 16-entry limit and a 256 MiB byte limit. The first thread builds each calibration map while holding the cache lock, later workers share the completed read-only maps, and the final log reports entries, memory, hits, misses, evictions and limits. A 1600x900 `CV_16SC2` map pair measured 8.24 MiB locally.
- 1600x900 cache contract: Explicitly tested a pkl whose K is declared at `intrinsic_width=3840`, `intrinsic_height=2160` while the cached image is `image_width=1600`, `image_height=900`. The visualizer independently scales K rows by `1600/3840` and `900/2160`, performs no redundant resize when `--camera-size 1600 900` is used, and keeps the final no-BEV video frame at 1600x940 including the 40px Header. The earlier source-resolution wording was a user-confirmed typo, not a separate supported resolution.
- Local verification: Mosaic outputs for 1-6 mixed-size panels were pixel-identical to the former algorithm; no-op padding identity and smaller-target rejection passed; OpenCV undistortion output/new-K parity passed; 24 concurrent accesses produced one miss and 23 hits; forced two-entry LRU eviction passed. A full two-worker synthetic sequence used the fake FFmpeg pipe, wrote two 1600x940 frames, and logged one 8.2 MiB cache entry with one hit/one miss. Follow-up checks passed for `py_compile`, CLI help, whitespace, all four embedded commands through the real argparse parser, `3840x2160` K scaling to a `1600x900` cache image, no-op `--camera-size 1600 900`, and the final `1600x940` Header canvas.
- What remains uncertain: These post-validation render/cache changes have not yet been rerun with the real internal FFmpeg binary and real 1600x900 cache dataset. The encoding command itself was not changed; run a short internal visual parity and memory/log check before replacing the original script.

### 2026-07-13 N7 PKL Data Gate And Box-Origin Publicization

- Box-origin contract: Added the pure-numpy `tools/n7_box_origin.py` as the single center/bottom/native conversion implementation. Dataset eval, the OpenCV and FFmpeg N7 visualizers, production mono/ONNX JSON+pkl output, and the portable eval exporter now delegate to it. New array outputs must declare `box_origin`; legacy arrays without metadata remain unchanged instead of guessing. The vendor `tools/utils.py` fixed-6V/corner-output path was deliberately not changed and is not the canonical mono board postprocess.
- Existing-pkl gate: Added `fastbev_pkl_data_gate.py` and extended `validate_n7_fastbev_pkl.py` to accept train/val/test together. It writes `data_gate.json` and `data_gate.md`, checks token/clip overlap, exact camera order, K/extrinsic/distortion/size fields, optional real image headers, GT array integrity, empty/class/box distributions, current pinhole-visible GT counts, pose coverage and history offsets. Use `--data-mode single-frame` for S0 and `--data-mode temporal` for B0; `--strict-data` makes FAIL nonzero.
- Converter gate: Formal conversion now requires manifests unless explicit debug `--allow-auto-discovery`; strict mode is default and only writes a failure summary, not a pkl, when critical clip/frame/camera/label/gate checks fail. `--allow-partial` is debug-only. Sidecars retain manifest provenance, failure categories/examples, raw image-frame/label/empty counts, `gt_box_origin=center`, and gate status. Temporal B0 generation should add `--require-pose`; S0 does not require pose but the external report still audits it.
- Local verification: `py_compile`/CLI help passed for all touched files. Synthetic three-split pkl data passed strict data+geometry gates; injected missing distortion plus token/clip leakage produced FAIL and nonzero strict exit. A synthetic converter run with three manifests produced three PASS pkls; removing one cam0 image produced a failure summary, no pkl, and nonzero exit. Center↔bottom round trip and every public consumer returned the same center z. The local converter smoke used a temporary `pyquaternion` stub because this modern container lacks that existing converter dependency.
- What remains uncertain: The workspace has no actual N7 pkl/images and no legacy training runtime. Run the gate on internal train/val (and later independent test) pkls, archive the JSON/Markdown, then sync any converter fixes revealed by real failure examples. Distortion-aware GT visibility remains a separate unresolved item; the gate labels its current visible-GT statistic as pinhole-only.

### 2026-07-14 N7 FFmpeg Visualizer Promotion

- What changed: After the user confirmed real internal FFmpeg visualization and playback, promoted the validated H.264 implementation to the canonical `tools/data_converter/n7/visualize_n7_fastbev_pkl.py` and removed the duplicate `visualize_n7_fastbev_pkl_ffmpeg.py` candidate. Embedded commands now use the canonical filename. Existing imports from mono/6V production inference and point-cloud diagnostics remain unchanged; the public drawing API is backward compatible, with only the optional trailing `legend_layout_items=None` parameter added to `render_info`.
- Runtime contract: Image-only rendering and imported drawing helpers do not require an FFmpeg executable. Video output now requires `ffmpeg` and defaults to `libx264`; `h264_nvenc` remains explicit and never silently falls back. The former OpenCV `mp4v` video backend is no longer retained.
- Local verification: `py_compile` passed for the canonical visualizer and its three import consumers. An isolated API check found every externally imported symbol in the promoted implementation and confirmed the `render_info` parameter prefix is unchanged. The local container still has no FFmpeg binary, so real H.264 generation relies on the completed internal test.

### 2026-07-14 Commit Identity And Baseline Submission Rule

- Commit author and committer must both be the user's GitHub identity `liujiaren19 <1334282612@qq.com>`. Never create another commit as `Codex` or `codex@openai.local`.
- Before every `git commit`, show the exact file scope and commit message to the user and wait for explicit approval. Verify both `git var GIT_AUTHOR_IDENT` and `git var GIT_COMMITTER_IDENT` immediately before committing.
- The user confirmed that current source code is synchronized with the internal repository. Markdown and Claude/Codex workflow files are local-only, so keep the reproducible internal code commit scope separate from any local documentation commit.

### 2026-07-14 N7 B0/S0 Baseline Commit Stack

- GitHub code commit A is `2d8ab7d9855f787d55da8521b1993c445351037d` (`固化 N7 6V 与单目前视四时序基线及数据评估链路`). It also makes the four-date `20260625` pkl the default 6V 704x256 dataset and records the known legacy-pkl gate failure `labels_without_frame=851/90`.
- GitHub code commit B is `c3ec682e78e273fa6142e13825f0cde4857d7883` (`新增 N7 原生单帧训练与 ONNX/LUT 导出支持`). It explicitly records that real S0 checkpoint export, vehicle-calibration LUT, decoded boxes, sigmoid-once, and PTH/ORT/board float alignment remain internal validation work.
- Both commits use `liujiaren19 <1334282612@qq.com>` for author and committer. The user separately confirmed that the corresponding two code commits were completed, checked, and pushed on the internal repository branch.

### 2026-07-14 S0 First-Round Evaluation And Documentation Refresh

- Evidence added: Parsed `work_dirs/n7_mono_704_256_single_frame/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260713/20260713_113748.log` and its `test_results/eval_summary.md`. The log confirms e1～e9 checkpoints and reaches e10 190/785; it does not prove e10 or e15 completion. The eval summary covers e1～e9.
- S0 current result: canonical mAP best is e8 `0.355078`, with BEV mAP@0.5 `0.3650`, mATE `0.8681`, mAOE `4.5918°`, and mASE `0.2297`. E9 has slightly lower canonical mAP `0.3548`, but higher BEV mAP `0.3713` and better aggregate ATE/AOE/ASE. E8 remains the checkpoint selected by the declared primary key until later epochs are available.
- B0/S0 comparison: At e5 S0/B0 canonical mAP is `0.3518/0.344934`; current-best S0 e8 exceeds B0 e5 by `0.010144`. The gain is modest and not universal: S0 e8 AOE is about `0.277°` worse. E8 truck Recall@2m is `0.7107/0.7124` at 40～60m/60～80m, consistent with the reported far-range weakness.
- Production qualitative evidence: The user reports S0 e8 is much better than B0 repeated-current-frame inference on straight roads within about 40m, but remains weak for road-end turns, ~90°/door-facing vehicles, and 40～80m. This is recorded as no-GT qualitative evidence only.
- Loss correction: B0 e15 `0.6496` is `positive_bag_loss`, not total loss. The same sample has `negative_bag_loss=0.1433` and total `loss=0.7928`. Experiment notes now use explicit positive/negative/total columns.
- Data gate status: `work_dirs/n7_pkl_gate_b0/data_gate.md` is recorded as an expected FAIL caused by legacy converter `labels_without_frame=851/90`; token/clip leakage is zero. Train also dropped 1237 empty-GT labels, and no independent test pkl exists.
- LUT status clarified: The current config-driven LUT builder with `n_times=1,sequential=False` selects only current cam0 and no history/prev pose. A production board-calibration-direct input path and parity comparison against pkl-derived LUT are still open; do not mark fixed-LUT board validation complete.
- Documentation changed only: refreshed `N7_FASTBEV_EXPERIMENT_SUMMARY.md`, `N7_FASTBEV_EXPERIMENT_DETAILS.md`, `N7_MONO_FRONT_SINGLE_FRAME_TODO.md`, and this handoff. No source code, commit, or push was requested/performed in this task.

### 2026-07-17 S0 Final Epoch13 Dataset/Production PTH Alignment

- Training status: The user confirmed the native single-frame S0 run completed all 15 epochs and selected epoch13 as the final best checkpoint. Exact epoch13 metrics and asset hashes still need to be copied into the experiment archive; do not retain the earlier epoch8-as-current-best wording for final model selection.
- What changed: Added `tools/compare_mono_front_dataset_production.py` to run one val image through the standard `CustomMultiViewDataset` test path and the production sample-construction path with the same FP32 PTH model instance. Temporary hooks capture normalized input, `neck_fuse_0` 2D feature, `neck_3d` input and pre-sigmoid bbox-head logits; both sides decode through canonical `bbox_head.get_bboxes()` and convert boxes through `tools/n7_box_origin.py` to center-origin. The tool writes manifests, per-stage hashes/diffs, optional full tensors and strict exit status. It also reproduces `tools/test.py` cleanup by removing `samples_per_gpu` before `build_dataset()`.
- Real internal result: On the N7 acquisition vehicle val data, epoch13 passed the pkl-calibration golden sample and the corresponding N7 `info.json` mode. The reported token prefix was `20251031_164821_10`; `first_mismatch=None`, `asset_relation=NUMERIC_CALIBRATION_MATCH`, `calibration_status=PASS`, and `downstream_status=PASS`. Five additional fixed, non-prediction-selected samples all reported `real PTH alignment: PASS` and `first_mismatch=None`.
- Production-vehicle numeric result: The user then generated a no-GT `infos.pkl` from one real production-vehicle image and its matching production-vehicle `info.json`, and fed it through the same epoch13 comparison. All asserted states passed, including the pkl reference, numeric calibration relation and info-json downstream path; the final check printed `PRODUCTION_VEHICLE_NUMERIC_ALIGNMENT=PASS`. Because both the temporary pkl and production sample derive from the same info.json, this closes production-image code-path consistency but is not independent evidence that the physical camera-to-lidar calibration is correct.
- Interpretation: Within the checked N7 samples, the standard dataset and production PTH paths agree through image loading, deterministic 1600x900-K to 704x256 post transform, normalization, distortion-aware backprojection, 2D/BEV features, raw cls/bbox/dir logits and canonical decoded boxes. This closes the N7 code-path alignment item; it does not prove production-vehicle calibration, physical projection accuracy, ONNX/LUT/board parity, or model accuracy.
- Local verification: `python -m py_compile tools/compare_mono_front_dataset_production.py`; `--help`; synthetic same-input PASS; injected-input strict FAIL with exit code 2 and `first_mismatch=input`; full tensor dump assertions for `[1,1,3,256,704]`, `[1,256,160,140]`, raw logit keys and `box_origin=center`; `git diff --check` and no-index whitespace check passed. The real PTH execution occurred only in the user's internal legacy environment.
- Next action: Independently validate the production-vehicle physical calibration with synchronized lidar points on both the raw-distorted and undistorted image views, checking center/edges and 20-40m behavior. After that, close epoch13 PTH-to-ONNX/LUT/canonical-box float parity.

### 2026-07-18 S0 Epoch13 ONNX / Torch CUDA LUT Alignment

- LUT root cause and fix: The original NumPy-built offline LUT differed from FastBEV's runtime Torch CUDA distortion projection at 3650 voxel sampling positions, with only 8 valid-mask differences. Rebuilding the same token LUT through the new `build_fastbev_lut.py --projection-backend torch --torch-device cuda:0` path produced `valid_count=80117`; the fixed-LUT geometry then matched the PTH dynamic-backprojection BEV exactly (`exact_equal=True`, max/mean absolute difference 0). This closes the N7 fixed-LUT geometry implementation issue for the tested token; NumPy and Torch projection backends must not be treated as integer-LUT equivalent.
- 2D ONNX result: PTH eager repeatability and PTH versus the export-wrapper eager path were bit-exact. ONNXRuntime CPU and the requested CUDA provider produced the same observed 2D output, but ONNX versus PTH feature still had max/mean absolute differences `0.0022243261/0.0003409632`. The export-time random-input 2D maximum was `0.0021201372`, so the real-image result is consistent with the known ONNX graph/runtime numerical envelope rather than a wrapper or input mismatch.
- 3D ONNX isolation: Feeding the golden PTH BEV directly to 3D ONNX produced raw bbox/cls/dir maximum differences `0.02209711/0.00929117/0.00585455`; the full 2D ONNX + exact LUT + 3D ONNX chain produced `0.04293680/0.01437950/0.00860071`. Replaying golden PTH logits through canonical postprocess remained bit-exact at 41 boxes, proving the canonical sigmoid-once/decode implementation itself is unchanged.
- Decoded-box behavior: Direct 3D ONNX returned 43 boxes versus the 41-box PTH reference; the full chain returned 42. Diagnostic matching found all 41 PTH boxes in both candidates. For the direct path, common boxes had center max/mean `0.001178m/0.000425m`, score max `0.000593`, plus two low-score extras (`0.07663` car and `0.06074` truck). For the full path, common boxes had center max/mean `0.001748m/0.000701m`, score max `0.000975`, plus one low-score car (`0.07664`). Therefore the chain is geometrically close but remains strict FAIL because small ONNX logit changes cross a discrete top-k/threshold/NMS selection boundary.
- Diagnostic tooling and real trace: `compare_mono_front_onnx_lut.py` now records a read-only `postprocess_trace.json`. It leaves `bbox_head.get_bboxes()` and alignment status unchanged, maps final canonical outputs back to original anchor+label pairs, and reports whether a selection diverged at `nms_pre`, `score_thr`, or class-wise NMS. The real internal trace passed and identified `nms_pre=1000` as the exact decoded-count divergence. Direct 3D ONNX moved source anchor 42073 from PTH rank 1001 (`0.07654374`) to rank 1000 (`0.07662670`), after which class-wise NMS retained the same anchor as both car and truck. The full chain moved source anchor 21651 from PTH rank 1002 (`0.07652451`) to rank 1000 (`0.07664063`) and retained one extra car. Both comparisons kept the same 41 reference anchor+label outputs and had no missing reference selection.
- Raw versus simplified ONNX: The user reran explicit raw/simplified combinations and compared their dumped tensors. Simplified versus raw 3D outputs from the same PTH BEV were bit-exact for bbox/cls/dir logits. Simplified versus raw 2D feature, fixed-LUT BEV and full-chain bbox/cls/dir logits were also all bit-exact (`exact_equal=True`, max/mean absolute difference 0). Therefore onnxsim is conclusively not the PTH/ONNX drift source for this export/runtime; the simplified ONNX files remain the deployment candidates unless the board converter imposes a separate compatibility constraint.
- Fixed-five N7 robustness result: The user ran sorted val indices `0,142,284,426,568` from the same `20251031_164821_10` clip. All five asset contracts passed and all five Torch CUDA LUTs reproduced the dynamic PTH BEV exactly. Across 225 PTH canonical anchor+label outputs, direct 3D ONNX retained all 225 with zero missing and two extras confined to index 0; the full 2D ONNX + LUT + 3D ONNX path retained all 225 with zero missing and one extra, also confined to index 0. The other four samples had identical output anchor+label sets. Full-chain common-box maxima were `0.004290m` center distance, `0.012285m` size, `0.000553rad` yaw (about `0.0317°`) and `0.001212` score. The single full-chain extra score was `0.076641` and its exact divergence remained `nms_pre`. This supports N7 functional parity for the fixed five samples while strict PTH/ONNX tensor/output-set parity remains FAIL. Because all five tokens are from one clip, this is not yet cross-sequence or cross-scene evidence.
- Local verification: `py_compile`, `--help`, report PASS self-test, injected-LUT strict FAIL with exit code 2, a synthetic `nms_pre` trace smoke, and `git diff --check` passed. This workspace still lacks legacy mmcv/mmdet runtime and real epoch13 assets; all real values above were supplied from the user's internal run.

### 2026-07-20 Production 9797 Single-Sample ONNX / Fixed-LUT Alignment

- Real production asset: The user selected native `3840x2160` image `1709058510666_1709058510666_1.jpg` from vehicle `9797_UKEF`; epoch13 PTH visualization contains one production-relevant car. The image has no independent GT, so it is suitable for execution-path parity and qualitative inspection, not accuracy measurement.
- LUT and asset result: The production golden tensor, exported ONNX assets and checkpoint passed `asset_contract`. A fixed LUT generated from the production `infos.pkl` with the Torch CUDA projection backend reproduced the PTH dynamic-backprojection geometry exactly (`lut_geometry=PASS`). Because that `infos.pkl` was constructed from the same production `info.json`, this validates the current PC simulation path but does not yet replace the still-open board-calibration-direct LUT input/parity check.
- ONNX numerical result: The production image PTH/ONNX 2D feature maximum absolute difference was `0.002673715353012085`; `head_logits_direct` remained a strict numerical FAIL, consistent with the already characterized ONNXRuntime/PTH float drift. These raw tensor differences must remain visible in the strict report and must not be hidden by relaxed data rewriting.
- Canonical selection result: The full `2D ONNX -> fixed LUT -> 3D ONNX -> canonical postprocess` trace selected the same 22 anchor/label outputs as PTH: comparison `PASS`, `first_divergence_stage=None`, common `22`, extra `0`, missing `0`. Thus this production sample has no discrete top-k, score-threshold or NMS selection divergence.
- Production car-only diagnostic: At the production display threshold, both direct 3D ONNX and the full chain retained the same car count with no missing or extra. Direct/full car errors were respectively: center `0.0003552224m/0.0001287363m`, size max `0.0001125336m/0.0000748634m`, yaw `0.0000343323rad/0.0000195503rad`, and score `0.0000004172/0.0000192523`. The full chain being slightly closer than the isolated direct chain is ordinary error cancellation and is not evidence that adding 2D ONNX improves accuracy.
- Interpretation: Mark this exact production sample as functional float-chain PASS while retaining strict raw-tensor FAIL. Production currently needs only `car`, but numeric validation continues to preserve the trained two-class car/truck head and canonical decode; car-only filtering belongs in a versioned output layer and must later be matched to the board implementation.
- Next action: Reuse the same production calibration/LUT across a fixed, non-result-selected multi-image set and aggregate common/missing/extra car outputs plus center/size/yaw/score errors. After production multi-sample closure, implement and compare the board-calibration-direct LUT path and board-consistent preprocessing/postprocessing. Before finishing the overall task, perform the explicitly recorded test-code/tooling cleanup review; do not remove or submit pre-existing local/vendor files implicitly.

### 2026-07-20 Production 9797 Ten-Sample Float-Chain Closure

- Sample sets: The user completed two fixed production sets from the same vehicle/calibration: five consecutive 10-FPS frames for short-term stability and five reproducibly random-selected frames for wider temporal coverage. Different-scene stratification is intentionally deferred. Across the ten samples, the PTH references contained 290 canonical selected anchor/label outputs and 41 production car outputs at score `>=0.2`.
- Aggregate result: All ten samples passed the asset contract, all Torch CUDA fixed LUTs exactly reproduced PTH dynamic geometry, and every full-chain postprocess trace passed with `first_divergence_stage=None`. The full chain retained all 290/290 canonical anchor/label outputs with total missing `0` and extra `0`; all 41 score-filtered car outputs also had matching counts with no reported missing or extra.
- Output-order diagnostic: One consecutive frame initially showed raw row-wise `xyz/lwh/yaw` differences of about `22.05m/1.16m/2.75rad` despite an identical 24/24 selected anchor/label set. Anchor+label tracing proved that exactly two output records exchanged array positions. After diagnostic alignment by `source_anchor_index + label`, all-output maximum center error was `0.0023438245m`; the production car set had missing `0`, extra `0`, center max `0.0011772905m`, size max `0.0004625320m`, yaw max `0.0002729893rad`, and score max `0.0002440661`. Therefore the large raw row-wise value was an ordering artifact, not a physical box displacement.
- Interpretation: Close production-vehicle PC float simulation as functional PASS for these ten no-GT images: `2D ONNX -> fixed LUT -> 3D ONNX -> PC canonical postprocess` preserves the selected detection set and production car outputs within small numerical error. Preserve the strict raw tensor and raw output-order failures in reports; anchor-aligned comparison is explicitly diagnostic and must not replace or hide raw-order reporting. If a downstream consumer treats output list order as an API contract, that ordering must be defined and matched separately during board postprocess validation.
- Remaining boundary: This result does not measure detection accuracy, does not independently prove physical camera/lidar calibration, and does not yet validate direct board-calibration LUT generation, board image preprocessing, board runtime precision/quantization, or board car-only postprocess. These are the next deployment stage. The final test-code/tooling cleanup reminder remains open until the whole board-alignment task is finished.

### 2026-07-20 Standalone Mono-Front Board Runtime Implementation

- What changed: Added `tools/run_mono_front_board_inference.py` as the standalone runtime entry, separate from the golden/comparison-heavy `tools/simulate_mono_front_board.py`. The new path is linear: deterministic image preprocessing, split 2D ONNX, per-vehicle board-bin LUT gather/scatter, 3D ONNX, pure-NumPy anchor decode/CPU rotated NMS, direction correction, center-origin output and an optional delayed-import CUDA NMS backend. It supports one vehicle directory or multiple vehicle-number directories and keeps production `car` filtering after the full model-class decode.
- Model assets: Added public `build_board_model_assets()`. Every run first validates `<weights>/board_model_spec.json` and `data/board_lut/common/<profile>/{points.npy,anchors.npy,asset_manifest.json}` by ONNX/file hashes and shapes; valid assets are skipped. The model spec records dynamic `class_names/num_classes`, head channel formulas, NCHW/NHWC output layout, anchor/test-NMS contract and sigmoid-once/center-origin semantics. Current two-class S0 resolves to 8 anchors/location and cls/bbox/dir channels `16/72/16`; a synthetic car-only config resolved to `8/72/16` while reusing the same 44800x9 anchors.
- Vehicle LUT layout: Extended `build_fastbev_lut.export_one_sequence()` with a default-off `compact_output` option. The standalone runtime uses it only when a vehicle's complete bin LUT is missing, writing `<lut-root>/<vehicle>/metadata.json`, `LUT/{gather_new_0.bin,scatter_nd_new_0.bin,featurePointLength.bin}` and `LUT_arr/{gather_0.npy,scatter_nd_0.npy}` without per-vehicle dense debug arrays. `featurePointLength.bin` remains calibration/vehicle-specific; fixed `points.npy` and `anchors.npy` live under `common/<profile>`.
- Local verification: `py_compile`, CLI `--help`, whitespace checks, public-asset generated-then-existing skip, anchor order/shape, dynamic car-only class contract, CPU rotated-NMS suppression, bottom-to-center conversion, partial-LUT nonzero failure, compact info.json-to-LUT Torch CUDA generation and a full synthetic ORT chain all passed. The full chain confirmed input/feature/BEV shapes `[1,3,256,704]`, `[1,64,64,176]`, `[1,256,160,140]`, sigmoid count one and center-origin output. The local environment lacks legacy `mmcv`, so real epoch13/production-image CPU-NMS versus canonical/CUDA-NMS box parity remains an internal validation item.
- Next action: Run the new runtime on the fixed production 9797 images with CPU NMS and CUDA NMS into separate output directories, compare `(source_anchor_index,label)` selections and numeric boxes, then use the CPU result as the no-CUDA board-postprocess baseline only if that real comparison passes. The final tooling/code cleanup reminder remains open.

### 2026-07-21 Board Runtime CPU/CUDA Closure And Runtime/Asset Separation

- Real internal result: The user completed the requested real production checks with both CPU and CUDA versions and reported that their outputs are consistent. This closes the prior CPU-versus-CUDA board-postprocess validation item for the checked production set. It remains a functional parity result for those no-GT images, not a detection-accuracy or physical-calibration result.
- Responsibility split: The former monolithic runtime was moved to the internal `tools/mono_front_board_core.py`. `tools/build_mono_front_board_assets.py` is now the only public generation entry: it prepares or validates `board_model_spec.json`, common anchors/points, and per-vehicle `LUT`/`LUT_arr`. `tools/run_mono_front_board_inference.py` is a concise read-only runtime and no longer accepts config, info.json, intrinsic/extrinsic or LUT-build options.
- Fail-closed assets: Runtime loading validates ONNX/common-asset hashes, LUT bin completeness, feature/voxel shapes, profile and geometry-contract hash. A missing/partial vehicle LUT raises a nonzero error and tells the operator to run the asset builder; inference never auto-generates or patches LUT files. Visualization calibration comes from the same LUT metadata, so projection cannot silently switch to another info.json revision.
- Runtime output: Each image writes a JSON prediction and, by default, an `original-K` undistorted front-view plus BEV visualization. The default visualization width is 1600 for practical 4K review; `--visual-width 0` preserves the source width. Classes and head channels remain model-spec driven, full model-class decode/NMS happens before the current production-default `car` business filter (`--classes all` exposes every trained class), sigmoid runs once, and public boxes remain center-origin.
- Local verification: `py_compile`, both CLI help commands, whitespace checks, model-asset existing skip, fresh CPU Torch LUT generation (`featurePointLength=80123`), full synthetic ORT runtime, visualization creation and missing-LUT nonzero behavior passed. The synthetic full chain retained `[1,3,256,704] -> [1,64,64,176] -> [1,256,160,140]` and `16/72/16` head shapes. The same synthetic ONNX/LUT input produced identical decoded JSON before and after the refactor.
- Cleanup reminder: After the complete board-deployment task is finished, review and consolidate the accumulated diagnostic/test tools as explicitly requested by the user. Do not delete, stage or submit pre-existing local/vendor files as part of that cleanup without separate scope confirmation.

### 2026-07-21 S0 PC Alignment Archive And ORT CUDA Correction

- Archive scope: The PC-side phase is ready to archive: epoch13 canonical PTH golden alignment, N7 and production-vehicle split-ONNX/fixed-LUT functional comparison, and a standalone CPU board-reference runtime are implemented and exercised. The overall deployment task is not complete because no real board pre/post implementation or board tensor/output dump has yet been compared against this reference.
- PTH golden closure: The final selected native single-frame S0 checkpoint is epoch13. The N7 val golden sample passed both exact pkl-calibration and matching N7 `info.json` production construction with `first_mismatch=None`; five fixed non-prediction-selected val samples also passed. The production-vehicle pkl/info-json construction passed its numeric code-path check, but both sides derive from the same calibration asset, so this is not independent physical-calibration evidence. Exact final epoch13 metric values and all final asset hashes still need to be copied into the experiment archive.
- Fixed-LUT/ONNX closure: Torch-CUDA LUT generation fixed the NumPy-versus-runtime integer sampling discrepancy and reproduced PTH dynamic geometry exactly for checked N7/production assets. N7 fixed-five testing retained all 225 PTH anchor/label outputs with zero missing; one sample had one full-chain low-score extra at the `nms_pre=1000` boundary. Production 9797 fixed ten-sample testing passed all asset/LUT contracts and retained all 290/290 canonical anchor/label outputs plus all 41 score-filtered car outputs with zero missing/extra. These are functional float-chain PASS results; raw 2D features and 3D logits remain strict numerical FAIL within the already-recorded ORT/PTH float envelope, and the N7 sample-0 discrete extra must remain visible rather than being hidden by matching.
- ORT CUDA correction: The internal environment has `onnxruntime-gpu==1.10.0`, PyTorch `1.10.0+cu113`, and an NVIDIA L20. Although ORT advertises `CUDAExecutionProvider`, real session inspection returned only `['CPUExecutionProvider']` and provider options only for CPU after logging `Failed to create CUDAExecutionProvider`. Therefore prior statements that requested CPU/CUDA ORT runs established CUDA parity are withdrawn: both observed ORT runs may have executed on CPU. This does not invalidate CPU ORT/PTH/LUT comparisons or Torch-CUDA LUT generation. Real CUDA ORT parity remains unverified and is not required for the CPU board-reference path.
- Standalone runtime state: `tools/run_mono_front_board_inference.py` is now intentionally self-contained and imports no project inference/core/runtime module. Its direct path is image read/preprocess -> named 2D ONNX -> fixed board-bin LUT -> named 3D ONNX -> sigmoid-once/anchor decode/direction/NMS -> center-origin business output -> front/BEV visualization. It requires prebuilt per-vehicle LUT assets and never auto-generates or repairs them. Current S0 anchors are correctly `(44800,9)` = `70*80*8` anchors with nine coder values.
- Runtime output/UX: The runtime supports `--stride`, production filename timestamp extraction, per-box class/score labels, an original-K undistorted front view, BEV `+X` up / `+Y` left legend, and default one-frame-per-line `business_predictions.jsonl`. Default business rows omit input image path/name and round bbox/score numerics to three decimals. Full per-frame decoded/tensor-shape/trace JSON is default-off behind `--save-frame-details`. Explicit `--provider cuda` now fails nonzero if ORT falls back to CPU; `auto` records the actual provider and the archival production command should use explicit `--provider cpu`.
- Verification: Latest local checks passed `py_compile`, CLI `--help`, `git diff --check`, and 22 focused standalone/core runtime tests. They cover independent imports, input shape/dtype, NCHW/NHWC handling in the analysis runtime, LUT ZC ordering/completeness, named 3D outputs, single/multi-class and circle-NMS branches, top-k tie diagnostics, center-origin behavior, timestamp/business JSONL formatting, three-decimal output, BEV axis directions/bounds, and explicit CUDA-fallback refusal. Real epoch13/9797 execution remains internal; after syncing the latest visual/provider changes, produce one clean final 9797 run with `--provider cpu` for the archive.
- What remains: (1) compare real board image decode/color/resize/normalize/layout against the standalone reference; (2) compare board 2D output, board LUT gather/scatter/ZC layout and board 3D raw outputs; (3) define and compare board sigmoid/top-k/score/NMS/yaw/center-origin/car-filter semantics, including output ordering and tie boundaries; (4) compare real board output dumps on the fixed N7 and 9797 image sets; (5) separately validate any FP16/INT8/model-converter effects; (6) add cross-scene production coverage if product acceptance requires it; (7) record final epoch13 metrics/hashes. No production GT exists in the checked set, so detection accuracy and independent physical calibration remain outside the completed claim.
- Cleanup gate: The user explicitly requested a final code/tooling consolidation after this workstream. Before any commit, perform a separately scoped review of the accumulated `compare_*`, `simulate_*`, board asset/core/runtime and historical diagnostic scripts; decide which are canonical, analysis-only or obsolete. Do not delete, stage or submit pre-existing local/vendor files without explicit approval. No commit was made for this archive update.

### 2026-07-21 Mono-Front Three-Entrypoint And Asset-Builder Consolidation

- Outcome: The N7 native S0 workflow is consolidated into three designated runtime/analysis entrypoints plus one public asset generator. Native S0 remains `n_images=1,n_times=1`; production inference never repeats one current image into four temporal views. No tracked or pre-existing untracked file was deleted, and no commit was created.
- Public responsibilities:
  - `tools/infer_mono_front_image.py`: production image inference for `pth|onnx-fp|onnx-int8` and `dynamic|fixed`, with canonical PC pipeline and `bbox_head.get_bboxes`; supports single image, one vehicle directory and multiple vehicle directories. Business filtering runs after full canonical decode/NMS.
  - `tools/run_mono_front_board_inference.py`: self-contained fixed-LUT ONNX board reference, with no project inference/core/runtime imports. Its linear path is image -> board preprocess -> named 2D ONNX -> prebuilt board-bin LUT -> named 3D ONNX -> board postprocess -> center-origin JSONL/visualization. It does not support PTH, dynamic LUT, asset generation or stage comparison.
  - `tools/analyze_mono_front_pipeline.py`: the designated complex diagnostic entry for `pth|onnx-fp|onnx-int8` x `dynamic|fixed` x `pc|board` preprocess x `pc|board` postprocess. It saves raw/dequant tensors, projection/LUT indices, raw logits, all decoded anchors, top-k/threshold/class-NMS trace, center boxes and business output, and classifies the first divergence boundary.
  - `tools/build_mono_front_board_assets.py`: the sole public asset-generation entry for `board_model_spec.json`, common anchors/points, FP/INT8 contracts and per-vehicle fixed LUT bin/npy/metadata.
- Unified schema: `board_model_spec.json` schema v2 now records config/checkpoint/model/ONNX SHA256, S0 image/time counts, preprocessing, input/feature/BEV shapes, exact tensor names/layouts/dtypes, dynamic classes/head channel contracts, anchors/points geometry and hashes, `sigmoid_count=1`, model NMS settings, `box_origin=center`, FP and real-INT8 ONNX contracts, and the fixed-LUT geometry contract hash. Runtime recomputes model/geometry contract hashes and fails closed on manifest content drift.
- INT8 contract: The builder reads actual ONNX graph and ORT session I/O rather than inferring quantization from filenames. Raw `int8/uint8` external tensors require recorded scale, zero point, per-tensor/per-channel mode, channel axis, ties-to-even rounding and clamp range. Float-external QDQ graphs are not manually quantized. The split 2D-feature/3D-input quantization domain is compared numerically while intentionally ignoring whether the boundary record came from a `QuantizeLinear` or `DequantizeLinear` node. Raw cls is dequantized before the one postprocess sigmoid. A cls output with upstream `Sigmoid` is rejected.
- File-level migration:
  - `compare_mono_front_dataset_production.py` remains the specialized legacy PTH golden-reference generator/reviewer; its embedded synthetic self-test moved to `tools/tests`.
  - Useful staged comparison, first-divergence, tensor dump and canonical trace behavior from `compare_mono_front_onnx_lut.py` and `simulate_mono_front_board.py` moved into the analyzer. Both old implementations remain as legacy/archive candidates and no longer contain public synthetic asset generation; the simulator no longer rebuilds LUTs.
  - Effective `mono_front_board_runtime.py` coverage moved to standalone/analyzer tests. The legacy implementation is retained with only cross-implementation equivalence guards until deletion is separately approved.
  - `mono_front_board_core.py` is asset-build implementation; `mono_front_board_assets.py` is the read-only asset loader. The standalone runtime imports neither.
  - `export_2d_to_3d_model.py` is not currently removable because both retained legacy compare/simulate tools still import it. `infer_6v_front_image_blackfill.py`, the old compare/simulate/runtime files and eventually that exporter remain report-only archive/deletion candidates; none was deleted.
  - The explicitly protected pkl/eval/export/download/point-cloud/visualization/LUT/box-origin tools were retained unchanged by this consolidation except for pre-existing worktree edits outside this task.
- Fail-closed behavior: Missing manifest/anchors/points/LUT arrays or any partial board-bin LUT, ONNX/asset/hash/geometry mismatch, mismatched named 3D output contract, pre-sigmoided cls output, invalid raw quantization metadata, source-image/LUT size mismatch and explicit CUDA-to-CPU fallback all raise a nonzero error. All three main entrypoints include the exact initialization message: `首次运行前先执行 tools/build_mono_front_board_assets.py 初始化模型和车辆资产。`
- Test migration and result: The two prior board test files were deduplicated and `tools/tests/test_mono_front_pipeline_contracts.py` was added. The 35-test suite covers NCHW/NHWC exactly once, named 3D outputs, ZC LUT order/completeness, single/multi-class and empty/single/multiple boxes, rotate/circle NMS, top-k tie boundaries, center origin, delayed car filtering, explicit CUDA fallback failure, float-external QDQ, raw/per-channel INT8 quant/dequant, contract tampering and analyzer raw/dequant dumps.
- FP equivalence: The same synthetic image/model/LUT input was run through the retained legacy runtime and the standalone implementation; preprocessing, 2D output, LUT BEV, named 3D raw logits and decoded outputs were exactly equal. Synthetic raw and simplified FP ONNX behavior was also exactly equal. This is refactor-equivalence evidence, not new real epoch13 accuracy evidence.
- Validation commands: All task-related Python files passed `py_compile`; the four required `--help` commands passed; `python -m unittest discover -s tools/tests -p 'test_*.py'` passed 35/35; focused FP/INT8/named-output/CUDA tests passed 6/6; tracked `git diff --check` and no-index checks for all task untracked files passed; the standalone AST import audit passed.
- Remaining real-data boundary: This workspace contains zero ONNX/PTH model artifacts and has no `mmcv`, so real epoch13/9797 execution was not repeated. No real INT8 ONNX is present; only manifest/CLI/fail-closed behavior and synthetic raw/QDQ INT8 graphs were validated. The required statement remains `未做真实 INT8 模型数值验证`. Real board image decode/preprocess, board tensor dumps, model-converter/runtime quantization and physical calibration parity remain external checks.

### 2026-07-22 Mono-Front Legacy Cleanup With Frozen Golden Fixtures

- Golden freeze: Before deleting the old runtime, froze its SHA256 (`6a05367ca6176c414550f1fd735db3158d033b214da73ef33d94ac79c18f694a`) and exact NumPy shape/dtype/byte hashes into `tools/tests/fixtures/mono_front_board_legacy_v1.json`. The fixture covers empty/single/multiclass/score-boundary postprocess cases plus the full synthetic FP chain from preprocessing through named 2D/3D ONNX, fixed LUT and decoded outputs.
- Test migration: Added `tools/tests/mono_front_golden.py` and replaced live imports of the old runtime with fixture comparisons. Renamed the runtime test to `test_mono_front_board_golden.py`; removed one now-duplicate postprocess comparison from the standalone suite. The new tests no longer execute legacy implementation code.
- Deleted after dependency audit: `tools/mono_front_board_runtime.py`, `tools/compare_mono_front_onnx_lut.py`, `tools/simulate_mono_front_board.py`, `tools/export_2d_to_3d_model.py`, and `tools/infer_6v_front_image_blackfill.py`. The first three were superseded by standalone/analyzer plus golden tests; the latter two had no remaining Python references after those deletions.
- Intentionally retained: `tools/compare_mono_front_dataset_production.py` remains because analyzer imports its `ForwardCapture` for direct PTH dynamic analysis. Pre-existing vendor/debug files such as `tools/utils.py`, `tools/vis.py`, BST files, local scripts and `.claude` were not deleted or included in this cleanup.
- Verification: Golden-focused tests passed 23/23 before deletion. After deleting all five legacy files, `python -m unittest discover -s tools/tests -p 'test_*.py'` passed 34/34. A repository-wide Python reference search found no remaining imports or calls to the deleted modules; only the fixture provenance string and historical documentation retain their names.

### 2026-07-22 Mono-Front Production Visualization Yaw Orientation Investigation

- Split-environment workflow: The current `/workspace/Fast-BEV_test_custom-fastbev-adapter` repository is the code-review/test mirror; checkpoint, production images and generated 9797 outputs remain in the internal `/mnt/liujiaren/fastbev-python-custom-fastbev-adapter` repository. Code inspection, pure geometry tests and minimal patches are completed in the workspace. Only the exact task files should then be synchronized internally; real commands run there and their JSON reports/hashes/images are returned for review. Do not substitute unrelated workspace data, copy model assets outward, or treat local synthetic checks as real-data closure.
- Root cause: The public N7 box contract is `[x,y,z_center,l,w,h,yaw]`, `+X` forward, `+Y` left and positive yaw from `+X` toward `+Y`. Both `visualize_n7_fastbev_pkl.corners_from_boxes()` and standalone `box_corners_3d()` instead copied the old clockwise-positive MMDetection3D row-vector formula. For `yaw=6.216`, the numeric forward is `[0.9977439161,-0.0671347743]`, but the old front edge was `[0.9977436,+0.0671347]`, so the BEV box was mirrored to the vehicle's left.
- Call-chain evidence: `infer_mono_front_image.py` converts prediction boxes to center-origin once and passes one `boxes` array to `render_info()`. The camera panel and BEV panel both derive corners through the same `corners_from_boxes()` function, with identical `[l,w,h]` indices and yaw index 6. There is no extra yaw negation/normalization, `pi/2` adjustment, repeated center/bottom z conversion or separate front-view box array. The apparent rightward lean in the perspective panel was therefore not proof of a second correct yaw implementation; both panels consumed the same incorrectly mirrored 3D corners.
- Minimal visualization-only fix: The two drawing corner functions now use `x'=x*cos-y*sin, y'=x*sin+y*cos`. Decode, direction correction, `bev_box_corners()` used by standalone CPU NMS, canonical `bbox_head.get_bboxes()`, business filtering, portable JSON/pkl writers, `n7_box_origin.py`, model outputs and calibration remain unchanged. In particular, original-K undistortion remains `cv2.undistort(image,K,D,None,K)`.
- Geometry coverage: Added `tools/tests/test_mono_front_visualization_geometry.py` for yaw `0`, `+pi/2`, `-pi/2`, `6.216`, exact wrapped yaw and rounded `-0.0672`. It checks front-edge indices `[0,1]`, derived forward vector, BEV up/left/right mapping, wrapped-corner equivalence, standalone/canonical visualizer equality and a same-corner synthetic pinhole projection. Added read-only `tools/diagnose_mono_front_visualization_yaw.py` to select the real output closest to yaw 6.216 and save the four bottom corners, forward vector, BEV pixels, original-K camera pixels/depth, box origins, summary/pkl/business numeric comparisons and input SHA256 values.
- Local verification: Targeted `py_compile` passed; the three new geometry tests passed; `python -m unittest discover -s tools/tests -p 'test_*.py'` passed 37/37; the diagnostic `--help` and a portable synthetic four-file run passed with `VISUALIZATION_YAW_GEOMETRY=PASS`; whitespace checks passed for the touched/new Python files. Real epoch13/9797 execution was not possible in the workspace.
- Internal evidence command:

  ```bash
  cd /mnt/liujiaren/fastbev-python-custom-fastbev-adapter
  run_dir=work_dirs/n7_board_runtime_post_cleanup_validation/9797_UKEF/production_pth_dynamic_first1
  python tools/diagnose_mono_front_visualization_yaw.py \
    --run-dir "$run_dir" \
    --target-yaw 6.216 \
    --camera-id cam0
  ```

  Return `visualization_yaw_geometry_report.json` and the final `VISUALIZATION_YAW_GEOMETRY=...` line. The report must show center origin, summary/pkl box equality, `[0,1]` as the front edge, derived forward equal to `[cos(yaw),sin(yaw)]`, and the same corrected corners projected into both coordinate systems.
- Internal visualization-only before/after command (does not rerun inference or rewrite structured outputs):

  ```bash
  cd /mnt/liujiaren/fastbev-python-custom-fastbev-adapter
  run_dir=work_dirs/n7_board_runtime_post_cleanup_validation/9797_UKEF/production_pth_dynamic_first1
  after_dir=work_dirs/n7_board_runtime_post_cleanup_validation/9797_UKEF/production_pth_dynamic_first1_yaw_visual_fix
  sha256sum \
    "$run_dir/business_predictions.jsonl" \
    "$run_dir/prediction_summary.json" \
    "$run_dir/pred_results.pkl" > /tmp/9797_yaw_visual_before.sha256
  python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
    --gt-pkl "$run_dir/infos.pkl" \
    --pred-pkl "$run_dir/pred_results.pkl" \
    --data-root / \
    --output-dir "$after_dir" \
    --camera-ids cam0 \
    --start-index 0 \
    --max-frames 1 \
    --score-thr 0.2 \
    --max-preds 100 \
    --hide-gt \
    --box-label-mode compact \
    --camera-size 1600 900 \
    --bev-range 0 -35 80 35 \
    --bev-heading-style front-edge \
    --undistort \
    --undistort-new-k original \
    --no-video
  sha256sum \
    "$run_dir/business_predictions.jsonl" \
    "$run_dir/prediction_summary.json" \
    "$run_dir/pred_results.pkl" > /tmp/9797_yaw_visual_after.sha256
  diff -u /tmp/9797_yaw_visual_before.sha256 /tmp/9797_yaw_visual_after.sha256
  find "$after_dir" -type f -name '*.jpg' -print
  ```

- Remaining real boundary: The corrected real 9797 image has not yet been returned, so physical camera/LiDAR calibration, raw-N7-to-FastBEV extrinsic correctness and edge distortion cannot be closed here. After syncing the exact visualization/test/diagnostic files, return the report, before/after images and unchanged-hash result. If the corrected shared corners still miss the physical vehicle, investigate calibration with synchronized lidar points; do not negate model yaw or change 9797 K/D to improve appearance.

### 2026-07-22 First Internal 9797 Yaw Diagnostic Version Mismatch

- First returned report: `VISUALIZATION_YAW_GEOMETRY=FAIL`, but all structured checks passed. The selected real box was exactly equal across summary/pkl/business output, with score `0.9794921875`, label `car`, `box_origin=center` and yaw `6.2160468101501465`. The only failed check was `front_edge_matches_numeric_forward`.
- Version-mismatch evidence: Numeric forward was `[0.9977470576,-0.0670880697]`, while the reported corner-derived front edge remained `[0.9977473617,+0.0670880824]`. Its four reported bottom corners exactly match the old clockwise formula, not the patched workspace implementation. Recomputing the same real box with the workspace fix gives front edge `[0.9977473617,-0.0670880824]`; maximum old/new bottom-corner coordinate delta is about `0.294472m`.
- Interpretation: This FAIL does not implicate box origin, structured output, yaw normalization or 9797 calibration. It proves the internal diagnostic loaded an unsynchronized `visualize_n7_fastbev_pkl.py` (most likely only the new diagnostic script was copied), or it ran from a different repository copy. Before rerunning, inspect the actual loaded file and require lines equivalent to `local[:,0]=x*c-y*s` and `local[:,1]=x*s+y*c`.
- Diagnostic hardening: The report now records absolute paths and SHA256 values for both the diagnostic script and the dynamically loaded visualizer, so subsequent results can prove exactly which implementation ran. Real visual/physical validation remains pending until the corrected visualizer is synchronized and the report becomes PASS.

### 2026-07-23 Real 9797 Yaw-Zero Projection Reference

- Real A/B setup: After synchronizing the corrected counter-clockwise visualization corners, the user confirmed structured output and visualization are normal. The selected real car remains center-origin at `[10.5607595,5.3739672,-1.4074830,4.3893299,1.8040502,1.4649718,6.2160468]`, whose normalized world yaw is `-3.8467525°`. A diagnostic-only yaw-zero box with identical center/dimensions was projected through the same calibration in both original-K undistorted and raw-distorted modes.
- Undistorted result: The actual projected center-axis angle was `9.6491992°`; the yaw-zero reference was already `11.0032795°`; their relative angle was only `-1.3540803°`. Raw-distorted result: actual `8.4418761°`, yaw-zero reference `10.2647351°`, relative `-1.8228590°`.
- Interpretation: The visually large absolute front-view slope is primarily perspective/camera geometry, not an enlarged yaw. The object is close and laterally offset (`x≈10.56m,y≈5.37m`), and its rear/front center-axis depths span about `8.69m→13.07m`, so even a yaw-zero box projects with a roughly `10–11°` image slope. Original-K undistortion changes the relative actual-versus-yaw-zero angle by about `0.47°` and reduces its magnitude rather than causing the apparent exaggeration. Keep original-K undistortion and the corrected shared corners unchanged.
- Acceptance boundary: Compare front-view yaw against the same-position yaw-zero projected axis, not against the image horizontal. This closes the reported BEV-versus-front visual-angle discrepancy as expected perspective behavior for the checked frame. Independent physical calibration accuracy still requires lidar/image correspondence or GT, but no yaw sign, box-origin, decode, K/D or retraining change is indicated by this result.

### 2026-07-23 Production Visualization Yaw Diagnostic Cleanup

- Closure: The visualization yaw-sign fix and the real 9797 yaw-zero projection explanation are complete. The one-off `tools/diagnose_mono_front_visualization_yaw.py` was removed after its real report had served its purpose. Its synthetic camera-projection test was also removed from the standalone test file.
- Retained regression: `tools/tests/test_mono_front_visualization_geometry.py` remains as a compact two-test coordinate contract. It checks both production visualizers against `[cos(yaw),sin(yaw)]`, cardinal BEV directions, `yaw=6.216` up-right behavior and wrapped-yaw corner/pixel equivalence. This is product regression coverage rather than disposable diagnostic code.
- Retained evidence: Keep the complete `work_dirs/n7_board_runtime_post_cleanup_validation/9797_UKEF/production_pth_dynamic_first1` group, including structured predictions, the successful geometry report, corrected visualization and `yaw_zero_reference` comparison outputs available internally. It is the canonical evidence bundle for future questions about this frame. Do not delete or rewrite it during general test cleanup.

### 2026-07-22 Mono-Front Analyzer FP32 And Internal ORT Provider Contract

- Analyzer PTH fix: Direct PTH analysis calls `extract_feat()` / `onnx_export_*()` below the model's standard `forward()` boundary. The inherited training config enables MMCV FP16, which can cast an intermediate input to `torch.cuda.HalfTensor` while a legacy-stack convolution still has `torch.cuda.FloatTensor` weights. `prepare_canonical_context()` now applies the analyzer-only config override `fp16=None`, so direct dynamic/fixed PTH analysis consistently uses FP32 weights and tensors. This does not change production inference precision policy, checkpoint contents, ONNX files or LUT assets.
- Analyzer postprocess fix: Canonical PC postprocess already reconstructs and validates each final output's original anchor through `trace_canonical_postprocess()`, but the mapping was previously retained only in the trace. It is now strictly checked for complete output order and copied into `decoded["source_anchor_indices"]` before the shared business filter runs. This fixes the analyzer-only `KeyError: 'source_anchor_indices'` without changing decoded boxes, scores, labels or NMS decisions.
- Internal ORT rule: In the current internal environment, ONNX Runtime can create and run only `CPUExecutionProvider` sessions reliably. `CUDAExecutionProvider` may be listed as available but real session creation fails (`Failed to create CUDAExecutionProvider`) or does not retain CUDA as the active provider. All current ONNX validation commands must therefore pass `--provider cpu` and record the actual session provider. Do not claim ORT CUDA parity; it remains unverified.
- Internal path-output rule: After `cd /mnt/liujiaren/fastbev-python-custom-fastbev-adapter`, commands, logs and user-facing printed output should use repository-root-relative paths for every file under that working tree (for example `work_dirs/...`, `data/board_lut/...` and `configs/...`). Keep absolute paths only for assets genuinely outside the repository, such as `/mnt/liujiaren/data/...`. This keeps internal logs portable and avoids repeating the repository prefix; it does not change how paths are resolved or hashed.
- Verification boundary: Local `py_compile`, focused unit tests and whitespace checks cover the code contract. The real epoch13 dynamic/fixed PTH analyzer pair must be rerun internally after syncing `tools/analyze_mono_front_pipeline.py`; subsequent ONNX analyzer/production checks must use explicit CPU provider.
- Internal PTH dynamic/fixed result: The real 9797 epoch13 analyzer runs completed with `backend=pth`, `preprocess=pc`, `postprocess=pc` and exit code 0 for both dynamic and fixed geometry. Their common tensors were bit-exact for raw image, normalized input, 2D feature, LUT-produced BEV, all three raw head outputs, anchors, all-anchor decode and final center boxes; `business_output.json` and `postprocess_trace.json` were also exactly equal. Dynamic projection matrices and fixed gather/scatter arrays were intentionally not compared directly because they encode the same geometry in different representations. This closes PTH fixed-LUT equivalence for the checked 9797 image, not ONNX or real board-runtime parity.
- Internal ONNX-FP fixed production result: The real 9797 `tools/infer_mono_front_image.py` run passed with `backend=onnx-fp`, `geometry=fixed`, explicit `--provider cpu` and exit/check codes 0. The one-image output contract passed for portable infos/prediction pkl, summary, car-only center-origin business JSONL and visualization creation. This confirms the production entry can execute the checked FP ONNX/LUT/calibration chain with CPU ORT; it does not yet establish exact PC-preprocess/postprocess versus board-preprocess/postprocess parity or real board-runtime parity.
- Internal PC/board postprocess envelope: The 9797 ONNX-FP fixed comparison found all tensors through raw head outputs and all-anchor decode bit-exact. Final PC-CUDA Torch versus board-CPU NumPy postprocess selected the same 22 boxes in the same source-anchor order, with identical x/y/yaw/velocity, scores, classes and one-car business selection. Only center z and decoded w/l/h differed by CPU/GPU FP32 rounding: max `9.5367431640625e-07`, mean `5.7497409859088936e-08`, all within `atol=1e-6, rtol=0`; yaw wrapped error was zero. The analyzer business comparator now keeps all discrete fields exact while applying the explicitly requested CLI tolerance only to score/box floats. This is an evidenced cross-runtime numeric envelope, not permission to relax tensor, selection, class, score, yaw or source-anchor contracts silently.
- Internal PC/board envelope gate result: After syncing the schema-aware business comparator, the real board/board analyzer rerun passed with explicit `atol=1e-6, rtol=0`, CPUExecutionProvider, stable code hashes and strict mode. `center_boxes_exact=False` remained visible with max diff `9.5367431640625e-07`; `business_output_exact=False` remained visible with max box diff `4.76837158203125e-07`. All other required checks were exact/true, including upstream tensors, business discrete fields, zero score difference, x/y/yaw/velocity, sigmoid count, center origin, selected count and source-anchor order. The accepted status is therefore `PASS` within the documented FP32 cross-runtime envelope, not bit-exact final-box parity.
- Internal standalone board-reference result: The cleaned `tools/run_mono_front_board_inference.py` executed the same real 9797 image with ONNX-FP, fixed LUT and explicit CPUExecutionProvider, returning exit code 0 with stable source hash. Its AST import audit confirmed no dependency on analyzer, production inference, the legacy PTH comparator or internal board asset/core helpers. With frame details enabled, decoded count, sigmoid/center-origin trace, selected source-anchor order, full-precision business rows and the default rounded timestamp JSONL were exactly equal to analyzer board/board output; visualization creation also passed. This closes the independent PC-side board-reference implementation for the checked image, but not real chip/runtime dump parity.
- Internal continuous-five batch result: Production `onnx-fp/fixed` and the standalone CPU board reference both completed the naturally sorted first five 9797 images with exit code 0 and stable source hashes. Both batch contracts reported exactly five frames/records/details in matching timestamp order. Per-frame class/count/origin and output structure matched; production canonical versus standalone board score/box floats stayed inside the documented `1e-6` envelope, while each standalone rounded JSONL row matched its full-precision detail exactly. This closes multi-frame selection/order/file-organization parity for the checked continuous set; visualization was intentionally disabled here because the prior single-frame visualization creation gate already passed.

### 2026-07-23 Mono-Front Final Test Cleanup Disposition

- Removed locally: Generated Python caches under the touched config, `mmdet3d`, `tools`, N7 converter and mono-front test directories. The one-off yaw diagnostic script and its disposable synthetic projection test had already been removed by the completed yaw investigation.
- Retained product/regression code: `tools/infer_mono_front_image.py`, `tools/build_mono_front_board_assets.py`, `tools/run_mono_front_board_inference.py`, `tools/analyze_mono_front_pipeline.py`, both board asset/core helpers, and `tools/compare_mono_front_dataset_production.py` because analyzer PTH dynamic still imports its `ForwardCapture`.
- Retained tests/evidence: Keep all 38 `tools/tests` regressions, `mono_front_board_legacy_v1.json`, the compact yaw geometry regression, the N7 epoch13 golden and current board assets. The count is 38 after the completed yaw cleanup intentionally removed one disposable synthetic camera-projection test while retaining two product coordinate-contract tests. Keep the full internal `work_dirs/n7_board_runtime_post_cleanup_validation/9797_UKEF/production_pth_dynamic_first1` group as the canonical structured/yaw evidence bundle. Do not delete the LUT archive during generic test cleanup.
- Internal removable outputs: The PTH dynamic/fixed comparison, ONNX-FP single-run check, PC/board analyzer comparison, standalone single-image comparison and continuous-five batch parity directories are reproducible validation scratch outputs after their results have been recorded above. Remove only those explicitly named directories; do not recursively delete the validation root or canonical evidence group.

### 2026-07-23 Internal Mono-Front Final Cleanup Completion

- Internal synchronization corrections: Restored the missing `tools/tests/test_mono_front_board_golden.py` and synchronized the cleaned two-test `tools/tests/test_mono_front_visualization_geometry.py`. The stale internal yaw test had still contained the removed synthetic camera-projection case, which explained the temporary count of 39; the obsolete `tools/tests/test_mono_front_board_runtime.py` had explained the earlier count of 41.
- Removed obsolete code/tests: Deleted the internal copies of `tools/infer_6v_front_image_blackfill.py` and `tools/tests/test_mono_front_board_runtime.py`; the remaining explicitly obsolete mono-front modules and one-off yaw diagnostic were confirmed absent. Final scope review also removed the standalone `tools/data_converter/n7/compare_n7_camera_undistortion.py` A/B diagnostic after its original-K conclusion had been recorded. `tools/mono_front_board_visualization.py` was confirmed absent; the standalone board reference intentionally keeps its compact visualization path self-contained. The deleted-module import audit passed.
- Removed reproducible outputs only: Deleted the six explicitly listed PTH/ONNX/standalone/continuous comparison scratch directories and generated Python/pytest caches. Preserved the complete `production_pth_dynamic_first1` canonical evidence group, the current model/ONNX/LUT/golden assets and both SHA256 manifests created before deletion.
- Final internal result: The post-cleanup suite returned `Ran 38 tests`, `OK`, exact test-count code 0, task-scoped diff check code 0, import audit pass, obsolete-file audit pass and `MONO_FRONT_FINAL_CLEANUP=PASS`. Three task files had only trailing whitespace in command examples and were corrected. A separate pre-existing trailing-whitespace finding in `tools/export_onnx.py` was outside this task and intentionally left untouched.
- Remaining validation boundary and next phase: Internal ONNX Runtime validation remains CPU-provider-only. Real chip tensor dumps, the actual board runtime and real INT8 numerical validation remain deferred (`未做真实 INT8 模型数值验证`) and are not implied by this cleanup closure. The mono-front code/fixture cleanup task is complete, so active work can return to model-side input geometry, augmentation, sampling/anchor/NMS and optimization experiments while keeping M4 board acceptance open.

### 2026-07-23 EXP-6V-B0 Final Accuracy Closeout

- Evidence scope: Inspected only the synchronized `20260717_173206.log` and `test_results/eval_summary.md` under `work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260717`. The epoch17-saved log SHA256 is `bf5a98f4908ff1ac5d0d49fec687ae564577a9760dfd0c63492b2c681ba0635b`; the still-e1～16 eval-summary SHA256 is `ad2254ef828efcacf4a5b66cd2f556a037799f6620d4d442b42c91bc6a2ed580`.
- Completeness: One continuous non-resumed log proves epoch1～17 complete and saved. The eval summary covers the formal selection window epoch1～16; epoch17 is intentionally excluded and will not be evaluated for this baseline.
- Selection: Canonical center-distance mAP and both BEV AP summaries peak at epoch6 (`0.453768`, `0.6453`, `0.4787`). Epoch2 is the 40～60m Recall@2m challenger (`0.7908` aggregate); epoch16 is the formal terminal. Epoch17 remains only an excluded raw continuation asset.
- Convergence: Epoch7～16 are 10 consecutive evaluated misses after the canonical best, so patience=5 is satisfied. Median final-20%-of-epoch loss falls from `0.6501` at epoch6 to `0.4096` at epoch17, while the latest evaluated epoch16 canonical mAP falls to `0.4192` and 40～60m Recall to `0.5333`; this supports overfit/selection degradation rather than a numeric crash. There are 14 isolated `grad_norm=inf` log points, including two in epoch17, but no NaN/Inf loss, OOM, traceback, data skip or recorded training interruption.
- Decision boundary: The run is closed at epoch16 for formal selection; do not evaluate epoch17 for selection and do not run epoch18～20.
- Data and reproducibility boundary: Resolved test reuses val, so this is a val baseline, not independent test. The user confirms PTH/pkl assets are preserved internally and accepts the 20260717 6V gate failure as incomplete clip start/end boundaries (`PASS_WITH_ACCEPTED_CLIP_BOUNDARY_EXCEPTION`); local paths/hashes and the gate report are not synchronized.
- S0/LR interpretation: Both 6V and S0 show training-loss/validation-metric decoupling, but S0 e6→e13 still improves mAP `0.3541→0.356425` while loss falls `0.9460→0.8474`; 6V e6→e16 degrades `0.453768→0.4192` while loss falls `0.6501→0.4212` and predictions halve. Treat 6V as stronger post-peak overfit/selection polarization, not unfinished convergence. Fix `force_resize` first, rerun the `8e-4` control, and test `4e-4` only if the same pattern recurs.
- Priority disposition: The post-`force_resize` 6V rerun is recorded in `N7_MONO_FRONT_SINGLE_FRAME_TODO.md` as P3 low priority. It must use a new experiment ID/work_dir and does not reopen or block the closed EXP-6V-B0 baseline.
- Config boundary: The run log resolves `AdamW2`, LR `8e-4`, weight decay `0.01` and backbone `lr_mult=0.1`. The repository dist config initially only overrode LR and would inherit standard Adam; another concurrent workstream later rebuilt it as AdamW2 with matching 20260717 data/schedule. This accuracy closeout did not modify the config, which remains a later reproducibility alignment rather than the original training snapshot; the log is the run-of-record.
- Documentation closeout: Synchronized the final e6/e2/e16 selection, accepted clip-boundary gate exception, internally preserved asset status, S0 comparison and post-`force_resize` LR decision across `N7_FASTBEV_EXPERIMENT_SUMMARY.md`, `N7_FASTBEV_EXPERIMENT_DETAILS.md`, and this handoff.

### 2026-07-23 N7 `force_resize` Two-Stage Geometry Local Closure

- Root cause: `RandomAugImageMultiViewImage` historically sampled train-time
  resize/crop/flip/rotate under `force_resize=True`, then overwrote every sampled
  value with fixed 704x256 stretch parameters. The native
  `intrinsic_width/height -> input_size` `post_rot` was correct and must remain;
  setting `force_resize=False` would instead pair native 1600x900 K with an
  identity post transform for cached 704x256 images.
- Implementation: `force_resize` now remains the deterministic base stage and
  `enable_random_aug` independently controls an optional train-only second
  affine. `None` preserves historical behavior (`force_resize` off -> random
  augmentation on; `force_resize` on -> off). When enabled, the random affine
  composes after the native-K base affine and updates image, `post_rot`,
  `post_tran` and `lidar2img` together. `n_images` defines actual camera slots;
  time-major views reuse parameters by camera slot, so 1x1, 1x4 and 6x4 require
  no fixed-six indexing. The disabled path intentionally retains the old
  discarded sampling call so subsequent NumPy RNG state also stays unchanged.
- Baseline/config contract: S0 single-frame, mono temporal and N7 6V train/test
  pipelines explicitly set `enable_random_aug=False` and their real
  `n_images`; the single-frame dist-train config inherits the same resolved
  settings. Evaluation remains deterministic even if the switch is true.
  `build_fastbev_lut.py` continues to reproduce only deterministic test
  geometry, validates native intrinsic sizes and checks pipeline/model
  `n_images` consistency.
- Regression coverage: Added frozen-legacy exact comparisons for image,
  `post_rot`, `post_tran`, `lidar2img`, normalized tensor, projected points and
  NumPy RNG state; fixed-seed reproducibility; min/max crop, flip and positive/
  negative rotation marker-projection checks; temporal camera sharing; dynamic
  1x1/1x4/6x4 organization; deterministic LUT equivalence; baseline resolved
  config checks; and a 6Vx4 disabled-path legacy comparison. Local
  `py_compile`, both new CLI help checks, `git diff --check`, and the complete
  `tools/tests` suite passed (`Ran 48 tests`, `OK`). The two NumPy 2
  `DeprecationWarning` messages come from the pre-existing Tensor-to-NumPy
  assignment in `img_transform` and do not change the exact comparison.
- AUG1 statistics: Added read-only
  `tools/data_converter/n7/analyze_n7_image_aug_targets.py`. It reproduces the
  current class/front-ROI/cam0 pinhole-visible selection, then uses the existing
  distortion-aware projection to report 704x256 target width/height/area,
  clipped area ratio and edge margins by class and distance. This workspace has
  no N7 train/val pkl, epoch13 PTH, `mmcv` or `mmdet`; therefore only synthetic
  tool regression was run. No real target statistic, AUG1 amplitude/config, or
  PTH/model result was fabricated.
- Next legacy step: Run the statistics tool on
  `data/N7_704_256/mono_front_pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_train_20260703.pkl`.
  Use the resulting car/truck 20-40m and 40-60m pixel/edge quantiles to choose
  conservative AUG1 ranges. Before any long training, run the retained
  `tools/compare_mono_front_dataset_production.py` against epoch13 from both the
  frozen baseline commit and this worktree with `--sample-index 0 --atol 0
  --rtol 0 --dump-tensors --strict`, then compare the dumped input, 2D feature,
  BEV input, raw cls/bbox/dir logits and decoded boxes. AUG1 must receive a new
  config/experiment ID/work_dir only after those real gates pass.
- Scope boundary: Distortion-aware GT filtering was not changed and remains a
  separate statistics-first task. No GEOM1, CBGS, anchor, NMS, LR, loss,
  checkpoint, ONNX, analyzer or board-reference change was mixed into this
  work. No commit or push was made.

### 2026-07-28 N7 `force_resize` Closeout Review Convergence

- Contract convergence (F1/F2): `MultiViewPipeline` now writes explicit
  `view_layout={n_images,n_times,sequential}` metadata after assembling the
  actual image list. `RandomAugImageMultiViewImage` validates that layout on
  every branch, detects explicit `n_images` conflicts and can infer camera
  slots from the metadata. `sample_augmentation()` now separates
  `is_train` (train/test size contract) from `randomize` (whether stochastic
  additions are sampled); `is_train=True, randomize=False` therefore keeps the
  deterministic train geometry and does not consume NumPy RNG state.
- Geometry/LUT convergence (F3/F6): The `force_resize` base stage now performs
  only the required native-size-to-input-size PIL resize and affine update;
  the redundant zero-pad/core-transform bypass was removed. The LUT builder
  explicitly warns when a train pipeline enables random image augmentation,
  because a fixed LUT reproduces deterministic test geometry only, and records
  `base_geometry_is_deterministic=true` in its metadata. The three N7 base
  configs keep augmentation explicitly disabled and declare their real camera
  counts; all three corresponding dist-train configs resolve to the same
  contract.
- Random-amplitude boundary: The S0 and mono-temporal configs currently use
  zero image-randomization amplitudes (`resize/crop/rot = 0`, `flip=False`), so
  changing only `enable_random_aug=True` would still be an identity transform;
  a future mono AUG1 config must set its reviewed non-zero amplitudes in the
  same isolated experiment change. The N7 6V 704x256 config is different: it
  retains the legacy non-zero 6V amplitudes but explicitly disables them, so
  turning on its switch alone would immediately change training and must not be
  treated as equivalent to the two mono configs.
- Frozen legacy evidence (T5): Removed the hand-written
  `_legacy_force_resize` oracle. The new
  `tools/tests/fixtures/force_resize_legacy_v1.json` was generated by AST-loading
  the real old class from commit
  `c1aea0d7c6beef930737bb3502d8c3aea7696d81`; the source-file SHA256 is
  `27c6da0c6a9b4662d28dcd8265066292919b0ec6a93702ef2fbf9dbaea157c52`.
  The fixture freezes image, affine/calibration matrices, normalized tensor,
  projected points, image shape and complete/next-value NumPy RNG evidence.
  It contains eight scenarios: 1x1, 1x4 and 6x4 under
  `force_resize=True`, each in train and test mode, plus the ordinary
  `force_resize=False` train and test paths. This is the full Cartesian set;
  the review text called it seven even though its listed cases add up to eight.
- Changed implementation/test files: `mmdet3d/datasets/pipelines/multi_view.py`,
  `mmdet3d/datasets/pipelines/transforms_3d.py`,
  `tools/data_converter/n7/build_fastbev_lut.py`, the three N7 base configs,
  `tools/tests/test_force_resize_image_augmentation.py`, and
  `tools/tests/fixtures/force_resize_legacy_v1.json`. The existing
  `tools/data_converter/n7/analyze_n7_image_aug_targets.py` and
  `tools/tests/test_n7_image_aug_target_stats.py` remain part of the earlier
  statistics-first force-resize workstream rather than this review delta.
- Local verification: Targeted `py_compile` passed; the focused force-resize
  suite passed 10/10; complete discovery passed 50/50 (`Ran 50 tests`, `OK`);
  both N7 CLI `--help` checks and `git diff --check` passed. Two NumPy 2
  `DeprecationWarning` messages still originate from the pre-existing
  tensor-to-NumPy assignment in `img_transform`; they do not affect exact
  comparisons. An independent direct AST comparison against the actual old
  class also found all eight frozen scenarios bit-exact, including subsequent
  NumPy RNG state.
- Cross-environment fixture portability: An internal legacy environment
  initially differed only on the derived `ordinary_6x4_train` projected-point
  SHA (`443353...` versus local NumPy 2.2.6 `358d17...`) while image,
  `post_rot/post_tran`, `lidar2img`, normalized tensor and RNG evidence matched.
  Reproduction showed that fixed-order scalar float32 accumulation yields the
  internal SHA and NumPy/BLAS `matmul` differs by at most
  `0.0001220703125 px`. The test-only `_project()` helper now uses fixed-order
  float32 accumulation and the fixture records that contract; all eight cases
  still pass when AST-loading the real class from `c1aea0d`, and the local
  focused/full suites remain 10/10 and 50/50. No production implementation or
  tolerance was changed.
- Real legacy-stack gate passed (T7): The user synchronized the candidate into
  the internal legacy environment, confirmed the complete suite reports
  `Ran 50 tests` / `OK`, and ran baseline-worktree versus current-worktree
  epoch13 comparisons with identical config, sample index 0, CUDA FP32,
  `--atol 0 --rtol 0 --dump-tensors --strict`. The epoch13 checkpoint SHA256
  was `897ae47de8889c4ff107e5cfa3f9b1d3f7a04f67362c8f1e050da7ec0dd851d7`
  and the validation pkl SHA256 was
  `ac19a123bd08a15fc9bc79b8829773d5428a41931d41a9ac5b6c783e498e0624`.
  Both pre and post commands exited 0, and the cross-stage comparator reported
  `T7 PASS`: dataset and production captures were bit-exact for input,
  `neck_fuse_0` feature, `neck_3d` input, every raw cls/bbox/dir logit tensor
  and decoded boxes. The discarded legacy RNG-sampling call (F7) remains in
  place; deleting it is outside this closeout and would separately change the
  cross-sample training RNG sequence.
- Deferred decisions: F4 remains the train-GT visibility contract after random
  crop/rotate. Reviewer measurement on 704x256 decomposed F5 into two
  independent pre-existing upstream mismatches: integer `resize_dims`
  truncation versus the floating analytic resize affine (measured maximum
  0.838 px, consistent with the theoretical 0.893/0.961 px bounds), and PIL
  `img.rotate()` NEAREST resampling versus the analytic `get_rot()` matrix
  (measured maximum 1.222 px at `r=1.0`). The new two-stage affine remained
  bit-identical to `c1aea0d` at the same amplitudes, so neither item is fixed in
  this closeout; resize truncation and rotate resampling must be evaluated as
  separate pre-AUG1 items. Ordinary `force_resize=False` temporal views still
  sample each frame independently to preserve upstream exact compatibility;
  choose any cross-time sharing policy explicitly before enabling that path for
  future temporal augmentation. If a later 1600x900 migration removes the
  `force_resize` branch, cross-time camera-slot sharing must be re-established
  explicitly or explicitly rejected before temporal random augmentation is
  enabled.
- Next phase boundary: Migrating cached front images from 704x256 to 1600x900,
  selecting resize/crop augmentation ranges and starting new GEOM experiments
  are deliberately not implemented here. They require new experiment IDs and
  work directories after the 704x256 T7 compatibility gate is closed. No
  commit or push was made.

### 2026-07-28 GEOM1-A Local Preparation And Internal Test Boundary

- Split-environment contract: This task intentionally uses the canonical local
  mirror at `/workspace/Fast-BEV_test_custom-fastbev-adapter` for code
  generation/review and the user's internal
  `/root/Fast-BEV_test_custom-fastbev-adapter` repository for data/model
  execution. The local mirror matches branch
  `feature/n7-mono-s0-optimization-v1` and commit `134d5f3`; N7 data, legacy
  `mmcv`/`mmdet` runtime, model assets and real outputs remain internal and are
  not expected to be locally accessible. The user will synchronize the exact
  task files, run the supplied gates internally and return reports/logs for
  review. Until those reports arrive, no real PKL, data-gate, F4,
  dataloader/model smoke, throughput or visualization result is claimed.
- Config preparation: Added independent `EXP-MONO-GEOM1-A` base/dist configs
  for native cam0 images. Both resolved train/test image transforms explicitly
  use `force_resize=False`, `enable_random_aug=False`, `n_images=1`,
  `n_times=1`, and 704x256 input. They re-declare the pipelines so config
  inheritance cannot retain the parent's cached `data_config`. The resolved
  geometry is exactly `1600x900 -> 704x396 -> crop(0,70,704,326)`,
  `post_rot=diag(0.44,0.44)`, `post_tran=[0,-70,0]`; test remains explicitly
  `test=val`. Model/ROI/voxel/anchor/classes, BEV augmentation, AdamW2
  `lr=1e-4`, schedule, 15 epochs and COCO initialization remain equal to S0.
  The isolated work dir is
  `work_dirs/n7_mono_1600_900_scale_crop/EXP-MONO-GEOM1-A`.
- Read-only gates/tooling: Added `compare_n7_pkl_migration.py` to hash PKL and
  manifests and reject token order/set, clip, timestamp, GT, K, distortion,
  extrinsic or any other non-image-path/size change. Added
  `analyze_n7_scale_crop_geometry.py` to compare direct stretch, center crop
  and multiple vertical offsets by car/truck, x-distance and yaw bins; it
  reports input/stride-4 target size, clipped ratio, edge margins, and exact
  train/eval keep-mask differences. Any ratio above its configured threshold
  is marked `F4_REVIEW_REQUIRED`. Added
  `visualize_n7_scale_crop_pipeline.py` to consume its deterministic far/edge/
  nonzero-yaw/crop-dropped selection and render native images through the real
  `RandomAugImageMultiViewImage`, not the qualitative LANCZOS comparison tool.
  Added `tools/benchmark_n7_training.py` to run a real legacy runner for one
  backward smoke iteration or warmup plus 50 measured iterations and archive
  data/iteration time, throughput and CUDA peak memory without starting an
  epoch-length or long training job.
- Local evidence: The existing production transform implementation was not
  modified. New regression coverage checks front_wide 1600x900 K, exact
  resize/crop/post affine, transformed image, lidar2img marker synchronization,
  train/test determinism, unconsumed NumPy RNG, LUT/test geometry and resolved
  S0-equivalent config contract. Focused tests passed 9/9; complete
  `tools/tests` discovery passed 59/59; targeted `py_compile`, all three new
  CLI help commands and `git diff --check` passed. The two existing NumPy 2
  warnings still come from tensor-to-NumPy assignments in `img_transform`.
- Required continuation: After the user synchronizes the task files into the
  internal `/root/...` repository containing `data/N7_1600_900`, copy/reuse the original
  `data/N7_704_256` manifests without auto-discovery or partial mode, generate
  only train/val PKLs, run strict data/geometry and migration gates, then run
  the F4 statistics. If GEOM1-A loses any material eval GT under the train
  crop, stop for an explicit F4 decision before model smoke or training. If F4
  passes, run legacy dataloader/forward/backward, 50-iteration S0 comparison
  and selected pipeline visualization. Do not launch the 15-epoch run without
  a subsequent user confirmation. No commit or push was made.

### 2026-07-28 `force_resize` Commit Confirmation And GEOM1-A Handoff

- Commit correction: The preceding closeout entry describes the state before
  approval. The user subsequently approved, committed and pushed the exact
  11-file scope as `134d5f3124e0d11f40dbcbade9e5bfe07c28ff96`
  (`修复 N7 force_resize 图像增强与几何同步`), with both author and
  committer set to `liujiaren19 <1334282612@qq.com>`. The current local branch
  and its origin both point to that commit.
- Closed status: The 704x256 compatibility work is complete, including the
  internal epoch13 T7 bit-exact result. Do not reopen its implementation or
  delete the retained F7 RNG-consumption call as part of GEOM1-A. The latter
  would alter cross-sample training RNG and requires a separately reviewed
  change if ever desired.
- Current active phase: The user reports that native 1600x900 cam0 images are
  already downloaded internally. Local base/dist configs, three read-only
  diagnostics and regression coverage are prepared but uncommitted; focused
  tests passed 9/9 and complete discovery passed 59/59. No real 1600x900 PKL,
  internal data/geometry/F4 gate report, legacy model smoke, 50-iteration
  profile, checkpoint or training log has yet been supplied. Therefore the
  project must describe GEOM1-A as locally prepared/internal execution pending,
  not as an active training run.
- Data contract: Generate one PKL set pointing to native 1600x900 images and
  reuse the frozen train/val manifests. Direct stretch and aspect-preserving
  scale-crop are online config choices, not separate PKLs or offline caches.
  Compare tokens, GT, calibration and split assignment against the 704x256
  baseline; expected differences are image paths, image width/height and
  related metadata only. Keep all old 704x256/golden assets.
- First experiment: Use S0 e13 as the existing stretch reference. Audit and
  use the prepared `force_resize=False, enable_random_aug=False` candidate implementing
  `1600x900 -> 704x396 -> vertical crop 704x256`. Before long training, pass
  strict data/geometry gates, quantify crop offsets and train/eval GT
  visibility, verify resolved train/test geometry, and measure dataloader/
  50-iteration throughput. Use a new experiment ID/work_dir and keep ROI,
  voxel, anchors, optimizer, LR, seed, schedule, initialization and split fixed.
- Decision rule: Do not start a parallel full 1600-online-stretch run. If the
  scale-crop result improves the required overall/range/production slices,
  adopt it without that extra ablation; run the stretch control only if the
  result is negative or ambiguous. Then address turning/lateral-yaw coverage
  as DATA1 and image augmentation as AUG1 in separate experiments.
- Parallel open gates: independent test, old-pkl `labels_without_frame`/empty
  GT disposition, synchronized lidar/image physical calibration,
  distortion-aware GT visibility, final asset hashes, board 4K preprocessing
  specification and real chip/runtime/INT8 validation remain open. The 6V-B1
  rerun is still P3 despite its force-resize prerequisite now being satisfied.
- Documentation synchronization: Updated
  `N7_FASTBEV_EXPERIMENT_SUMMARY.md`,
  `N7_FASTBEV_EXPERIMENT_DETAILS.md`,
  `N7_MONO_FRONT_SINGLE_FRAME_TODO.md` and this handoff to reflect the current
  commit and task ordering. No source/config/test file was changed and no
  commit or push was made by this documentation-only update.

### 2026-07-28 GEOM1-A Standalone Run-Of-Record Config

- Config flattening: Replaced the prepared six-level GEOM1-A inheritance chain
  with one complete formal training config,
  `configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18_n7_1600x900_scale_crop_dist_train.py`.
  It has no `_base_` assignment and explicitly freezes the full model, N7
  dataset/pipelines, GEOM1-A image contract, optimizer/LR schedule, 15-epoch
  limit, COCO initialization, FP16/runtime, seed 0, experiment ID and isolated
  work dir. The old S0/6V/paper configs and the committed `force_resize`
  implementation were not modified.
- Migration evidence: Before flattening, the nested resolved config was frozen
  with full SHA256
  `d1c6078bf4ade8954dcb722448732dc7bdb9be8a248f13b7482de6967995f9b4`.
  Its 28 effective model/data/pipeline/optimizer/runtime fields have SHA256
  `dcfd9882e1e8d2f75348d8d693d0bc411adbb7930e47be306da18b8c67d961d4`;
  the standalone config produces the same effective SHA exactly. The unused
  inherited `custom_dataset_common` orphan still pointed to `N7_704_256` and
  was intentionally not copied; every active standalone dataset path points to
  `N7_1600_900`.
- Candidate cleanup: Removed the untracked intermediate
  `custom_fastbev_mono_front_single_frame_r18_n7_1600x900_scale_crop.py`.
  Internal synchronization must also delete that obsolete file if an earlier
  preparation copy is present. Training, eval, LUT, visualization and smoke
  commands should all use the standalone `...scale_crop_dist_train.py` file.
- Regression update: `tools/tests/test_n7_scale_crop_geometry.py` now enforces
  no `_base_` assignment, absence of the obsolete intermediate config and
  stale 704x256 helper, exact effective resolved SHA equivalence, explicit
  seed 0, and the existing deterministic geometry/LUT/data-path contract.
- Local verification: Targeted `py_compile` passed; the focused GEOM1-A suite
  passed 8/8; complete `tools/tests` discovery passed 60/60; tracked
  `git diff --check` passed and task-untracked no-index whitespace checks were
  clean. The two existing NumPy 2 `DeprecationWarning` messages remain in the
  unchanged `img_transform` tensor-to-NumPy assignments.
- Remaining boundary: No internal data/model operation was run by this local
  flattening. PKL migration, strict data/geometry/F4 gates, legacy dataloader,
  forward/backward, 50-iteration throughput and selected real visualization
  still require the user's internal repository and assets. Do not start the
  15-epoch GEOM1-A training before those results are reviewed and the user
  explicitly confirms it.

### 2026-07-28 GEOM1-A Final Cleanup Reminder

- User-required terminal step: After the complete GEOM1-A data gates, smoke,
  throughput, 15-epoch training, evaluation and conclusion archive are
  finished, perform a dedicated cleanup audit before closing the task.
- Cleanup candidates: Generated Python/test caches, synthetic fixtures that
  are no longer regression evidence, one-off diagnostics, superseded test
  helpers, temporary visualization selections/images, benchmark scratch
  work_dirs and other reproducible intermediate data that have no continuing
  product or audit value.
- Mandatory preservation: Do not delete or overwrite the native 1600x900
  source data, final train/val PKLs and manifests, strict gate/migration/F4
  reports and hashes, resolved run-of-record config, selected visual evidence,
  throughput summaries, training logs/checkpoints, S0 epoch13 golden, 9797 or
  board evidence, or the old 704x256 baseline assets.
- Safety rule: Classify every candidate as retain/archive/delete, show the
  exact paths, sizes and recoverability to the user, and wait for explicit
  deletion approval. Do not recursively clean a broad data/work_dirs root and
  do not mix unrelated dirty/untracked files into this cleanup.

### 2026-07-29 GEOM1-A Native-Image Data Gate

- Internal PKLs generated from the frozen 704x256 train/val clip manifests
  contain 200874 train infos and 24909 val infos. Train/val token and clip
  overlap are both zero; camera/image/intrinsic metadata is consistently cam0
  and 1600x900, and the 200-sample geometry checks for both splits report zero
  failures and zero maximum error.
- The strict validator intentionally retains `FAIL` because converter
  provenance records 851 train plus 90 val labels at clip boundaries without
  a corresponding image. These counts match the known legacy condition and
  were explicitly accepted by the user for this experiment; the 1237 dropped
  empty-GT train labels remain a recorded warning pending the old/new PKL
  equality check. Do not edit or relabel the raw strict report as PASS.
- The user confirmed the completed `--check-image-files` scan checked all
  200874 train and 24909 val PKL images, with zero missing files, zero JPEG
  header-size mismatches and zero decode errors. The experiment-level result is
  therefore `PASS_WITH_ACCEPTED_CLIP_BOUNDARY_EXCEPTION`; rerunning the costly
  full image scan is unnecessary.
- Next hard gate is the strict old-704/new-1600 PKL migration comparison. Long
  training remains blocked until migration equality, F4 train/eval visibility,
  real-pipeline visualization, legacy forward/backward smoke and 50-iteration
  throughput checks are reviewed.

### 2026-07-29 GEOM1-A F4 Full-Data Analyzer Scaling

- The initial read-only F4 analyzer retained one Python dictionary for every
  visible GT under every resize/crop strategy and repeated projection work for
  each offset. At the native train/val scale this could create tens of millions
  of rows, excessive memory/CSV output and avoidable repeated projection cost.
- `analyze_n7_scale_crop_geometry.py` now computes eval/train keep masks and yaw
  bucket counts over every selected frame, projects pinhole centers/corners once
  per frame for reuse by all contracts, and projects distortion-aware box-edge
  samples once per sampled metric frame. Pixel-size/crop/margin quantiles use a
  deterministic frame-stride sample (`--metric-stride`, default 100); reports
  state exact population counts and sampled metric counts separately.
- The strict global F4 decision is explicitly bound to the run-of-record
  `scale_crop_top_70` strategy. Other requested offsets remain visible as
  `INFO_DIFFERENCE` rows but cannot falsely fail the center-crop candidate.
- This is an analysis-only scalability change. It does not alter dataset,
  training/eval filter, image transform, PKL, config or model behavior. A new
  regression proves batched edge metrics equal the original single-box method
  and that visibility uses all frames while pixel metrics are sampled.
- Verification after the change: focused GEOM1-A tests passed 10/10, complete
  `tools/tests` discovery passed 62/62, `py_compile` and no-index whitespace
  checks passed. Internal execution must sync the updated analyzer (and test if
  desired) before launching the formal F4 scan.

### 2026-07-29 GEOM1-A Migration PASS And F4 Review

- Internal strict old-704/new-1600 migration comparison completed with
  `N7_PKL_MIGRATION_COMPARE=PASS failures=0`. The new native-image PKLs are
  therefore accepted as label/token/calibration/split preserving; retain the
  generated migration report and hashes.
- The optimized F4 run processed all 225783 infos for exact visibility/yaw and
  2259 deterministic metric frames, producing 104660 sampled pixel records.
  The raw zero-tolerance status is `F4_REVIEW_REQUIRED` for the gated
  `scale_crop_top_70` contract: train loses 101 of 958707 eval-visible car/truck
  ROI targets (0.0105%) and val loses 10 of 85239 (0.0117%). Combined mismatch
  is 111/1043946 = 0.01063%, or 99.98937% keep-mask agreement.
- Direct stretch has exact train/eval visibility agreement. No tested crop
  offset has exact agreement; top=140 has the smallest combined mismatch (55)
  but would move the native principal point from crop-output y=128 to y=58 and
  violate the frozen centered GEOM1-A geometry. Do not select it based on only
  56 fewer edge targets, and do not modify eval GT visibility to manufacture a
  PASS.
- Current recommendation is to preserve center crop and classify the 0.01063%
  difference as a bounded edge-crop exception only if exact class/distance/yaw
  buckets show no material truck, middle/far-range or lateral-yaw concentration.
  Long training remains paused at this F4 decision until that distribution is
  reviewed and the exception is explicitly recorded.
- Exact F4 bucket follow-up found all 111 top=70 eval-only targets in 0-20m:
  65 car and 46 truck; 107 have yaw in [-45,45] degrees, with only two each in
  [-90,-45] and [135,180]. There are zero 20-80m losses and no lateral-yaw
  concentration. Two deterministic pixel-metric samples are at x=0.45m and
  x=1.54m, y about 4.5m; their box edges cross the camera near plane and produce
  extremely negative distortion-aware margins. Classify the experiment-level
  result as `PASS_WITH_ACCEPTED_NEAR_FIELD_CROP_EXCEPTION`, while preserving
  the raw zero-tolerance `F4_REVIEW_REQUIRED` report unchanged. Continue with
  real train/test pipeline visualization and smoke; do not alter GT filtering,
  eval visibility, crop offset or the centered GEOM1-A config.
- Internal real-pipeline visualization then passed for both paths. The train
  selection contained 40 each of far/top-edge/bottom-edge/nonzero-yaw and two
  sampled crop-dropped targets; rendering used eight per category and produced
  the expected 34 images. Val contained 40 in each non-dropped category and no
  sampled dropped target, producing the expected 32 images. Both commands
  reported `N7_SCALE_CROP_PIPELINE_VIS=PASS`, confirming the resolved real
  `RandomAugImageMultiViewImage` can read native images and synchronously render
  the selected post-crop projections. Preserve both visualization reports and
  selected output images as gate evidence.
- Internal one-GPU 10-warmup/50-measure training benchmarks both passed on a
  separate single-card 20GB test machine at batch 64/workers 8; these were not
  run on the target four-L20 training host. S0 reports iter mean 11.810291s,
  data mean 0.094683s and
  peak allocated memory 9926.1 MiB; GEOM1-A reports 13.276409s, 0.199413s and
  9929.3 MiB. Thus end-to-end iter+data rises 11.904974s -> 13.475822s
  (+13.19%), corresponding throughput falls 5.3759 -> 4.7492 samples/s
  (-11.66%), while memory changes only +3.2 MiB (+0.032%). Wall time rises
  13m08.9s -> 15m04.7s (+14.68%). Treat the relative A/B result as a valid
  correctness/overhead gate for that machine, but do not extrapolate its
  absolute wall time to four L20s. Target-host CPU/JPEG/storage and DDP behavior
  must be measured from its initial 50-100 iterations. Concurrent native JPEG
  decode/resize worker CPU contention can also lengthen the timed model section
  even though post-pipeline tensor shapes are identical.
- User selected the target four-L20 run-of-record directory
  `work_dirs/n7_mono_1600_900_scale_crop/EXP-MONO-GEOM1-A/20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260729`.
  Keep the inherited S0 `fp16=dict(loss_scale='dynamic')` contract even though
  target memory is sufficient; disabling FP16 would change numerical precision,
  loss scaling, Tensor Core execution and speed in addition to image geometry.
  Any FP32 diagnosis must be a separate, matched S0/GEOM control rather than a
  mutation of the GEOM1-A run-of-record.
- The formal four-L20 GEOM1-A run has started in that directory with per-GPU
  batch 64, workers 8 and the retained dynamic FP16 contract. Initial logging
  converged from the expected dataloader cold start (`data_time=2.6s`) to
  0.055-0.099s, while iter time stabilized near 8.35s; framework memory reports
  9986 MiB and `nvidia-smi` reports 13412 MiB per process/device including CUDA,
  NCCL and reserved/cache overhead. Early ETA converged toward roughly 1 day
  6-8 hours. Treat the run as healthy unless loss/grad/loss-scale or utilization
  evidence says otherwise.
- Do not use spare memory on the same four GPUs to overlap another distributed
  training job. GPU compute/memory bandwidth, NCCL, host JPEG workers and storage
  would contend, invalidating timing and increasing failure risk. A second run
  is acceptable only on separately isolated GPUs/host. Also do not schedule a
  confounded `force_resize + image augmentation` experiment: first evaluate
  GEOM1-A; if negative/ambiguous, use native online stretch with image random
  augmentation still disabled, or if GEOM1-A wins, test AUG1 on the selected
  geometry. If middle/far improves but lateral yaw remains weak, prioritize DATA1.
- Historical-session audit for the user's batch-size question found prior
  four-GPU 6V-temporal experiments that raised per-GPU batch 24 -> 28 -> 30.
  Batch 28 logged time 23.125s, data_time 2.015s and memory 31225 MiB; batch 30
  still logged 20.686s after warmup with memory 33430 MiB, versus roughly 16s
  for the optimized batch-24 path. The larger batches did not improve
  throughput. This workload is heavier than mono S0, so it is evidence against
  assuming spare memory implies speed, not a direct mono scaling curve.
- Keep the active GEOM1-A run at 64/GPU (global 256), matching the S0 epoch13
  golden and LR 1e-4. Changing it would require restarting, reduce optimizer
  steps per 15 epochs, change the fixed 1000-iteration warmup fraction and
  potentially require a separately validated LR/schedule. If future throughput
  optimization is desired, benchmark larger batches in scratch runs on the same
  target host and compare samples/second; do not mutate or resume the formal run
  with a different batch.
- Keep the active run's 15-epoch limit unchanged. Because its poly LR hook is
  iteration-based (`by_epoch=False`), changing max epochs to 20 changes max
  iterations and therefore changes the LR values throughout epochs 1-15; a
  20-epoch run cannot be made comparable by simply ignoring epochs 16-20. The
  active process has already resolved 15 epochs, so editing the config would not
  affect it, while stop/restart or resume under a new horizon would introduce a
  different schedule or an LR discontinuity. Evaluate all 15 checkpoints first.
  Only if canonical and required range metrics are still establishing new bests
  at epoch15 should a separately named five-epoch extension/fine-tune schedule
  be designed; do not mutate GEOM1-A in place. Historical S0 peaked at epoch13
  and used epoch15 only as terminal archive, so there is no current evidence for
  a blind 20-epoch extension.

### 2026-07-29 GEOM1-A Documentation Sync And Training Watch

- Synchronized `N7_FASTBEV_EXPERIMENT_SUMMARY.md`,
  `N7_FASTBEV_EXPERIMENT_DETAILS.md` and
  `N7_MONO_FRONT_SINGLE_FRAME_TODO.md` to the completed migration/data/F4/
  visualization/50-iteration gates and the active four-L20 training state.
  The documents distinguish the preserved raw strict statuses from the two
  experiment-level accepted exceptions; they do not relabel either raw report
  as an unconditional PASS and do not claim model accuracy before validation.
- Current wait boundary: leave the run-of-record process/config unchanged for
  approximately one day. On the next review, first collect process health,
  continuous log tail, current epoch/iter, LR, loss components, grad norm,
  dynamic loss-scale/overflow evidence, checkpoint list/sizes, free disk and GPU
  utilization. Then validate available checkpoints on the independent eval
  host and compare canonical mAP, BEV@0.5, mATE/mAOE/mASE, class metrics and
  0-80m Recall@2m against S0 epoch13 plus the epoch10 range challenger.
- Do not launch another job on the same four GPUs, change batch/FP16/epoch
  horizon, or clean any test/data/report/checkpoint evidence while the run is
  active. Cleanup remains a separate post-experiment, user-approved inventory
  operation. No commit or push was performed for this documentation update.

### 2026-07-29 N7 Eval GT Population And Velocity Metrics

- Added nuScenes-style velocity evaluation for the N7 9D box contract
  `[x,y,z,l,w,h,yaw,vx,vy]`. `mAVE@2m`/official alias `mAVE` is the
  class-balanced mean `||v_pred-v_gt||2` over the existing unique
  center-distance TP matches <=2m; no second velocity-specific matching is
  introduced. vx/vy follow the Fast-BEV lidar frame (x forward, y left), and
  errors are reported in m/s.
- Dataset metrics now include overall and per-class AVE, velocity TP coverage,
  velocity-L2 p50/p90, vx/vy MAE p50/p90 and signed bias, plus the same fields
  for the existing 0-20/20-40/40-60/60-80m GT-x buckets. Missing 7D or
  non-finite velocity fields are excluded and exposed by `velocity_tp`; they
  are not silently treated as zero error. The metric schema is bumped to v3.
- `eval_summary.md` now writes the shared eval GT population near the top,
  includes `mAVE@2m`, `eval_gt` and `eval_det` in NuScenes-Like Overall, adds
  car/truck GT beside detection counts, and adds per-class/range velocity
  absolute-error and signed-bias tables. CSV/JSON gain matching fields and a
  new `eval_velocity_summary.csv`.
- Local verification: focused velocity tests passed 3 plus one legacy
  integration test skipped because local modern CUDA environment lacks mmcv;
  complete `tools/tests` discovery passed 66 tests with that one expected
  skip. Relevant `py_compile` and whitespace checks must remain green after
  final diff review. The two pre-existing GEOM1-A tests were normalized to the
  user-selected full run work_dir without changing the training config.
- Internal action: sync `mmdet3d/datasets/custom_multiview_dataset.py`,
  `tools/eval_epoch_checkpoints.py`, `tools/n7_eval_velocity.py`,
  `tools/tests/test_n7_eval_velocity.py` and the adjusted
  `tools/tests/test_n7_scale_crop_geometry.py`; restart only the eval watcher,
  not training. Recompute epoch1 metrics from the existing result pkl with
  `--epochs 1 --skip-inference --rerun-eval`; no inference rerun is required.
  Confirm `eval_gt=85239`, finite car/truck velocity coverage and non-empty
  mAVE/detail tables before treating the feature as internally validated.

### 2026-07-29 N7 Eval Velocity Preservation And Watcher Argument Fix

- Internal epoch1 re-evaluation exposed an implementation issue rather than
  proving that the historical result pkl lacked velocity: N7 eval called the
  shared box-origin converter with `box_dim=7`, so it discarded prediction
  `vx/vy` even when `bbox3d2result` had preserved the model's 9D boxes.
  `_parse_det_result` now retains all available dimensions while the shared
  helper keeps its historical 7D default for other geometry callers.
- The epoch2 watcher failure happened before model construction because it
  forwarded malformed `--cfg-options data.test.samples_per_g`; MMCV requires
  every item to be `key=value`. The formal config already has
  `data.test.samples_per_gpu=8`, so the override is unnecessary. The watcher
  now validates cfg/eval options before entering its polling loop, preventing
  repeated checkpoint retries for the same CLI typo. The HAMI 48GB/20GB limit
  inconsistency message was not the exit cause in this trace, though it should
  still be watched after inference actually starts.
- Regression coverage now checks malformed cfg-option rejection and, in the
  legacy integration environment, prediction 9D velocity preservation through
  `_parse_det_result`. Local focused tests passed 4 with one expected legacy
  skip; complete `tools/tests` discovery passed 67 with that one skip;
  `py_compile` and `git diff --check` passed.
- Internal recovery: sync the updated dataset/eval script (and velocity helper
  plus regression test if they are not already present), stop only the failed
  eval watcher, rerun epoch1 evaluation from its existing result pkl, then
  restart watch mode without `--cfg-options`. If epoch1 still reports zero
  velocity TP after this fix, inspect the stored prediction box dimension;
  only a genuinely 7D result pkl requires inference to be rerun.

### 2026-07-31 GEOM1-A Final Result And Next Optimization Route

- Completion: The formal four-L20 GEOM1-A run completed all 15 epochs and all
  checkpoint evaluations. The synchronized log ends with the epoch15
  checkpoint save; `eval_summary.md` contains a fixed `eval_gt=85239` for every
  epoch and selects epoch11 by canonical mAP.
- Result: e11 has mAP `0.381993`, BEV@0.5 `0.4016`, mATE `0.8615m`, mAOE
  `4.6376°` and mASE `0.2113`. Against S0 e13, mAP is `+0.025568` and BEV is
  `+0.0243`; car 40-60/60-80m Recall rises `+0.0425/+0.0314`, and truck rises
  `+0.0531/+0.0677`.
- Boundary: Orientation does not improve. Overall AOE worsens `+0.5831°`;
  car/truck AOE worsen `+0.4641°/+0.7020°`. Existing range x-MAE is conditional
  on <=2m TP matches and cannot rule out the reported one-body-length depth
  error among unmatched cases.
- Frozen roles: e11 is canonical and the next fine-tune candidate; e13 is the
  GEOM orientation/scale challenger (`mAOE=4.4920°`, `mASE=0.2045`); e15 is
  terminal. Do not create EXT5. Keep S0 e13 as production fallback until
  targeted production and board regression pass.
- Next route: first add yaw-binned/production evaluation, then prioritize
  `DATA1-CITY-YAW` and `DATA3-BANKED-RING` real GT. Run `AUG1` independently on
  the selected base. Keep BEVFusion endurance-road data as explicitly managed
  `DATA2-ENDURANCE-PSEUDO`, after real-GT work.
- Documentation synchronization touched only the four canonical Markdown
  files. No source/config/test file was changed and no commit or push was made.

### 2026-08-01 GEOM1-A Submission Boundary And Reproducibility Audit

- Git baseline: `HEAD` and `origin/feature/n7-mono-s0-optimization-v1` are both
  `134d5f3124e0d11f40dbcbade9e5bfe07c28ff96`; the commit exists locally and on
  the remote, and is the verified `force_resize` closeout. The worktree still
  contains mixed historical changes, so no broad staging or cleanup is allowed.
- Actual experiment contract: the standalone run-of-record config is
  `configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18_n7_1600x900_scale_crop_dist_train.py`,
  SHA256 `d134b0bdbc4af0ec3aea391b884c20f46e792248eceecefa742f0639992b89fc`.
  The prior `8b4ecfdd...439e` value was a pre-run version before the unique
  work_dir was written and is no longer the final config hash. The run is
  `n_images=1,n_times=1,sequential=False`, not temporal `1x4`; it uses native
  1600x900 images, 704x396 resize, crop `(0,70,704,326)`, final 704x256,
  `force_resize=False` and `enable_random_aug=False`.
- Reproduction evidence: the formal command used `CONFIG=<standalone config>`,
  the unique `WORK_DIR`, `TRAIN_BATCH=64`, `EVAL_BATCH=8`, `WORKERS=8`,
  `NO_VALIDATE=1`, and `bash tools/dist_train_ljr.sh` from
  `/mnt/liujiaren/fastbev-python-custom-fastbev-adapter`. The resolved log
  confirms four L20s, seed 0, 15 epochs, the 20260728 native train/val PKLs and
  the COCO Cascade Mask R-CNN R18/FPN initialization path.
- Locally verified assets: the training log SHA256 is
  `1abbd30779fe65b6ff5e255e5eb3751c61427f29ee86f45753395f8b3ee9079b`, the
  `eval_summary.md` SHA256 is
  `8e75a795c571f8e59ca468300232be2184ba20cb542d2b789f7943cf5ceb30f0`, and the
  retained `data_gate.md` SHA256 is
  `aa012f55c7315b98dc55aa4fe9c92b5501fabaf0c590734aba846eb15d4aeef6`.
  e11/e13/e15 PTH, per-epoch result/metrics/log files, train/val PKLs,
  pretrained weights, calibration and cached images are absent locally; their
  existence/hash must be filled from the internal asset store and was not
  fabricated from naming conventions.
- Proposed source boundaries: put the standalone config, formal F4 analyzer and
  scale/crop regression in the core commit because the regression imports the
  analyzer. Put strict migration comparison/test, real-pipeline visual gate and
  training benchmark in the data/gate commit. Keep the dataset 9D-box change,
  epoch evaluator, velocity helper and its tests together in a separate
  velocity commit. Put the four canonical experiment documents in the final
  result commit.
- Exclusions: `AGENTS.md`, `.claude/`, `CLAUDE.md`, inference Unicode-output
  changes, BST/vendor/debug files, Ozone download, destructive-capable camera
  alignment cleanup and qualitative resize-strategy comparison are not part of
  the GEOM1-A submission. No Phase-B data, PKL or training config was modified
  or generated during this audit. No file was staged and no commit was created.
- Submission order was subsequently changed by the user to an internal-first
  rule for every future commit. The exact candidate scope must first be synced
  to the internal canonical repo, validated in the legacy/real-data environment,
  reviewed and committed there. Only the internally committed final contents
  may then be synced back and committed in this local mirror. This container
  currently cannot access the known internal repo paths, so it must remain
  before `git add`/`git commit` until the internal result is returned.

### 2026-08-01 GEOM1-A Internal-First Submission Completion

- Internal validation: the user synchronized all non-Markdown candidate files
  to `/mnt/liujiaren/fastbev-python-custom-fastbev-adapter`. `py_compile`
  passed; PKL migration pytest passed 2/2; velocity pytest passed 5/5 including
  the legacy integration path; and complete unittest discovery passed 68/68.
  Geometry pytest initially exposed accidental collection of imported helper
  `test_image_aug_params`; aliasing it to `_test_image_aug_params` preserved the
  ten intended tests and the user confirmed the rerun passed.
- Internal submission: the user confirmed all intended internal changes were
  committed before local submission. Codex/Claude conversation-oriented
  Markdown is intentionally local-only and is not part of internal sync or
  internal commits.
- Local mirror commits completed before final documentation/push:
  `230d5ae` freezes the standalone GEOM1-A config, F4 analyzer and geometry
  regression; `e8fa7d5` adds migration and training-gate tools; `23fa1b7`
  records session naming and internal-first submission rules; `36945bd` adds
  epoch evaluation/velocity metrics; and `7f148f4` adds Unicode-safe inference
  output naming and regression coverage.
- Approved divergence: the internal evaluator and inference script contain
  additional usage examples. The user reviewed those diffs and explicitly
  approved leaving them internal-only instead of copying them back. The local
  functional implementation and tests remain the locally validated versions;
  exact file equality is not claimed for those approved example-only changes.
- Scope integrity: unrelated data cleaning/download, qualitative resize
  comparison, BST/vendor/debug, local Claude configuration and generated model/
  data/work-dir assets remain untracked or unstaged. No Phase-B data, PKL or
  training config was created before Phase-A submission.
