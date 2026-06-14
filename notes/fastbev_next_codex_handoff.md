# Fast-BEV Next Codex Handoff

This document is the concise handoff for the next Codex session. It extracts the
current effective conclusions from the recent discussion and re-prioritizes the
work. Prefer reading this file first, then inspect `git log --oneline -3` and the
specific files mentioned below. Do not reread the full chat unless blocked.

## Repository State

```text
repo: /root/autodl-tmp/Fast-BEV
branch: test/custom-fastbev-adapter
latest pushed commit: 224b20e Add BEVFusion pseudo label converter
previous commit: e41d058 Validate front monocular Fast-BEV smoke path
```

The latest pushed branch contains:

```text
configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_1iter.py
configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_debug.py
tools/data_converter/bevfusion_pred_to_custom_labels.py
notes/fastbev_custom_handoff.md
```

Current AutoDL instance may be no-GPU. Training work should be done on a GPU
machine. The no-GPU instance is still useful for code inspection, converter
work, and documentation.

## Updated Strategic Priority

The original three goals should now be ordered and scoped as follows.

### Priority 1: Real Six-Camera Fast-BEV To Front-Monocular Fast-BEV

This is still the highest-priority technical target, but the current state is
only a smoke-path adaptation. The previous work proved that the pipeline can run
with only `CAM_FRONT` on nuScenes mini, but it did not complete the model-side
monocular design.

Current completed part:

```text
Dataset/config now can feed only CAM_FRONT.
Training input can be 1 camera x 4 temporal frames instead of 6 cameras x 4.
A 1-iter training and 1-sample inference smoke test succeeded on nuScenes mini.
```

Important limitation:

```text
Only the data/config side was adapted. The Fast-BEV view transform / voxel
projection code was not redesigned for a front-only camera model.
```

Specifically, the current Round 1 configs still use the inherited full BEV grid:

```text
point_cloud_range = [-50, -50, -5, 50, 50, 3]
n_voxels ~= 200 x 200 x 4
voxel_size ~= 0.5 x 0.5 x 1.5
```

With only `CAM_FRONT`, Fast-BEV still builds/fills the full BEV volume. Only
voxels that project into the front camera frustum receive image features;
side/rear areas have no real image evidence but may still be part of the loss
unless labels are filtered.

Next required work for Priority 1:

```text
1. Decide the front-monocular BEV training region.
   Candidate: x_forward 0..80 or 0..100 m, y_left -35..35 m, z as before.

2. Add a front ROI config, not just camera filtering.
   Update point_cloud_range, n_voxels, voxel_size if needed, anchor ranges,
   object range filters, and eval/test ranges consistently.

3. Filter GT to front-camera-visible or at least front-ROI boxes.
   Do not train mono-front on full 360 labels; missing side/rear image evidence
   will create bad supervision.

4. Verify the model-side backprojection behavior.
   Inspect the Fast-BEV view-transform/backprojection code and confirm that
   `n_images=1` produces correct volume shapes and masks. Add lightweight shape
   assertions or debug prints if needed.

5. Run a real small training experiment.
   Use nuScenes mini full train split first, not just 1 sample. Target is not
   high AP; target is stable loss, non-empty predictions, and sane geometry.
```

The next Codex should not claim monocular adaptation is complete until the
front ROI / GT filtering / model-side shape checks are done.

### Priority 2: Adapt 9797 BEVFusion Pseudo Labels To The Custom Data Interface

The immediate inserted business task is to use BEVFusion pseudo labels generated
from a separate 9797 collection vehicle and feed them into the existing custom
Fast-BEV data interface.

User-provided expected data layout on the internal/GPU machine:

```text
data/gt/20260514_9797/20260514103014_1.dat_img/   # 5000+ front images
data/gt/20260514_9797/name.pkl                    # image names corresponding to pred.pkl
data/gt/20260514_9797/pred.pkl                    # BEVFusion predictions
data/gt/20260514_9797/9797_3.txt                  # 9797 front camera/LiDAR calib txt
```

A converter was added:

```text
tools/data_converter/bevfusion_pred_to_custom_labels.py
```

Purpose:

```text
name.pkl + pred.pkl -> custom 3d_od JSON labels
```

The converter supports common BEVFusion/MMDet3D result structures:

```text
boxes_3d / bboxes_3d / boxes / bboxes
scores_3d / scores
labels_3d / labels
nested pts_bbox
nested pred_instances_3d
```

It extracts only the basename from paths in `name.pkl`, because the stored paths
may not match the local `dat_img` directory.

Default temporary coordinate conversion in the converter:

```text
source: BEVFusion/MMDet3D LiDAR frame, x front, y left, z up
output: current custom raw label frame, x left, y rear, z up

raw_x_left = fastbev_y_left
raw_y_rear = -fastbev_x_front
raw_z_up = fastbev_z_up
raw_yaw = normalize(fastbev_yaw - pi/2)
```

This conversion is a temporary engineering approximation. It assumes the 9797
LiDAR frame can be treated as equivalent to the N7 LiDAR-label frame after axis
conversion. That is not physically rigorous.

First command on the internal machine should be an inspect/dry-run:

```bash
python tools/data_converter/bevfusion_pred_to_custom_labels.py \
  --name-pkl data/gt/20260514_9797/name.pkl \
  --pred-pkl data/gt/20260514_9797/pred.pkl \
  --image-dir data/gt/20260514_9797/20260514103014_1.dat_img \
  --calib-txt data/gt/20260514_9797/9797_3.txt \
  --out-dir data/gt/20260514_9797/output/20260514103014_1/3D_OD/lidar \
  --score-thr 0.3 \
  --front-range 0 100 -35 35 \
  --inspect \
  --dry-run
```

If the pkl structure is compatible, remove `--dry-run` to write labels.

Required validation before training:

```text
1. Inspect pkl structure and verify boxes/scores/labels are parsed correctly.
2. Generate a small subset of JSON files first.
3. Visualize several frames by projecting/plotting boxes.
4. Check box centers, dimensions, yaw direction, and global offset.
5. Only then generate all labels and feed them to the custom converter/training.
```

Critical caveat:

```text
9797 data appears to have front images only. If so, do not train the original
six-camera model unless the other camera inputs exist or are explicitly handled.
Use mono-front training for this data, or adapt the data loader to a valid camera
set.
```

### Priority 3: Unify All Data Into A Rear-Axle-Ground Ego Coordinate System

This remains important but should not be mixed into the first two validation
steps. It is the final coordinate cleanup round.

Target final convention:

```text
origin: rear axle center projected to ground
x: forward
y: left
z: up
```

Why this is needed:

```text
N7 labels currently use the N7 top LiDAR origin with x left, y rear, z up.
9797 BEVFusion pseudo labels use the 9797 top LiDAR/ego origin, likely x front,
y left, z up.
These are not the same physical frame. Axis conversion alone does not account
for LiDAR mounting offset, height, yaw bias, or vehicle reference point.
```

Do not pretend these frames are identical in final experiments. The temporary
approximation is acceptable only to answer: "Can BEVFusion pseudo labels enter
Fast-BEV and provide a useful training signal?"

Final coordinate migration should require per-vehicle extrinsics:

```text
T_N7_lidar_to_rear_axle_ground_ego
T_9797_lidar_to_rear_axle_ground_ego
camera_to_ego for each vehicle/camera
```

Then both N7 and 9797 labels should be represented in the same ego frame before
mixed training/evaluation.

## Recommended Next Execution Plan

### Step A: Finish Priority 1 Model-Side Mono Adaptation

On a GPU machine:

```text
1. Pull latest branch.
2. Run the existing debug config to verify current behavior.
3. Add front ROI config and GT filter.
4. Add shape checks for n_images=1 and temporal frames.
5. Train on nuScenes mini for a few epochs.
6. Verify non-empty predictions and reasonable front-scene geometry.
```

Existing debug config:

```text
configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_debug.py
```

### Step B: Validate 9797 Pseudo Label Conversion

On the internal machine with real files:

```text
1. Run converter with --inspect --dry-run.
2. Fix parser if pred.pkl has unexpected structure.
3. Generate labels for a small subset.
4. Visualize boxes against front images / BEV.
5. Generate all JSON labels only after geometry looks correct.
```

### Step C: Decide Training Target For 9797

If 9797 has only front images:

```text
Use mono-front model path first.
```

If 9797 has six synchronized cameras and calibration:

```text
It can be used for six-camera Fast-BEV, but use 9797 calibration and document
that labels are in the 9797 local frame unless/until ego migration is done.
```

### Step D: Postpone Unified Ego Migration

Only after pseudo labels show value:

```text
1. Collect/verify N7 and 9797 LiDAR-to-ego extrinsics.
2. Define exact rear-axle-ground ego frame.
3. Convert labels, camera extrinsics, anchors/ranges, and visualization tools.
4. Re-run both mono and multi-camera training under the unified frame.
```

## Files To Inspect First

```text
notes/fastbev_custom_handoff.md
tools/data_converter/custom_fastbev_converter.py
tools/data_converter/bevfusion_pred_to_custom_labels.py
configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_debug.py
mmdet3d/datasets/nuscenes_dataset.py
mmdet3d/datasets/nuscenes_monocular_dataset.py
```

## Current Known Risks

```text
1. Current mono adaptation is not complete model-side adaptation.
2. 9797 pseudo-label coordinate conversion currently assumes equal LiDAR origins.
3. 9797 appears front-only; original six-camera training may not be directly valid.
4. pred.pkl structure has not been tested on the real internal file yet.
5. 9797_3.txt calibration parser is best-effort metadata preservation; box conversion
   currently assumes BEVFusion predictions are already in LiDAR coordinates.
6. Do not include secrets, GitHub tokens, SSH passwords, or Jupyter passwords in docs.
```
