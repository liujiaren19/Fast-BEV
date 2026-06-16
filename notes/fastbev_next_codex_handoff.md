# Fast-BEV Next Codex Handoff

This document is the concise handoff for the next Codex session. Read this file
first, then inspect `git log --oneline -8` and the files named below. Do not
reread the full old chat unless blocked.

## Repository State

```text
repo: /root/autodl-tmp/Fast-BEV
branch: test/custom-fastbev-adapter
recent relevant commits before this handoff update:
  1e9cabf Translate N7 converter comments to Chinese
  4de368d Default N7 frame matching to exact timestamps
  c7ba19d Translate N7 converter header comments
  0f588eb Add pose-aware N7 temporal conversion
  384742c Fix custom Fast-BEV OD training config
  1490384 Adapt front mono ROI and N7 OD converter
  224b20e Add BEVFusion pseudo label converter
  e41d058 Validate front monocular Fast-BEV smoke path
```

Current AutoDL instance may be no-GPU. CPU mode is enough for converter,
visualization-code, config inspection, and small pkl validation. Training and
model-side validation should run on a GPU machine.

## Updated Business Priority

The current business order is now:

```text
1. Adapt self-collected N7 data into Fast-BEV training.
2. Adapt six-camera Fast-BEV into true front-monocular Fast-BEV, covering data,
   model training, and actual training/effect validation.
3. Adapt 9797 Nanning proving-ground front camera + LiDAR data. Use BEVFusion
   pseudo labels as monocular Fast-BEV training labels for scene adaptation.
```

The rear-axle-ground ego migration is still important, but it is not ahead of
these three business goals unless a coordinate bug blocks training.

## Priority 1: N7 Self-Collected Data To Fast-BEV Training

Primary target:

```text
Convert N7 self-collected 3D obstacle labels and six synchronized camera frames
into a pkl that CustomMultiViewDataset can train with.
```

Current main converter:

```text
new_tool/unified_processor_raw.py
```

Current visualizer:

```text
new_tool/draw_gt_pkl.py
```

Current known N7 conventions handled by the converter:

```text
Raw label frame: x left, y rear, z up, origin at N7 top main lidar.
Fast-BEV/MMDet3D LiDAR frame: x front, y left, z up.
Axis remap: x_fastbev=-y_raw, y_fastbev=x_raw, z_fastbev=z_raw.
```

Current converter behavior:

```text
1. Only 3D obstacle OD data is supported. Old AVM/parking-lot branches should
   stay removed.
2. Default label/frame association is exact: label timestamp -> frames/<ts>.
3. Nearest frame matching is optional via --frame-match-mode nearest.
4. If slamResult/baidu_ins/odom_lidar_reference.txt exists, pose is read from it.
5. If odom is missing, empty, or no pose falls within --max-pose-match-us, the
   converter falls back to label JSON 3d_od.lidar_pose.
6. odom_lidar_reference.txt timestamps are already synchronized with lidar.
7. Output pkl writes clip-local lidar2global_* and ego2global_* fields for
   temporal compensation. The word global means clip-local reference frame, not
   geographic global coordinates.
8. ego is currently treated as the top main lidar frame, so sensor2ego equals
   sensor2lidar. This is intentional for the current training path because
   lidar2ego is identity under this temporary convention.
```

Important current limitation:

```text
The pkl is in top-lidar-origin Fast-BEV lidar coordinates, not yet rear-axle-
ground ego coordinates. Do not mix vehicles as if their lidar origins were the
same physical point.
```

Next required Priority 1 work:

```text
1. Put a real continuous N7 clip under data/nuscenes with actual six-camera
   image files. The currently uploaded six labels are not enough by themselves.
2. Run new_tool/unified_processor_raw.py on real N7 data and generate pkl.
3. Visualize BEV boxes and camera projections with new_tool/draw_gt_pkl.py.
4. Verify camera image paths, sensor2lidar projection, yaw direction, box center,
   dimensions, velocity, class mapping, and adjacent-frame pose compensation.
5. Run CustomMultiViewDataset with n_times=1 and n_times>1 to confirm pkl loading,
   image stacking, GT loading, and temporal compensation.
6. Run a small Fast-BEV training job on GPU using the N7 pkl.
7. Only after that decide whether rear-axle-ground ego migration is needed before
   bigger experiments.
```

Practical command shape for conversion:

```bash
python new_tool/unified_processor_raw.py \
  --data-path data/nuscenes \
  --datasets 20251203 \
  --sets train \
  --info-json data/info_json/2025_04_18_2k_byd_info.json \
  --output-dir data/nuscenes \
  --extra-tag custom_fastbev \
  --max-adj 60
```

Use `--frame-match-mode nearest` only for old exports where `frames/<lidar_ts>`
does not exist.

## Priority 2: Six-Camera Fast-BEV To True Front-Monocular Fast-BEV

This is no longer the first business item, but it must still be a real model
adaptation, not only a smoke-path config.

Current completed part:

```text
Dataset/config can feed only CAM_FRONT on nuScenes mini.
A smoke path has shown that 1 camera x temporal frames can enter training.
```

Current limitation:

```text
The data/config path was adapted, but the front-monocular BEV design is not
complete. Full 360-degree BEV/loss supervision with only CAM_FRONT is not a valid
final mono setup.
```

Required work before claiming mono adaptation is complete:

```text
1. Define the front-camera training ROI.
   Candidate: x_forward 0..80 or 0..100 m, y_left -35..35 m, z as before.
2. Add a front ROI config, including point_cloud_range, n_voxels, voxel_size,
   anchor ranges, object range filters, train/test/eval ranges.
3. Filter GT to front ROI or front-camera-visible boxes. Do not supervise side
   and rear objects that the mono camera cannot observe.
4. Inspect Fast-BEV backprojection/view-transform code with n_images=1. Confirm
   volume shape, masks, and temporal dimensions.
5. Run real small training experiments, not just 1-iter smoke tests. Minimum
   target is stable loss, non-empty predictions, and sane front-scene geometry.
6. Compare n_times=1 and n_times>1 when possible, because temporal support is a
   useful part of matching original Fast-BEV behavior.
```

Existing starting configs:

```text
configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_1iter.py
configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_debug.py
```

## Priority 3: 9797 BEVFusion Pseudo Labels To Mono Fast-BEV Training

Business target:

```text
Use Nanning proving-ground 9797 vehicle front camera + LiDAR data. BEVFusion
has generated pseudo 3D labels; convert them into the custom 3D_OD JSON/pkl path
and train/front-adapt monocular Fast-BEV for that scene.
```

Current converter:

```text
tools/data_converter/bevfusion_pred_to_custom_labels.py
```

Expected internal data layout from earlier context:

```text
data/gt/20260514_9797/20260514103014_1.dat_img/   # front images
data/gt/20260514_9797/name.pkl                    # image names for pred.pkl
data/gt/20260514_9797/pred.pkl                    # BEVFusion predictions
data/gt/20260514_9797/9797_3.txt                  # 9797 front camera/LiDAR calib txt
```

First required validation:

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

Critical caveats:

```text
1. The current 9797 conversion assumes BEVFusion/MMDet3D LiDAR boxes are in
   x front, y left, z up and converts them to the temporary raw 3D_OD frame
   x left, y rear, z up.
2. This is only a temporary engineering bridge. It does not account for the
   9797 LiDAR origin, mounting offset, height, yaw bias, or rear-axle ego frame.
3. 9797 appears front-camera-only. It should feed the monocular Fast-BEV path,
   not the original six-camera Fast-BEV path, unless more cameras are provided.
4. pred.pkl/name.pkl must be inspected on the real internal machine before bulk
   label generation.
```

Required validation before training:

```text
1. Inspect real pred.pkl/name.pkl structures with --inspect --dry-run.
2. Generate a small subset of JSON labels.
3. Visualize BEV and camera projection for several frames.
4. Check class mapping, scores, box centers, dimensions, yaw direction, and
   front-range filtering.
5. Convert to pkl through the same custom data path only after geometry looks
   correct.
6. Train using the mono-front Fast-BEV config produced by Priority 2.
```

## Deferred Coordinate Cleanup: Rear-Axle-Ground Ego

Target final convention:

```text
origin: rear axle center projected to ground
x: forward
y: left
z: up
```

This migration should be done after the N7 training path and mono path are
functionally validated, unless coordinate inconsistency blocks them.

Need per-vehicle transforms:

```text
T_N7_lidar_to_rear_axle_ground_ego
T_9797_lidar_to_rear_axle_ground_ego
camera_to_ego for each vehicle/camera
```

When this migration is done, update all related fields together:

```text
1. GT boxes and velocities.
2. sensor2lidar and/or sensor2ego depending on final dataset semantics.
3. lidar2ego, ego2global, lidar2global.
4. point_cloud_range, anchors, filters, and visualization tools.
```

Do not only change box coordinates. The pkl must remain internally self-
consistent.

## Files To Inspect First

```text
new_tool/unified_processor_raw.py
new_tool/draw_gt_pkl.py
mmdet3d/datasets/custom_multiview_dataset.py
configs/fastbev/round1/custom_n7_6v_fastbev_r18_debug.py
configs/fastbev/round1/custom_n7_6v_fastbev_r18_smoke.py
configs/fastbev/round1/nuscenes_mini_front_mono_fastbev_r18_debug.py
tools/data_converter/bevfusion_pred_to_custom_labels.py
notes/fastbev_custom_handoff.md
```

## Current Known Risks

```text
1. Current N7 pkl converter is validated on synthetic fixtures and limited
   uploaded labels, but still needs a real image-backed continuous clip test.
2. Current N7 ego is top-lidar-origin, not rear-axle-ground origin.
3. Camera sensor2ego equals sensor2lidar only because ego currently equals lidar.
4. Mono adaptation is not complete until ROI, GT filtering, model-side shape
   checks, and real small training are done.
5. 9797 pseudo-label coordinate conversion is a temporary approximation.
6. 9797 data appears front-only, so it depends on the mono path.
7. Do not include secrets, GitHub tokens, SSH passwords, or Jupyter passwords in
   docs or commits.
```
