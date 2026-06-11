# Fast-BEV Custom Dataset Handoff

This note is intended for another Codex session to continue the custom dataset
adaptation with low context/token cost. Read this file first, then inspect the
current git diff instead of replaying the full chat history.

## Goal

Adapt this Fast-BEV fork for an in-house multi-camera dataset:

1. Train on custom collected data.
2. Support front-camera monocular training/inference.
3. Evaluate detection metrics on the custom dataset without depending on the
   official nuScenes database or maps.

## Custom Data Layout

Dataset root is expected to be `data/nuscenes`.

Images:

```text
data/nuscenes/<day>/<session>/parsed_data/<clip>/frames/<frame_ts>/images/<cam_id>/<image_ts>.jpg
```

Example:

```text
data/nuscenes/20250401/20250401_145924/parsed_data/20250401_145924_0/frames/1743490778325134/images/cam0/1743490778325000.jpg
```

Labels:

```text
data/nuscenes/<day>/<session>/output/<clip>/3D_OD/lidar/<label_ts>.json
```

Calibration:

```text
data/info_json/2025_04_18_2k_byd_info.json
```

Camera mapping:

```text
cam0  -> front_wide  -> front
cam11 -> right_front -> front-right
cam9  -> left_front  -> front-left
cam3  -> back        -> rear
cam8  -> left_back   -> rear-left
cam10 -> right_back  -> rear-right
```

## Coordinate Convention

The raw custom main-lidar frame is:

```text
X: vehicle left
Y: vehicle rear
Z: vehicle roof/up
origin: main lidar
```

Fast-BEV/MMDetection3D LiDAR convention used by the adaptation is:

```text
X: vehicle front
Y: vehicle left
Z: up
```

Conversion:

```text
x_fast = -y_raw
y_fast =  x_raw
z_fast =  z_raw
yaw_fast = yaw_raw + pi / 2
```

This is implemented in `tools/data_converter/custom_fastbev_converter.py` via
`RAW_TO_FASTBEV`.

## Main Files Added/Changed

Added:

```text
tools/data_converter/custom_fastbev_converter.py
mmdet3d/datasets/custom_multiview_dataset.py
configs/fastbev/custom/custom_fastbev_6v_r18.py
configs/fastbev/custom/custom_fastbev_mono_front_r18.py
notes/fastbev_custom_handoff.md
```

Changed:

```text
tools/create_data.py
tools/data_converter/__init__.py
mmdet3d/datasets/__init__.py
mmdet3d/datasets/pipelines/multi_view.py
mmdet3d/models/detectors/fastbev.py
```

## What Was Implemented

- Custom info converter:
  `tools/data_converter/custom_fastbev_converter.py`
- Dataset:
  `CustomMultiViewDataset`
- Configs:
  - six-camera: `configs/fastbev/custom/custom_fastbev_6v_r18.py`
  - front monocular: `configs/fastbev/custom/custom_fastbev_mono_front_r18.py`
- FastBEV model now has configurable `n_images`, defaulting to `6`.
- `MultiViewPipeline` now supports `n_images != 6`.
- `tools/create_data.py` now supports `custom_fastbev` and uses lazy imports so
  custom conversion does not require compiled mmdet3d ops.

## Environment Created During Validation

A conda environment was created:

```bash
conda activate fastbev-mmdet
```

Key versions:

```text
python 3.8
torch 1.10.0+cpu
mmcv-full 1.4.0
mmdet 2.20.0
mmseg 0.21.1
numpy 1.23.5
opencv-python-headless 4.13.0
```

Notes:

- `opencv-python` GUI package was removed because it required `libGL.so.1`.
- `opencv-python-headless` is intentionally used.
- This machine has no `nvcc`, so mmdet3d CUDA/C++ ops were not compiled.
  Full training/dataset building that imports all ops requires a CUDA/nvcc
  environment or prebuilt compatible extensions.

## Verified

Config parsing:

```bash
conda run -n fastbev-mmdet python -c "from mmcv import Config; paths=['configs/fastbev/custom/custom_fastbev_6v_r18.py','configs/fastbev/custom/custom_fastbev_mono_front_r18.py']; [print(p, Config.fromfile(p).model.n_images) for p in paths]"
```

Expected output includes:

```text
configs/fastbev/custom/custom_fastbev_6v_r18.py 6
configs/fastbev/custom/custom_fastbev_mono_front_r18.py 1
```

Mock custom conversion was run successfully and generated:

```text
*_infos_train.pkl
*_infos_val.pkl
*_infos_trainval.pkl
```

The mock coordinate check confirmed:

```text
raw location (x=4, y=-15, z=-1.2)
converted location (x=15, y=4, z=-1.2)
raw velocity (vx=0.1, vy=-0.2)
converted velocity (vx=0.2, vy=0.1)
yaw = raw_yaw + pi/2
```

## Real Data Conversion Command

Run this once real data exists under `data/nuscenes`:

```bash
conda run -n fastbev-mmdet python tools/create_data.py custom_fastbev \
  --root-path data/nuscenes \
  --out-dir data/nuscenes \
  --extra-tag custom_fastbev \
  --calib-path data/info_json/2025_04_18_2k_byd_info.json
```

Optional front-only conversion:

```bash
conda run -n fastbev-mmdet python tools/create_data.py custom_fastbev \
  --root-path data/nuscenes \
  --out-dir data/nuscenes \
  --extra-tag custom_fastbev_front \
  --calib-path data/info_json/2025_04_18_2k_byd_info.json \
  --camera-ids cam0
```

If using the current configs without editing `ann_prefix`, generate the default
six-camera pkl with `--extra-tag custom_fastbev`.

## Training Configs

Six-camera:

```bash
python tools/train.py configs/fastbev/custom/custom_fastbev_6v_r18.py
```

Front monocular:

```bash
python tools/train.py configs/fastbev/custom/custom_fastbev_mono_front_r18.py
```

## Important Follow-Ups

1. Run the converter on real data and inspect pkl counts/class distribution.
2. Verify timestamp matching tolerance. Current default is `--max-match-us 80000`.
3. Visualize projected boxes to confirm calibration direction. The converter
   assumes camera `extrinsic.to_lidar_main` is camera-to-main-lidar in raw
   custom coordinates.
4. Confirm whether the label `z` is box center height. Current code treats
   labels as center-based boxes and passes `origin=(0.5, 0.5, 0.5)`.
5. For temporal Fast-BEV quality, add real per-frame ego/lidar pose alignment if
   available. Current adjacent frames are loaded without ego-motion
   compensation.
6. Compile mmdet3d ops in a CUDA/nvcc environment before full training.

## Suggested First Commands For Next Codex

```bash
git status --short
git diff --stat
sed -n '1,260p' notes/fastbev_custom_handoff.md
```

Avoid rereading the full previous chat unless absolutely necessary.
