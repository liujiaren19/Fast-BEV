import argparse
import glob
import json
import math
import os
import os.path as osp
from copy import deepcopy

import mmcv
import numpy as np
from pyquaternion import Quaternion


CAMERA_ID_TO_SENSOR = {
    'cam0': 'front_wide',
    'cam11': 'right_front',
    'cam9': 'left_front',
    'cam3': 'back',
    'cam8': 'left_back',
    'cam10': 'right_back',
}

CAMERA_ORDER = ['cam0', 'cam11', 'cam9', 'cam3', 'cam8', 'cam10']

CLASS_MAPPING = {
    'car': 'car',
    'smallMot': 'car',
    'bigMot': 'truck',
    'truck': 'truck',
    'bus': 'bus',
    'otherMot': 'construction_vehicle',
    'bicycle': 'bicycle',
    'nonMot': 'bicycle',
    'rider': 'motorcycle',
    'pedestrian': 'pedestrian',
    'trafficCone': 'traffic_cone',
    'cicularBarrel': 'barrier',
    'smallColumn': 'barrier',
    'waterFilledbarrier': 'barrier',
    'barrier': 'barrier',
}

# Raw vehicle/lidar frame: x left, y rear, z up.
# MMDet3D LiDAR frame: x front, y left, z up.
RAW_TO_FASTBEV = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _timestamp_from_path(path):
    return int(osp.splitext(osp.basename(path))[0])


def _norm_yaw(yaw):
    return (yaw + math.pi) % (2 * math.pi) - math.pi


def _quat_to_mat(q):
    q = np.asarray(q, dtype=np.float32)
    if np.allclose(q, 0):
        return np.eye(3, dtype=np.float32)
    return Quaternion(q[0], q[1], q[2], q[3]).rotation_matrix.astype(np.float32)


def _find_sensor(calib, sensor_name):
    for sensor in calib.get('sensors', []):
        if sensor.get('name') == sensor_name:
            return sensor
    raise KeyError(f'Cannot find sensor {sensor_name} in calibration json')


def _build_camera_info(sensor, image_path, data_root):
    raw_ext = sensor['extrinsic']['to_lidar_main']
    tran_raw = np.asarray(raw_ext[:3], dtype=np.float32)
    rot_cam_to_raw = _quat_to_mat(raw_ext[3:7])

    sensor2lidar_rotation = RAW_TO_FASTBEV @ rot_cam_to_raw
    sensor2lidar_translation = RAW_TO_FASTBEV @ tran_raw

    intrinsic = np.asarray(sensor['intrinsic']['K'], dtype=np.float32)
    rel_image_path = osp.relpath(image_path, data_root)
    return dict(
        data_path=rel_image_path,
        sensor_name=sensor.get('name', ''),
        cam_intrinsic=intrinsic,
        sensor2lidar_rotation=sensor2lidar_rotation.astype(np.float32),
        sensor2lidar_translation=sensor2lidar_translation.astype(np.float32),
        distortion=np.asarray(sensor.get('intrinsic', {}).get('D', []),
                              dtype=np.float32),
        width=sensor.get('width', 0),
        height=sensor.get('height', 0),
    )


def _collect_frames(clip_dir):
    frames_root = osp.join(clip_dir, 'frames')
    frame_dirs = sorted(glob.glob(osp.join(frames_root, '*')))
    frames = []
    for frame_dir in frame_dirs:
        if not osp.isdir(frame_dir):
            continue
        try:
            frame_ts = int(osp.basename(frame_dir))
        except ValueError:
            continue
        images = {}
        for cam_id in CAMERA_ORDER:
            img_dir = osp.join(frame_dir, 'images', cam_id)
            image_files = sorted(
                glob.glob(osp.join(img_dir, '*.jpg')) +
                glob.glob(osp.join(img_dir, '*.jpeg')) +
                glob.glob(osp.join(img_dir, '*.png')))
            if image_files:
                images[cam_id] = image_files[0]
        frames.append(dict(timestamp=frame_ts, images=images))
    return frames


def _closest_frame(frames, timestamp, max_match_us):
    if not frames:
        return None
    closest = min(frames, key=lambda x: abs(x['timestamp'] - timestamp))
    if max_match_us is not None and abs(closest['timestamp'] - timestamp) > max_match_us:
        return None
    return closest


def _annotation_to_box(anno):
    loc_raw = np.array([
        anno['location']['x'],
        anno['location']['y'],
        anno['location']['z'],
    ], dtype=np.float32)
    loc = RAW_TO_FASTBEV @ loc_raw

    size = anno['size']
    yaw_raw = float(anno.get('rotation', {}).get('yaw', 0.0))
    yaw = _norm_yaw(yaw_raw + math.pi / 2.0)

    velocity = anno.get('velocity', {})
    vel_raw = np.array([
        velocity.get('vx', 0.0),
        velocity.get('vy', 0.0),
        velocity.get('vz', 0.0),
    ], dtype=np.float32)
    vel = RAW_TO_FASTBEV @ vel_raw

    # MMDet3D LiDAR boxes are [x, y, z, dx, dy, dz, yaw].
    return (
        [loc[0], loc[1], loc[2],
         float(size['l']), float(size['w']), float(size['h']), yaw],
        [vel[0], vel[1]],
    )


def _parse_label(label_path, classes):
    label = _load_json(label_path)
    payload = label.get('3d_od', label)
    annotations = payload.get('annotations', [])

    gt_boxes, gt_names, gt_velocity, track_ids = [], [], [], []
    for anno in annotations:
        mapped_name = CLASS_MAPPING.get(anno.get('type'), anno.get('type'))
        if mapped_name not in classes:
            continue
        box, vel = _annotation_to_box(anno)
        gt_boxes.append(box)
        gt_names.append(mapped_name)
        gt_velocity.append(vel)
        track_ids.append(anno.get('track_id', -1))

    if gt_boxes:
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        gt_velocity = np.asarray(gt_velocity, dtype=np.float32)
    else:
        gt_boxes = np.zeros((0, 7), dtype=np.float32)
        gt_velocity = np.zeros((0, 2), dtype=np.float32)

    return dict(
        timestamp=int(payload.get('frame_timestamp', _timestamp_from_path(label_path))),
        gt_boxes=gt_boxes,
        gt_names=np.asarray(gt_names),
        gt_velocity=gt_velocity,
        track_ids=np.asarray(track_ids, dtype=np.int64),
    )


def _adjacent_view(info):
    return dict(
        token=info['token'],
        timestamp=info['timestamp'],
        cams=info['cams'],
    )


def _link_adjacent_infos(infos, max_adjacent=10):
    by_clip = {}
    for info in infos:
        by_clip.setdefault(info['clip_id'], []).append(info)

    for clip_infos in by_clip.values():
        clip_infos.sort(key=lambda x: x['timestamp'])
        for idx, info in enumerate(clip_infos):
            prev_infos = clip_infos[max(0, idx - max_adjacent):idx][::-1]
            next_infos = clip_infos[idx + 1:idx + 1 + max_adjacent]
            info['prev'] = [_adjacent_view(x) for x in prev_infos]
            info['next'] = [_adjacent_view(x) for x in next_infos]


def _iter_clips(root_path):
    day_dirs = sorted(glob.glob(osp.join(root_path, '*')))
    for day_dir in day_dirs:
        if not osp.isdir(day_dir):
            continue
        for session_dir in sorted(glob.glob(osp.join(day_dir, '*'))):
            parsed_root = osp.join(session_dir, 'parsed_data')
            if not osp.isdir(parsed_root):
                continue
            for clip_dir in sorted(glob.glob(osp.join(parsed_root, '*'))):
                if not osp.isdir(clip_dir):
                    continue
                yield day_dir, session_dir, clip_dir


def create_custom_fastbev_infos(root_path,
                                info_prefix,
                                calib_path,
                                out_dir=None,
                                classes=None,
                                camera_ids=None,
                                max_match_us=80000,
                                train_ratio=0.8,
                                max_adjacent=10):
    if out_dir is None:
        out_dir = root_path
    classes = classes or [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    camera_ids = camera_ids or CAMERA_ORDER
    calib = _load_json(calib_path)

    infos = []
    skipped_no_frame = 0
    skipped_no_camera = 0
    for _, session_dir, clip_dir in _iter_clips(root_path):
        clip_id = osp.basename(clip_dir)
        session_name = osp.basename(session_dir)
        label_dir = osp.join(session_dir, 'output', clip_id, '3D_OD', 'lidar')
        label_files = sorted(glob.glob(osp.join(label_dir, '*.json')))
        if not label_files:
            continue

        frames = _collect_frames(clip_dir)
        for label_path in label_files:
            label_ts = _timestamp_from_path(label_path)
            frame = _closest_frame(frames, label_ts, max_match_us)
            if frame is None:
                skipped_no_frame += 1
                continue

            cams = {}
            for cam_id in camera_ids:
                if cam_id not in frame['images']:
                    continue
                sensor = _find_sensor(calib, CAMERA_ID_TO_SENSOR[cam_id])
                cams[cam_id] = _build_camera_info(
                    sensor, frame['images'][cam_id], root_path)
            if len(cams) != len(camera_ids):
                skipped_no_camera += 1
                continue

            ann = _parse_label(label_path, classes)
            info = dict(
                token=f'{clip_id}_{label_ts}',
                timestamp=label_ts,
                frame_timestamp=frame['timestamp'],
                clip_id=clip_id,
                session=session_name,
                cams=cams,
                lidar_path='',
                sweeps=[],
                gt_boxes=ann['gt_boxes'],
                gt_names=ann['gt_names'],
                gt_velocity=ann['gt_velocity'],
                track_ids=ann['track_ids'],
                num_lidar_pts=np.ones(len(ann['gt_names']), dtype=np.int32),
                valid_flag=np.ones(len(ann['gt_names']), dtype=np.bool_),
            )
            infos.append(info)

    infos = sorted(infos, key=lambda x: x['timestamp'])
    _link_adjacent_infos(infos, max_adjacent=max_adjacent)

    split = int(len(infos) * train_ratio)
    train_infos = deepcopy(infos[:split])
    val_infos = deepcopy(infos[split:])
    trainval_infos = deepcopy(infos)

    metadata = dict(
        version='custom-fastbev',
        classes=classes,
        camera_ids=camera_ids,
        coordinate='mmdet3d_lidar:x_front_y_left_z_up',
        source_coordinate='custom_lidar:x_left_y_back_z_up',
        raw_to_fastbev=RAW_TO_FASTBEV.tolist(),
        max_match_us=max_match_us,
        skipped_no_frame=skipped_no_frame,
        skipped_no_camera=skipped_no_camera,
    )

    mmcv.mkdir_or_exist(out_dir)
    paths = {
        'train': osp.join(out_dir, f'{info_prefix}_infos_train.pkl'),
        'val': osp.join(out_dir, f'{info_prefix}_infos_val.pkl'),
        'trainval': osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl'),
    }
    mmcv.dump(dict(infos=train_infos, metadata=metadata), paths['train'])
    mmcv.dump(dict(infos=val_infos, metadata=metadata), paths['val'])
    mmcv.dump(dict(infos=trainval_infos, metadata=metadata), paths['trainval'])

    print(f'Generated {len(train_infos)} train / {len(val_infos)} val infos')
    print(f'Skipped {skipped_no_frame} labels without matched frame')
    print(f'Skipped {skipped_no_camera} frames without required cameras')
    for split_name, path in paths.items():
        print(f'{split_name}: {path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create Fast-BEV infos for custom multi-view data')
    parser.add_argument('--root-path', default='data/nuscenes')
    parser.add_argument('--calib-path', required=True)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--extra-tag', default='custom_fastbev')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--max-match-us', type=int, default=80000)
    parser.add_argument('--max-adjacent', type=int, default=10)
    parser.add_argument(
        '--camera-ids',
        nargs='+',
        default=CAMERA_ORDER,
        choices=CAMERA_ORDER)
    parser.add_argument(
        '--classes',
        nargs='+',
        default=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_custom_fastbev_infos(
        root_path=args.root_path,
        info_prefix=args.extra_tag,
        calib_path=args.calib_path,
        out_dir=args.out_dir,
        classes=args.classes,
        camera_ids=args.camera_ids,
        max_match_us=args.max_match_us,
        train_ratio=args.train_ratio,
        max_adjacent=args.max_adjacent)
