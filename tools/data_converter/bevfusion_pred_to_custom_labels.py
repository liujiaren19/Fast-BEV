import argparse
import json
import math
import os
import os.path as osp
import pickle
from pathlib import Path

import numpy as np


FASTBEV_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# FastBEV/MMDet3D LiDAR frame: x front, y left, z up.
# Custom raw main-lidar label frame: x left, y rear, z up.
FASTBEV_TO_RAW = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
RAW_TO_FASTBEV = FASTBEV_TO_RAW.T


def norm_yaw(yaw):
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_numpy(x):
    """Convert common tensor/box wrappers to numpy without importing torch."""
    if x is None:
        return None
    if hasattr(x, 'tensor'):
        x = x.tensor
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        x = x.numpy()
    return np.asarray(x)


def short_repr(obj, depth=0):
    if depth > 2:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {k: short_repr(v, depth + 1) for k, v in list(obj.items())[:12]}
    if isinstance(obj, (list, tuple)):
        return [short_repr(x, depth + 1) for x in list(obj)[:3]]
    arr = to_numpy(obj)
    if isinstance(arr, np.ndarray):
        return f'{type(obj).__name__}/array shape={arr.shape} dtype={arr.dtype}'
    return repr(obj)[:160]


def inspect_pickles(name_path, pred_path):
    names = load_pickle(name_path)
    preds = load_pickle(pred_path)
    print('name type:', type(names), 'len:', len(names) if hasattr(names, '__len__') else 'NA')
    print('pred type:', type(preds), 'len:', len(preds) if hasattr(preds, '__len__') else 'NA')
    if hasattr(names, '__len__') and len(names):
        print('name[0]:', repr(names[0]))
    if hasattr(preds, '__len__') and len(preds):
        print('pred[0] summary:')
        print(json.dumps(short_repr(preds[0]), indent=2, ensure_ascii=False))


def basename_no_ext(path_like):
    return osp.splitext(osp.basename(str(path_like)))[0]


def image_basename(path_like):
    return osp.basename(str(path_like))


def timestamp_from_name(path_like, fallback_idx):
    stem = basename_no_ext(path_like)
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)
    return int(fallback_idx)


def find_image_path(image_dir, name):
    base = image_basename(name)
    direct = osp.join(image_dir, base)
    if osp.exists(direct):
        return direct
    stem = basename_no_ext(base)
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        p = osp.join(image_dir, stem + ext)
        if osp.exists(p):
            return p
        p = osp.join(image_dir, stem + ext.upper())
        if osp.exists(p):
            return p
    return direct


def first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def first_present_attr(obj, names):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def parse_calib_txt(path):
    """Best-effort parser for a camera calibration txt.

    The converter does not need calibration when BEVFusion boxes are already in
    LiDAR coordinates. This parser preserves numeric calibration in the output
    metadata for traceability and handles common K/D/extrinsic text files.
    """
    if not path:
        return {}
    text = Path(path).read_text(errors='ignore')
    nums = np.fromstring(
        ''.join(ch if (ch.isdigit() or ch in '.-+eE') else ' ' for ch in text),
        sep=' ', dtype=np.float64)
    calib = {'path': path, 'num_values': int(nums.size)}
    if nums.size >= 9:
        calib['K_guess'] = nums[:9].reshape(3, 3).tolist()
    if nums.size >= 14:
        calib['D_guess'] = nums[9:14].tolist()
    if nums.size >= 30:
        maybe_ext = nums[14:30].reshape(4, 4)
        calib['extrinsic_4x4_guess'] = maybe_ext.tolist()
    elif nums.size >= 21:
        maybe_ext = nums[9:21].reshape(3, 4)
        calib['extrinsic_3x4_guess'] = maybe_ext.tolist()
    return calib


def unwrap_prediction(pred):
    """Return boxes, scores, labels from common BEVFusion/MMDet3D outputs."""
    if isinstance(pred, dict) and 'pts_bbox' in pred:
        pred = pred['pts_bbox']
    if isinstance(pred, dict) and 'pred_instances_3d' in pred:
        pred = pred['pred_instances_3d']

    if isinstance(pred, dict):
        boxes = first_present(pred, ['boxes_3d', 'bboxes_3d', 'boxes', 'bboxes'])
        scores = first_present(pred, ['scores_3d', 'scores'])
        labels = first_present(pred, ['labels_3d', 'labels'])
    else:
        boxes = first_present_attr(pred, ['boxes_3d', 'bboxes_3d', 'boxes', 'bboxes'])
        scores = first_present_attr(pred, ['scores_3d', 'scores'])
        labels = first_present_attr(pred, ['labels_3d', 'labels'])

    boxes = to_numpy(boxes)
    scores = to_numpy(scores)
    labels = to_numpy(labels)

    if boxes is None:
        boxes = np.zeros((0, 7), dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if scores is None:
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
    if labels is None:
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)
    return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int64)


def fastbev_box_to_custom_annotation(box, score, label, idx, class_names):
    """Convert [x,y,z,l,w,h,yaw,...] from x-front/y-left/z-up to custom JSON."""
    center_fastbev = np.asarray(box[:3], dtype=np.float32)
    center_raw = FASTBEV_TO_RAW @ center_fastbev

    # Existing custom converter applies yaw_fastbev = yaw_raw + pi/2.
    yaw_fastbev = float(box[6]) if len(box) > 6 else 0.0
    yaw_raw = norm_yaw(yaw_fastbev - math.pi / 2.0)

    if len(box) >= 9:
        vel_fastbev = np.asarray([box[7], box[8], 0.0], dtype=np.float32)
        vel_raw = FASTBEV_TO_RAW @ vel_fastbev
    else:
        vel_raw = np.zeros(3, dtype=np.float32)

    label_int = int(label)
    obj_type = class_names[label_int] if 0 <= label_int < len(class_names) else str(label_int)
    return {
        'location': {
            'x': float(center_raw[0]),
            'y': float(center_raw[1]),
            'z': float(center_raw[2]),
        },
        'rotation': {'yaw': yaw_raw},
        'size': {
            'l': float(box[3]) if len(box) > 3 else 0.0,
            'w': float(box[4]) if len(box) > 4 else 0.0,
            'h': float(box[5]) if len(box) > 5 else 0.0,
        },
        'velocity': {
            'vx': float(vel_raw[0]),
            'vy': float(vel_raw[1]),
            'vz': float(vel_raw[2]),
        },
        'acceleration': {'ax': 0.0, 'ay': 0.0, 'az': 0.0},
        'type': obj_type,
        'track_id': int(idx),
        'score_3d': float(score),
    }


def convert_bevfusion_predictions(name_path,
                                  pred_path,
                                  image_dir,
                                  out_dir,
                                  calib_path=None,
                                  class_names=None,
                                  score_thr=0.0,
                                  front_range=None,
                                  dry_run=False):
    names = load_pickle(name_path)
    preds = load_pickle(pred_path)
    if len(names) != len(preds):
        raise ValueError(f'name/pred length mismatch: {len(names)} vs {len(preds)}')

    class_names = class_names or FASTBEV_CLASSES
    calib = parse_calib_txt(calib_path)
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    total_boxes = 0
    kept_boxes = 0
    for frame_idx, (name, pred) in enumerate(zip(names, preds)):
        boxes, scores, labels = unwrap_prediction(pred)
        timestamp = timestamp_from_name(name, frame_idx)
        image_path = find_image_path(image_dir, name)

        annotations = []
        for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            total_boxes += 1
            if float(score) < score_thr:
                continue
            if front_range is not None:
                x, y = float(box[0]), float(box[1])
                xmin, xmax, ymin, ymax = front_range
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    continue
            annotations.append(
                fastbev_box_to_custom_annotation(box, score, label, box_idx, class_names))
            kept_boxes += 1

        payload = {
            '3d_od': {
                'frame_timestamp': int(timestamp),
                'image_path': image_path,
                'source_image_name': str(name),
                'coordinate': 'custom_lidar:x_left_y_rear_z_up',
                'source_coordinate': 'mmdet3d_lidar:x_front_y_left_z_up',
                'fastbev_to_raw': FASTBEV_TO_RAW.tolist(),
                'calibration': calib,
                'lidar_pose': {},
                'lidar_velocity': {'vx': 0.0, 'vy': 0.0, 'vz': 0.0},
                'lidar_acceleration': {'ax': 0.0, 'ay': 0.0, 'az': 0.0},
                'annotations': annotations,
            }
        }
        out_path = osp.join(out_dir, f'{timestamp}.json')
        if not dry_run:
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        written += 1

    print(f'frames: {len(names)}')
    print(f'written json: {written} -> {out_dir}')
    print(f'boxes total/kept: {total_boxes}/{kept_boxes}')
    print(f'score_thr: {score_thr}')
    if front_range is not None:
        print(f'front_range in source FastBEV coords: {front_range}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert BEVFusion pred.pkl/name.pkl to custom Fast-BEV 3d_od JSON labels')
    parser.add_argument('--name-pkl', required=True)
    parser.add_argument('--pred-pkl', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--calib-txt', default=None)
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument(
        '--front-range', type=float, nargs=4, default=None,
        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
        help='Optional filter in BEVFusion/MMDet3D coords: x front, y left.')
    parser.add_argument('--class-names', nargs='+', default=FASTBEV_CLASSES)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.inspect:
        inspect_pickles(args.name_pkl, args.pred_pkl)
    convert_bevfusion_predictions(
        name_path=args.name_pkl,
        pred_path=args.pred_pkl,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        calib_path=args.calib_txt,
        class_names=args.class_names,
        score_thr=args.score_thr,
        front_range=args.front_range,
        dry_run=args.dry_run)
