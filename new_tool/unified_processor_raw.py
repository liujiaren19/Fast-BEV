#!/usr/bin/env python3
"""Convert N7 3D_OD data to Fast-BEV CustomMultiViewDataset pkl files.

This script is intentionally OD-only. It removes the old AVM/parking-slot path
and emits the pkl schema consumed by mmdet3d.datasets.CustomMultiViewDataset.

Output coordinate:
    mmdet3d_lidar: x front, y left, z up, origin at N7 top lidar.

Source 3D_OD labels are treated as:
    custom_lidar: x left, y rear/back, z up, origin at N7 top lidar.

The rear-axle-ground ego migration is not applied here. That migration should be
done as a separate all-geometry change because it must update labels, camera
extrinsics, ranges, anchors, and visualization together.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import os.path as osp
import pickle
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CAMERA_ID_TO_SENSOR = {
    'cam0': 'front_wide',
    'cam11': 'right_front',
    'cam9': 'left_front',
    'cam3': 'back',
    'cam8': 'left_back',
    'cam10': 'right_back',
}
CAMERA_ORDER = ['cam0', 'cam11', 'cam9', 'cam3', 'cam8', 'cam10']

FASTBEV_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

CLASS_MAPPING = {
    'car': 'car',
    'smallMot': 'car',
    'van': 'truck',
    'van-less': 'truck',
    'truck': 'truck',
    'bigMot': 'truck',
    'tanker': 'truck',
    'bus': 'bus',
    'otherMot': 'construction_vehicle',
    'construction_vehicle': 'construction_vehicle',
    'bicycle': 'bicycle',
    'nonMot': 'bicycle',
    'no_motor_bike': 'bicycle',
    'tricycle': 'bicycle',
    'rider': 'motorcycle',
    'ride_person': 'motorcycle',
    'motorcycle': 'motorcycle',
    'pedestrian': 'pedestrian',
    'person': 'pedestrian',
    'trafficCone': 'traffic_cone',
    'traffic_cone': 'traffic_cone',
    'cicularBarrel': 'barrier',
    'smallColumn': 'barrier',
    'waterFilledbarrier': 'barrier',
    'barrier': 'barrier',
}

# Source N7/custom label frame: x left, y rear/back, z up.
# Fast-BEV/MMDet3D LiDAR frame: x front, y left, z up.
RAW_TO_FASTBEV = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


@dataclass(frozen=True)
class ClipRef:
    dataset: str
    sequence: str
    clip: str


@dataclass
class ConversionStats:
    clips_total: int = 0
    clips_missing_paths: int = 0
    labels_total: int = 0
    labels_without_frame: int = 0
    labels_without_cameras: int = 0
    labels_without_gt: int = 0
    objects_total: int = 0
    objects_kept: int = 0
    objects_unknown_class: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)

    def merge(self, other: 'ConversionStats') -> None:
        for key, value in other.as_dict().items():
            setattr(self, key, getattr(self, key) + value)


class TimestampMatcher:
    def __init__(self, frame_root: Path):
        self.items: List[Tuple[int, Path]] = []
        if frame_root.exists():
            for item in frame_root.iterdir():
                if not item.is_dir():
                    continue
                try:
                    ts = int(item.name)
                except ValueError:
                    continue
                self.items.append((ts, item))
        self.items.sort(key=lambda x: x[0])
        self.timestamps = [x[0] for x in self.items]

    def closest(self, timestamp: int, max_diff_us: int) -> Optional[Tuple[int, Path]]:
        if not self.items:
            return None
        pos = np.searchsorted(self.timestamps, timestamp)
        candidates = []
        for idx in (pos - 1, pos):
            if 0 <= idx < len(self.items):
                ts, path = self.items[idx]
                diff = abs(ts - timestamp)
                if diff <= max_diff_us:
                    candidates.append((diff, ts, path))
        if not candidates:
            return None
        _, ts, path = min(candidates, key=lambda x: x[0])
        return ts, path


def load_json(path: Path) -> Dict:
    with path.open('r') as f:
        return json.load(f)


def norm_yaw(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def timestamp_from_path(path: Path) -> int:
    return int(path.stem)


def normalize_rotation_matrix(mat: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(mat, dtype=np.float64))
    rot = u @ vh
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vh
    return rot.astype(np.float32)


def quat_xyzw_to_matrix(q_xyzw: Sequence[float]) -> np.ndarray:
    qx, qy, qz, qw = [float(x) for x in q_xyzw]
    q = Quaternion(qw, qx, qy, qz)
    return normalize_rotation_matrix(q.rotation_matrix)


def rotation_to_wxyz(rot: np.ndarray) -> List[float]:
    qx, qy, qz, qw = R.from_matrix(normalize_rotation_matrix(rot)).as_quat()
    return [float(qw), float(qx), float(qy), float(qz)]


def parse_sensor_to_lidar(raw_ext: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw_ext, dtype=np.float64)
    if arr.shape == (4, 4):
        rot = normalize_rotation_matrix(arr[:3, :3])
        tran = arr[:3, 3].astype(np.float32)
        return rot, tran
    flat = arr.reshape(-1)
    if flat.size != 7:
        raise ValueError(f'expected 7-vector or 4x4 extrinsic, got shape {arr.shape}')
    tran = flat[:3].astype(np.float32)
    rot = quat_xyzw_to_matrix(flat[3:7])
    return rot, tran


def find_sensor(calib: Dict, sensor_name: str) -> Dict:
    for sensor in calib.get('sensors', []):
        if sensor.get('name') == sensor_name:
            return sensor
    raise KeyError(f'Cannot find sensor {sensor_name} in calibration json')


def lidar_main_metadata(calib: Dict) -> Dict:
    """Preserve the N7 lidar_main mixed to_ego convention for traceability."""
    try:
        sensor = find_sensor(calib, 'lidar_main')
        to_ego = np.asarray(sensor.get('extrinsic', {}).get('to_ego', []), dtype=np.float64)
        if to_ego.shape != (4, 4):
            return {}
        ego_to_lidar_rot = normalize_rotation_matrix(to_ego[:3, :3])
        lidar_to_ego_rot = ego_to_lidar_rot.T
        lidar_to_ego_tran = to_ego[:3, 3].astype(np.float32)
        return {
            'note': 'N7 lidar_main to_ego uses ego_to_lidar rotation but lidar_to_ego translation.',
            'ego_to_lidar_rotation_from_json': ego_to_lidar_rot.tolist(),
            'lidar_to_ego_rotation_fixed': lidar_to_ego_rot.tolist(),
            'lidar_to_ego_translation_from_json': lidar_to_ego_tran.tolist(),
        }
    except Exception as exc:
        return {'parse_error': str(exc)}


def build_camera_info(sensor: Dict, image_path: Path, data_root: Path) -> Dict:
    raw_ext = sensor.get('extrinsic', {}).get('to_lidar_main')
    if raw_ext is None:
        raise ValueError(f'sensor {sensor.get("name")} lacks to_lidar_main extrinsic')
    sensor_to_raw_rot, sensor_to_raw_tran = parse_sensor_to_lidar(raw_ext)
    sensor_to_fastbev_rot = RAW_TO_FASTBEV @ sensor_to_raw_rot
    sensor_to_fastbev_tran = RAW_TO_FASTBEV @ sensor_to_raw_tran

    try:
        data_path = osp.relpath(str(image_path), str(data_root))
    except ValueError:
        data_path = str(image_path)

    intrinsic = np.asarray(sensor.get('intrinsic', {}).get('K', []), dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f'sensor {sensor.get("name")} has invalid K shape {intrinsic.shape}')

    distortion = np.asarray(sensor.get('intrinsic', {}).get('D', []), dtype=np.float32).reshape(-1)
    return {
        'data_path': data_path,
        'sensor_name': sensor.get('name', ''),
        'cam_intrinsic': intrinsic,
        'sensor2lidar_rotation': sensor_to_fastbev_rot.astype(np.float32),
        'sensor2lidar_translation': sensor_to_fastbev_tran.astype(np.float32),
        'sensor2ego_rotation': rotation_to_wxyz(sensor_to_fastbev_rot),
        'sensor2ego_translation': sensor_to_fastbev_tran.astype(np.float32).tolist(),
        'distortion': distortion,
        'width': sensor.get('width', 0),
        'height': sensor.get('height', 0),
    }


def image_files(cam_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP'):
        files.extend(Path(x) for x in glob.glob(str(cam_dir / ext)))
    return sorted(files)


def collect_frame_images(frame_dir: Path, camera_ids: Sequence[str]) -> Optional[Dict[str, Path]]:
    images_root = frame_dir / 'images'
    images: Dict[str, Path] = {}
    for cam_id in camera_ids:
        files = image_files(images_root / cam_id)
        if not files:
            return None
        image_path = files[0]
        if not image_path.exists() or image_path.stat().st_size <= 0:
            return None
        images[cam_id] = image_path
    return images


def annotation_payload(label: Dict) -> Dict:
    return label.get('3d_od', label)


def annotation_to_box(anno: Dict) -> Tuple[List[float], List[float]]:
    loc_raw = np.array([
        anno.get('location', {}).get('x', 0.0),
        anno.get('location', {}).get('y', 0.0),
        anno.get('location', {}).get('z', 0.0),
    ], dtype=np.float32)
    loc = RAW_TO_FASTBEV @ loc_raw

    size = anno.get('size', {})
    yaw_raw = float(anno.get('rotation', {}).get('yaw', 0.0))
    yaw = norm_yaw(yaw_raw + math.pi / 2.0)

    vel_raw = np.array([
        anno.get('velocity', {}).get('vx', 0.0),
        anno.get('velocity', {}).get('vy', 0.0),
        anno.get('velocity', {}).get('vz', 0.0),
    ], dtype=np.float32)
    vel = RAW_TO_FASTBEV @ vel_raw

    box = [
        float(loc[0]), float(loc[1]), float(loc[2]),
        float(size.get('l', 0.0)),
        float(size.get('w', 0.0)),
        float(size.get('h', 0.0)),
        yaw,
    ]
    return box, [float(vel[0]), float(vel[1])]


def parse_label(label_path: Path, classes: Sequence[str], stats: ConversionStats) -> Dict:
    payload = annotation_payload(load_json(label_path))
    annotations = payload.get('annotations', [])
    timestamp = int(payload.get('frame_timestamp', timestamp_from_path(label_path)))

    gt_boxes: List[List[float]] = []
    gt_names: List[str] = []
    gt_velocity: List[List[float]] = []
    track_ids: List[int] = []

    for anno in annotations:
        stats.objects_total += 1
        raw_name = anno.get('type', 'unknown')
        mapped_name = CLASS_MAPPING.get(raw_name, raw_name)
        if mapped_name not in classes:
            stats.objects_unknown_class += 1
            continue
        box, velocity = annotation_to_box(anno)
        gt_boxes.append(box)
        gt_names.append(mapped_name)
        gt_velocity.append(velocity)
        track_ids.append(int(anno.get('track_id', -1)))
        stats.objects_kept += 1

    if gt_boxes:
        boxes_arr = np.asarray(gt_boxes, dtype=np.float32)
        velocity_arr = np.asarray(gt_velocity, dtype=np.float32)
    else:
        boxes_arr = np.zeros((0, 7), dtype=np.float32)
        velocity_arr = np.zeros((0, 2), dtype=np.float32)

    return {
        'timestamp': timestamp,
        'gt_boxes': boxes_arr,
        'gt_names': np.asarray(gt_names),
        'gt_velocity': velocity_arr,
        'track_ids': np.asarray(track_ids, dtype=np.int64),
    }


def resolve_clip_paths(data_root: Path, ref: ClipRef) -> Optional[Tuple[Path, Path]]:
    base = data_root / ref.dataset
    if not base.exists():
        base = data_root
    candidates = [
        (
            base / ref.sequence / 'output' / ref.clip / '3D_OD' / 'lidar',
            base / ref.sequence / 'parsed_data' / ref.clip / 'frames',
        ),
        (
            base / 'output' / ref.sequence / ref.clip / '3D_OD' / 'lidar',
            base / 'parsed_data' / ref.sequence / ref.clip / 'frames',
        ),
        (
            base / 'output' / ref.clip / '3D_OD' / 'lidar',
            base / 'parsed_data' / ref.clip / 'frames',
        ),
        (
            base / ref.sequence / 'output' / ref.clip / '3D_OD' / 'lidar',
            base / ref.sequence / 'parsed_data' / ref.clip / 'frames',
        ),
    ]
    for label_dir, frames_dir in candidates:
        if label_dir.exists() and frames_dir.exists():
            return label_dir, frames_dir
    return None


def parse_manifest(path: Path) -> List[ClipRef]:
    refs: List[ClipRef] = []
    with path.open('r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [x for x in line.split('/') if x]
            if len(parts) != 3:
                raise ValueError(f'{path}:{line_no}: expected dataset/sequence/clip, got {line!r}')
            refs.append(ClipRef(dataset=parts[0], sequence=parts[1], clip=parts[2]))
    return refs


def manifest_candidates(data_root: Path, dataset_name: str, set_name: str) -> List[Path]:
    name = f'{dataset_name}_{set_name}_clips.txt'
    return [
        data_root / name,
        data_root / 'manifests' / name,
        data_root / dataset_name / name,
    ]


def discover_refs(data_root: Path, dataset_name: str) -> List[ClipRef]:
    base = data_root / dataset_name
    if not base.exists():
        base = data_root
    refs = set()
    for label_dir_str in glob.glob(str(base / '**' / '3D_OD' / 'lidar'), recursive=True):
        label_dir = Path(label_dir_str)
        if not any(label_dir.glob('*.json')):
            continue
        parents = label_dir.parents
        # Layout: dataset/sequence/output/clip/3D_OD/lidar
        if len(parents) >= 4 and parents[2].name == 'output':
            sequence = '' if parents[3] == base else parents[3].name
            refs.add(ClipRef(dataset=dataset_name, sequence=sequence, clip=parents[1].name))
        # Layout: dataset/output/sequence/clip/3D_OD/lidar
        if len(parents) >= 5 and parents[3].name == 'output':
            refs.add(ClipRef(dataset=dataset_name, sequence=parents[2].name, clip=parents[1].name))
    return sorted(refs, key=lambda x: (x.dataset, x.sequence, x.clip))


def load_refs_for_set(data_root: Path, dataset_name: str, set_name: str) -> List[ClipRef]:
    for manifest in manifest_candidates(data_root, dataset_name, set_name):
        if manifest.exists():
            refs = parse_manifest(manifest)
            logger.info('Loaded %d clips from %s', len(refs), manifest)
            return refs
    refs = discover_refs(data_root, dataset_name)
    if refs:
        logger.info('Discovered %d clips under %s for dataset %s', len(refs), data_root, dataset_name)
    return refs


def adjacent_view(info: Dict) -> Dict:
    return {
        'token': info['token'],
        'timestamp': info['timestamp'],
        'cams': info['cams'],
    }


def link_adjacent_infos(infos: List[Dict], max_adjacent: int) -> None:
    by_clip: Dict[str, List[Dict]] = {}
    for info in infos:
        by_clip.setdefault(info['clip_id'], []).append(info)

    for clip_infos in by_clip.values():
        clip_infos.sort(key=lambda x: x['timestamp'])
        for idx, info in enumerate(clip_infos):
            prev_infos = clip_infos[max(0, idx - max_adjacent):idx][::-1]
            next_infos = clip_infos[idx + 1:idx + 1 + max_adjacent]
            info['prev'] = [adjacent_view(x) for x in prev_infos]
            info['next'] = [adjacent_view(x) for x in next_infos]


def build_info_for_label(
    label_path: Path,
    frame_ts: int,
    frame_dir: Path,
    ref: ClipRef,
    calib: Dict,
    data_root: Path,
    camera_ids: Sequence[str],
    classes: Sequence[str],
    keep_empty: bool,
    stats: ConversionStats,
) -> Optional[Dict]:
    ann = parse_label(label_path, classes, stats)
    if len(ann['gt_names']) == 0 and not keep_empty:
        stats.labels_without_gt += 1
        return None

    image_map = collect_frame_images(frame_dir, camera_ids)
    if image_map is None:
        stats.labels_without_cameras += 1
        return None

    cams = {}
    for cam_id in camera_ids:
        sensor_name = CAMERA_ID_TO_SENSOR[cam_id]
        sensor = find_sensor(calib, sensor_name)
        cams[cam_id] = build_camera_info(sensor, image_map[cam_id], data_root)

    label_ts = ann['timestamp']
    token = f'{ref.dataset}_{ref.sequence}_{ref.clip}_{label_ts}'
    clip_uid = f'{ref.dataset}/{ref.sequence}/{ref.clip}'
    return {
        'token': token,
        'timestamp': label_ts,
        'frame_timestamp': frame_ts,
        'clip_id': clip_uid,
        'dataset': ref.dataset,
        'sequence': ref.sequence,
        'clip': ref.clip,
        'cams': cams,
        'lidar_path': '',
        'sweeps': [],
        'gt_boxes': ann['gt_boxes'],
        'gt_names': ann['gt_names'],
        'gt_velocity': ann['gt_velocity'],
        'track_ids': ann['track_ids'],
        'num_lidar_pts': np.ones(len(ann['gt_names']), dtype=np.int32),
        'num_radar_pts': np.zeros(len(ann['gt_names']), dtype=np.int32),
        'valid_flag': np.ones(len(ann['gt_names']), dtype=np.bool_),
    }


def process_clip(
    data_root: Path,
    ref: ClipRef,
    calib: Dict,
    camera_ids: Sequence[str],
    classes: Sequence[str],
    max_match_us: int,
    max_adjacent: int,
    keep_empty: bool,
) -> Tuple[List[Dict], ConversionStats]:
    stats = ConversionStats(clips_total=1)
    paths = resolve_clip_paths(data_root, ref)
    if paths is None:
        stats.clips_missing_paths += 1
        return [], stats
    label_dir, frames_dir = paths
    matcher = TimestampMatcher(frames_dir)
    infos: List[Dict] = []

    for label_path in sorted(label_dir.glob('*.json')):
        stats.labels_total += 1
        try:
            label_ts = timestamp_from_path(label_path)
        except ValueError:
            payload = annotation_payload(load_json(label_path))
            label_ts = int(payload.get('frame_timestamp', 0))
        matched = matcher.closest(label_ts, max_match_us)
        if matched is None:
            stats.labels_without_frame += 1
            continue
        frame_ts, frame_dir = matched
        try:
            info = build_info_for_label(
                label_path=label_path,
                frame_ts=frame_ts,
                frame_dir=frame_dir,
                ref=ref,
                calib=calib,
                data_root=data_root,
                camera_ids=camera_ids,
                classes=classes,
                keep_empty=keep_empty,
                stats=stats,
            )
        except Exception as exc:
            logger.debug('Skip %s: %s', label_path, exc)
            continue
        if info is not None:
            infos.append(info)

    infos.sort(key=lambda x: x['timestamp'])
    link_adjacent_infos(infos, max_adjacent=max_adjacent)
    return infos, stats


def make_metadata(
    classes: Sequence[str],
    camera_ids: Sequence[str],
    calib: Dict,
    stats: ConversionStats,
    set_name: str,
    datasets: Sequence[str],
    max_match_us: int,
    max_adjacent: int,
) -> Dict:
    return {
        'version': 'custom-fastbev-n7-od',
        'created': datetime.now().isoformat(),
        'set': set_name,
        'datasets': list(datasets),
        'classes': list(classes),
        'camera_ids': list(camera_ids),
        'coordinate': 'mmdet3d_lidar:x_front_y_left_z_up; origin=N7_top_lidar',
        'source_coordinate': 'custom_lidar:x_left_y_rear_z_up; origin=N7_top_lidar',
        'raw_to_fastbev': RAW_TO_FASTBEV.tolist(),
        'rear_axle_ground_ego_applied': False,
        'max_match_us': max_match_us,
        'max_adjacent': max_adjacent,
        'stats': stats.as_dict(),
        'lidar_main_mixed_to_ego': lidar_main_metadata(calib),
    }


def save_pkl(infos: List[Dict], metadata: Dict, out_path: Path, dry_run: bool) -> None:
    if dry_run:
        logger.info('Dry-run: would write %d infos to %s', len(infos), out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        pickle.dump({'infos': infos, 'metadata': metadata}, f)
    logger.info('Wrote %d infos to %s', len(infos), out_path)


def convert_one_set(
    data_root: Path,
    dataset_names: Sequence[str],
    set_name: str,
    calib: Dict,
    out_dir: Path,
    extra_tag: str,
    camera_ids: Sequence[str],
    classes: Sequence[str],
    max_match_us: int,
    max_adjacent: int,
    keep_empty: bool,
    separate: bool,
    dry_run: bool,
) -> None:
    per_dataset: Dict[str, List[Dict]] = {}
    per_dataset_stats: Dict[str, ConversionStats] = {}

    for dataset_name in dataset_names:
        refs = load_refs_for_set(data_root, dataset_name, set_name)
        if not refs:
            logger.warning('No clips found for dataset=%s set=%s', dataset_name, set_name)
            continue
        infos: List[Dict] = []
        stats = ConversionStats()
        for ref in tqdm(refs, desc=f'{dataset_name}/{set_name}'):
            clip_infos, clip_stats = process_clip(
                data_root=data_root,
                ref=ref,
                calib=calib,
                camera_ids=camera_ids,
                classes=classes,
                max_match_us=max_match_us,
                max_adjacent=max_adjacent,
                keep_empty=keep_empty,
            )
            infos.extend(clip_infos)
            stats.merge(clip_stats)
        infos.sort(key=lambda x: x['timestamp'])
        per_dataset[dataset_name] = infos
        per_dataset_stats[dataset_name] = stats
        logger.info('dataset=%s set=%s infos=%d stats=%s', dataset_name, set_name, len(infos), stats.as_dict())

        if separate:
            metadata = make_metadata(classes, camera_ids, calib, stats, set_name, [dataset_name], max_match_us, max_adjacent)
            save_pkl(infos, metadata, out_dir / f'{extra_tag}_{dataset_name}_infos_{set_name}.pkl', dry_run)

    if separate:
        return

    merged_infos: List[Dict] = []
    merged_stats = ConversionStats()
    for dataset_name in dataset_names:
        merged_infos.extend(deepcopy(per_dataset.get(dataset_name, [])))
        if dataset_name in per_dataset_stats:
            merged_stats.merge(per_dataset_stats[dataset_name])
    merged_infos.sort(key=lambda x: x['timestamp'])
    metadata = make_metadata(classes, camera_ids, calib, merged_stats, set_name, dataset_names, max_match_us, max_adjacent)
    if not merged_infos:
        logger.warning('set=%s produced no infos; skipping output', set_name)
        return
    save_pkl(merged_infos, metadata, out_dir / f'{extra_tag}_infos_{set_name}.pkl', dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert N7 3D_OD data to Fast-BEV CustomMultiViewDataset pkl files')
    parser.add_argument('--data-type', default='od', choices=['od'], help='Only od is supported.')
    parser.add_argument('--data-path', required=True, help='N7 data root, usually ./data/nuscenes')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset tags or manifest prefixes')
    parser.add_argument('--sets', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--info-json', required=True, help='N7 sensor calibration json')
    parser.add_argument('--output-dir', default='./data/nuscenes')
    parser.add_argument('--extra-tag', default='custom_fastbev', help='Output prefix: {tag}_infos_{set}.pkl')
    parser.add_argument('--template-pkl', default=None, help='Deprecated and ignored; kept for old commands.')
    parser.add_argument('--max-match-us', type=int, default=50000, help='Max label/frame timestamp diff in microseconds')
    parser.add_argument('--max-adj', '--max-adjacent', dest='max_adjacent', type=int, default=60)
    parser.add_argument('--interval', type=int, default=3, help='Kept in metadata for compatibility; adjacent frames are stored densely.')
    parser.add_argument('--camera-ids', nargs='+', default=CAMERA_ORDER, choices=CAMERA_ORDER)
    parser.add_argument('--classes', nargs='+', default=FASTBEV_CLASSES)
    parser.add_argument('--keep-empty', action='store_true', help='Keep frames with no valid 3D boxes')
    parser.add_argument('--separate', '-s', action='store_true', help='Write one pkl per dataset/set instead of merging datasets per set')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--multiprocessing', action='store_true', help='Accepted for old commands; currently ignored.')
    parser.add_argument('--debug', action='store_true', help='Accepted for old commands; currently no extra raw dump.')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    data_root = Path(args.data_path)
    out_dir = Path(args.output_dir)
    calib_path = Path(args.info_json)

    if not data_root.exists():
        raise FileNotFoundError(f'data path does not exist: {data_root}')
    if not calib_path.exists():
        raise FileNotFoundError(f'calibration json does not exist: {calib_path}')
    if args.template_pkl:
        logger.info('--template-pkl is ignored by this OD-only converter: %s', args.template_pkl)
    if args.multiprocessing:
        logger.info('--multiprocessing is accepted for compatibility but not used')

    calib = load_json(calib_path)
    logger.info('data_root=%s', data_root)
    logger.info('datasets=%s sets=%s', args.datasets, args.sets)
    logger.info('camera_ids=%s', args.camera_ids)
    logger.info('coordinate=mmdet3d_lidar:x_front_y_left_z_up origin=N7_top_lidar')

    for set_name in args.sets:
        convert_one_set(
            data_root=data_root,
            dataset_names=args.datasets,
            set_name=set_name,
            calib=calib,
            out_dir=out_dir,
            extra_tag=args.extra_tag,
            camera_ids=args.camera_ids,
            classes=args.classes,
            max_match_us=args.max_match_us,
            max_adjacent=args.max_adjacent,
            keep_empty=args.keep_empty,
            separate=args.separate,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
