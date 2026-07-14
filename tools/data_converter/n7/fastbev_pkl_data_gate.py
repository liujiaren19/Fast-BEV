#!/usr/bin/env python3
"""N7 Fast-BEV pkl 的训练前数据门禁与审计报告。

这里放置纯数据契约检查，既能审计 converter 新生成的内存 payload，也能由
``validate_n7_fastbev_pkl.py`` 对已有 train/val/test pkl 做全量检查。几何矩阵
等价性仍由 ``fastbev_geometry.py`` 负责；本模块关注 split、字段完整性、统计
分布和时序数据质量。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tools.data_converter.n7.fastbev_geometry import compute_lidar2img
except ModuleNotFoundError:
    from fastbev_geometry import compute_lidar2img


DATA_GATE_SCHEMA_VERSION = 1
BOX_FIELDS = ('x', 'y', 'z', 'l', 'w', 'h', 'yaw')
CAMERA_REQUIRED_FIELDS = (
    'data_path',
    'cam_intrinsic',
    'sensor2lidar_rotation',
    'sensor2lidar_translation',
    'distortion',
    'intrinsic_width',
    'intrinsic_height',
    'image_width',
    'image_height',
)
CONVERTER_CRITICAL_STATS = (
    'clips_missing_paths',
    'labels_without_frame',
    'labels_without_cameras',
    'labels_failed',
    'objects_unknown_class',
)
SPLIT_PATTERN = re.compile(r'(?:^|[_-])(train|val|test)(?:[_\-.]|$)', re.IGNORECASE)


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _append_issue(report: Dict, level: str, message: str, max_examples: int) -> None:
    count_key = 'failed_checks' if level == 'failure' else 'warning_checks'
    list_key = 'failures' if level == 'failure' else 'warnings'
    report[count_key] += 1
    if len(report[list_key]) < max_examples:
        report[list_key].append(str(message))


def _failure(report: Dict, message: str, max_examples: int) -> None:
    _append_issue(report, 'failure', message, max_examples)


def _warning(report: Dict, message: str, max_examples: int) -> None:
    _append_issue(report, 'warning', message, max_examples)


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _split_name(path: Path, metadata: Dict, fallback_index: int) -> str:
    value = str(metadata.get('set', '') or '').strip().lower()
    if value:
        return value
    match = SPLIT_PATTERN.search(path.name)
    if match:
        return match.group(1).lower()
    return 'input_{}'.format(fallback_index)


def _clip_key(info: Dict) -> str:
    dataset = str(info.get('dataset', '') or '')
    sequence = str(info.get('sequence', '') or '')
    clip = str(info.get('clip', info.get('clip_id', '')) or '')
    return '/'.join((dataset, sequence, clip))


def _has_pose(info: Dict) -> bool:
    rotation = info.get('lidar2global_rotation')
    translation = info.get('lidar2global_translation')
    if rotation is None or translation is None:
        return False
    try:
        rotation = np.asarray(rotation, dtype=np.float64).reshape(-1)
        translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return (
        rotation.size == 4 and translation.size == 3 and
        np.isfinite(rotation).all() and np.isfinite(translation).all() and
        float(np.linalg.norm(rotation)) > 0.0
    )


def _numeric_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            'count': 0,
            'min': None,
            'mean': None,
            'p50': None,
            'p90': None,
            'p99': None,
            'max': None,
        }
    percentiles = np.percentile(array, [50, 90, 99])
    return {
        'count': int(array.size),
        'min': float(np.min(array)),
        'mean': float(np.mean(array)),
        'p50': float(percentiles[0]),
        'p90': float(percentiles[1]),
        'p99': float(percentiles[2]),
        'max': float(np.max(array)),
    }


def _box_array(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(-1, array.shape[-1])
    return array


def _box_distribution(chunks: Sequence[np.ndarray]) -> Dict[str, Dict]:
    if not chunks:
        return {name: _numeric_summary([]) for name in BOX_FIELDS}
    array = np.concatenate(chunks, axis=0)
    return {
        name: _numeric_summary(array[:, index])
        for index, name in enumerate(BOX_FIELDS)
    }


def _gt_box_corners(boxes: np.ndarray) -> np.ndarray:
    """按当前 N7 LiDAR yaw 约定生成重心 origin box 的八角点。"""
    if len(boxes) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    signs = np.asarray([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ], dtype=np.float32)
    local = signs[None, :, :] * boxes[:, None, 3:6] * 0.5
    cos_yaw = np.cos(boxes[:, 6])[:, None]
    sin_yaw = np.sin(boxes[:, 6])[:, None]
    local_x = local[:, :, 0].copy()
    local_y = local[:, :, 1].copy()
    local[:, :, 0] = local_x * cos_yaw + local_y * sin_yaw
    local[:, :, 1] = -local_x * sin_yaw + local_y * cos_yaw
    return local + boxes[:, None, :3]


def _pinhole_visible_mask(boxes: np.ndarray, cam_info: Dict, min_depth: float = 0.1) -> np.ndarray:
    """按当前 dataset 的 pinhole 中心/角点口径统计几何可见 GT。

    这里只用于数据分布审计，不作为 distortion-aware 修复的替代实现。
    """
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.bool_)
    intrinsic_width = int(cam_info['intrinsic_width'])
    intrinsic_height = int(cam_info['intrinsic_height'])
    image_width = int(cam_info['image_width'])
    image_height = int(cam_info['image_height'])
    intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32).copy()
    intrinsic[0, :] *= image_width / float(intrinsic_width)
    intrinsic[1, :] *= image_height / float(intrinsic_height)
    projection = compute_lidar2img(cam_info, intrinsic_override=intrinsic)
    points = np.concatenate(
        [boxes[:, None, :3], _gt_box_corners(boxes)], axis=1)
    points_h = np.concatenate(
        [points, np.ones((*points.shape[:2], 1), dtype=np.float32)], axis=2)
    projected = points_h @ projection[:3, :4].T
    depth = projected[:, :, 2]
    safe_depth = np.where(np.abs(depth) > np.finfo(np.float32).eps, depth, 1.0)
    xs = projected[:, :, 0] / safe_depth
    ys = projected[:, :, 1] / safe_depth
    visible = (
        (depth > float(min_depth)) &
        (xs >= 0.0) & (xs < float(image_width)) &
        (ys >= 0.0) & (ys < float(image_height))
    )
    return visible.any(axis=1)


def _resolve_image_path(data_path: str, data_root: Optional[Path]) -> Path:
    path = Path(str(data_path))
    if path.is_absolute() or data_root is None:
        return path
    return data_root / path


def _check_camera_contract(
    report: Dict,
    cam_info: Dict,
    context: str,
    size_hists: Dict[str, Counter],
    expected_image_size: Optional[Tuple[int, int]],
    expected_intrinsic_size: Optional[Tuple[int, int]],
    check_image_files: bool,
    data_root: Optional[Path],
    max_examples: int,
) -> bool:
    if not isinstance(cam_info, dict):
        _failure(report, '{} camera info is not dict'.format(context), max_examples)
        return False
    missing = [key for key in CAMERA_REQUIRED_FIELDS if key not in cam_info]
    if missing:
        _failure(report, '{} missing camera fields {}'.format(context, missing), max_examples)
        return False

    valid = True
    try:
        intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float64)
        rotation = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float64)
        translation = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float64).reshape(-1)
        distortion = np.asarray(cam_info['distortion'], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        _failure(report, '{} invalid camera arrays: {}'.format(context, exc), max_examples)
        return False

    for name, array, shape in (
            ('cam_intrinsic', intrinsic, (3, 3)),
            ('sensor2lidar_rotation', rotation, (3, 3)),
            ('sensor2lidar_translation', translation, (3,))):
        if array.shape != shape or not np.isfinite(array).all():
            _failure(
                report,
                '{} invalid {} shape/finite: {}'.format(context, name, array.shape),
                max_examples)
            valid = False
    if distortion.size == 0 or not np.isfinite(distortion).all():
        _failure(report, '{} distortion is empty or non-finite'.format(context), max_examples)
        valid = False

    if rotation.shape == (3, 3) and np.isfinite(rotation).all():
        orthogonal_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
        if orthogonal_error > 1e-3 or determinant_error > 1e-3:
            _failure(
                report,
                '{} rotation is not orthonormal: orth_err={:.6g} det_err={:.6g}'.format(
                    context, orthogonal_error, determinant_error),
                max_examples)
            valid = False
    if intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all():
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or abs(intrinsic[2, 2]) < 1e-9:
            _failure(report, '{} invalid focal length/K[2,2]'.format(context), max_examples)
            valid = False

    intrinsic_width = _safe_int(cam_info.get('intrinsic_width'))
    intrinsic_height = _safe_int(cam_info.get('intrinsic_height'))
    image_width = _safe_int(cam_info.get('image_width'))
    image_height = _safe_int(cam_info.get('image_height'))
    if min(intrinsic_width, intrinsic_height, image_width, image_height) <= 0:
        _failure(
            report,
            '{} non-positive sizes intrinsic={}x{} image={}x{}'.format(
                context, intrinsic_width, intrinsic_height, image_width, image_height),
            max_examples)
        valid = False
    size_hists['intrinsic_size']['{}x{}'.format(intrinsic_height, intrinsic_width)] += 1
    size_hists['image_size']['{}x{}'.format(image_height, image_width)] += 1
    size_hists['distortion_length'][str(int(distortion.size))] += 1
    if expected_image_size is not None and (image_height, image_width) != expected_image_size:
        _failure(
            report,
            '{} image size {}x{} != expected {}x{}'.format(
                context, image_height, image_width,
                expected_image_size[0], expected_image_size[1]),
            max_examples)
        valid = False
    if (expected_intrinsic_size is not None and
            (intrinsic_height, intrinsic_width) != expected_intrinsic_size):
        _failure(
            report,
            '{} intrinsic size {}x{} != expected {}x{}'.format(
                context, intrinsic_height, intrinsic_width,
                expected_intrinsic_size[0], expected_intrinsic_size[1]),
            max_examples)
        valid = False

    data_path = str(cam_info.get('data_path', '') or '').strip()
    if not data_path:
        _failure(report, '{} empty data_path'.format(context), max_examples)
        valid = False
    elif check_image_files:
        image_path = _resolve_image_path(data_path, data_root)
        report['image_file_checks'] += 1
        if not image_path.is_file():
            report['missing_image_files'] += 1
            _failure(report, '{} missing image {}'.format(context, image_path), max_examples)
            valid = False
        else:
            try:
                from PIL import Image
                with Image.open(image_path) as image:
                    actual_width, actual_height = image.size
                if (actual_height, actual_width) != (image_height, image_width):
                    report['image_header_mismatches'] += 1
                    _failure(
                        report,
                        '{} image header {}x{} != pkl {}x{}'.format(
                            context, actual_height, actual_width, image_height, image_width),
                        max_examples)
                    valid = False
            except Exception as exc:
                report['image_decode_errors'] += 1
                _failure(
                    report, '{} cannot read image header: {}'.format(context, exc),
                    max_examples)
                valid = False
    return valid


def audit_fastbev_payload(
    infos: Sequence[Dict],
    metadata: Dict,
    *,
    source_path: Path,
    split_name: str,
    expected_camera_ids: Optional[Sequence[str]] = None,
    mode: str = 'auto',
    expected_image_size: Optional[Tuple[int, int]] = None,
    expected_intrinsic_size: Optional[Tuple[int, int]] = None,
    visibility_camera_id: Optional[str] = None,
    check_image_files: bool = False,
    data_root: Optional[Path] = None,
    require_complete_history: bool = False,
    max_examples: int = 50,
) -> Dict:
    """全量审计一个 converter pkl payload，返回 JSON 可序列化报告。"""
    if mode not in ('auto', 'single-frame', 'temporal'):
        raise ValueError('unsupported data gate mode: {}'.format(mode))
    metadata_camera_ids = [str(value) for value in metadata.get('camera_ids', []) or []]
    expected_camera_ids = (
        [str(value) for value in expected_camera_ids]
        if expected_camera_ids is not None else metadata_camera_ids)
    expected_prev_count = len(metadata.get('adjacent_time_offsets_us', []) or [])
    if mode == 'auto':
        effective_mode = 'temporal' if expected_prev_count > 0 else 'single-frame'
    else:
        effective_mode = mode
    if visibility_camera_id is None:
        visibility_camera_id = 'cam0' if 'cam0' in expected_camera_ids else (
            expected_camera_ids[0] if expected_camera_ids else None)

    report = {
        'source_path': str(source_path),
        'split': split_name,
        'mode': effective_mode,
        'num_infos': int(len(infos)),
        'num_tokens': 0,
        'num_clips': 0,
        'num_gt': 0,
        'empty_gt_frames': 0,
        'nonempty_gt_frames': 0,
        'expected_camera_ids': list(expected_camera_ids),
        'metadata_camera_ids': metadata_camera_ids,
        'gt_box_origin': metadata.get('gt_box_origin', metadata.get('box_origin')),
        'camera_count_hist': {},
        'camera_id_frame_counts': {},
        'frames_with_camera_mismatch': 0,
        'invalid_gt_frames': 0,
        'invalid_boxes': 0,
        'class_object_counts': {},
        'class_frame_counts': {},
        'pinhole_visibility_camera': visibility_camera_id,
        'pinhole_visible_gt': 0,
        'pinhole_visible_frames': 0,
        'pinhole_visible_class_counts': {},
        'key_pose_coverage': 0.0,
        'key_frames_with_pose': 0,
        'key_frames_without_pose': 0,
        'adjacent_frames_checked': 0,
        'adjacent_frames_without_pose': 0,
        'expected_prev_count': int(expected_prev_count),
        'prev_count_hist': {},
        'frames_with_prev_shortage': 0,
        'actual_prev_offset_us': {},
        'target_prev_offset_us': {},
        'prev_offset_abs_error_us': {},
        'pose_time_diff_abs_us': {},
        'pose_source_counts': {},
        'image_size_hist': {},
        'intrinsic_size_hist': {},
        'distortion_length_hist': {},
        'image_file_checks': 0,
        'missing_image_files': 0,
        'image_header_mismatches': 0,
        'image_decode_errors': 0,
        'box_distribution': {},
        'box_distribution_by_class': {},
        'converter_stats': dict(metadata.get('stats', {}) or {}),
        'converter_keep_empty': metadata.get('keep_empty'),
        'manifest_sources': list(metadata.get('manifest_sources', []) or []),
        'auto_discovery_used': bool(metadata.get('auto_discovery_used', False)),
        'failed_checks': 0,
        'warning_checks': 0,
        'failures': [],
        'warnings': [],
    }
    if not infos:
        _failure(report, 'pkl contains no infos', max_examples)
    if not expected_camera_ids:
        _failure(report, 'expected camera ids are unknown; pass --expected-camera-ids', max_examples)
    if metadata_camera_ids and expected_camera_ids != metadata_camera_ids:
        _failure(
            report,
            'metadata camera_ids {} != expected {}'.format(
                metadata_camera_ids, expected_camera_ids),
            max_examples)
    if report['gt_box_origin'] is None:
        _warning(
            report,
            'GT box origin metadata is absent (legacy pkl); N7 converter contract expects center',
            max_examples)
    elif str(report['gt_box_origin']).lower() != 'center':
        _failure(
            report,
            'GT box origin must be center, got {}'.format(report['gt_box_origin']),
            max_examples)

    converter_stats = report['converter_stats']
    if converter_stats:
        for key in CONVERTER_CRITICAL_STATS:
            value = _safe_int(converter_stats.get(key, 0))
            if value > 0:
                _failure(
                    report, 'converter stats {}={} > 0'.format(key, value),
                    max_examples)
        failure_counts = converter_stats.get('failure_counts', {}) or {}
        if sum(_safe_int(value) for value in failure_counts.values()) > 0:
            _failure(
                report, 'converter recorded failures: {}'.format(failure_counts),
                max_examples)
        labels_without_gt = _safe_int(converter_stats.get('labels_without_gt', 0))
        if labels_without_gt > 0 and metadata.get('keep_empty') is not True:
            _warning(
                report,
                'converter dropped {} empty-GT labels; current pkl cannot recover these negatives'.format(
                    labels_without_gt),
                max_examples)
    else:
        _warning(report, 'metadata.stats is missing; raw/drop counts are unavailable', max_examples)
    if report['auto_discovery_used']:
        _failure(report, 'converter used automatic clip discovery instead of manifest', max_examples)
    elif not report['manifest_sources']:
        _warning(
            report,
            'manifest provenance is absent (legacy pkl); rely on cross-pkl overlap audit',
            max_examples)

    tokens = set()
    clips = set()
    camera_count_hist = Counter()
    camera_id_frame_counts = Counter()
    size_hists = {
        'image_size': Counter(),
        'intrinsic_size': Counter(),
        'distortion_length': Counter(),
    }
    class_object_counts = Counter()
    class_frame_counts = Counter()
    visible_class_counts = Counter()
    prev_count_hist = Counter()
    all_box_chunks: List[np.ndarray] = []
    class_box_chunks: Dict[str, List[np.ndarray]] = defaultdict(list)
    actual_prev_offsets: List[float] = []
    target_prev_offsets: List[float] = []
    prev_offset_errors: List[float] = []
    pose_time_diffs: List[float] = []
    pose_source_counts = Counter()

    for index, info in enumerate(infos):
        context = 'split={} index={}'.format(split_name, index)
        if not isinstance(info, dict):
            _failure(report, '{} info is not dict: {}'.format(context, type(info)), max_examples)
            continue
        token = str(info.get('token', '') or '')
        if not token:
            _failure(report, '{} missing token'.format(context), max_examples)
        elif token in tokens:
            _failure(report, '{} duplicate token={}'.format(context, token), max_examples)
        else:
            tokens.add(token)
        clip_key = _clip_key(info)
        if not clip_key.strip('/'):
            _failure(report, '{} missing dataset/sequence/clip identity'.format(context), max_examples)
        else:
            clips.add(clip_key)
        try:
            timestamp = int(info['timestamp'])
        except (KeyError, TypeError, ValueError, OverflowError):
            timestamp = 0
            _failure(report, '{} invalid timestamp'.format(context), max_examples)

        cams = info.get('cams', {}) or {}
        if not isinstance(cams, dict):
            cams = {}
            _failure(report, '{} cams is not dict'.format(context), max_examples)
        camera_ids = [str(value) for value in cams.keys()]
        camera_count_hist[str(len(camera_ids))] += 1
        camera_id_frame_counts.update(camera_ids)
        if camera_ids != expected_camera_ids:
            report['frames_with_camera_mismatch'] += 1
            _failure(
                report,
                '{} camera_ids {} != expected {}'.format(context, camera_ids, expected_camera_ids),
                max_examples)

        valid_cameras = {}
        for cam_id in expected_camera_ids:
            if cam_id not in cams:
                continue
            cam_context = '{} token={} cam={}'.format(context, token, cam_id)
            valid_cameras[cam_id] = _check_camera_contract(
                report=report,
                cam_info=cams[cam_id],
                context=cam_context,
                size_hists=size_hists,
                expected_image_size=expected_image_size,
                expected_intrinsic_size=expected_intrinsic_size,
                check_image_files=check_image_files,
                data_root=data_root,
                max_examples=max_examples)

        try:
            boxes = _box_array(info.get('gt_boxes', []))
        except (TypeError, ValueError) as exc:
            boxes = np.zeros((0, 7), dtype=np.float32)
            report['invalid_gt_frames'] += 1
            _failure(report, '{} invalid gt_boxes: {}'.format(context, exc), max_examples)
        names = np.asarray(info.get('gt_names', []), dtype=object).reshape(-1)
        try:
            velocities = np.asarray(info.get('gt_velocity', []), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            velocities = np.zeros((0, 2), dtype=np.float32)
            _failure(report, '{} invalid gt_velocity: {}'.format(context, exc), max_examples)
        if velocities.size == 0:
            velocities = np.zeros((0, 2), dtype=np.float32)
        elif velocities.ndim == 1:
            velocities = velocities.reshape(1, -1)
        if velocities.ndim != 2 or velocities.shape[1] != 2:
            _failure(
                report, '{} gt_velocity shape {} != Nx2'.format(context, velocities.shape),
                max_examples)
        try:
            track_ids = np.asarray(info.get('track_ids', []), dtype=object).reshape(-1)
        except (TypeError, ValueError) as exc:
            track_ids = np.asarray([], dtype=object)
            _failure(report, '{} invalid track_ids: {}'.format(context, exc), max_examples)
        counts = {
            'gt_boxes': int(len(boxes)),
            'gt_names': int(len(names)),
            'gt_velocity': int(len(velocities)),
            'track_ids': int(len(track_ids)),
        }
        if len(set(counts.values())) != 1:
            report['invalid_gt_frames'] += 1
            _failure(report, '{} GT field count mismatch {}'.format(context, counts), max_examples)
        gt_count = min(counts.values()) if counts else 0
        boxes = boxes[:gt_count]
        names = names[:gt_count]
        if boxes.ndim != 2 or boxes.shape[1] < 7:
            report['invalid_gt_frames'] += 1
            _failure(report, '{} gt_boxes shape {} lacks 7 dims'.format(context, boxes.shape), max_examples)
            boxes = np.zeros((0, 7), dtype=np.float32)
            names = names[:0]
            gt_count = 0
        else:
            boxes = boxes[:, :7]
            finite = np.isfinite(boxes).all(axis=1)
            positive_size = (boxes[:, 3:6] > 0).all(axis=1)
            invalid_box_mask = ~(finite & positive_size)
            invalid_count = int(invalid_box_mask.sum())
            if invalid_count:
                report['invalid_boxes'] += invalid_count
                _failure(
                    report, '{} contains {} non-finite/non-positive boxes'.format(
                        context, invalid_count),
                    max_examples)
            valid_mask = ~invalid_box_mask
            boxes = boxes[valid_mask]
            names = names[valid_mask]
            gt_count = len(boxes)

        report['num_gt'] += int(gt_count)
        if gt_count == 0:
            report['empty_gt_frames'] += 1
        else:
            report['nonempty_gt_frames'] += 1
            all_box_chunks.append(boxes)
            names_as_str = np.asarray([str(name) for name in names], dtype=object)
            frame_names = set(names_as_str.tolist())
            class_frame_counts.update(frame_names)
            class_object_counts.update(names_as_str.tolist())
            metadata_classes = set(str(value) for value in metadata.get('classes', []) or [])
            unknown_names = sorted(frame_names - metadata_classes) if metadata_classes else []
            if unknown_names:
                _failure(
                    report, '{} GT classes absent from metadata: {}'.format(context, unknown_names),
                    max_examples)
            for class_name in frame_names:
                class_box_chunks[class_name].append(boxes[names_as_str == class_name])

            if (visibility_camera_id in cams and
                    valid_cameras.get(visibility_camera_id, False)):
                try:
                    visible_mask = _pinhole_visible_mask(
                        boxes, cams[visibility_camera_id])
                    visible_count = int(visible_mask.sum())
                    report['pinhole_visible_gt'] += visible_count
                    if visible_count:
                        report['pinhole_visible_frames'] += 1
                        visible_class_counts.update(names_as_str[visible_mask].tolist())
                except Exception as exc:
                    _warning(
                        report,
                        '{} pinhole visibility audit failed: {}'.format(context, exc),
                        max_examples)

        key_has_pose = _has_pose(info)
        if key_has_pose:
            report['key_frames_with_pose'] += 1
        else:
            report['key_frames_without_pose'] += 1
        if info.get('pose_source'):
            pose_source_counts[str(info['pose_source'])] += 1
        if info.get('pose_time_diff_us') is not None:
            try:
                pose_time_diffs.append(abs(float(info['pose_time_diff_us'])))
            except (TypeError, ValueError, OverflowError):
                _failure(report, '{} invalid pose_time_diff_us'.format(context), max_examples)

        prev_infos = info.get('prev') or []
        if isinstance(prev_infos, dict):
            prev_infos = [prev_infos]
        if not isinstance(prev_infos, (list, tuple)):
            prev_infos = []
            _failure(report, '{} prev is not list/dict'.format(context), max_examples)
        prev_count = len(prev_infos)
        prev_count_hist[str(prev_count)] += 1
        if expected_prev_count and prev_count < expected_prev_count:
            report['frames_with_prev_shortage'] += 1
        for prev_index, prev_info in enumerate(prev_infos):
            report['adjacent_frames_checked'] += 1
            prev_context = '{} prev[{}]'.format(context, prev_index)
            if not isinstance(prev_info, dict):
                report['adjacent_frames_without_pose'] += 1
                _failure(report, '{} is not dict'.format(prev_context), max_examples)
                continue
            if not _has_pose(prev_info):
                report['adjacent_frames_without_pose'] += 1
            if prev_info.get('pose_source'):
                pose_source_counts[str(prev_info['pose_source'])] += 1
            prev_cams = prev_info.get('cams', {}) or {}
            if [str(value) for value in prev_cams.keys()] != expected_camera_ids:
                _failure(
                    report,
                    '{} camera_ids {} != expected {}'.format(
                        prev_context, list(prev_cams.keys()), expected_camera_ids),
                    max_examples)
            # 相邻帧会被训练 dataloader 实际读取，因此也检查完整相机契约；尺寸
            # histogram 只统计 key frame，避免历史复用导致分布重复计数。
            dummy_hists = {
                'image_size': Counter(),
                'intrinsic_size': Counter(),
                'distortion_length': Counter(),
            }
            for cam_id in expected_camera_ids:
                if cam_id not in prev_cams:
                    continue
                _check_camera_contract(
                    report=report,
                    cam_info=prev_cams[cam_id],
                    context='{} cam={}'.format(prev_context, cam_id),
                    size_hists=dummy_hists,
                    expected_image_size=expected_image_size,
                    expected_intrinsic_size=expected_intrinsic_size,
                    check_image_files=False,
                    data_root=data_root,
                    max_examples=max_examples)
            try:
                prev_timestamp = int(prev_info['timestamp'])
                actual_offset = timestamp - prev_timestamp
                actual_prev_offsets.append(float(actual_offset))
                target_offset = prev_info.get('target_time_offset_us')
                if target_offset is not None:
                    target_offset = float(target_offset)
                    target_prev_offsets.append(target_offset)
                    prev_offset_errors.append(abs(float(actual_offset) - target_offset))
            except (KeyError, TypeError, ValueError, OverflowError):
                _failure(report, '{} invalid timestamp/offset'.format(prev_context), max_examples)
            if prev_info.get('pose_time_diff_us') is not None:
                try:
                    pose_time_diffs.append(abs(float(prev_info['pose_time_diff_us'])))
                except (TypeError, ValueError, OverflowError):
                    _failure(
                        report, '{} invalid pose_time_diff_us'.format(prev_context),
                        max_examples)

    report['num_tokens'] = len(tokens)
    report['num_clips'] = len(clips)
    report['camera_count_hist'] = _counter_dict(camera_count_hist)
    report['camera_id_frame_counts'] = _counter_dict(camera_id_frame_counts)
    report['image_size_hist'] = _counter_dict(size_hists['image_size'])
    report['intrinsic_size_hist'] = _counter_dict(size_hists['intrinsic_size'])
    report['distortion_length_hist'] = _counter_dict(size_hists['distortion_length'])
    report['class_object_counts'] = _counter_dict(class_object_counts)
    report['class_frame_counts'] = _counter_dict(class_frame_counts)
    report['pinhole_visible_class_counts'] = _counter_dict(visible_class_counts)
    report['prev_count_hist'] = _counter_dict(prev_count_hist)
    report['box_distribution'] = _box_distribution(all_box_chunks)
    report['box_distribution_by_class'] = {
        class_name: _box_distribution(chunks)
        for class_name, chunks in sorted(class_box_chunks.items())
    }
    report['actual_prev_offset_us'] = _numeric_summary(actual_prev_offsets)
    report['target_prev_offset_us'] = _numeric_summary(target_prev_offsets)
    report['prev_offset_abs_error_us'] = _numeric_summary(prev_offset_errors)
    report['pose_time_diff_abs_us'] = _numeric_summary(pose_time_diffs)
    report['pose_source_counts'] = _counter_dict(pose_source_counts)
    report['key_pose_coverage'] = (
        report['key_frames_with_pose'] / float(len(infos)) if infos else 0.0)

    if len(report['image_size_hist']) > 1:
        _warning(
            report, 'multiple key-frame image sizes: {}'.format(report['image_size_hist']),
            max_examples)
    if len(report['intrinsic_size_hist']) > 1:
        _warning(
            report, 'multiple intrinsic sizes: {}'.format(report['intrinsic_size_hist']),
            max_examples)
    if len(report['distortion_length_hist']) > 1:
        _warning(
            report, 'multiple distortion lengths: {}'.format(report['distortion_length_hist']),
            max_examples)
    max_pose_match_us = _safe_int(metadata.get('max_pose_match_us', 0))
    pose_diff_max = report['pose_time_diff_abs_us'].get('max')
    if (max_pose_match_us > 0 and pose_diff_max is not None and
            pose_diff_max > max_pose_match_us):
        _failure(
            report,
            'pose_time_diff_us max {} exceeds metadata max_pose_match_us {}'.format(
                pose_diff_max, max_pose_match_us),
            max_examples)
    if effective_mode == 'temporal':
        if report['key_frames_without_pose']:
            _failure(
                report,
                'temporal mode has {} key frames without pose'.format(
                    report['key_frames_without_pose']),
                max_examples)
        if report['adjacent_frames_without_pose']:
            _failure(
                report,
                'temporal mode has {} adjacent frames without pose'.format(
                    report['adjacent_frames_without_pose']),
                max_examples)
        if require_complete_history and report['frames_with_prev_shortage']:
            _failure(
                report,
                '{} frames have fewer than {} histories'.format(
                    report['frames_with_prev_shortage'], expected_prev_count),
                max_examples)
        elif report['frames_with_prev_shortage']:
            _warning(
                report,
                '{} frames have history shortage; dataset will reuse earliest/current frame'.format(
                    report['frames_with_prev_shortage']),
                max_examples)
    elif report['key_frames_without_pose'] or report['adjacent_frames_without_pose']:
        _warning(
            report,
            'single-frame mode does not require pose; missing key/adjacent pose={}/{} is retained for audit'.format(
                report['key_frames_without_pose'], report['adjacent_frames_without_pose']),
            max_examples)

    report['status'] = 'PASS' if report['failed_checks'] == 0 else 'FAIL'
    return report


def audit_fastbev_splits(
    payloads: Sequence[Tuple[Path, Sequence[Dict], Dict]],
    *,
    expected_camera_ids: Optional[Sequence[str]] = None,
    mode: str = 'auto',
    expected_image_size: Optional[Tuple[int, int]] = None,
    expected_intrinsic_size: Optional[Tuple[int, int]] = None,
    visibility_camera_id: Optional[str] = None,
    check_image_files: bool = False,
    data_root: Optional[Path] = None,
    require_complete_history: bool = False,
    required_splits: Optional[Sequence[str]] = None,
    max_examples: int = 50,
) -> Dict:
    """审计若干 train/val/test payload，并检查 token/clip 零重叠。"""
    report = {
        'schema_version': DATA_GATE_SCHEMA_VERSION,
        'generated_at': datetime.now().isoformat(),
        'status': 'PASS',
        'failed_checks': 0,
        'warning_checks': 0,
        'failures': [],
        'warnings': [],
        'split_reports': [],
        'split_overlap': [],
        'required_splits': list(required_splits or []),
    }
    split_inputs = []
    split_names = []
    for index, (path, infos, metadata) in enumerate(payloads):
        split_name = _split_name(Path(path), metadata, index)
        split_names.append(split_name)
        split_report = audit_fastbev_payload(
            infos=infos,
            metadata=metadata,
            source_path=Path(path),
            split_name=split_name,
            expected_camera_ids=expected_camera_ids,
            mode=mode,
            expected_image_size=expected_image_size,
            expected_intrinsic_size=expected_intrinsic_size,
            visibility_camera_id=visibility_camera_id,
            check_image_files=check_image_files,
            data_root=data_root,
            require_complete_history=require_complete_history,
            max_examples=max_examples)
        report['split_reports'].append(split_report)
        report['failed_checks'] += split_report['failed_checks']
        report['warning_checks'] += split_report['warning_checks']
        token_set = {
            str(info.get('token')) for info in infos
            if isinstance(info, dict) and info.get('token') is not None
        }
        clip_set = {
            _clip_key(info) for info in infos
            if isinstance(info, dict) and _clip_key(info).strip('/')
        }
        split_inputs.append((split_name, Path(path), token_set, clip_set))

    duplicate_split_names = [
        name for name, count in Counter(split_names).items() if count > 1
    ]
    if duplicate_split_names:
        _failure(
            report,
            'multiple input pkls resolve to the same split names: {}'.format(
                sorted(duplicate_split_names)),
            max_examples)
    if required_splits:
        missing_splits = sorted(set(required_splits) - set(split_names))
        if missing_splits:
            _failure(
                report, 'required splits are missing: {}'.format(missing_splits),
                max_examples)

    for left, right in combinations(split_inputs, 2):
        left_name, left_path, left_tokens, left_clips = left
        right_name, right_path, right_tokens, right_clips = right
        token_overlap = sorted(left_tokens & right_tokens)
        clip_overlap = sorted(left_clips & right_clips)
        overlap_row = {
            'left_split': left_name,
            'right_split': right_name,
            'left_path': str(left_path),
            'right_path': str(right_path),
            'token_overlap_count': len(token_overlap),
            'clip_overlap_count': len(clip_overlap),
            'token_overlap_examples': token_overlap[:max_examples],
            'clip_overlap_examples': clip_overlap[:max_examples],
        }
        report['split_overlap'].append(overlap_row)
        if token_overlap:
            _failure(
                report,
                '{} vs {} token overlap={} examples={}'.format(
                    left_name, right_name, len(token_overlap), token_overlap[:5]),
                max_examples)
        if clip_overlap:
            _failure(
                report,
                '{} vs {} clip overlap={} examples={}'.format(
                    left_name, right_name, len(clip_overlap), clip_overlap[:5]),
                max_examples)

    report['status'] = 'PASS' if report['failed_checks'] == 0 else 'FAIL'
    return report


def _fmt(value, digits=3) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return ('{:.%df}' % digits).format(value)
    return str(value)


def data_gate_markdown(report: Dict) -> str:
    """把完整 JSON 报告整理成便于实验归档的 Markdown 摘要。"""
    lines = [
        '# N7 Fast-BEV pkl data gate',
        '',
        '- status: `{}`'.format(report['status']),
        '- generated_at: `{}`'.format(report['generated_at']),
        '- failed_checks: `{}`'.format(report['failed_checks']),
        '- warning_checks: `{}`'.format(report['warning_checks']),
        '',
        '## Split summary',
        '',
        '| split | infos | clips | GT | empty frames | cameras | key pose | prev shortage | status |',
        '|---|---:|---:|---:|---:|---|---:|---:|---|',
    ]
    for item in report['split_reports']:
        lines.append(
            '| {} | {} | {} | {} | {} | {} | {:.2%} | {} | {} |'.format(
                item['split'], item['num_infos'], item['num_clips'], item['num_gt'],
                item['empty_gt_frames'], ','.join(item['expected_camera_ids']),
                item['key_pose_coverage'], item['frames_with_prev_shortage'], item['status']))

    lines.extend([
        '',
        '## Split overlap',
        '',
        '| pair | token overlap | clip overlap |',
        '|---|---:|---:|',
    ])
    if report['split_overlap']:
        for row in report['split_overlap']:
            lines.append(
                '| {} vs {} | {} | {} |'.format(
                    row['left_split'], row['right_split'],
                    row['token_overlap_count'], row['clip_overlap_count']))
    else:
        lines.append('| n/a | 0 | 0 |')

    for item in report['split_reports']:
        lines.extend([
            '',
            '## {}'.format(item['split']),
            '',
            '- source: `{}`'.format(item['source_path']),
            '- gt_box_origin: `{}`'.format(item['gt_box_origin']),
            '- image_size_hist: `{}`'.format(item['image_size_hist']),
            '- intrinsic_size_hist: `{}`'.format(item['intrinsic_size_hist']),
            '- distortion_length_hist: `{}`'.format(item['distortion_length_hist']),
            '- prev_count_hist: `{}`'.format(item['prev_count_hist']),
            '- pose_source_counts: `{}`'.format(item['pose_source_counts']),
            '- pinhole visible via `{}`: frames={}, GT={}'.format(
                item['pinhole_visibility_camera'], item['pinhole_visible_frames'],
                item['pinhole_visible_gt']),
            '- converter_stats: `{}`'.format(item['converter_stats']),
            '',
            '### Classes',
            '',
            '| class | objects | frames | pinhole visible |',
            '|---|---:|---:|---:|',
        ])
        classes = sorted(set(item['class_object_counts']) | set(item['class_frame_counts']))
        for class_name in classes:
            lines.append(
                '| {} | {} | {} | {} |'.format(
                    class_name,
                    item['class_object_counts'].get(class_name, 0),
                    item['class_frame_counts'].get(class_name, 0),
                    item['pinhole_visible_class_counts'].get(class_name, 0)))
        lines.extend([
            '',
            '### Box distribution (all classes)',
            '',
            '| field | min | mean | p50 | p90 | p99 | max |',
            '|---|---:|---:|---:|---:|---:|---:|',
        ])
        for field_name in BOX_FIELDS:
            values = item['box_distribution'].get(field_name, {})
            lines.append(
                '| {} | {} | {} | {} | {} | {} | {} |'.format(
                    field_name, _fmt(values.get('min')), _fmt(values.get('mean')),
                    _fmt(values.get('p50')), _fmt(values.get('p90')),
                    _fmt(values.get('p99')), _fmt(values.get('max'))))
        lines.extend([
            '',
            '### Box distribution by class',
            '',
            '| class | field | mean | p50 | p90 | p99 |',
            '|---|---|---:|---:|---:|---:|',
        ])
        for class_name, distribution in item['box_distribution_by_class'].items():
            for field_name in BOX_FIELDS:
                values = distribution.get(field_name, {})
                lines.append(
                    '| {} | {} | {} | {} | {} | {} |'.format(
                        class_name, field_name, _fmt(values.get('mean')),
                        _fmt(values.get('p50')), _fmt(values.get('p90')),
                        _fmt(values.get('p99'))))
        if item['failures']:
            lines.extend(['', '### Failures', ''])
            lines.extend('- {}'.format(message) for message in item['failures'])
        if item['warnings']:
            lines.extend(['', '### Warnings', ''])
            lines.extend('- {}'.format(message) for message in item['warnings'])

    if report['failures']:
        lines.extend(['', '## Global failures', ''])
        lines.extend('- {}'.format(message) for message in report['failures'])
    if report['warnings']:
        lines.extend(['', '## Global warnings', ''])
        lines.extend('- {}'.format(message) for message in report['warnings'])
    lines.append('')
    return '\n'.join(lines)


def write_data_gate_report(output_dir: Path, report: Dict) -> Tuple[Path, Path]:
    """同时写出完整 JSON 和便于人工阅读的 Markdown。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'data_gate.json'
    markdown_path = output_dir / 'data_gate.md'
    with json_path.open('w', encoding='utf-8') as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
    with markdown_path.open('w', encoding='utf-8') as file_obj:
        file_obj.write(data_gate_markdown(report))
    return json_path, markdown_path


__all__ = [
    'DATA_GATE_SCHEMA_VERSION',
    'audit_fastbev_payload',
    'audit_fastbev_splits',
    'data_gate_markdown',
    'write_data_gate_report',
]
