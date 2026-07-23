#!/usr/bin/env python3
"""N7 点云辅助 GT 可视化和诊断。

第一版只做离线诊断，不修改 pkl，也不参与训练/eval 过滤逻辑。脚本读取
Fast-BEV N7 pkl、可选原始点云和 cam0 图片，输出逐帧 BEV/前视图以及逐 GT
统计 CSV/JSON，便于人工判断后续是否需要清洗 GT 或增加可见性字段。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from tools.data_converter.n7.fastbev_geometry import compute_lidar2img, to_numpy
    from tools.data_converter.n7.visualize_n7_fastbev_pkl import (
        BOX_EDGES,
        FILTERED_GT_COLOR,
        MONO_FRONT_GT_FILTER_RANGE,
        RawDefaultsHelpFormatter,
        add_top_header,
        bev_to_pixel,
        camera_display_name,
        camera_info_for_loaded_image,
        camera_info_for_pkl_image_size,
        class_color,
        clipped_image_point,
        corners_from_boxes,
        draw_bev_box_by_source,
        draw_bev_box_heading,
        draw_bev_grid,
        draw_label_tag,
        draw_projected_box,
        frame_stem_from_info,
        get_class_names,
        gt_filter_range_mask,
        gt_visible_camera_mask,
        gt_used_mask,
        info_output_dir,
        load_fastbev_pkl,
        make_metric_bev_range,
        placeholder_image,
        project_lidar_points_distorted,
        project_lidar_points_pinhole,
        resize_to_width,
        resize_to_width_with_scale,
        resolve_image_path,
        scaled_camera_info,
        selected_infos,
        undistort_if_requested,
    )
except ModuleNotFoundError:
    from fastbev_geometry import compute_lidar2img, to_numpy
    from visualize_n7_fastbev_pkl import (
        BOX_EDGES,
        FILTERED_GT_COLOR,
        MONO_FRONT_GT_FILTER_RANGE,
        RawDefaultsHelpFormatter,
        add_top_header,
        bev_to_pixel,
        camera_display_name,
        camera_info_for_loaded_image,
        camera_info_for_pkl_image_size,
        class_color,
        clipped_image_point,
        corners_from_boxes,
        draw_bev_box_by_source,
        draw_bev_box_heading,
        draw_bev_grid,
        draw_label_tag,
        draw_projected_box,
        frame_stem_from_info,
        get_class_names,
        gt_filter_range_mask,
        gt_visible_camera_mask,
        gt_used_mask,
        info_output_dir,
        load_fastbev_pkl,
        make_metric_bev_range,
        placeholder_image,
        project_lidar_points_distorted,
        project_lidar_points_pinhole,
        resize_to_width,
        resize_to_width_with_scale,
        resolve_image_path,
        scaled_camera_info,
        selected_infos,
        undistort_if_requested,
    )


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


USAGE_EXAMPLES = r"""示例：
  # cam0 mono-front：读取 pkl、图片和原始点云，输出 BEV/前视图与 CSV/JSON。
  python tools/data_converter/n7/n7_pointcloud_gt_diagnostics.py \
--pkl data/N7_704_256/pkl/custom_fastbev_xxx_infos_val_20260703.pkl \
--data-root data/N7_704_256 \
--pointcloud-root /path/to/N7_raw_pointcloud \
--output-dir work_dirs/n7_pc_gt_diag \
--camera-id cam0 --bev-range 0 -35 80 35 --max-frames 100

  # 内网点云结构不在默认候选路径内时，使用占位符显式指定。
  python tools/data_converter/n7/n7_pointcloud_gt_diagnostics.py \
--pkl data/N7_704_256/pkl/custom_fastbev_xxx_infos_val_20260703.pkl \
--data-root data/N7_704_256 \
--pointcloud-root /mnt/n7_raw \
--pointcloud-glob '{sequence}/parsed_data/{clip}/lidar_main/{timestamp}.pkl' \
--output-dir work_dirs/n7_pc_gt_diag --frame-indices 0,10,20-30

  # 如果传入的点云已经是 Fast-BEV lidar 坐标，避免再次应用 raw_to_fastbev。
  python tools/data_converter/n7/n7_pointcloud_gt_diagnostics.py \
--pkl /tmp/smoke.pkl --data-root /tmp/smoke --pointcloud-root /tmp/smoke \
--pointcloud-coordinate fastbev --output-dir /tmp/n7_diag_smoke --frame-indices 0
"""


SUPPORTED_POINT_EXTS = ('.pkl', '.pcd', '.bin', '.npy', '.npz', '.txt', '.csv')
DEFAULT_CLASS_NAMES = ['car', 'truck']
DEFAULT_BEV_RANGE = (0.0, -35.0, 80.0, 35.0)
DEFAULT_GT_FILTER_RANGE = MONO_FRONT_GT_FILTER_RANGE
RAW_GT_COLOR = (205, 205, 205)
ROI_COLOR = (0, 220, 255)
POINT_COLOR = (88, 88, 88)


CSV_FIELDS = [
    'frame_index', 'frame_token', 'timestamp', 'dataset', 'sequence', 'clip',
    'gt_index', 'gt_token', 'track_id', 'class',
    'used_by_mono_front', 'filtered_reason',
    'class_in_use', 'in_roi', 'cam0_geometry_visible',
    'center_distance_m',
    'box_x', 'box_y', 'box_z', 'box_l', 'box_w', 'box_h', 'box_yaw',
    'box_point_count',
    'box_point_lidar_x_min', 'box_point_lidar_x_max',
    'box_point_lidar_z_min', 'box_point_lidar_z_max',
    'box_point_cam_depth_min', 'box_point_cam_depth_max',
    'box_point_density',
    'projected_2d_bbox_x1', 'projected_2d_bbox_y1',
    'projected_2d_bbox_x2', 'projected_2d_bbox_y2',
    'projected_2d_bbox_area',
    'pointcloud_path', 'pointcloud_status',
]


def parse_frame_indices(text: Optional[str]) -> Optional[List[int]]:
    """解析 ``0,2,10-20`` 形式的帧索引列表。"""
    if text is None or str(text).strip() == '':
        return None
    indices = set()
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            left, right = part.split('-', 1)
            start, end = int(left), int(right)
            if end < start:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return sorted(i for i in indices if i >= 0)


def selected_frame_items(
    infos: Sequence[Dict],
    frame_indices: Optional[Sequence[int]],
    start_index: int,
    stride: int,
    max_frames: Optional[int],
) -> List[Tuple[int, Dict]]:
    if frame_indices is not None:
        items = [(idx, infos[idx]) for idx in frame_indices if 0 <= idx < len(infos)]
        return items[:max_frames] if max_frames is not None else items
    return list(selected_infos(infos, start_index, stride, max_frames))


def as_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def sanitize_path_part(value: Any) -> str:
    text = str(value or '')
    keep = []
    for ch in text:
        keep.append(ch if (ch.isalnum() or ch in ('-', '_', '.')) else '_')
    return ''.join(keep).strip('_') or 'unknown'


def format_info_fields(info: Dict) -> Dict[str, str]:
    timestamp = info.get('timestamp')
    frame_timestamp = info.get('frame_timestamp', timestamp)
    fields = {
        'dataset': sanitize_path_part(info.get('dataset') or 'unknown_dataset'),
        'sequence': sanitize_path_part(info.get('sequence') or 'unknown_sequence'),
        'clip': sanitize_path_part(info.get('clip') or info.get('clip_id') or 'unknown_clip'),
        'timestamp': sanitize_path_part(timestamp if timestamp is not None else 'unknown_ts'),
        'frame_timestamp': sanitize_path_part(frame_timestamp if frame_timestamp is not None else timestamp),
        'token': sanitize_path_part(info.get('token') or timestamp or 'unknown_token'),
    }
    return fields


def unique_existing_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    result = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            result.append(path)
    return result


def glob_pattern(base: Optional[Path], pattern: str, fields: Dict[str, str]) -> List[Path]:
    formatted = pattern.format(**fields)
    path = Path(formatted)
    if not path.is_absolute() and base is not None:
        path = base / path
    if any(ch in str(path) for ch in '*?[]'):
        matches = [Path(item) for item in sorted(glob.glob(str(path), recursive=True))]
    else:
        matches = [path]
    return [match for match in matches if match.is_file()]


def candidate_pointcloud_dirs(root: Path, info: Dict) -> List[Path]:
    fields = format_info_fields(info)
    dataset = fields['dataset']
    sequence = fields['sequence']
    clip = fields['clip']
    ts = fields['timestamp']
    frame_ts = fields['frame_timestamp']
    timestamps = [ts]
    if frame_ts != ts:
        timestamps.append(frame_ts)

    dirs: List[Path] = []
    for stamp in timestamps:
        dirs.extend([
            root / dataset / sequence / 'parsed_data' / clip / 'frames' / stamp,
            root / dataset / sequence / 'parsed_data' / clip / 'frames' / stamp / 'lidar',
            root / dataset / sequence / 'parsed_data' / clip / 'frames' / stamp / 'lidar_main',
            root / dataset / sequence / 'parsed_data' / clip / 'frames' / stamp / 'pointcloud',
            root / dataset / 'parsed_data' / sequence / clip / 'frames' / stamp,
            root / dataset / 'parsed_data' / sequence / clip / 'frames' / stamp / 'lidar',
            root / dataset / 'parsed_data' / clip / 'frames' / stamp,
            root / dataset / 'parsed_data' / clip / 'frames' / stamp / 'lidar',
            root / sequence / 'parsed_data' / clip / 'frames' / stamp,
            root / sequence / 'parsed_data' / clip / 'frames' / stamp / 'lidar',
            root / sequence / 'parsed_data' / clip / 'frames' / stamp / 'lidar_main',
        ])
    dirs.extend([
        root / dataset / sequence / 'parsed_data' / clip / 'lidar',
        root / dataset / sequence / 'parsed_data' / clip / 'lidar_main',
        root / dataset / sequence / 'parsed_data' / clip / 'pointcloud',
        root / dataset / sequence / 'parsed_data' / clip / 'points',
        root / dataset / sequence / clip / 'lidar',
        root / dataset / sequence / clip / 'pointcloud',
        root / dataset / sequence / 'pointcloud' / clip,
        root / dataset / sequence / 'lidar' / clip,
        root / sequence / 'parsed_data' / clip / 'lidar',
        root / sequence / 'parsed_data' / clip / 'lidar_main',
        root / sequence / 'parsed_data' / clip / 'pointcloud',
        root / sequence / 'parsed_data' / clip / 'points',
    ])
    return dirs


def find_pointcloud_in_dir(directory: Path, info: Dict) -> List[Path]:
    if not directory.is_dir():
        return []
    fields = format_info_fields(info)
    stems = [
        fields['timestamp'],
        fields['frame_timestamp'],
        fields['token'],
        'lidar',
        'lidar_main',
        'pointcloud',
        'points',
    ]
    candidates = []
    for stem in stems:
        for ext in SUPPORTED_POINT_EXTS:
            candidates.append(directory / f'{stem}{ext}')
    existing = unique_existing_paths(candidates)
    if existing:
        return existing

    try:
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_POINT_EXTS)
    except OSError:
        return []
    if len(files) == 1:
        return files
    token_parts = [fields['timestamp'], fields['frame_timestamp'], fields['token']]
    return [path for path in files if any(part and part in path.stem for part in token_parts)]


def resolve_pointcloud_path(
    info: Dict,
    data_root: Optional[Path],
    pointcloud_root: Optional[Path],
    pointcloud_glob: Optional[str],
) -> Optional[Path]:
    fields = format_info_fields(info)
    roots = [root for root in (pointcloud_root, data_root) if root is not None]

    if pointcloud_glob:
        for root in roots or [None]:
            matches = glob_pattern(root, pointcloud_glob, fields)
            if matches:
                return matches[0]

    path_keys = (
        'lidar_path', 'pointcloud_path', 'point_cloud_path',
        'pcd_path', 'bin_path', 'velodyne_path')
    raw_paths = [info.get(key) for key in path_keys if info.get(key)]
    for raw_path in raw_paths:
        path = Path(str(raw_path))
        candidates = [path] if path.is_absolute() else [root / path for root in roots]
        existing = unique_existing_paths(candidates)
        if existing:
            return existing[0]

    for root in roots:
        for directory in candidate_pointcloud_dirs(root, info):
            matches = find_pointcloud_in_dir(directory, info)
            if matches:
                return matches[0]
    return None


def read_ascii_points(path: Path, delimiter: Optional[str]) -> np.ndarray:
    arr = np.loadtxt(str(path), delimiter=delimiter, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError(f'{path} 至少需要 3 列 xyz')
    return arr[:, :3].astype(np.float32)


def pcd_dtype(fields: Sequence[str], sizes: Sequence[int], types: Sequence[str], counts: Sequence[int]) -> np.dtype:
    dtype_fields = []
    for name, size, typ, count in zip(fields, sizes, types, counts):
        if typ == 'F':
            base = {4: np.float32, 8: np.float64}.get(size)
        elif typ == 'U':
            base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(size)
        elif typ == 'I':
            base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}.get(size)
        else:
            base = None
        if base is None:
            raise ValueError(f'Unsupported PCD field type={typ} size={size}')
        if count <= 1:
            dtype_fields.append((name, base))
        else:
            dtype_fields.append((name, base, (count,)))
    return np.dtype(dtype_fields)


def read_pcd_points(path: Path) -> np.ndarray:
    with path.open('rb') as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'{path} 缺少 PCD DATA header')
            decoded = line.decode('utf-8', errors='ignore').strip()
            header_lines.append(decoded)
            if decoded.upper().startswith('DATA '):
                break
        data_bytes = f.read()

    header: Dict[str, List[str]] = {}
    data_type = 'ascii'
    for line in header_lines:
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        key = parts[0].upper()
        values = parts[1:]
        if key == 'DATA':
            data_type = values[0].lower() if values else 'ascii'
        else:
            header[key] = values

    fields = header.get('FIELDS', [])
    if not all(name in fields for name in ('x', 'y', 'z')):
        raise ValueError(f'{path} PCD FIELDS 缺少 x/y/z')
    points_count = int(header.get('POINTS', header.get('WIDTH', ['0']))[0])
    if points_count <= 0:
        width = int(header.get('WIDTH', ['0'])[0])
        height = int(header.get('HEIGHT', ['1'])[0])
        points_count = width * height

    if data_type == 'ascii':
        text = data_bytes.decode('utf-8', errors='ignore')
        arr = np.fromstring(text, sep=' ', dtype=np.float32)
        if len(fields) == 0 or arr.size % len(fields) != 0:
            raise ValueError(f'{path} ASCII PCD 数据列数和 FIELDS 不匹配')
        arr = arr.reshape(-1, len(fields))
        xyz_idx = [fields.index(name) for name in ('x', 'y', 'z')]
        return arr[:, xyz_idx].astype(np.float32)

    if data_type != 'binary':
        raise ValueError(f'{path} 暂不支持 DATA {data_type}')

    sizes = [int(x) for x in header.get('SIZE', ['4'] * len(fields))]
    types = header.get('TYPE', ['F'] * len(fields))
    counts = [int(x) for x in header.get('COUNT', ['1'] * len(fields))]
    dtype = pcd_dtype(fields, sizes, types, counts)
    arr = np.frombuffer(data_bytes, dtype=dtype, count=points_count)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1)
    return xyz.astype(np.float32)


def array_to_xyz(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype.names:
        names = set(arr.dtype.names)
        if {'x', 'y', 'z'}.issubset(names):
            return np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)
        return None
    if arr.ndim == 1:
        if arr.size >= 3 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        else:
            return None
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3].astype(np.float32)
    return None


def extract_xyz_from_pickle_payload(payload: Any, depth: int = 0) -> Optional[np.ndarray]:
    """从常见点云 pkl payload 中提取 Nx3 xyz。

    内网导出的点云 pkl 具体 key 可能随任务变化，这里只做轻量、可解释的兼容：
    先找常见点云字段，再兜底递归检查 dict/list/tuple 中第一个 Nx3-like 数组。
    """
    if depth > 3:
        return None
    direct = array_to_xyz(payload)
    if direct is not None:
        return direct
    if isinstance(payload, dict):
        lowered = {str(key).lower(): key for key in payload.keys()}
        for key in ('points', 'pointcloud', 'point_cloud', 'lidar_points',
                    'lidar', 'lidar_main', 'xyz', 'data'):
            real_key = lowered.get(key)
            if real_key is None:
                continue
            result = extract_xyz_from_pickle_payload(payload[real_key], depth + 1)
            if result is not None:
                return result
        if all(key in lowered for key in ('x', 'y', 'z')):
            x = np.asarray(payload[lowered['x']]).reshape(-1)
            y = np.asarray(payload[lowered['y']]).reshape(-1)
            z = np.asarray(payload[lowered['z']]).reshape(-1)
            count = min(x.size, y.size, z.size)
            if count > 0:
                return np.stack([x[:count], y[:count], z[:count]], axis=1).astype(np.float32)
        for value in payload.values():
            result = extract_xyz_from_pickle_payload(value, depth + 1)
            if result is not None:
                return result
    if isinstance(payload, (list, tuple)):
        for value in payload:
            result = extract_xyz_from_pickle_payload(value, depth + 1)
            if result is not None:
                return result
    return None


def read_pickle_points(path: Path) -> np.ndarray:
    with path.open('rb') as f:
        payload = pickle.load(f)
    points = extract_xyz_from_pickle_payload(payload)
    if points is None:
        raise ValueError(f'{path} pkl 中没有找到 Nx3 点云数组或 x/y/z 字段')
    return points.astype(np.float32)


def read_pointcloud(path: Path, bin_dim: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == '.pkl':
        points = read_pickle_points(path)
    elif suffix == '.pcd':
        points = read_pcd_points(path)
    elif suffix == '.npy':
        arr = np.load(str(path))
        points = np.asarray(arr, dtype=np.float32)
    elif suffix == '.npz':
        payload = np.load(str(path))
        key = 'points' if 'points' in payload.files else payload.files[0]
        points = np.asarray(payload[key], dtype=np.float32)
    elif suffix == '.bin':
        raw = np.fromfile(str(path), dtype=np.float32)
        if raw.size == 0:
            points = np.zeros((0, 3), dtype=np.float32)
        else:
            dim = int(bin_dim)
            if dim <= 0:
                for candidate in (4, 5, 6, 3):
                    if raw.size % candidate == 0:
                        dim = candidate
                        break
                if dim <= 0:
                    raise ValueError(f'{path} 无法自动推断 bin 点维度')
            if raw.size % dim != 0:
                raise ValueError(f'{path} float32 数量 {raw.size} 不能整除 --pointcloud-bin-dim={dim}')
            points = raw.reshape(-1, dim)
    elif suffix in ('.txt', '.csv'):
        points = read_ascii_points(path, delimiter=',' if suffix == '.csv' else None)
    else:
        raise ValueError(f'Unsupported pointcloud extension: {path.suffix}')

    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f'{path} 点云至少需要 3 列 xyz')
    points = np.asarray(points[:, :3], dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def transform_points_to_fastbev(
    points: np.ndarray,
    metadata: Dict,
    coordinate_mode: str,
) -> Tuple[np.ndarray, str]:
    if points.size == 0:
        return points.reshape(0, 3).astype(np.float32), 'empty'
    if coordinate_mode == 'fastbev':
        return points.astype(np.float32), 'fastbev_no_transform'
    if coordinate_mode == 'raw':
        should_transform = True
    elif coordinate_mode == 'auto':
        should_transform = 'raw_to_fastbev' in metadata
    else:
        raise ValueError(f'Unsupported coordinate mode: {coordinate_mode}')

    if not should_transform:
        return points.astype(np.float32), 'auto_no_raw_to_fastbev'
    matrix = np.asarray(metadata.get('raw_to_fastbev'), dtype=np.float32)
    if matrix.shape == (3, 3):
        return (points @ matrix.T).astype(np.float32), 'raw_to_fastbev_3x3'
    if matrix.shape == (4, 4):
        homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        return (homo @ matrix.T)[:, :3].astype(np.float32), 'raw_to_fastbev_4x4'
    logger.warning('metadata.raw_to_fastbev shape=%s 不支持，点云保持原坐标', matrix.shape)
    return points.astype(np.float32), 'raw_to_fastbev_invalid_shape'


def load_frame_pointcloud(
    info: Dict,
    metadata: Dict,
    data_root: Optional[Path],
    pointcloud_root: Optional[Path],
    pointcloud_glob: Optional[str],
    bin_dim: int,
    coordinate_mode: str,
) -> Tuple[np.ndarray, Optional[Path], str, str]:
    path = resolve_pointcloud_path(info, data_root, pointcloud_root, pointcloud_glob)
    if path is None:
        return np.zeros((0, 3), dtype=np.float32), None, 'missing', 'none'
    try:
        points_raw = read_pointcloud(path, bin_dim=bin_dim)
        points, transform_status = transform_points_to_fastbev(points_raw, metadata, coordinate_mode)
        return points, path, 'loaded', transform_status
    except Exception as exc:
        logger.warning('读取点云失败 token=%s path=%s error=%s', info.get('token'), path, exc)
        return np.zeros((0, 3), dtype=np.float32), path, f'error:{exc}', 'error'


def box_point_mask(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.bool_)
    x, y, z, length, width, height, yaw = [float(v) for v in box[:7]]
    if length <= 0 or width <= 0 or height <= 0:
        return np.zeros((points.shape[0],), dtype=np.bool_)
    delta = points - np.array([x, y, z], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    local_x = delta[:, 0] * c - delta[:, 1] * s
    local_y = delta[:, 0] * s + delta[:, 1] * c
    local_z = delta[:, 2]
    return (
        (np.abs(local_x) <= length * 0.5) &
        (np.abs(local_y) <= width * 0.5) &
        (np.abs(local_z) <= height * 0.5)
    )


def camera_depth_for_points(points: np.ndarray, cam_info: Optional[Dict]) -> np.ndarray:
    if cam_info is None or points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    sensor2lidar_r = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float64)
    sensor2lidar_t = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float64).reshape(3)
    lidar2cam_r = np.linalg.inv(sensor2lidar_r)
    cam_points = (points.astype(np.float64) - sensor2lidar_t) @ lidar2cam_r.T
    return cam_points[:, 2].astype(np.float32)


def projected_2d_bbox_stats(
    box: np.ndarray,
    cam_info: Optional[Dict],
    min_depth: float,
) -> Tuple[Optional[List[float]], Optional[float]]:
    if cam_info is None:
        return None, None
    draw_cam_info, image_shape = camera_info_for_pkl_image_size(cam_info)
    lidar2img = compute_lidar2img(draw_cam_info)
    corners = corners_from_boxes(box.reshape(1, -1))[0]
    uv, depth = project_lidar_points_pinhole(corners, lidar2img)
    h, w = image_shape
    valid = (
        (depth > float(min_depth)) &
        np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    )
    if not np.any(valid):
        return None, 0.0
    uv_valid = uv[valid]
    x1 = float(np.clip(np.min(uv_valid[:, 0]), 0, w - 1))
    y1 = float(np.clip(np.min(uv_valid[:, 1]), 0, h - 1))
    x2 = float(np.clip(np.max(uv_valid[:, 0]), 0, w - 1))
    y2 = float(np.clip(np.max(uv_valid[:, 1]), 0, h - 1))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return [x1, y1, x2, y2], float(area)


def dim_color(color: Tuple[int, int, int], factor: float = 0.45) -> Tuple[int, int, int]:
    return tuple(int(np.clip(round(c * factor + 50), 0, 255)) for c in color)


def gt_tokens(info: Dict, count: int) -> List[str]:
    for key in ('gt_tokens', 'instance_tokens', 'track_ids'):
        values = info.get(key)
        if values is not None and len(values) >= count:
            return [str(v) for v in list(values)[:count]]
    frame_token = str(info.get('token', 'frame'))
    return [f'{frame_token}:{idx}' for idx in range(count)]


def build_gt_diagnostics(
    frame_index: int,
    info: Dict,
    boxes: np.ndarray,
    names: Sequence[str],
    class_names: Sequence[str],
    used_mask: np.ndarray,
    range_mask: np.ndarray,
    visible_mask: np.ndarray,
    points: np.ndarray,
    cam_info: Optional[Dict],
    camera_id: str,
    pointcloud_path: Optional[Path],
    pointcloud_status: str,
    min_depth: float,
) -> List[Dict[str, Any]]:
    track_values = list(info.get('track_ids', []))
    tokens = gt_tokens(info, len(names))
    rows: List[Dict[str, Any]] = []
    cam_depths_all = camera_depth_for_points(points, cam_info)
    for gt_idx, (box, name) in enumerate(zip(boxes, names)):
        class_in_use = str(name) in class_names
        reasons = []
        if not class_in_use:
            reasons.append('not_class')
        if not bool(range_mask[gt_idx]):
            reasons.append('outside_roi')
        if not bool(visible_mask[gt_idx]):
            reasons.append(f'not_{camera_display_name(camera_id).lower()}_geometry_visible')
        if bool(used_mask[gt_idx]):
            reasons = []

        inside = box_point_mask(points, box)
        box_points = points[inside]
        box_depths = cam_depths_all[inside] if cam_depths_all.shape[0] == points.shape[0] else np.zeros((0,), dtype=np.float32)
        bbox2d, bbox_area = projected_2d_bbox_stats(box, cam_info, min_depth)
        volume = max(float(box[3] * box[4] * box[5]), 1e-6)
        row = {
            'frame_index': int(frame_index),
            'frame_token': str(info.get('token', '')),
            'timestamp': int(info['timestamp']) if info.get('timestamp') is not None else None,
            'dataset': str(info.get('dataset', '')),
            'sequence': str(info.get('sequence', '')),
            'clip': str(info.get('clip', info.get('clip_id', ''))),
            'gt_index': int(gt_idx),
            'gt_token': tokens[gt_idx],
            'track_id': str(track_values[gt_idx]) if gt_idx < len(track_values) else '',
            'class': str(name),
            'used_by_mono_front': bool(used_mask[gt_idx]),
            'filtered_reason': ';'.join(reasons),
            'class_in_use': bool(class_in_use),
            'in_roi': bool(range_mask[gt_idx]),
            'cam0_geometry_visible': bool(visible_mask[gt_idx]),
            'center_distance_m': float(np.linalg.norm(box[:2])),
            'box': [float(v) for v in box[:7]],
            'box_x': float(box[0]),
            'box_y': float(box[1]),
            'box_z': float(box[2]),
            'box_l': float(box[3]),
            'box_w': float(box[4]),
            'box_h': float(box[5]),
            'box_yaw': float(box[6]),
            'box_point_count': int(box_points.shape[0]),
            'box_point_lidar_x_min': as_float_or_none(np.min(box_points[:, 0])) if box_points.size else None,
            'box_point_lidar_x_max': as_float_or_none(np.max(box_points[:, 0])) if box_points.size else None,
            'box_point_lidar_z_min': as_float_or_none(np.min(box_points[:, 2])) if box_points.size else None,
            'box_point_lidar_z_max': as_float_or_none(np.max(box_points[:, 2])) if box_points.size else None,
            'box_point_cam_depth_min': as_float_or_none(np.min(box_depths)) if box_depths.size else None,
            'box_point_cam_depth_max': as_float_or_none(np.max(box_depths)) if box_depths.size else None,
            'box_point_density': float(box_points.shape[0] / volume),
            'projected_2d_bbox': bbox2d,
            'projected_2d_bbox_area': bbox_area,
            'projected_2d_bbox_x1': bbox2d[0] if bbox2d else None,
            'projected_2d_bbox_y1': bbox2d[1] if bbox2d else None,
            'projected_2d_bbox_x2': bbox2d[2] if bbox2d else None,
            'projected_2d_bbox_y2': bbox2d[3] if bbox2d else None,
            'pointcloud_path': str(pointcloud_path) if pointcloud_path is not None else '',
            'pointcloud_status': pointcloud_status,
        }
        rows.append(row)
    return rows


def draw_roi_rectangle(image: np.ndarray, display_range: Tuple[float, float, float, float], roi_range: Sequence[float]) -> None:
    values = list(roi_range)
    if len(values) == 6:
        x_min, y_min, _, x_max, y_max, _ = values
    else:
        x_min, y_min, x_max, y_max = values[:4]
    xy = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ], dtype=np.float32)
    pts = bev_to_pixel(xy, display_range, image.shape[0]).astype(np.int32)
    cv2.polylines(image, [pts], True, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.polylines(image, [pts], True, ROI_COLOR, 2, cv2.LINE_AA)
    label_pt = pts[0] + np.array([6, -8])
    cv2.putText(image, 'front ROI', tuple(label_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ROI_COLOR, 1, cv2.LINE_AA)


def draw_distance_labels(image: np.ndarray, display_range: Tuple[float, float, float, float]) -> None:
    x_min, y_min, x_max, y_max = display_range
    size = image.shape[0]
    y_anchor = min(max(0.0, y_min), y_max)
    x_start = int(math.ceil(max(0.0, x_min) / 10.0) * 10)
    x_end = int(math.floor(x_max / 10.0) * 10)
    for x in range(x_start, x_end + 1, 10):
        pt = bev_to_pixel(np.array([[x, y_anchor]], dtype=np.float32), display_range, size)[0].astype(int)
        cv2.circle(image, tuple(pt), 3, (190, 190, 190), -1, cv2.LINE_AA)
        cv2.putText(
            image, f'{x}m', tuple(pt + np.array([5, -5])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)


def draw_bev_points(
    image: np.ndarray,
    points: np.ndarray,
    display_range: Tuple[float, float, float, float],
    max_points: int,
    rng: np.random.Generator,
) -> int:
    if points.shape[0] == 0:
        return 0
    x_min, y_min, x_max, y_max = display_range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    pts = points[mask]
    if pts.shape[0] == 0:
        return 0
    if max_points > 0 and pts.shape[0] > max_points:
        choice = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[choice]
    pix = np.rint(bev_to_pixel(pts[:, :2], display_range, image.shape[0])).astype(np.int32)
    inside = (
        (pix[:, 0] >= 0) & (pix[:, 0] < image.shape[1]) &
        (pix[:, 1] >= 0) & (pix[:, 1] < image.shape[0])
    )
    pix = pix[inside]
    if pix.shape[0] == 0:
        return 0
    z = pts[inside, 2]
    z_norm = np.clip((z + 3.0) / 6.0, 0.0, 1.0)
    colors = np.stack([
        np.full_like(z_norm, 85.0),
        75.0 + 90.0 * z_norm,
        120.0 + 90.0 * (1.0 - z_norm),
    ], axis=1).astype(np.uint8)
    image[pix[:, 1], pix[:, 0]] = colors
    return int(pix.shape[0])


def draw_raw_gt_outlines(
    image: np.ndarray,
    boxes: np.ndarray,
    display_range: Tuple[float, float, float, float],
) -> None:
    for corners in corners_from_boxes(boxes):
        pts = bev_to_pixel(corners[:4, :2], display_range, image.shape[0]).astype(np.int32)
        cv2.polylines(image, [pts], True, (35, 35, 35), 3, cv2.LINE_AA)
        cv2.polylines(image, [pts], True, RAW_GT_COLOR, 1, cv2.LINE_AA)


def render_bev_diagnostic(
    info: Dict,
    boxes: np.ndarray,
    names: Sequence[str],
    class_names: Sequence[str],
    used_mask: np.ndarray,
    points: np.ndarray,
    display_range: Tuple[float, float, float, float],
    roi_range: Sequence[float],
    size: int,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    draw_bev_grid(image, display_range)
    drawn_points = draw_bev_points(image, points, display_range, max_points, rng)
    draw_roi_rectangle(image, display_range, roi_range)
    draw_distance_labels(image, display_range)

    if boxes.shape[0] > 0:
        draw_raw_gt_outlines(image, boxes, display_range)
    corners = corners_from_boxes(boxes)
    for idx, (box, name, used) in enumerate(zip(boxes, names, used_mask)):
        color = class_color(str(name), class_names)
        source = 'GT' if bool(used) else 'Filtered'
        draw_color = color if bool(used) else dim_color(color)
        pts = bev_to_pixel(corners[idx, :4, :2], display_range, size).astype(np.int32)
        draw_bev_box_by_source(image, pts, draw_color, source)
        draw_bev_box_heading(image, corners[idx], box, display_range, draw_color, size, 'front-edge')
        prefix = 'U' if bool(used) else 'F'
        label_pt = bev_to_pixel(np.array([[box[0], box[1]]], dtype=np.float32), display_range, size)[0].astype(int)
        cv2.putText(
            image, f'{prefix}:{name}', tuple(label_pt + np.array([4, -4])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, draw_color, 1, cv2.LINE_AA)

    x_min, y_min, x_max, y_max = display_range
    summary = (
        f'BEV points={drawn_points} raw_gt={len(boxes)} used={int(used_mask.sum())} '
        f'filtered={int(len(used_mask) - used_mask.sum())} '
        f'x[{x_min:.0f},{x_max:.0f}] y[{y_min:.0f},{y_max:.0f}]'
    )
    cv2.rectangle(image, (0, 0), (size, 34), (0, 0, 0), -1)
    cv2.putText(image, summary, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    legend = [
        ('raw GT', RAW_GT_COLOR),
        ('used car/truck', class_color(class_names[0], class_names)),
        ('filtered', FILTERED_GT_COLOR),
        ('ROI', ROI_COLOR),
        ('point cloud', POINT_COLOR),
    ]
    return add_top_header(image, str(info.get('token', '')), legend)


def project_points_for_camera(
    points: np.ndarray,
    cam_info: Dict,
    image: np.ndarray,
    undistort: bool,
    undistort_alpha: float,
    undistort_new_k: str = 'optimal',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, intrinsic = undistort_if_requested(
        image, cam_info, undistort, undistort_alpha,
        new_k_mode=undistort_new_k)
    if undistort:
        lidar2img = compute_lidar2img(cam_info, intrinsic_override=intrinsic)
        uv, depth = project_lidar_points_pinhole(points, lidar2img)
    else:
        uv, depth = project_lidar_points_distorted(points, cam_info, intrinsic)
    return image, uv, depth


def draw_depth_points(
    image: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    max_points: int,
    rng: np.random.Generator,
) -> int:
    if uv.shape[0] == 0:
        return 0
    h, w = image.shape[:2]
    valid = (
        (depth > float(min_depth)) &
        (depth <= float(max_depth)) &
        np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) &
        (uv[:, 0] >= 0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    idx = np.where(valid)[0]
    if idx.size == 0:
        return 0
    if max_points > 0 and idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    d = depth[idx]
    norm = np.clip(d / max(float(max_depth), 1.0), 0.0, 1.0)
    colors = cv2.applyColorMap((255.0 * (1.0 - norm)).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    pix = np.rint(uv[idx]).astype(np.int32)
    for point, color in zip(pix, colors.reshape(-1, 3)):
        cv2.circle(image, tuple(point.tolist()), 1, tuple(int(v) for v in color), -1, cv2.LINE_AA)
    return int(idx.size)


def render_camera_diagnostic(
    info: Dict,
    data_root: Optional[Path],
    camera_id: str,
    boxes: np.ndarray,
    names: Sequence[str],
    class_names: Sequence[str],
    used_mask: np.ndarray,
    points: np.ndarray,
    camera_width: int,
    draw_projected_points: bool,
    max_camera_points: int,
    max_point_depth: float,
    undistort: bool,
    undistort_alpha: float,
    undistort_new_k: str,
    min_depth: float,
    max_edge_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    cams = info.get('cams', {}) or {}
    display_name = camera_display_name(camera_id)
    if camera_id not in cams:
        image = placeholder_image(camera_width, int(camera_width * 9 / 16), f'missing {display_name}')
        return add_top_header(image, f'{info.get("token", "")} {display_name}')

    cam_info = cams[camera_id]
    image_path = resolve_image_path(data_root or Path('.'), cam_info['data_path'])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        logger.warning('Cannot read image: %s', image_path)
        image = placeholder_image(camera_width, int(camera_width * 9 / 16), f'missing {display_name}')
        return add_top_header(image, f'{info.get("token", "")} {display_name}')

    draw_cam_info = camera_info_for_loaded_image(cam_info, image)
    image, sx, sy = resize_to_width_with_scale(image, camera_width)
    draw_cam_info = scaled_camera_info(draw_cam_info, sx, sy)
    image, intrinsic = undistort_if_requested(
        image, draw_cam_info, undistort, undistort_alpha,
        new_k_mode=undistort_new_k)

    if undistort:
        lidar2img = compute_lidar2img(draw_cam_info, intrinsic_override=intrinsic)

        def project_fn(query_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return project_lidar_points_pinhole(query_points, lidar2img)
    else:
        def project_fn(query_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return project_lidar_points_distorted(query_points, draw_cam_info, intrinsic)

    projected_count = 0
    if draw_projected_points and points.shape[0] > 0:
        sample_points = points
        if max_camera_points > 0 and sample_points.shape[0] > max_camera_points * 3:
            choice = rng.choice(sample_points.shape[0], size=max_camera_points * 3, replace=False)
            sample_points = sample_points[choice]
        uv, depth = project_fn(sample_points)
        projected_count = draw_depth_points(
            image, uv, depth, min_depth=min_depth, max_depth=max_point_depth,
            max_points=max_camera_points, rng=rng)

    for idx, (corners, name, used) in enumerate(zip(corners_from_boxes(boxes), names, used_mask)):
        color = class_color(str(name), class_names)
        color = color if bool(used) else dim_color(color)
        prefix = 'U' if bool(used) else 'F'
        label = f'{prefix}:{name}'
        draw_projected_box(
            image, corners, project_fn, label, color,
            min_depth, max_edge_px, use_corner_clipline=bool(undistort))

    projection_mode = f'undist={undistort_new_k}' if undistort else 'raw-distorted'
    draw_label_tag(image, f'{display_name} pc={projected_count} {projection_mode}')
    legend = [
        ('used GT', class_color(class_names[0], class_names)),
        ('filtered GT', FILTERED_GT_COLOR),
        ('depth points', (64, 180, 255)),
    ]
    return add_top_header(image, f'{info.get("token", "")} {display_name}', legend)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f'Failed to write image: {path}')


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='N7 点云辅助 GT 可视化/诊断：输出逐帧图和逐 GT CSV/JSON。',
        formatter_class=RawDefaultsHelpFormatter,
        epilog=USAGE_EXAMPLES)
    parser.add_argument('--pkl', required=True, help='Fast-BEV N7 info pkl 路径')
    parser.add_argument('--data-root', default=None, help='pkl 中相对图片路径的根目录；不传时按当前目录解析')
    parser.add_argument('--pointcloud-root', default=None, help='原始点云根目录；不传时尝试使用 --data-root 和 pkl 中 lidar_path')
    parser.add_argument('--pointcloud-glob', default=None,
                        help='点云路径模板，支持 {dataset}/{sequence}/{clip}/{timestamp}/{frame_timestamp}/{token} 和 glob 通配符')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--camera-id', default='cam0', help='用于前视图和几何可见性判断的相机 id')
    parser.add_argument('--frame-indices', default=None, help='逗号/范围形式指定帧，例如 0,5,10-20；优先级高于 --start-index/--stride')
    parser.add_argument('--start-index', type=int, default=0, help='未指定 --frame-indices 时，从第几个 info 开始')
    parser.add_argument('--stride', type=int, default=1, help='未指定 --frame-indices 时的抽帧步长')
    parser.add_argument('--max-frames', type=int, default=100, help='最多处理多少帧；设为 -1 表示全部')
    parser.add_argument('--bev-range', nargs=4, type=float, default=DEFAULT_BEV_RANGE,
                        metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'), help='BEV 显示范围；脚本会扩成等比例显示范围')
    parser.add_argument('--gt-filter-range', nargs='+', type=float, default=list(DEFAULT_GT_FILTER_RANGE),
                        help='mono-front used/filtered 判断 ROI，可填 4 维 x_min y_min x_max y_max 或 6 维 point_cloud_range')
    parser.add_argument('--class-names', nargs='+', default=DEFAULT_CLASS_NAMES,
                        help='当前 mono-front used GT 的类别口径')
    parser.add_argument('--pointcloud-coordinate', choices=['auto', 'raw', 'fastbev'], default='auto',
                        help='点云坐标系；auto 会在 pkl metadata 存在 raw_to_fastbev 时把原始点云转到 Fast-BEV lidar 坐标')
    parser.add_argument('--pointcloud-bin-dim', type=int, default=4,
                        help='.bin 点云每点 float32 维度；<=0 时在 4/5/6/3 中自动推断')
    parser.add_argument('--bev-size', type=int, default=900, help='BEV 图尺寸')
    parser.add_argument('--camera-width', type=int, default=1280, help='单目前视图输出宽度')
    parser.add_argument('--max-bev-points', type=int, default=200000, help='BEV 叠加点云最大采样点数；<=0 表示不限制')
    parser.add_argument('--max-camera-points', type=int, default=60000, help='前视图投影点云最大采样点数；<=0 表示不限制')
    parser.add_argument('--no-projected-points', action='store_true', help='前视图不叠加点云深度点')
    parser.add_argument('--max-point-depth', type=float, default=100.0, help='前视图叠加点云的最大相机深度')
    parser.add_argument('--undistort', dest='undistort', action='store_true', default=True,
                        help='画框/点云前先对图片去畸变')
    parser.add_argument('--raw-distorted', dest='undistort', action='store_false', default=argparse.SUPPRESS,
                        help='不去畸变，在原始畸变图上使用畸变投影')
    parser.add_argument('--undistort-alpha', type=float, default=0.0, help='OpenCV 去畸变 alpha')
    parser.add_argument(
        '--undistort-new-k', choices=['original', 'optimal'], default='optimal',
        help=('去畸变输出 K；original 等价于 cv2.undistort(image,K,D,None,K)，'
              'optimal 保持历史行为并使用 getOptimalNewCameraMatrix；alpha 仅对 optimal 生效'))
    parser.add_argument('--min-depth', type=float, default=0.1, help='相机几何可见和画框的最小深度')
    parser.add_argument('--max-corner-edge-px', type=float, default=0.0, help='原始畸变图角点连线最大像素长度；0 表示自动')
    parser.add_argument('--image-ext', default='jpg', choices=['jpg', 'png'], help='输出图片格式')
    parser.add_argument('--random-seed', type=int, default=0, help='点云采样随机种子')
    parser.add_argument('--no-render', action='store_true', help='只输出 CSV/JSON，不保存图片')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    if args.gt_filter_range is not None and len(args.gt_filter_range) not in (4, 6):
        parser.error('--gt-filter-range 只能填写 4 个或 6 个数字')
    return args


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    pkl_path = Path(args.pkl)
    data_root = Path(args.data_root) if args.data_root else None
    pointcloud_root = Path(args.pointcloud_root) if args.pointcloud_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infos, metadata = load_fastbev_pkl(pkl_path)
    pkl_class_names = get_class_names(metadata, infos)
    class_names = [str(x) for x in (args.class_names or pkl_class_names)]
    frame_indices = parse_frame_indices(args.frame_indices)
    max_frames = None if args.max_frames is not None and args.max_frames < 0 else args.max_frames
    items = selected_frame_items(infos, frame_indices, args.start_index, args.stride, max_frames)
    display_bev_range = make_metric_bev_range(tuple(args.bev_range))
    rng = np.random.default_rng(args.random_seed)

    logger.info('Loaded %d infos from %s', len(infos), pkl_path)
    logger.info('Selected %d frames', len(items))
    logger.info('class_names=%s pkl_class_names=%s', class_names, pkl_class_names)
    logger.info('camera_id=%s bev_range=%s display_bev_range=%s gt_filter_range=%s',
                args.camera_id, args.bev_range, display_bev_range, args.gt_filter_range)

    all_rows: List[Dict[str, Any]] = []
    frame_summaries: List[Dict[str, Any]] = []
    rendered = 0
    missing_pointcloud = 0
    loaded_pointcloud = 0

    for frame_index, info in items:
        raw_boxes = to_numpy(info.get('gt_boxes', np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
        if raw_boxes.size == 0:
            boxes = np.zeros((0, 7), dtype=np.float32)
        elif raw_boxes.ndim == 1:
            boxes = raw_boxes.reshape(1, -1)[:, :7]
        else:
            boxes = raw_boxes[:, :7]
        raw_names = [str(x) for x in list(info.get('gt_names', []))]
        count = min(boxes.shape[0], len(raw_names))
        boxes = boxes[:count]
        names = raw_names[:count]

        cams = info.get('cams', {}) or {}
        cam_info = cams.get(args.camera_id)
        range_mask = gt_filter_range_mask(boxes, args.gt_filter_range)
        if cam_info is None or boxes.shape[0] == 0:
            visible_mask = np.zeros((boxes.shape[0],), dtype=np.bool_)
        else:
            visible_mask = gt_visible_camera_mask(info, boxes, [args.camera_id], args.min_depth)
        if cam_info is None:
            used_mask = np.zeros((boxes.shape[0],), dtype=np.bool_)
        else:
            used_mask = gt_used_mask(
                info, boxes, names, class_names, args.gt_filter_range,
                [args.camera_id], args.min_depth)

        points, pc_path, pc_status, transform_status = load_frame_pointcloud(
            info=info,
            metadata=metadata,
            data_root=data_root,
            pointcloud_root=pointcloud_root,
            pointcloud_glob=args.pointcloud_glob,
            bin_dim=args.pointcloud_bin_dim,
            coordinate_mode=args.pointcloud_coordinate)
        if pc_status == 'loaded':
            loaded_pointcloud += 1
        else:
            missing_pointcloud += 1

        rows = build_gt_diagnostics(
            frame_index=frame_index,
            info=info,
            boxes=boxes,
            names=names,
            class_names=class_names,
            used_mask=used_mask,
            range_mask=range_mask,
            visible_mask=visible_mask,
            points=points,
            cam_info=cam_info,
            camera_id=args.camera_id,
            pointcloud_path=pc_path,
            pointcloud_status=pc_status,
            min_depth=args.min_depth)
        all_rows.extend(rows)

        stem = f'{frame_index:06d}_{frame_stem_from_info(info, frame_index)}'
        frame_dir = info_output_dir(output_dir / 'frames', info)
        frame_summary = {
            'frame_index': frame_index,
            'token': info.get('token'),
            'timestamp': info.get('timestamp'),
            'raw_gt_count': int(len(boxes)),
            'used_gt_count': int(used_mask.sum()),
            'filtered_gt_count': int(len(used_mask) - used_mask.sum()),
            'pointcloud_path': str(pc_path) if pc_path is not None else '',
            'pointcloud_status': pc_status,
            'pointcloud_transform_status': transform_status,
            'point_count': int(points.shape[0]),
            'outputs': {},
        }

        if not args.no_render:
            bev_image = render_bev_diagnostic(
                info=info,
                boxes=boxes,
                names=names,
                class_names=class_names,
                used_mask=used_mask,
                points=points,
                display_range=display_bev_range,
                roi_range=args.gt_filter_range,
                size=args.bev_size,
                max_points=args.max_bev_points,
                rng=rng)
            cam_image = render_camera_diagnostic(
                info=info,
                data_root=data_root,
                camera_id=args.camera_id,
                boxes=boxes,
                names=names,
                class_names=class_names,
                used_mask=used_mask,
                points=points,
                camera_width=args.camera_width,
                draw_projected_points=not args.no_projected_points,
                max_camera_points=args.max_camera_points,
                max_point_depth=args.max_point_depth,
                undistort=args.undistort,
                undistort_alpha=args.undistort_alpha,
                undistort_new_k=args.undistort_new_k,
                min_depth=args.min_depth,
                max_edge_px=args.max_corner_edge_px,
                rng=rng)
            target_h = max(bev_image.shape[0], cam_image.shape[0])
            bev_resized = resize_to_width(bev_image, int(round(bev_image.shape[1] * target_h / bev_image.shape[0])))
            cam_resized = resize_to_width(cam_image, int(round(cam_image.shape[1] * target_h / cam_image.shape[0])))
            combined = np.concatenate([cam_resized, bev_resized], axis=1)

            bev_path = frame_dir / f'{stem}_bev.{args.image_ext}'
            cam_path = frame_dir / f'{stem}_{args.camera_id}.{args.image_ext}'
            combined_path = frame_dir / f'{stem}_combined.{args.image_ext}'
            save_image(bev_path, bev_image)
            save_image(cam_path, cam_image)
            save_image(combined_path, combined)
            frame_summary['outputs'] = {
                'bev': str(bev_path),
                'camera': str(cam_path),
                'combined': str(combined_path),
            }
            rendered += 1
        frame_summaries.append(frame_summary)

    csv_path = output_dir / 'gt_diagnostics.csv'
    json_path = output_dir / 'gt_diagnostics.json'
    summary_path = output_dir / 'summary.json'
    write_csv(csv_path, all_rows)
    with json_path.open('w') as f:
        json.dump(json_safe(all_rows), f, ensure_ascii=False, indent=2)
    summary = {
        'pkl': str(pkl_path),
        'data_root': str(data_root) if data_root is not None else '',
        'pointcloud_root': str(pointcloud_root) if pointcloud_root is not None else '',
        'pointcloud_glob': args.pointcloud_glob,
        'camera_id': args.camera_id,
        'undistort': bool(args.undistort),
        'undistort_new_k': args.undistort_new_k if args.undistort else 'not_applicable',
        'undistort_alpha': float(args.undistort_alpha) if args.undistort else None,
        'class_names': class_names,
        'bev_range_cli': list(args.bev_range),
        'bev_range_display': list(display_bev_range),
        'gt_filter_range': list(args.gt_filter_range),
        'frames_selected': len(items),
        'frames_rendered': rendered,
        'gt_rows': len(all_rows),
        'pointcloud_loaded_frames': loaded_pointcloud,
        'pointcloud_missing_or_error_frames': missing_pointcloud,
        'outputs': {
            'csv': str(csv_path),
            'json': str(json_path),
        },
        'frames': frame_summaries,
    }
    with summary_path.open('w') as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    logger.info('Wrote %d GT rows to %s', len(all_rows), csv_path)
    logger.info('Wrote summary to %s', summary_path)
    if missing_pointcloud:
        logger.warning('Pointcloud missing/error frames: %d / %d', missing_pointcloud, len(items))


if __name__ == '__main__':
    main()
