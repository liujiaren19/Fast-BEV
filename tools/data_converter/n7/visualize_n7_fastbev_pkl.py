#!/usr/bin/env python3
"""可视化 N7 converter 生成的 Fast-BEV pkl。

该脚本不依赖 nuScenes dataset 类，只依赖 N7 OD converter 写入的字段：

    {
        "infos": [
            {
                "token": "...",
                "timestamp": 171...,
                "cams": {
                    "cam0": {
                        "data_path": "relative/or/absolute/image.jpg",
                        "cam_intrinsic": [[...], [...], [...]],
                        "sensor2lidar_rotation": [[...], [...], [...]],
                        "sensor2lidar_translation": [x, y, z],
                        "distortion": [...]
                    },
                    ...
                },
                "gt_boxes": [[x, y, z, l, w, h, yaw], ...],
                "gt_names": ["car", ...],
                "gt_velocity": [[vx, vy], ...]
            },
            ...
        ],
        "metadata": {...}
    }

该可视化脚本默认理解的坐标约定：

    x：车头向前
    y：车身向左
    z：向上
    原点：默认是 N7 顶部主 lidar，除非 converter metadata 另有说明

投影逻辑刻意和 ``CustomMultiViewDataset._lidar2img_from_cam_info`` 保持一致。
如果这里投影出来的 GT box 不正确，训练时 Fast-BEV 使用同一份 pkl 时大概率
也会有相同的投影几何问题。

常用命令：

    # 704x256 缓存图：图片已经离线 resize，但 info_json/pkl 中 K 仍对应 1600x900。
    # 因此 converter 需要写入实际图片尺寸，visualizer 会按该尺寸自动缩放 K。
    python tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py \
        --data-path data/N7_704_256 \
        --datasets 20251031 \
        --sets train val \
        --info-json data/info_json/2025_04_18_2k_byd_info.json \
        --output-dir data/N7_704_256/pkl \
        --extra-tag custom_fastbev \
        --image-size 256 704

    # 只做 pkl 和几何校验，不生成图片，适合批量转换后快速检查。
    python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
        --pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl \
        --data-root data/N7_704_256 \
        --output-dir work_dirs/vis_n7_704_check \
        --no-render \
        --check-geometry \
        --check-temporal-geometry \
        --strict-geometry

    # 抽帧生成视频，不额外落逐帧图片。
    python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
        --pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl \
        --data-root data/N7_704_256 \
        --output-dir work_dirs/vis_n7_704 \
        --max-frames 300 \
        --stride 5 \
        --video-only \
        --bev-range -50 -50 50 50

如果 pkl 中的图片路径是相对路径，``--data-root`` 必须和 converter 的
``--data-path`` 使用同一个根目录。默认在原始畸变图上绘制“带畸变投影”的
3D 框；只有启用 ``--undistort`` 时，脚本才会先去畸变图片并用 new_K 投影。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RawDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """保留 epilog 示例换行，同时继续展示 argparse 默认值。"""

cv2.setNumThreads(4)

# converter 默认六目相机的人眼检查顺序：
#   第一行：左前、前视广角、右前
#   第二行：左后、后视、右后
DEFAULT_MOSAIC_ORDER = ['cam9', 'cam0', 'cam11', 'cam8', 'cam3', 'cam10']

# 当前 converter pkl 仍使用 N7 标定中的 cam id 作为 key，避免影响 dataset/config。
# 可视化时只把显示名映射成 nuScenes 常用名称，方便和原版 Fast-BEV 习惯对齐。
CAMERA_DISPLAY_NAMES = {
    'cam9': 'CAM_FRONT_LEFT',
    'cam0': 'CAM_FRONT',
    'cam11': 'CAM_FRONT_RIGHT',
    'cam8': 'CAM_BACK_LEFT',
    'cam3': 'CAM_BACK',
    'cam10': 'CAM_BACK_RIGHT',
}

# OpenCV 使用 BGR 通道顺序，这里的颜色不是 RGB。
DEFAULT_COLORS = [
    (0, 255, 0), (0, 128, 255), (255, 0, 0), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (180, 80, 0), (80, 180, 0),
    (180, 0, 180), (255, 255, 255),
]

# corners_from_boxes 返回的角点顺序：
#   底面：0--1      顶面：4--5
#         |  |            |  |
#         3--2            7--6
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
FRONT_EDGES = {(0, 1), (4, 5)}
UNDISTORT_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def load_fastbev_pkl(path: Path) -> Tuple[List[Dict], Dict]:
    """读取 converter 新格式 pkl，也兼容旧版只保存 info 列表的 pkl。"""
    with path.open('rb') as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        infos = payload.get('infos', [])
        metadata = payload.get('metadata', {})
    elif isinstance(payload, list):
        infos = payload
        metadata = {}
    else:
        raise TypeError(f'Unsupported pkl payload type: {type(payload)!r}')

    if not isinstance(infos, list):
        raise TypeError(f'Expected infos to be list, got {type(infos)!r}')
    return infos, metadata


def sanitize_filename(text: str, max_len: int = 180) -> str:
    text = re.sub(r'[^0-9A-Za-z_.-]+', '_', str(text))
    return text[:max_len]


def camera_display_name(cam_id: str) -> str:
    """返回可视化中显示的相机名称，未知相机保留原始 cam id。"""
    return CAMERA_DISPLAY_NAMES.get(str(cam_id), str(cam_id))


def info_ref_parts(info: Dict) -> Tuple[str, str, str]:
    """从 info 中提取输出目录需要的 dataset/sequence/clip 三层引用。"""
    dataset = sanitize_filename(info.get('dataset') or 'unknown_dataset')
    sequence = sanitize_filename(info.get('sequence') or 'unknown_sequence')
    clip = sanitize_filename(info.get('clip') or info.get('clip_id') or 'unknown_clip')
    return dataset, sequence, clip


def info_output_dir(output_dir: Path, info: Dict) -> Path:
    """按 dataset/sequence/clip 层级生成当前帧的输出目录。"""
    dataset, sequence, clip = info_ref_parts(info)
    return output_dir / dataset / sequence / clip


def frame_stem_from_info(info: Dict, fallback_index: int) -> str:
    """帧文件名只使用 lidar 时间戳；缺失时间戳时才退回 token 或索引。"""
    timestamp = info.get('timestamp')
    if timestamp is not None:
        return sanitize_filename(str(int(timestamp)))
    token = info.get('token')
    if token is not None:
        return sanitize_filename(str(token))
    return f'{fallback_index:06d}'

def resolve_image_path(data_root: Path, data_path: str) -> Path:
    """解析 pkl 中记录的相机图片路径。

    converter 会尽量把图片路径写成相对 ``--data-path`` 的路径，因此这里用
    ``--data-root`` 作为根目录还原真实图片位置。
    """
    path = Path(data_path)
    if path.is_absolute():
        return path
    return data_root / path


def to_numpy(value, dtype=np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def get_class_names(metadata: Dict, infos: Sequence[Dict]) -> List[str]:
    if metadata.get('classes'):
        return list(metadata['classes'])
    names: List[str] = []
    for info in infos:
        for name in info.get('gt_names', []):
            name = str(name)
            if name not in names:
                names.append(name)
    return names or ['unknown']


def class_color(name: str, class_names: Sequence[str]) -> Tuple[int, int, int]:
    try:
        idx = class_names.index(name)
    except ValueError:
        idx = len(class_names)
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]


def camera_dimension_fields(cam_info: Dict, image_shape: Optional[Tuple[int, int]] = None) -> Dict[str, int]:
    """整理相机内参源尺寸和当前图片尺寸。

    ``intrinsic_width/height`` 表示 ``cam_intrinsic`` 所在的图像坐标尺寸；
    ``image_width/height`` 表示 pkl 中 ``data_path`` 实际图片尺寸。704x256 离线
    缓存图会出现二者不一致的情况，可视化时必须先把 K 从源尺寸缩到实际图片
    尺寸，再投影到图上。
    """
    intrinsic_width = int(cam_info.get('intrinsic_width', cam_info.get('width', 0)) or 0)
    intrinsic_height = int(cam_info.get('intrinsic_height', cam_info.get('height', 0)) or 0)
    image_width = int(cam_info.get('image_width', intrinsic_width) or intrinsic_width or 0)
    image_height = int(cam_info.get('image_height', intrinsic_height) or intrinsic_height or 0)

    if image_shape is not None:
        loaded_height, loaded_width = [int(v) for v in image_shape[:2]]
        if image_width > 0 and image_height > 0 and (image_width != loaded_width or image_height != loaded_height):
            logger.debug(
                'pkl image size %sx%s differs from loaded image size %sx%s; use loaded size for visualization',
                image_width, image_height, loaded_width, loaded_height)
        image_width, image_height = loaded_width, loaded_height

    return dict(
        intrinsic_width=intrinsic_width,
        intrinsic_height=intrinsic_height,
        image_width=image_width,
        image_height=image_height,
    )


def choose_camera_order(info: Dict, requested: Optional[Sequence[str]]) -> List[str]:
    cams = info.get('cams', {})
    if requested:
        missing = [cam for cam in requested if cam not in cams]
        if missing:
            logger.warning('Requested cameras missing in token=%s: %s', info.get('token'), missing)
        return [cam for cam in requested if cam in cams]
    ordered = [cam for cam in DEFAULT_MOSAIC_ORDER if cam in cams]
    ordered.extend([cam for cam in cams.keys() if cam not in ordered])
    return ordered


def dataset_lidar2img_from_cam_info(
    cam_info: Dict,
    sensor2lidar_r: Optional[np.ndarray] = None,
    sensor2lidar_t: Optional[np.ndarray] = None,
    intrinsic_override: Optional[np.ndarray] = None,
    temporal_compensated: bool = False,
) -> Tuple[np.ndarray, Dict, Dict]:
    """复刻 ``CustomMultiViewDataset._lidar2img_from_cam_info`` 的几何计算。

    converter 写入的 ``sensor2lidar_rotation`` 和 ``sensor2lidar_translation``
    表示“相机坐标到 lidar 坐标”的外参。Fast-BEV dataset 在训练时会把它反
    过来得到 lidar 到相机，再和相机内参组成 ``lidar2img``。这里保留同一套
    OpenMMLab 行向量写法，用于检查 pkl 和训练代码是否真的自洽。
    """
    intrinsic = to_numpy(intrinsic_override if intrinsic_override is not None else cam_info['cam_intrinsic'])
    if sensor2lidar_r is None:
        sensor2lidar_r = to_numpy(cam_info['sensor2lidar_rotation'])
    else:
        sensor2lidar_r = to_numpy(sensor2lidar_r)
    if sensor2lidar_t is None:
        sensor2lidar_t = to_numpy(cam_info['sensor2lidar_translation']).reshape(3)
    else:
        sensor2lidar_t = to_numpy(sensor2lidar_t).reshape(3)

    lidar2cam_r = np.linalg.inv(sensor2lidar_r)
    lidar2cam_t = sensor2lidar_t @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    distortion = np.asarray(cam_info.get('distortion', []), dtype=np.float32).reshape(-1)
    dimension_fields = camera_dimension_fields(cam_info)
    lidar2img_aug = dict(
        intrin=intrinsic.astype(np.float32),
        rot=sensor2lidar_r.astype(np.float32),
        tran=sensor2lidar_t.astype(np.float32),
        post_rot=np.eye(3, dtype=np.float32),
        post_tran=np.zeros(3, dtype=np.float32),
        temporal_compensated=temporal_compensated,
        distortion=distortion,
        **dimension_fields,
    )
    lidar2img_extra = dict(
        distortion=distortion,
        temporal_compensated=temporal_compensated,
        **dimension_fields,
    )
    return lidar2img, lidar2img_aug, lidar2img_extra


def direct_lidar2img_from_sensor2lidar(
    cam_info: Dict,
    sensor2lidar_r: Optional[np.ndarray] = None,
    sensor2lidar_t: Optional[np.ndarray] = None,
    intrinsic_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """用更直观的列向量物理公式重新计算 ``lidar2img``。

    如果 ``x_lidar = R_cam_to_lidar @ x_cam + t_cam_to_lidar``，那么
    ``x_cam = inv(R_cam_to_lidar) @ (x_lidar - t_cam_to_lidar)``。这个函数按
    该公式直接构造 ``K @ [R_lidar_to_cam | t_lidar_to_cam]``。它不复用
    dataset 的行向量写法，因此可以作为独立对照，检查 converter 写入的外参
    和 Fast-BEV dataset 读取外参的约定是否一致。
    """
    intrinsic = to_numpy(intrinsic_override if intrinsic_override is not None else cam_info['cam_intrinsic'])
    if sensor2lidar_r is None:
        sensor2lidar_r = to_numpy(cam_info['sensor2lidar_rotation'])
    else:
        sensor2lidar_r = to_numpy(sensor2lidar_r)
    if sensor2lidar_t is None:
        sensor2lidar_t = to_numpy(cam_info['sensor2lidar_translation']).reshape(3)
    else:
        sensor2lidar_t = to_numpy(sensor2lidar_t).reshape(3)

    lidar2cam_r = np.linalg.inv(sensor2lidar_r)
    lidar2cam = np.eye(4, dtype=np.float32)
    lidar2cam[:3, :3] = lidar2cam_r
    lidar2cam[:3, 3] = -lidar2cam_r @ sensor2lidar_t

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    return (viewpad @ lidar2cam).astype(np.float32)


def compute_lidar2img(cam_info: Dict, intrinsic_override: Optional[np.ndarray] = None) -> np.ndarray:
    """返回和 Fast-BEV dataset 训练路径一致的 ``lidar2img`` 矩阵。"""
    lidar2img, _, _ = dataset_lidar2img_from_cam_info(cam_info, intrinsic_override=intrinsic_override)
    return lidar2img


def undistort_if_requested(
    image: np.ndarray,
    cam_info: Dict,
    enabled: bool,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """按需对图片去畸变，并返回投影时应该使用的内参。

    默认不去畸变，因为“原始图片 + 原始 K”最能检查训练输入是否自洽。启用
    ``--undistort`` 后，OpenCV 会返回 ``new_K``，后续投影也使用这个新内参，
    这样 3D 框才能和去畸变后的图片对齐。
    """
    intrinsic = to_numpy(cam_info['cam_intrinsic'])
    if not enabled:
        return image, intrinsic

    distortion = np.asarray(cam_info.get('distortion', []), dtype=np.float64).reshape(-1)
    if distortion.size == 0 or np.allclose(distortion, 0):
        return image, intrinsic

    h, w = image.shape[:2]
    dist_full = np.zeros(8, dtype=np.float64)
    dist_full[:min(distortion.size, dist_full.size)] = distortion[:dist_full.size]
    cache_key = (h, w, tuple(intrinsic.reshape(-1)), tuple(dist_full), float(alpha))

    if cache_key not in UNDISTORT_CACHE:
        new_k, _ = cv2.getOptimalNewCameraMatrix(intrinsic, dist_full, (w, h), alpha, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(intrinsic, dist_full, None, new_k, (w, h), cv2.CV_16SC2)
        UNDISTORT_CACHE[cache_key] = (map1, map2, new_k.astype(np.float32))

    map1, map2, new_k = UNDISTORT_CACHE[cache_key]
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR), new_k


def corners_from_boxes(boxes: np.ndarray) -> np.ndarray:
    """把 ``[x, y, z, l, w, h, yaw]`` 格式的 3D 框转成 lidar 坐标系 8 个角点。"""
    if boxes.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)

    boxes = boxes[:, :7].astype(np.float32)
    corners = np.zeros((boxes.shape[0], 8, 3), dtype=np.float32)
    for i, box in enumerate(boxes):
        x, y, z, length, width, height, yaw = [float(v) for v in box]
        if length <= 0 or width <= 0 or height <= 0:
            continue
        local = np.array([
            [ length / 2,  width / 2, -height / 2],
            [ length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2,  width / 2, -height / 2],
            [ length / 2,  width / 2,  height / 2],
            [ length / 2, -width / 2,  height / 2],
            [-length / 2, -width / 2,  height / 2],
            [-length / 2,  width / 2,  height / 2],
        ], dtype=np.float32)
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        local[:, :2] = local[:, :2] @ rot.T
        local += np.array([x, y, z], dtype=np.float32)
        corners[i] = local
    return corners


def opencv_distortion_coeffs(cam_info: Dict) -> np.ndarray:
    """把 pkl 中的畸变参数整理成 OpenCV projectPoints 可接受的长度。"""
    distortion = np.asarray(cam_info.get('distortion', []), dtype=np.float64).reshape(-1)
    if distortion.size in (0, 4, 5, 8, 12, 14):
        return distortion
    dist_full = np.zeros(8, dtype=np.float64)
    dist_full[:min(distortion.size, dist_full.size)] = distortion[:dist_full.size]
    return dist_full


def lidar_points_to_camera(points: np.ndarray, cam_info: Dict) -> np.ndarray:
    """把 lidar 坐标系点转换到相机坐标系。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    sensor2lidar_r = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float64)
    sensor2lidar_t = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float64).reshape(3)
    lidar2cam_r = np.linalg.inv(sensor2lidar_r)
    return (points - sensor2lidar_t) @ lidar2cam_r.T


def project_lidar_points_pinhole(points: np.ndarray, lidar2img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """使用普通针孔模型投影 lidar 点。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    proj = (lidar2img @ points_h.T).T
    depth = proj[:, 2]
    uv = proj[:, :2] / np.maximum(depth[:, None], 1e-6)
    return uv, depth


def project_lidar_points_distorted(points: np.ndarray, cam_info: Dict, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """在原始畸变图上投影 lidar 点。\n\n    训练侧对原始图像使用畸变参数做 backprojection，因此可视化原图时也必须把\n    目标框投影点经过同一组畸变参数，否则框会更接近去畸变图而不是原图。\n    """
    cam_points = lidar_points_to_camera(points, cam_info)
    depth = cam_points[:, 2].astype(np.float32)
    distortion = opencv_distortion_coeffs(cam_info)
    if distortion.size == 0 or np.allclose(distortion, 0):
        lidar2img = compute_lidar2img(cam_info, intrinsic_override=intrinsic)
        return project_lidar_points_pinhole(points, lidar2img)

    uv, _ = cv2.projectPoints(
        cam_points.reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(intrinsic, dtype=np.float64),
        distortion,
    )
    return uv.reshape(-1, 2).astype(np.float32), depth


def clipped_image_point(point: np.ndarray, limit: float = 1e6) -> Tuple[int, int]:
    """把投影点限制到可安全传给 OpenCV 的整数范围。"""
    point = np.clip(np.asarray(point, dtype=np.float64), -limit, limit)
    return tuple(np.round(point).astype(np.int32).tolist())


def draw_plain_clipped_line(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """标准图像线段裁剪：用于去畸变图上的 8 点连线。"""
    h, w = image.shape[:2]
    ok, clipped_pt1, clipped_pt2 = cv2.clipLine(
        (0, 0, w, h), clipped_image_point(start), clipped_image_point(end))
    if ok:
        cv2.line(image, clipped_pt1, clipped_pt2, color, thickness, lineType=cv2.LINE_AA)


def draw_corner_edges(
    image: np.ndarray,
    corner_uv: np.ndarray,
    corner_depth: np.ndarray,
    color: Tuple[int, int, int],
    min_depth: float,
    use_clipline: bool,
    max_edge_px: float,
) -> None:
    """绘制 8 个投影角点之间的边线。

    去畸变图中，投影边是普通针孔直线，可以直接用 ``clipLine`` 表示截断目标；
    原始畸变图中，边缘投影可能异常拉伸，因此只画图内短边，避免飞线。
    """
    h, w = image.shape[:2]
    finite = np.isfinite(corner_uv[:, 0]) & np.isfinite(corner_uv[:, 1])
    depth_valid = corner_depth > min_depth
    inside = ((corner_uv[:, 0] >= 0) & (corner_uv[:, 0] < w) &
              (corner_uv[:, 1] >= 0) & (corner_uv[:, 1] < h))
    auto_max_edge = max(w, h) * 0.35 if max_edge_px <= 0 else max_edge_px
    for start, end in BOX_EDGES:
        if not (depth_valid[start] and depth_valid[end] and finite[start] and finite[end]):
            continue
        thickness = 3 if (start, end) in FRONT_EDGES else 2
        if use_clipline:
            draw_plain_clipped_line(image, corner_uv[start], corner_uv[end], color, thickness)
            continue
        if not (inside[start] and inside[end]):
            continue
        if float(np.linalg.norm(corner_uv[end] - corner_uv[start])) > auto_max_edge:
            continue
        cv2.line(
            image,
            clipped_image_point(corner_uv[start]),
            clipped_image_point(corner_uv[end]),
            color,
            thickness,
            lineType=cv2.LINE_AA)


def draw_projected_box(
    image: np.ndarray,
    corners: np.ndarray,
    project_fn,
    label: str,
    color: Tuple[int, int, int],
    min_depth: float,
    max_edge_px: float,
    use_corner_clipline: bool,
) -> None:
    """绘制投影后的 3D 框。

    原始畸变图只绘制图内且长度合理的角点短边，主要用于快速检查框和原图是否
    整体贴合；去畸变图使用 OpenCV ``clipLine`` 绘制 8 点连线，适合查看边缘截断
    目标。这里不再保留 sampled 曲线投影等排查路径，避免默认工具变慢、参数变多。
    """
    corner_uv, corner_depth = project_fn(corners)
    draw_corner_edges(image, corner_uv, corner_depth, color, min_depth, use_corner_clipline, max_edge_px)

    valid = (corner_depth > min_depth) & np.isfinite(corner_uv[:, 0]) & np.isfinite(corner_uv[:, 1])
    if not np.any(valid):
        return
    h, w = image.shape[:2]
    valid_uv = corner_uv[valid]
    inside = ((valid_uv[:, 0] >= 0) & (valid_uv[:, 0] < w) &
              (valid_uv[:, 1] >= 0) & (valid_uv[:, 1] < h))
    if not np.any(inside):
        return
    anchor = valid_uv[inside][np.argmin(valid_uv[inside][:, 1])]
    x, y = np.round(anchor).astype(int).tolist()
    cv2.putText(image, label, (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_boxes_on_camera(
    image: np.ndarray,
    boxes: np.ndarray,
    names: Sequence[str],
    cam_info: Dict,
    class_names: Sequence[str],
    undistort: bool,
    undistort_alpha: float,
    min_depth: float,
    max_edge_px: float,
) -> np.ndarray:
    """在相机图上绘制 3D GT 框。

    N7 原始图片带畸变，训练侧临时在 backprojection 中使用畸变参数，因此默认可视化
    原图时也对投影点施加畸变；启用 ``--undistort`` 时，图片先去畸变，再用新内参走
    标准 pinhole 投影，并用边界裁剪显示截断目标。
    """
    image, intrinsic = undistort_if_requested(image, cam_info, undistort, undistort_alpha)
    if undistort:
        lidar2img = compute_lidar2img(cam_info, intrinsic_override=intrinsic)

        def project_fn(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return project_lidar_points_pinhole(points, lidar2img)

        use_corner_clipline = True
    else:
        def project_fn(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return project_lidar_points_distorted(points, cam_info, intrinsic)

        use_corner_clipline = False

    for corners, name in zip(corners_from_boxes(boxes), names):
        draw_projected_box(
            image, corners, project_fn, str(name), class_color(str(name), class_names),
            min_depth, max_edge_px, use_corner_clipline)
    return image


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return image
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / float(w)
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def resize_to_width_with_scale(image: np.ndarray, width: int) -> Tuple[np.ndarray, float, float]:
    """按目标宽度缩放图像，并返回 x/y 方向缩放比例。"""
    if width <= 0:
        return image, 1.0, 1.0
    h, w = image.shape[:2]
    if w == width:
        return image, 1.0, 1.0
    scale = width / float(w)
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)
    return resized, width / float(w), new_h / float(h)


def scaled_camera_info(cam_info: Dict, sx: float, sy: float) -> Dict:
    """缩放相机内参，使投影坐标和已经缩放后的可视化图片一致。"""
    if abs(sx - 1.0) < 1e-8 and abs(sy - 1.0) < 1e-8:
        return cam_info
    scaled = dict(cam_info)
    intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32).copy()
    intrinsic[0, :] *= float(sx)
    intrinsic[1, :] *= float(sy)
    scaled['cam_intrinsic'] = intrinsic

    # 这些尺寸字段描述当前 K 对应的图像坐标系。缩放 K 后同步更新，避免
    # 后续逻辑把已经缩放过的 K 再当作原始标定尺寸处理。
    for width_key in ('intrinsic_width', 'image_width', 'width'):
        if width_key in scaled and scaled[width_key]:
            scaled[width_key] = int(round(float(scaled[width_key]) * float(sx)))
    for height_key in ('intrinsic_height', 'image_height', 'height'):
        if height_key in scaled and scaled[height_key]:
            scaled[height_key] = int(round(float(scaled[height_key]) * float(sy)))
    return scaled


def camera_info_for_loaded_image(cam_info: Dict, image: np.ndarray) -> Dict:
    """把 pkl 中的相机内参适配到当前读入图片的实际尺寸。

    converter 保存的 ``cam_intrinsic`` 可以对应 1600x900 或 2560x1440 等标定
    尺寸，而 ``data_path`` 指向的图片可能已经离线 resize 成 704x256。训练侧会
    在 pipeline 中通过 ``post_rot`` 完成这一步；可视化脚本直接在图片上画框，
    因此这里等价地把 K 先缩到当前图片尺寸。
    """
    dims = camera_dimension_fields(cam_info, image.shape[:2])
    intrinsic_width = dims['intrinsic_width']
    intrinsic_height = dims['intrinsic_height']
    image_width = dims['image_width']
    image_height = dims['image_height']
    if intrinsic_width <= 0 or intrinsic_height <= 0 or image_width <= 0 or image_height <= 0:
        return cam_info

    sx = image_width / float(intrinsic_width)
    sy = image_height / float(intrinsic_height)
    scaled = scaled_camera_info(cam_info, sx, sy)
    scaled['intrinsic_width'] = image_width
    scaled['intrinsic_height'] = image_height
    scaled['image_width'] = image_width
    scaled['image_height'] = image_height
    scaled['width'] = image_width
    scaled['height'] = image_height
    return scaled


def pad_to_shape(image: np.ndarray, height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    canvas[:h, :w] = image
    return canvas


def placeholder_image(width: int, height: int, text: str) -> np.ndarray:
    """生成缺图占位图。"""
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = max(12, (width - text_size[0]) // 2)
    y = max(28, height // 2)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2, cv2.LINE_AA)
    return image


def draw_label_tag(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int] = (8, 8),
    bg_color: Tuple[int, int, int] = (0, 110, 230),
) -> None:
    """绘制醒目的矩形角标，宽度由文字自动决定。"""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 8, 5
    w = text_size[0] + pad_x * 2
    h = text_size[1] + pad_y * 2 + baseline
    cv2.rectangle(image, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.putText(image, text, (x + pad_x, y + pad_y + text_size[1]), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def add_top_header(image: np.ndarray, text: str) -> np.ndarray:
    """给整张可视化图增加顶部标题栏，避免遮挡任何相机画面。"""
    h, w = image.shape[:2]
    header_h = 40
    header = np.full((header_h, w, 3), 24, dtype=np.uint8)
    cv2.putText(header, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def render_camera_panel(
    info: Dict,
    cam_id: str,
    data_root: Path,
    boxes: np.ndarray,
    names: Sequence[str],
    class_names: Sequence[str],
    camera_width: int,
    undistort: bool,
    undistort_alpha: float,
    min_depth: float,
    max_edge_px: float,
    draw_fullres: bool,
) -> np.ndarray:
    cam_info = info['cams'][cam_id]
    image_path = resolve_image_path(data_root, cam_info['data_path'])
    display_name = camera_display_name(cam_id)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        logger.warning('Cannot read image: %s', image_path)
        image = placeholder_image(camera_width, int(camera_width * 9 / 16), f'missing {display_name}')
    else:
        # 先把标定 K 适配到 data_path 当前图片尺寸；704x256 缓存图会依赖这一步。
        draw_cam_info = camera_info_for_loaded_image(cam_info, image)
        if not draw_fullres:
            # 先缩小再画框可以显著提升检查速度；同步缩放内参后，投影结果和缩放图一致。
            image, sx, sy = resize_to_width_with_scale(image, camera_width)
            draw_cam_info = scaled_camera_info(draw_cam_info, sx, sy)
        image = draw_boxes_on_camera(
            image, boxes, names, draw_cam_info, class_names,
            undistort, undistort_alpha, min_depth, max_edge_px)
        if draw_fullres:
            image = resize_to_width(image, camera_width)
    draw_label_tag(image, display_name)
    return image


def make_camera_mosaic(panels: Sequence[np.ndarray]) -> np.ndarray:
    if not panels:
        return placeholder_image(960, 540, 'no cameras')
    if len(panels) == 6:
        rows = [panels[:3], panels[3:]]
    else:
        cols = int(math.ceil(math.sqrt(len(panels))))
        rows = [panels[i:i + cols] for i in range(0, len(panels), cols)]

    row_images = []
    for row in rows:
        max_h = max(img.shape[0] for img in row)
        max_w = max(img.shape[1] for img in row)
        row_images.append(np.concatenate([pad_to_shape(img, max_h, max_w) for img in row], axis=1))
    max_row_w = max(img.shape[1] for img in row_images)
    return np.concatenate([pad_to_shape(img, img.shape[0], max_row_w) for img in row_images], axis=0)


def make_metric_bev_range(bev_range: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """把 BEV 显示范围扩成 x/y 等跨度，保证米制比例不变形。"""
    x_min, y_min, x_max, y_max = [float(x) for x in bev_range]
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    span = max(x_span, y_span)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    return (
        x_center - span / 2.0,
        y_center - span / 2.0,
        x_center + span / 2.0,
        y_center + span / 2.0,
    )


def infer_bev_range(infos: Sequence[Dict], margin: float = 5.0) -> Tuple[float, float, float, float]:
    """根据整个 pkl 的 GT box 自动推断一个稳定的 BEV 显示范围。"""
    all_xy = []
    for info in infos:
        boxes = to_numpy(info.get('gt_boxes', np.zeros((0, 7), dtype=np.float32)))
        if boxes.size:
            all_xy.append(corners_from_boxes(boxes)[:, :, :2].reshape(-1, 2))
    if not all_xy:
        return make_metric_bev_range((-50.0, -50.0, 80.0, 50.0))
    xy = np.concatenate(all_xy, axis=0)
    x_min, y_min = np.min(xy, axis=0) - margin
    x_max, y_max = np.max(xy, axis=0) + margin
    x_min, y_min = min(float(x_min), -10.0), min(float(y_min), -30.0)
    x_max, y_max = max(float(x_max), 50.0), max(float(y_max), 30.0)
    if x_max - x_min < 60:
        center = (x_max + x_min) / 2
        x_min, x_max = center - 30, center + 30
    if y_max - y_min < 60:
        center = (y_max + y_min) / 2
        y_min, y_max = center - 30, center + 30
    return make_metric_bev_range((x_min, y_min, x_max, y_max))


def bev_to_pixel(xy: np.ndarray, bev_range: Tuple[float, float, float, float], size: int) -> np.ndarray:
    x_min, y_min, x_max, y_max = bev_range
    x, y = xy[:, 0], xy[:, 1]
    # 图像上方表示车辆前方（+x），图像左侧表示车辆左侧（+y）。
    u = (y_max - y) / max(y_max - y_min, 1e-6) * (size - 1)
    v = (x_max - x) / max(x_max - x_min, 1e-6) * (size - 1)
    return np.stack([u, v], axis=1)


def draw_ego_marker(image: np.ndarray, bev_range: Tuple[float, float, float, float]) -> None:
    """用一个小车形状标记自车位置，车头朝 BEV 图上方。"""
    size = image.shape[0]
    car_xy = np.array([
        [2.4, 0.0],
        [-1.6, 1.0],
        [-1.1, 0.35],
        [-1.1, -0.35],
        [-1.6, -1.0],
    ], dtype=np.float32)
    pts = bev_to_pixel(car_xy, bev_range, size).astype(np.int32)
    cv2.fillPoly(image, [pts], (230, 230, 230), lineType=cv2.LINE_AA)
    cv2.polylines(image, [pts], True, (20, 20, 20), 2, cv2.LINE_AA)


def draw_axis_legend(image: np.ndarray) -> None:
    """在角落绘制固定尺寸坐标系图例，不占用主 BEV 空间。"""
    h, w = image.shape[:2]
    origin = np.array([w - 42, h - 42], dtype=np.int32)
    x_end = origin + np.array([0, -30], dtype=np.int32)
    y_end = origin + np.array([-30, 0], dtype=np.int32)
    cv2.arrowedLine(image, tuple(origin), tuple(x_end), (0, 255, 0), 2, tipLength=0.28)
    cv2.arrowedLine(image, tuple(origin), tuple(y_end), (255, 0, 0), 2, tipLength=0.28)
    cv2.putText(image, '+x', tuple(x_end + np.array([-8, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, '+y', tuple(y_end + np.array([-26, 4])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)


def draw_bev_grid(image: np.ndarray, bev_range: Tuple[float, float, float, float]) -> None:
    size = image.shape[0]
    x_min, y_min, x_max, y_max = bev_range
    grid_color = (45, 45, 45)
    for x in range(int(math.floor(x_min / 10) * 10), int(math.ceil(x_max / 10) * 10) + 1, 10):
        pts = bev_to_pixel(np.array([[x, y_min], [x, y_max]], dtype=np.float32), bev_range, size).astype(int)
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), grid_color, 1)
    for y in range(int(math.floor(y_min / 10) * 10), int(math.ceil(y_max / 10) * 10) + 1, 10):
        pts = bev_to_pixel(np.array([[x_min, y], [x_max, y]], dtype=np.float32), bev_range, size).astype(int)
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), grid_color, 1)
    draw_ego_marker(image, bev_range)
    draw_axis_legend(image)


def draw_bev_box_heading(
    image: np.ndarray,
    box_corners: np.ndarray,
    box: np.ndarray,
    bev_range: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
    size: int,
    heading_style: str,
) -> None:
    """绘制 BEV 框朝向。\n\n    默认用前边加粗表示朝向，避免小车框被中心箭头遮挡；需要和旧效果对比时\n    可以通过 ``--bev-heading-style arrow`` 重新打开箭头。\n    """
    if heading_style == 'none':
        return
    if heading_style == 'front-edge':
        front_pts = bev_to_pixel(box_corners[:2, :2], bev_range, size).astype(np.int32)
        cv2.line(image, tuple(front_pts[0]), tuple(front_pts[1]), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(image, tuple(front_pts[0]), tuple(front_pts[1]), color, 1, cv2.LINE_AA)
        return
    if heading_style == 'arrow':
        center = np.array([[box[0], box[1]]], dtype=np.float32)
        yaw = float(box[6])
        front = center + np.array([[math.cos(yaw), math.sin(yaw)]], dtype=np.float32) * max(float(box[3]) * 0.45, 1.2)
        arrow = bev_to_pixel(np.concatenate([center, front], axis=0), bev_range, size).astype(np.int32)
        cv2.arrowedLine(image, tuple(arrow[0]), tuple(arrow[1]), color, 2, tipLength=0.22)


def render_bev_panel(
    boxes: np.ndarray,
    names: Sequence[str],
    class_names: Sequence[str],
    bev_range: Tuple[float, float, float, float],
    size: int,
    heading_style: str,
) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    draw_bev_grid(image, bev_range)
    for box_corners, box, name in zip(corners_from_boxes(boxes), boxes, names):
        color = class_color(str(name), class_names)
        pts = bev_to_pixel(box_corners[:4, :2], bev_range, size).astype(np.int32)
        cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
        draw_bev_box_heading(image, box_corners, box, bev_range, color, size, heading_style)
        label_pt = bev_to_pixel(np.array([[box[0], box[1]]], dtype=np.float32), bev_range, size)[0].astype(int)
        cv2.putText(image, str(name), tuple(label_pt + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    x_min, y_min, x_max, y_max = bev_range
    summary = f'BEV x[{x_min:.1f},{x_max:.1f}] y[{y_min:.1f},{y_max:.1f}] equal-scale'
    cv2.rectangle(image, (0, 0), (size, 34), (0, 0, 0), -1)
    cv2.putText(image, summary, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def render_info(
    info: Dict,
    data_root: Path,
    class_names: Sequence[str],
    camera_ids: Optional[Sequence[str]],
    camera_width: int,
    bev_range: Tuple[float, float, float, float],
    bev_size: int,
    bev_heading_style: str,
    undistort: bool,
    undistort_alpha: float,
    min_depth: float,
    max_edge_px: float,
    draw_fullres: bool,
    no_bev: bool,
) -> np.ndarray:
    boxes = to_numpy(info.get('gt_boxes', np.zeros((0, 7), dtype=np.float32)))
    names = [str(x) for x in list(info.get('gt_names', []))]
    panels = [
        render_camera_panel(
            info, cam_id, data_root, boxes, names, class_names, camera_width,
            undistort, undistort_alpha, min_depth, max_edge_px, draw_fullres)
        for cam_id in choose_camera_order(info, camera_ids)
    ]
    mosaic = make_camera_mosaic(panels)
    timestamp = info.get('timestamp', 'unknown')
    ref_text = '/'.join(info_ref_parts(info))
    header_text = f'{ref_text} | ts={timestamp} | boxes={len(boxes)}'
    if no_bev:
        return add_top_header(mosaic, header_text)
    bev = render_bev_panel(boxes, names, class_names, bev_range, bev_size, bev_heading_style)
    bev = cv2.resize(bev, (mosaic.shape[0], mosaic.shape[0]), interpolation=cv2.INTER_AREA)
    return add_top_header(np.concatenate([mosaic, bev], axis=1), header_text)


def selected_infos(infos: Sequence[Dict], start_index: int, stride: int, max_frames: Optional[int]) -> Iterable[Tuple[int, Dict]]:
    count = 0
    for idx in range(max(start_index, 0), len(infos), max(stride, 1)):
        if max_frames is not None and count >= max_frames:
            break
        yield idx, infos[idx]
        count += 1


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    """把 pkl 中 nuScenes 风格的 ``[w, x, y, z]`` 四元数转成旋转矩阵。

    可视化脚本尽量不 import dataset，避免环境依赖影响离线检查，所以这里保留
    一个很小的本地实现。
    """
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise ValueError('zero-norm quaternion')
    w, x, y, z = quat / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def info_lidar2global(info: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """读取单帧 lidar 到 clip 参考坐标系的位姿。

    返回的位姿已经是 Fast-BEV 训练坐标轴下的 ``global_from_lidar``。这里的
    global 可以只是每个 clip 的局部参考系；时序补偿只需要相对运动，不依赖
    真实全局地图坐标。返回 ``None`` 表示可以逐帧可视化，但无法按原版 Fast-BEV
    的方式做相邻帧运动补偿。
    """
    rot = info.get('lidar2global_rotation', info.get('ego2global_rotation'))
    tran = info.get('lidar2global_translation', info.get('ego2global_translation'))
    if rot is None or tran is None:
        return None
    return quat_wxyz_to_matrix(rot), np.asarray(tran, dtype=np.float32).reshape(3)


def check_temporal_poses(infos: Sequence[Dict], max_items: int) -> None:
    """打印关键帧和历史帧的相对位姿，用于快速检查时序字段。

    这个检查不能替代图像叠框观察，只用于确认 pkl 内存在有效 pose，且 ``prev``
    帧能转换到关键帧 lidar 坐标系。静止车辆的相对运动可能接近 0；运动 clip
    中这些数值应当随时间平滑变化。
    """
    checked = 0
    for info in infos:
        if checked >= max_items:
            break
        if not info.get('prev'):
            continue
        key_pose = info_lidar2global(info)
        prev_pose = info_lidar2global(info['prev'][0])
        if key_pose is None or prev_pose is None:
            logger.warning('token=%s has prev but missing pose fields', info.get('token'))
            checked += 1
            continue
        key_r, key_t = key_pose
        prev_r, prev_t = prev_pose
        key_from_prev_r = key_r.T @ prev_r
        key_from_prev_t = key_r.T @ (prev_t - key_t)
        yaw = math.atan2(key_from_prev_r[1, 0], key_from_prev_r[0, 0])
        logger.info(
            'pose_check token=%s prev=%s dt_us=%s key_from_prev_t=%s yaw=%.6f rad',
            info.get('token'),
            info['prev'][0].get('token'),
            int(info.get('timestamp', 0)) - int(info['prev'][0].get('timestamp', 0)),
            np.array2string(key_from_prev_t, precision=4),
            yaw)
        checked += 1
    if checked == 0:
        logger.warning('No temporal pose pairs checked; pkl may have only one frame or no prev links.')


def sensor2reference_lidar_for_dataset(
    cam_info: Dict,
    frame_info: Dict,
    ref_info: Optional[Dict],
    temporal_compensate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """复刻 dataset 中相机外参补偿到关键帧 lidar 的逻辑。

    key frame 的相机外参已经直接表达在当前帧 lidar 坐标系下，因此不需要补偿。
    对于 prev/next 相邻帧，训练时 Fast-BEV 希望把相邻帧图像反投影到关键帧 BEV
    体素中，所以需要用两帧 ``lidar2global`` 位姿计算 ``key_from_adj``，再左乘
    到相邻帧的 ``adj_lidar_from_adj_camera`` 外参上。
    """
    sensor2lidar_r = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float32)
    sensor2lidar_t = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float32).reshape(3)

    if not temporal_compensate:
        return sensor2lidar_r, sensor2lidar_t, False, 'temporal_disabled'
    if ref_info is None or frame_info.get('token') == ref_info.get('token'):
        return sensor2lidar_r, sensor2lidar_t, False, 'key_frame'

    ref_pose = info_lidar2global(ref_info)
    frame_pose = info_lidar2global(frame_info)
    if ref_pose is None or frame_pose is None:
        return sensor2lidar_r, sensor2lidar_t, False, 'missing_pose'

    ref_to_global_r, ref_to_global_t = ref_pose
    frame_to_global_r, frame_to_global_t = frame_pose
    key_from_adj_r = ref_to_global_r.T @ frame_to_global_r
    key_from_adj_t = ref_to_global_r.T @ (frame_to_global_t - ref_to_global_t)

    compensated_r = key_from_adj_r @ sensor2lidar_r
    compensated_t = key_from_adj_r @ sensor2lidar_t + key_from_adj_t
    return compensated_r.astype(np.float32), compensated_t.astype(np.float32), True, 'pose_compensated'


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    """返回两个矩阵或向量的最大绝对误差。

    空畸变数组是合法输入；形状不同则直接视为无穷大误差，方便上层报告
    字段缺失或长度不一致的问题。
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float('inf')
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def append_failure(report: Dict, message: str) -> None:
    """记录失败项，同时限制日志列表长度，避免大 pkl 输出过长。"""
    report['failed_checks'] += 1
    if len(report['failures']) < 50:
        report['failures'].append(message)


def append_warning(report: Dict, message: str) -> None:
    """记录警告项，同时限制日志列表长度。"""
    report['warning_checks'] += 1
    if len(report['warnings']) < 50:
        report['warnings'].append(message)


def update_error(report: Dict, error_name: str, error_value: float) -> None:
    """更新某类误差和全局最大误差。"""
    report['max_error'] = max(report['max_error'], float(error_value))
    report['max_errors'][error_name] = max(report['max_errors'].get(error_name, 0.0), float(error_value))


def check_one_camera_geometry(
    report: Dict,
    info: Dict,
    cam_id: str,
    ref_info: Optional[Dict],
    relation: str,
    tolerance: float,
    temporal_compensate: bool,
) -> None:
    """检查单个相机在 key frame 或相邻帧中的几何一致性。

    这里同时做三类检查：
    1. 用 converter 字段的列向量公式直接算出的 ``lidar2img``；
    2. 用 Fast-BEV dataset 行向量写法算出的 ``lidar2img``；
    3. dataset 传给后续 pipeline/backprojection 的 ``lidar2img_aug`` 字段。
    三者一致，才说明 pkl 写入的内外参和训练代码读取方式没有坐标约定错位。
    """
    cam_info = info['cams'][cam_id]
    sensor_r, sensor_t, compensated, reason = sensor2reference_lidar_for_dataset(
        cam_info=cam_info,
        frame_info=info,
        ref_info=ref_info,
        temporal_compensate=temporal_compensate,
    )
    dataset_lidar2img, dataset_aug, dataset_extra = dataset_lidar2img_from_cam_info(
        cam_info,
        sensor2lidar_r=sensor_r,
        sensor2lidar_t=sensor_t,
        temporal_compensated=compensated,
    )
    direct_lidar2img = direct_lidar2img_from_sensor2lidar(
        cam_info,
        sensor2lidar_r=sensor_r,
        sensor2lidar_t=sensor_t,
    )

    prefix = f"{relation} token={info.get('token')} cam={cam_id} reason={reason}"
    lidar2img_error = max_abs_diff(dataset_lidar2img, direct_lidar2img)
    update_error(report, 'lidar2img_dataset_vs_direct', lidar2img_error)
    if lidar2img_error > tolerance:
        append_failure(report, f'{prefix} lidar2img 误差 {lidar2img_error:.6g} > {tolerance}')

    rot_error = max_abs_diff(dataset_aug['rot'], sensor_r)
    tran_error = max_abs_diff(dataset_aug['tran'], sensor_t)
    intrin_error = max_abs_diff(dataset_aug['intrin'], cam_info['cam_intrinsic'])
    post_rot_error = max_abs_diff(dataset_aug['post_rot'], np.eye(3, dtype=np.float32))
    post_tran_error = max_abs_diff(dataset_aug['post_tran'], np.zeros(3, dtype=np.float32))
    distortion = np.asarray(cam_info.get('distortion', []), dtype=np.float32).reshape(-1)
    aug_distortion_error = max_abs_diff(dataset_aug['distortion'], distortion)
    extra_distortion_error = max_abs_diff(dataset_extra['distortion'], distortion)
    dimension_fields = camera_dimension_fields(cam_info)
    dimension_errors = {}
    for field_name, expected_value in dimension_fields.items():
        # 新增尺寸字段需要和 CustomMultiViewDataset 传给 pipeline 的字段一致。
        aug_value = int(dataset_aug.get(field_name, 0) or 0)
        extra_value = int(dataset_extra.get(field_name, 0) or 0)
        dimension_errors[f'aug_{field_name}'] = abs(aug_value - int(expected_value))
        dimension_errors[f'extra_{field_name}'] = abs(extra_value - int(expected_value))

    update_error(report, 'aug_rot', rot_error)
    update_error(report, 'aug_tran', tran_error)
    update_error(report, 'aug_intrin', intrin_error)
    update_error(report, 'aug_post_rot', post_rot_error)
    update_error(report, 'aug_post_tran', post_tran_error)
    update_error(report, 'aug_distortion', aug_distortion_error)
    update_error(report, 'extra_distortion', extra_distortion_error)
    for name, value in dimension_errors.items():
        update_error(report, name, float(value))

    for name, value in [
        ('aug_rot', rot_error),
        ('aug_tran', tran_error),
        ('aug_intrin', intrin_error),
        ('aug_post_rot', post_rot_error),
        ('aug_post_tran', post_tran_error),
        ('aug_distortion', aug_distortion_error),
        ('extra_distortion', extra_distortion_error),
        *dimension_errors.items(),
    ]:
        if value > tolerance:
            append_failure(report, f'{prefix} {name} 误差 {value:.6g} > {tolerance}')

    if bool(dataset_aug['temporal_compensated']) != bool(dataset_extra['temporal_compensated']):
        append_failure(report, f'{prefix} lidar2img_aug 和 lidar2img_extra 的 temporal_compensated 不一致')
    if relation != 'key' and reason == 'missing_pose':
        append_warning(report, f'{prefix} 缺少 pose，dataset 会退化为不做相邻帧运动补偿')
    if relation != 'key' and reason == 'pose_compensated':
        report['temporal_compensated_cameras'] += 1
    if relation == 'key':
        report['key_camera_checks'] += 1
    else:
        report['adjacent_camera_checks'] += 1


def check_geometry_consistency(
    infos: Sequence[Dict],
    camera_ids: Optional[Sequence[str]],
    max_infos: int,
    tolerance: float,
    include_temporal: bool,
    strict: bool,
) -> Dict:
    """检查 pkl 几何字段和 Fast-BEV dataset 使用方式是否一致。

    key frame 检查是必须的：它确认 converter 写入的 ``sensor2lidar`` 和
    ``cam_intrinsic`` 能被 dataset 正确解释。时序检查是可选的：它会继续检查
    ``prev`` 和 ``next`` 中的相邻帧是否能通过 pose 补偿到关键帧 lidar 坐标系。
    """
    report = dict(
        checked_infos=0,
        key_camera_checks=0,
        adjacent_camera_checks=0,
        temporal_compensated_cameras=0,
        failed_checks=0,
        warning_checks=0,
        max_error=0.0,
        max_errors={},
        failures=[],
        warnings=[],
        tolerance=float(tolerance),
        include_temporal=bool(include_temporal),
    )

    for info in infos[:max(max_infos, 0)]:
        report['checked_infos'] += 1
        for cam_id in choose_camera_order(info, camera_ids):
            check_one_camera_geometry(
                report=report,
                info=info,
                cam_id=cam_id,
                ref_info=info,
                relation='key',
                tolerance=tolerance,
                temporal_compensate=True,
            )

        if not include_temporal:
            continue
        for relation in ('prev', 'next'):
            for adj_info in info.get(relation, []):
                for cam_id in choose_camera_order(adj_info, camera_ids):
                    check_one_camera_geometry(
                        report=report,
                        info=adj_info,
                        cam_id=cam_id,
                        ref_info=info,
                        relation=relation,
                        tolerance=tolerance,
                        temporal_compensate=True,
                    )

    logger.info(
        'geometry_check infos=%d key_cams=%d adj_cams=%d temporal_compensated=%d max_error=%.6g failures=%d warnings=%d',
        report['checked_infos'],
        report['key_camera_checks'],
        report['adjacent_camera_checks'],
        report['temporal_compensated_cameras'],
        report['max_error'],
        report['failed_checks'],
        report['warning_checks'],
    )
    for message in report['warnings'][:10]:
        logger.warning('geometry_check warning: %s', message)
    for message in report['failures'][:10]:
        logger.error('geometry_check failure: %s', message)
    if strict and report['failed_checks'] > 0:
        raise AssertionError(f"geometry_check failed: {report['failed_checks']} failed checks")
    return report


def write_geometry_report(output_dir: Path, report: Dict) -> Path:
    """把几何一致性检查结果写入 JSON，方便离线保存和接手人复核。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'geometry_check.json'
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='可视化并校验 Fast-BEV CustomMultiViewDataset 使用的 pkl 真值。',
        formatter_class=RawDefaultsHelpFormatter,
        epilog=(
            '示例：\n'
            '  # 704x256 缓存图，只做几何校验，不生成图片\n'
            '  python tools/data_converter/n7/visualize_n7_fastbev_pkl.py '
            '--pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl '
            '--data-root data/N7_704_256 '
            '--output-dir work_dirs/vis_n7_704_check '
            '--no-render --check-geometry --check-temporal-geometry --strict-geometry\n\n'
            '  # 抽帧生成视频，不保存逐帧 jpg/png\n'
            '  python tools/data_converter/n7/visualize_n7_fastbev_pkl.py '
            '--pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl '
            '--data-root data/N7_704_256 '
            '--output-dir work_dirs/vis_n7_704 '
            '--max-frames 300 --stride 5 --video-only --bev-range -50 -50 50 50\n\n'
            '  # 少量抽帧查看去畸变图上的 3D 框贴合情况\n'
            '  python tools/data_converter/n7/visualize_n7_fastbev_pkl.py '
            '--pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl '
            '--data-root data/N7_704_256 '
            '--output-dir work_dirs/vis_n7_704_undistort '
            '--max-frames 50 --stride 20 --undistort --no-video'
        ))
    parser.add_argument('--pkl', required=True, help='converter 输出的 pkl 路径')
    parser.add_argument('--data-root', required=True, help='converter --data-path 使用的数据根目录；相对图片路径会在该目录下解析')
    parser.add_argument('--output-dir', required=True, help='输出可视化图片帧和可选视频的目录')
    parser.add_argument('--camera-ids', nargs='+', default=None, help='指定要绘制的相机 id；默认使用六目检查顺序')
    parser.add_argument('--start-index', type=int, default=0, help='从第几个 info 开始可视化')
    parser.add_argument('--stride', type=int, default=1, help='每隔多少个 info 可视化一帧')
    parser.add_argument('--max-frames', type=int, default=100, help='最多渲染多少帧；设为 -1 表示全部渲染')
    parser.add_argument('--camera-width', type=int, default=640, help='拼接前每个相机小图的宽度')
    parser.add_argument('--draw-fullres', action='store_true', help='在原始分辨率上画框/去畸变后再缩放；速度慢，仅用于对比旧逻辑')
    parser.add_argument('--no-render', action='store_true', help='只执行 pkl 读取、pose/geometry 检查，不生成可视化图片或视频')
    parser.add_argument('--bev-size', type=int, default=700, help='BEV 面板在缩放到拼图高度前的尺寸')
    parser.add_argument('--bev-range-mode', default='fastbev', choices=['fastbev', 'auto'], help='BEV 显示范围来源；fastbev 使用 x/y 都为 ±50m，auto 根据 pkl 中所有 GT box 自动推断')
    parser.add_argument('--bev-range', nargs=4, type=float, metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'), default=None, help='固定 BEV 显示范围；填写后优先级高于 --bev-range-mode')
    parser.add_argument('--bev-heading-style', default='front-edge', choices=['front-edge', 'arrow', 'none'], help='BEV 目标朝向显示方式；front-edge 用前边加粗，arrow 使用中心箭头，none 不画朝向')
    parser.add_argument('--no-bev', action='store_true', help='只保存相机拼图，不拼接 BEV 面板')
    parser.add_argument('--undistort', action='store_true', help='画框前先对图片去畸变，并使用 OpenCV 返回的新内参投影')
    parser.add_argument('--undistort-alpha', type=float, default=0.0, help='OpenCV 去畸变 alpha；0 裁掉无效区域，1 保留完整视野')
    parser.add_argument('--max-corner-edge-px', type=float, default=0.0, help='原始畸变图中角点连线最大像素长度；0 表示按图像尺寸自动设置。--undistort 时不使用该限制')
    parser.add_argument('--min-depth', type=float, default=0.1, help='不绘制深度小于该阈值的投影边')
    parser.add_argument('--fps', type=int, default=10, help='输出视频帧率')
    parser.add_argument('--no-video', action='store_true', help='不输出 visualization.mp4 视频')
    parser.add_argument('--video-only', action='store_true', help='只输出 visualization.mp4，不保存 frames 图片；不能和 --no-video 同时使用')
    parser.add_argument('--workers', type=int, default=1, help='并行渲染帧数；仅在 --no-video 时启用，建议从 2 或 4 开始')
    parser.add_argument('--image-ext', default='jpg', choices=['jpg', 'png'], help='逐帧可视化图片格式')
    parser.add_argument('--check-poses', action='store_true', help='渲染前打印关键帧和历史帧的相对位姿')
    parser.add_argument('--pose-check-count', type=int, default=10, help='启用 --check-poses 时最多打印多少组 key/prev 位姿')
    parser.add_argument('--check-geometry', action='store_true', help='渲染前检查 converter 字段和 Fast-BEV dataset 使用的内外参是否一致')
    parser.add_argument('--check-temporal-geometry', action='store_true', help='启用 --check-geometry 时继续检查 prev/next 相邻帧的时序补偿几何')
    parser.add_argument('--geometry-check-count', type=int, default=50, help='启用 --check-geometry 时最多检查多少个 key frame info')
    parser.add_argument('--geometry-tol', type=float, default=1e-3, help='几何一致性检查允许的最大绝对误差')
    parser.add_argument('--strict-geometry', action='store_true', help='几何检查失败时直接退出，适合自动化校验')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    if args.video_only and args.no_video:
        parser.error('--video-only 和 --no-video 不能同时使用')
    return args


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    pkl_path = Path(args.pkl)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infos, metadata = load_fastbev_pkl(pkl_path)
    class_names = get_class_names(metadata, infos)
    max_frames = None if args.max_frames is not None and args.max_frames < 0 else args.max_frames
    if args.bev_range:
        bev_range = make_metric_bev_range(tuple(args.bev_range))
    elif args.bev_range_mode == 'fastbev':
        bev_range = make_metric_bev_range((-50.0, -50.0, 50.0, 50.0))
    else:
        bev_range = infer_bev_range(infos)

    logger.info('Loaded %d infos from %s', len(infos), pkl_path)
    logger.info('metadata.coordinate=%s', metadata.get('coordinate', 'unknown'))
    logger.info('data_root=%s', data_root)
    logger.info('class_names=%s', class_names)
    logger.info('bev_range=%s', bev_range)
    if args.check_poses:
        check_temporal_poses(infos, args.pose_check_count)
    if args.check_geometry:
        geometry_report = check_geometry_consistency(
            infos=infos,
            camera_ids=args.camera_ids,
            max_infos=args.geometry_check_count,
            tolerance=args.geometry_tol,
            include_temporal=args.check_temporal_geometry,
            strict=args.strict_geometry,
        )
        report_path = write_geometry_report(output_dir, geometry_report)
        logger.info('Geometry report saved to %s', report_path)
    if args.no_render:
        logger.info('Skip rendering because --no-render is set.')
        return

    writers: Dict[Path, Tuple[cv2.VideoWriter, Tuple[int, int]]] = {}
    rendered = 0
    saved_frame_dirs = set()
    iterator = list(selected_infos(infos, args.start_index, args.stride, max_frames))

    def render_and_save_frame(item, need_canvas=False):
        """渲染并保存单帧；多线程模式下不返回大图，减少内存占用。"""
        pkl_index, info = item
        canvas = render_info(
            info=info,
            data_root=data_root,
            class_names=class_names,
            camera_ids=args.camera_ids,
            camera_width=args.camera_width,
            bev_range=bev_range,
            bev_size=args.bev_size,
            bev_heading_style=args.bev_heading_style,
            undistort=args.undistort,
            undistort_alpha=args.undistort_alpha,
            min_depth=args.min_depth,
            max_edge_px=args.max_corner_edge_px,
            draw_fullres=args.draw_fullres,
            no_bev=args.no_bev,
        )
        ref_output_dir = info_output_dir(output_dir, info)
        frame_dir = ref_output_dir / 'frames'
        saved_frame_dir = None
        if args.video_only:
            # 只生成视频时不落逐帧图片，避免内网批量检查时产生大量小文件。
            ref_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / f'{frame_stem_from_info(info, pkl_index)}.{args.image_ext}'
            cv2.imwrite(str(frame_path), canvas)
            saved_frame_dir = frame_dir
        if need_canvas:
            return saved_frame_dir, ref_output_dir, canvas
        return saved_frame_dir, ref_output_dir, None

    workers = max(int(args.workers), 1)
    if args.no_video and workers > 1:
        logger.info('Parallel rendering enabled: workers=%d', workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(render_and_save_frame, item, False) for item in iterator]
            for future in tqdm(as_completed(futures), total=len(futures), desc='visualizing'):
                frame_dir, _, _ = future.result()
                if frame_dir is not None:
                    saved_frame_dirs.add(frame_dir)
                rendered += 1
    else:
        if (not args.no_video) and workers > 1:
            logger.info('Ignore --workers because video writing is sequential; use --no-video to enable parallel image rendering.')
        for item in tqdm(iterator, total=len(iterator), desc='visualizing'):
            frame_dir, ref_output_dir, canvas = render_and_save_frame(item, need_canvas=True)
            if frame_dir is not None:
                saved_frame_dirs.add(frame_dir)

            if not args.no_video:
                video_path = ref_output_dir / 'visualization.mp4'
                h, w = canvas.shape[:2]
                if ref_output_dir not in writers:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writers[ref_output_dir] = (cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h)), (w, h))
                    logger.info('Video writer: %s %dx%d @ %dfps', video_path, w, h, args.fps)
                writer, writer_size = writers[ref_output_dir]
                if writer_size == (w, h):
                    writer.write(canvas)
                else:
                    logger.warning('Skip video frame with changed size for %s: got %dx%d expected %dx%d',
                                   ref_output_dir, w, h, writer_size[0], writer_size[1])
            rendered += 1

    for writer, _ in writers.values():
        writer.release()
    logger.info('Rendered %d frames under %s', rendered, output_dir)
    for frame_dir in sorted(saved_frame_dirs):
        logger.info('Frame directory: %s', frame_dir)


if __name__ == '__main__':
    main()
