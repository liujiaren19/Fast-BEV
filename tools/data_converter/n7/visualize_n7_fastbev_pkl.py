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
也会有相同的投影几何问题。批量检查 pkl 几何/时序字段时请使用
``tools/data_converter/n7/validate_n7_fastbev_pkl.py``，不要把可视化输出作为
唯一的数据正确性校验。

常用命令见文件开头的 ``USAGE_EXAMPLES``，命令行 ``--help`` 也会同步展示同一份示例。

如果 pkl 中的图片路径是相对路径，``--data-root`` 必须和 converter 的
``--data-path`` 使用同一个根目录。默认先去畸变图片并用 OpenCV 返回的
new_K 投影，这样更适合人工检查目标框贴合；如果需要检查板端原始畸变输入
上的投影效果，可以显式添加 ``--raw-distorted``。
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from tools.data_converter.n7.fastbev_geometry import (
        camera_dimension_fields,
        choose_camera_order,
        compute_lidar2img,
        to_numpy,
    )
except ModuleNotFoundError:
    # 支持从仓库根目录执行，也支持直接在 tools/data_converter/n7 目录附近调试脚本。
    from fastbev_geometry import (
        camera_dimension_fields,
        choose_camera_order,
        compute_lidar2img,
        to_numpy,
    )


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RawDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """保留 epilog 示例换行，同时继续展示 argparse 默认值。"""

cv2.setNumThreads(4)

USAGE_EXAMPLES = r"""示例：
  # 抽帧生成 GT 视频，不保存逐帧 jpg/png。
  python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
--gt-pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl \
--data-root data/N7_704_256 \
--output-dir work_dirs/vis_n7_704 \
--max-frames 300 --stride 5 --video-only --bev-range -50 -50 50 50

  # test.py --out 生成预测 pkl 后，同时绘制 GT 和预测结果。
  # --gt-pkl 必须和 test/eval 使用的是同一份 val/test pkl；预测结果按 index 对齐。
  python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
--gt-pkl data/N7_704_256/pkl/custom_fastbev_20251017-20251030-20251031-20251203_infos_val_20260625.pkl \
--pred-pkl work_dirs/n7_6v_704_256/20251017_20251030_20251031_20251203_gpu4_batch24_work_8_260626/test_results/latest_val_results.pkl \
--data-root data/N7_704_256 \
--output-dir work_dirs/vis_n7_val_pred \
--max-frames 300 --stride 5 --score-thr 0.2 --max-preds 100 --video-only --bev-range -50 -50 50 50

  # 只在最终显示层把 704x256 相机图拉回 16:9，方便人眼查看。
  python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
--gt-pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl \
--data-root data/N7_704_256 \
--output-dir work_dirs/vis_n7_704_display_16x9 \
--max-frames 100 --stride 10 --display-aspect 16:9 --no-video

  # 如果要检查原始畸变图上的投影效果，显式加 --raw-distorted。
  python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
--gt-pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl \
--data-root data/N7_704_256 \
--output-dir work_dirs/vis_n7_704_raw_distorted \
--max-frames 50 --stride 20 --raw-distorted --no-video
"""

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

# 预测框单独使用一组高对比色，避免和 GT 类别颜色混淆。
PRED_COLORS = [
    (0, 0, 255),      # 红色：car 等主类预测
    (255, 0, 255),    # 洋红：truck 等第二类预测
    (255, 255, 255),
    (0, 255, 255),
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


def load_prediction_results(path: Path) -> List:
    """读取 ``tools/test.py --out`` 保存的预测 pkl。

    MMDetection3D 默认会把结果保存成 list，每个元素对应 dataset 中同 index 的
    一帧。少数脚本可能额外包一层 dict，这里做轻量兼容，但仍要求最终结果是
    list，便于和 ``infos[i]`` 稳定对齐。
    """
    with path.open('rb') as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        for key in ('results', 'outputs', 'predictions'):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise TypeError(f'Prediction dict from {path} does not contain results/outputs/predictions list')
    if isinstance(payload, list):
        return payload
    raise TypeError(f'Unsupported prediction payload type from {path}: {type(payload)!r}')


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


def compact_info_ref(info: Dict) -> str:
    """返回适合顶部标题栏展示的短引用。

    N7 的 clip 名通常已经包含日期和 sequence 信息，预测可视化时继续显示
    dataset/sequence/clip 会挤占图例空间，所以紧凑模式只显示 clip。
    """
    _, _, clip = info_ref_parts(info)
    return clip


def count_boxes_in_bev_range(boxes: np.ndarray, bev_range: Tuple[float, float, float, float]) -> int:
    """统计中心点落在当前 BEV 显示范围内的 box 数量。

    这里按中心点统计，而不是按角点裁剪统计。原因是可视化标题只需要说明当前
    BEV 范围覆盖了多少个 GT 目标，中心点判断最稳定，也和训练里常见的 range
    过滤语义更接近。
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] < 2:
        return 0
    x_min, y_min, x_max, y_max = [float(x) for x in bev_range]
    centers = boxes[:, :2]
    mask = (centers[:, 0] >= x_min) & (centers[:, 0] <= x_max) & (centers[:, 1] >= y_min) & (centers[:, 1] <= y_max)
    return int(mask.sum())


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


def tensor_like_to_numpy(value, dtype=None) -> np.ndarray:
    """把 torch Tensor / LiDARInstance3DBoxes / numpy-like 对象转成 ndarray。"""
    if value is None:
        arr = np.asarray([])
    else:
        if hasattr(value, 'tensor'):
            value = value.tensor
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            arr = value.numpy()
        else:
            arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def is_lidar_instance_boxes(value) -> bool:
    """判断对象是否是 MMDetection3D 的 LiDARInstance3DBoxes。

    这里不直接 import mmdet3d，避免离线可视化在轻量环境中因为 mmdet3d/cuda
    扩展缺失而失败。test.py 保存的 pred pkl 会保留 boxes_3d 的 Python 类，
    因此用类名判断足够稳定。
    """
    return value is not None and value.__class__.__name__ == 'LiDARInstance3DBoxes'


def prediction_boxes_to_numpy(boxes) -> np.ndarray:
    """把预测框转成可视化使用的 ``[x, y, z_center, l, w, h, yaw]``。

    GT pkl 中的 ``gt_boxes`` 会在 CustomMultiViewDataset 中以
    ``origin=(0.5, 0.5, 0.5)`` 包成 LiDARInstance3DBoxes，说明 pkl 里的 z
    按 3D 框重心保存；而 test.py 的预测结果已经是 LiDARInstance3DBoxes，
    其 ``tensor[:, 2]`` 是 MMDetection3D lidar 约定下的底中心 z。为了让 GT
    和 Pred 共用同一套本地角点生成逻辑，预测框在这里补回 ``h/2``。
    """
    boxes_np = tensor_like_to_numpy(boxes, dtype=np.float32)
    if is_lidar_instance_boxes(boxes) and boxes_np.ndim >= 2 and boxes_np.shape[-1] >= 6:
        boxes_np = boxes_np.copy()
        boxes_np[..., 2] += boxes_np[..., 5] * 0.5
    return boxes_np


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


def pred_color(name: str, class_names: Sequence[str]) -> Tuple[int, int, int]:
    """返回预测框颜色；按类别区分，同时和 GT 颜色保持明显差异。"""
    try:
        idx = class_names.index(name)
    except ValueError:
        idx = len(class_names)
    return PRED_COLORS[idx % len(PRED_COLORS)]


def unwrap_prediction_result(result):
    """兼容 FreeAnchor3DHead 常见输出和带 ``pts_bbox`` 包装的输出。"""
    if isinstance(result, dict) and 'pts_bbox' in result:
        return result['pts_bbox']
    return result


def prediction_result_to_draw_items(
    result,
    class_names: Sequence[str],
    score_thr: float,
    max_preds: int,
) -> Tuple[np.ndarray, List[str], List[str], List[Tuple[int, int, int]]]:
    """把单帧预测结果整理成可视化绘制需要的 boxes/names/labels/colors。

    返回的 ``names`` 是类别名，用于颜色和统计；``labels`` 是实际画到图上的文字，
    默认形如 ``P:car 0.82``。预测 pkl 必须来自同一份 test pkl，否则 index 对齐
    不成立。
    """
    result = unwrap_prediction_result(result)
    if not isinstance(result, dict):
        return np.zeros((0, 7), dtype=np.float32), [], [], []

    boxes = result.get('boxes_3d', result.get('bboxes_3d'))
    scores = result.get('scores_3d', result.get('scores'))
    labels = result.get('labels_3d', result.get('labels'))
    if boxes is None or scores is None or labels is None:
        return np.zeros((0, 7), dtype=np.float32), [], [], []

    boxes_np = prediction_boxes_to_numpy(boxes)
    if boxes_np.ndim == 0:
        return np.zeros((0, 7), dtype=np.float32), [], [], []
    if boxes_np.ndim == 1:
        boxes_np = boxes_np.reshape(1, -1)
    elif boxes_np.ndim > 2:
        boxes_np = boxes_np.reshape(-1, boxes_np.shape[-1])
    if boxes_np.shape[1] < 7:
        return np.zeros((0, 7), dtype=np.float32), [], [], []
    boxes_np = boxes_np[:, :7]
    scores_np = tensor_like_to_numpy(scores, dtype=np.float32).reshape(-1)
    labels_np = tensor_like_to_numpy(labels, dtype=np.int64).reshape(-1)

    count = min(len(boxes_np), len(scores_np), len(labels_np))
    if count == 0:
        return np.zeros((0, 7), dtype=np.float32), [], [], []
    boxes_np = boxes_np[:count]
    scores_np = scores_np[:count]
    labels_np = labels_np[:count]

    valid = np.isfinite(scores_np) & (scores_np >= float(score_thr))
    valid &= np.isfinite(boxes_np).all(axis=1)
    valid &= (boxes_np[:, 3] > 0) & (boxes_np[:, 4] > 0) & (boxes_np[:, 5] > 0)
    keep = np.where(valid)[0]
    if keep.size == 0:
        return np.zeros((0, 7), dtype=np.float32), [], [], []

    keep = keep[np.argsort(-scores_np[keep])]
    if max_preds > 0:
        keep = keep[:max_preds]

    pred_boxes = boxes_np[keep]
    pred_names: List[str] = []
    pred_labels: List[str] = []
    pred_colors: List[Tuple[int, int, int]] = []
    for label_id, score in zip(labels_np[keep], scores_np[keep]):
        if 0 <= int(label_id) < len(class_names):
            name = str(class_names[int(label_id)])
        else:
            name = f'class_{int(label_id)}'
        pred_names.append(name)
        pred_labels.append(f'P:{name} {float(score):.2f}')
        pred_colors.append(pred_color(name, class_names))
    return pred_boxes, pred_names, pred_labels, pred_colors


def undistort_if_requested(
    image: np.ndarray,
    cam_info: Dict,
    enabled: bool,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """按需对图片去畸变，并返回投影时应该使用的内参。

    默认去畸变，因为人工检查目标框贴合时，去畸变图上的 pinhole 投影更稳定，
    边缘截断目标也更不容易出现长飞线。显式使用 ``--raw-distorted`` 时，会退回
    到原始畸变图和原始 K 的显示方式，用于检查板端原始输入链路。
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
    """把 ``[x, y, z_center, l, w, h, yaw]`` 转成 lidar 坐标系 8 个角点。

    yaw 旋转方向按 MMDetection3D 的 ``LiDARInstance3DBoxes.corners`` 实现，
    即行向量形式下 ``x'=x*cos+y*sin, y'=-x*sin+y*cos``。之前使用普通数学
    正向旋转，BEV 上 GT/Pred 仍会相互贴合，但投影到相机时左右边可能偏掉。
    """
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
        local_xy = local[:, :2].copy()
        local[:, 0] = local_xy[:, 0] * c + local_xy[:, 1] * s
        local[:, 1] = -local_xy[:, 0] * s + local_xy[:, 1] * c
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

    if not str(label):
        return
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
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """在相机图上绘制 3D 框。

    N7 原始图片带畸变，训练侧临时在 backprojection 中使用畸变参数。为了方便
    人工检查，默认先去畸变再用新内参走标准 pinhole 投影，并用边界裁剪显示
    截断目标；显式使用 ``--raw-distorted`` 时才在原始畸变图上叠框。
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

    if labels is None:
        labels = [str(name) for name in names]
    if colors is None:
        colors = [class_color(str(name), class_names) for name in names]

    for corners, name, label, color in zip(corners_from_boxes(boxes), names, labels, colors):
        draw_projected_box(
            image, corners, project_fn, str(label), color,
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


def resize_to_display_aspect(image: np.ndarray, display_aspect: str) -> np.ndarray:
    """仅在最终显示层调整相机小图比例。

    这个函数故意放在投影和画框之后使用：图像、2D 框和分数文字会作为一个整体
    被拉伸到目标显示比例，因此不会引入新的投影误差。它只改善人眼观看 704x256
    这类强制 resize 图像时的视觉比例，不代表训练输入尺寸发生变化。
    """
    if display_aspect == 'native':
        return image
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    if display_aspect == '16:9':
        target_h = max(1, int(round(w * 9.0 / 16.0)))
    else:
        raise ValueError(f'Unsupported display_aspect: {display_aspect}')
    if target_h == h:
        return image
    return cv2.resize(image, (w, target_h), interpolation=cv2.INTER_LINEAR)


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


def fit_text_to_width(
    text: str,
    max_width: int,
    font: int,
    scale: float,
    thickness: int,
) -> str:
    """把标题文字截断到指定宽度，避免长路径挤出画布。"""
    text = str(text)
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text
    suffix = '...'
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = text[:mid] + suffix
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            low = mid
        else:
            high = mid - 1
    return text[:max(low, 0)] + suffix


def make_legend_rows(
    legend_items: Sequence[Tuple[str, Tuple[int, int, int]]],
    width: int,
    font: int,
    scale: float,
    thickness: int,
) -> List[List[Tuple[str, Tuple[int, int, int], int]]]:
    """将图例按画布宽度自动换行。"""
    rows: List[List[Tuple[str, Tuple[int, int, int], int]]] = []
    current: List[Tuple[str, Tuple[int, int, int], int]] = []
    current_w = 0
    max_w = max(80, width - 24)
    for label, color in legend_items:
        item_w = legend_item_width(str(label), font, scale, thickness)
        if current and current_w + item_w > max_w:
            rows.append(current)
            current = []
            current_w = 0
        current.append((str(label), color, item_w))
        current_w += item_w
    if current:
        rows.append(current)
    return rows


def legend_item_width(label: str, font: int, scale: float, thickness: int) -> int:
    """计算单个图例项宽度。"""
    text_w = cv2.getTextSize(str(label), font, scale, thickness)[0][0]
    return 18 + 6 + text_w + 18


def draw_legend_row(
    image: np.ndarray,
    row: Sequence[Tuple[str, Tuple[int, int, int], int]],
    x: int,
    y: int,
    font: int,
    scale: float,
    thickness: int,
) -> None:
    """绘制一行图例。"""
    for label, color, item_w in row:
        cv2.rectangle(image, (x, y - 12), (x + 16, y + 2), color, -1)
        cv2.rectangle(image, (x, y - 12), (x + 16, y + 2), (245, 245, 245), 1)
        cv2.putText(image, str(label), (x + 22, y + 2), font, scale, (235, 235, 235), thickness, cv2.LINE_AA)
        x += item_w


def add_top_header(
    image: np.ndarray,
    text: str,
    legend_items: Optional[Sequence[Tuple[str, Tuple[int, int, int]]]] = None,
) -> np.ndarray:
    """给整张可视化图增加顶部标题栏，避免遮挡任何相机画面。

    预测可视化会把类别和 GT/Pred 来源放到顶部图例里，画面中的框只保留
    必要文本，避免一帧里目标多时遮挡图像内容。图例优先和标题放在同一行，
    只有宽度不足时才换行。
    """
    _, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.64
    title_thickness = 2
    legend_scale = 0.52
    legend_thickness = 1
    legend_items = list(legend_items or [])
    legend_widths = [legend_item_width(label, font, legend_scale, legend_thickness) for label, _ in legend_items]
    legend_total_w = sum(legend_widths)
    title_gap = 28 if legend_items else 0
    title_full_w = cv2.getTextSize(str(text), font, title_scale, title_thickness)[0][0]
    single_line = bool(legend_items) and title_full_w + title_gap + legend_total_w <= w - 24

    if single_line:
        title_max_w = w - 24 - title_gap - legend_total_w
        legend_rows = [[(label, color, item_w) for (label, color), item_w in zip(legend_items, legend_widths)]]
        header_h = 40
    else:
        title_max_w = w - 24
        legend_rows = make_legend_rows(legend_items, w, font, legend_scale, legend_thickness)
        header_h = 40 + 24 * len(legend_rows)
    header = np.full((header_h, w, 3), 24, dtype=np.uint8)

    title = fit_text_to_width(text, title_max_w, font, title_scale, title_thickness)
    cv2.putText(header, title, (12, 27), font, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)

    if single_line and legend_rows:
        title_w = cv2.getTextSize(title, font, title_scale, title_thickness)[0][0]
        draw_legend_row(header, legend_rows[0], 12 + title_w + title_gap, 25, font, legend_scale, legend_thickness)
    else:
        y = 58
        for row in legend_rows:
            draw_legend_row(header, row, 12, y, font, legend_scale, legend_thickness)
            y += 24
    return np.concatenate([header, image], axis=0)


def render_camera_panel(
    info: Dict,
    cam_id: str,
    data_root: Path,
    boxes: np.ndarray,
    names: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
    class_names: Sequence[str],
    camera_width: int,
    undistort: bool,
    undistort_alpha: float,
    min_depth: float,
    max_edge_px: float,
    draw_fullres: bool,
    display_aspect: str,
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
            undistort, undistort_alpha, min_depth, max_edge_px,
            labels=labels, colors=colors)
        if draw_fullres:
            image = resize_to_width(image, camera_width)
    image = resize_to_display_aspect(image, display_aspect)
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


def blend_polygon_fill(
    image: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """在 BEV 图上绘制半透明填充，用于重合框时保留 GT 存在感。"""
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0, dst=image)


def draw_dashed_polyline(
    image: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    dash_len: float = 10.0,
    gap_len: float = 6.0,
) -> None:
    """绘制闭合虚线框。OpenCV 没有内置虚线多边形，这里按边分段绘制。"""
    if len(pts) < 2:
        return
    pts = np.asarray(pts, dtype=np.float32)
    for idx in range(len(pts)):
        start = pts[idx]
        end = pts[(idx + 1) % len(pts)]
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length <= 1e-3:
            continue
        direction = vec / length
        distance = 0.0
        while distance < length:
            seg_start = start + direction * distance
            seg_end = start + direction * min(distance + dash_len, length)
            cv2.line(
                image,
                tuple(np.round(seg_start).astype(int)),
                tuple(np.round(seg_end).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )
            distance += dash_len + gap_len


def draw_bev_box_by_source(
    image: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int],
    source: str,
) -> None:
    """按来源绘制 BEV box。

    GT 使用半透明填充 + 虚线轮廓，Pred 使用实线轮廓。这样 GT/Pred 完全
    重合时，Pred 的实线仍可能覆盖边线，但 GT 的淡色填充会保留下来。
    """
    source = str(source)
    if source == 'GT':
        blend_polygon_fill(image, pts, color, alpha=0.18)
        draw_dashed_polyline(image, pts, (245, 245, 245), thickness=3, dash_len=10, gap_len=6)
        draw_dashed_polyline(image, pts, color, thickness=2, dash_len=10, gap_len=6)
        return
    cv2.polylines(image, [pts], True, (245, 245, 245), 4, cv2.LINE_AA)
    cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)


def render_bev_panel(
    boxes: np.ndarray,
    names: Sequence[str],
    class_names: Sequence[str],
    bev_range: Tuple[float, float, float, float],
    size: int,
    heading_style: str,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[Tuple[int, int, int]]] = None,
    sources: Optional[Sequence[str]] = None,
) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    draw_bev_grid(image, bev_range)
    if labels is None:
        labels = [str(name) for name in names]
    if colors is None:
        colors = [class_color(str(name), class_names) for name in names]
    if sources is None:
        sources = [''] * len(names)
    for box_corners, box, name, label, color, source in zip(corners_from_boxes(boxes), boxes, names, labels, colors, sources):
        pts = bev_to_pixel(box_corners[:4, :2], bev_range, size).astype(np.int32)
        draw_bev_box_by_source(image, pts, color, source)
        draw_bev_box_heading(image, box_corners, box, bev_range, color, size, heading_style)
        if str(label):
            label_pt = bev_to_pixel(np.array([[box[0], box[1]]], dtype=np.float32), bev_range, size)[0].astype(int)
            cv2.putText(image, str(label), tuple(label_pt + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    x_min, y_min, x_max, y_max = bev_range
    summary = f'BEV x[{x_min:.1f},{x_max:.1f}] y[{y_min:.1f},{y_max:.1f}] equal-scale'
    cv2.rectangle(image, (0, 0), (size, 34), (0, 0, 0), -1)
    cv2.putText(image, summary, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def render_info(
    info: Dict,
    data_root: Path,
    class_names: Sequence[str],
    pred_result,
    pred_score_thr: float,
    max_preds: int,
    draw_gt: bool,
    draw_pred: bool,
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
    box_label_mode: str,
    display_aspect: str,
) -> np.ndarray:
    box_layers: List[np.ndarray] = []
    name_layers: List[str] = []
    label_layers: List[str] = []
    source_layers: List[str] = []
    color_layers: List[Tuple[int, int, int]] = []

    raw_gt_boxes = to_numpy(info.get('gt_boxes', np.zeros((0, 7), dtype=np.float32)))
    raw_gt_names = [str(x) for x in list(info.get('gt_names', []))]
    gt_total_count = min(len(raw_gt_boxes), len(raw_gt_names)) if raw_gt_boxes.size else 0
    gt_stat_boxes = raw_gt_boxes[:gt_total_count] if gt_total_count > 0 else np.zeros((0, 7), dtype=np.float32)
    gt_in_bev_count = count_boxes_in_bev_range(gt_stat_boxes, bev_range)
    gt_boxes = gt_stat_boxes
    gt_names = raw_gt_names[:gt_total_count]
    if draw_gt and gt_total_count > 0:
        box_layers.append(gt_boxes)
        name_layers.extend(gt_names)
        if box_label_mode == 'compact':
            label_layers.extend(['' for _ in gt_names])
        elif box_label_mode == 'none':
            label_layers.extend(['' for _ in gt_names])
        else:
            label_layers.extend([f'G:{name}' for name in gt_names])
        source_layers.extend(['GT' for _ in gt_names])
        color_layers.extend([class_color(name, class_names) for name in gt_names])

    pred_count = 0
    if draw_pred and pred_result is not None:
        pred_boxes, pred_names, pred_labels, pred_colors = prediction_result_to_draw_items(
            pred_result, class_names, pred_score_thr, max_preds)
        pred_count = len(pred_boxes)
        if pred_count > 0:
            box_layers.append(pred_boxes)
            name_layers.extend(pred_names)
            if box_label_mode == 'compact':
                # 紧凑模式下预测框只显示 score，类别和来源由顶部图例说明。
                label_layers.extend([label.rsplit(' ', 1)[-1] for label in pred_labels])
            elif box_label_mode == 'none':
                label_layers.extend(['' for _ in pred_labels])
            else:
                label_layers.extend(pred_labels)
            source_layers.extend(['Pred' for _ in pred_names])
            color_layers.extend(pred_colors)

    if box_layers:
        boxes = np.concatenate(box_layers, axis=0).astype(np.float32)
    else:
        boxes = np.zeros((0, 7), dtype=np.float32)

    legend_items: List[Tuple[str, Tuple[int, int, int]]] = []
    if box_label_mode == 'compact':
        seen = set()
        for source, name, color in zip(source_layers, name_layers, color_layers):
            key = (source, name)
            if key in seen:
                continue
            seen.add(key)
            legend_items.append((f'{source} {name}', color))

    panels = [
        render_camera_panel(
            info, cam_id, data_root, boxes, name_layers, label_layers, color_layers, class_names, camera_width,
            undistort, undistort_alpha, min_depth, max_edge_px, draw_fullres, display_aspect)
        for cam_id in choose_camera_order(info, camera_ids)
    ]
    mosaic = make_camera_mosaic(panels)
    timestamp = info.get('timestamp', 'unknown')
    ref_text = compact_info_ref(info) if box_label_mode == 'compact' else '/'.join(info_ref_parts(info))
    header_text = f'{ref_text} | ts={timestamp} | GT={gt_in_bev_count}/{gt_total_count}'
    if draw_pred:
        header_text += f' | Pred={pred_count} score>={pred_score_thr:.2f}'
    if no_bev:
        return add_top_header(mosaic, header_text, legend_items=legend_items)
    bev = render_bev_panel(
        boxes, name_layers, class_names, bev_range, bev_size, bev_heading_style,
        labels=label_layers, colors=color_layers, sources=source_layers)
    bev = cv2.resize(bev, (mosaic.shape[0], mosaic.shape[0]), interpolation=cv2.INTER_AREA)
    return add_top_header(np.concatenate([mosaic, bev], axis=1), header_text, legend_items=legend_items)


def selected_infos(infos: Sequence[Dict], start_index: int, stride: int, max_frames: Optional[int]) -> Iterable[Tuple[int, Dict]]:
    count = 0
    for idx in range(max(start_index, 0), len(infos), max(stride, 1)):
        if max_frames is not None and count >= max_frames:
            break
        yield idx, infos[idx]
        count += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='可视化并校验 Fast-BEV CustomMultiViewDataset 使用的 pkl 真值。',
        formatter_class=RawDefaultsHelpFormatter,
        epilog=USAGE_EXAMPLES)
    parser.add_argument('--gt-pkl', '--pkl', dest='gt_pkl', required=True,
                        help='converter 输出的 GT/info pkl 路径；--pkl 是兼容旧命令的别名')
    parser.add_argument('--pred-pkl', default=None, help='tools/test.py --out 保存的预测结果 pkl；按 index 和 --gt-pkl 中 infos 对齐')
    parser.add_argument('--data-root', required=True, help='converter --data-path 使用的数据根目录；相对图片路径会在该目录下解析')
    parser.add_argument('--output-dir', required=True, help='输出可视化图片帧和可选视频的目录')
    parser.add_argument('--camera-ids', nargs='+', default=None, help='指定要绘制的相机 id；默认使用六目检查顺序')
    parser.add_argument('--start-index', type=int, default=0, help='从第几个 info 开始可视化')
    parser.add_argument('--stride', type=int, default=1, help='每隔多少个 info 可视化一帧')
    parser.add_argument('--max-frames', type=int, default=100, help='最多渲染多少帧；设为 -1 表示全部渲染')
    parser.add_argument('--score-thr', type=float, default=0.2, help='绘制预测框的最小 score；只在 --pred-pkl 存在时生效')
    parser.add_argument('--max-preds', type=int, default=100, help='每帧最多绘制多少个预测框；<=0 表示不限制')
    parser.add_argument('--hide-gt', action='store_true', help='不绘制 GT 框，只看预测结果')
    parser.add_argument('--hide-pred', action='store_true', help='读取 --pred-pkl 时不绘制预测框，仅保留 GT；用于快速对比')
    parser.add_argument('--box-label-mode', default='auto', choices=['auto', 'compact', 'full', 'none'],
                        help='框文字显示模式；auto 会在存在预测 pkl 时默认 compact，GT-only 时默认 full')
    parser.add_argument('--camera-width', type=int, default=640, help='拼接前每个相机小图的宽度')
    parser.add_argument('--display-aspect', default='native', choices=['native', '16:9'],
                        help='最终显示层的相机小图比例；native 保持真实训练/缓存图比例，16:9 只拉伸输出画面')
    parser.add_argument('--draw-fullres', action='store_true', help='在原始分辨率上画框/去畸变后再缩放；速度慢，仅用于对比旧逻辑')
    parser.add_argument('--no-render', action='store_true', help='只执行 pkl 读取和参数解析，不生成可视化图片或视频')
    parser.add_argument('--bev-size', type=int, default=700, help='BEV 面板在缩放到拼图高度前的尺寸')
    parser.add_argument('--bev-range-mode', default='fastbev', choices=['fastbev', 'auto'], help='BEV 显示范围来源；fastbev 使用 x/y 都为 ±50m，auto 根据 pkl 中所有 GT box 自动推断')
    parser.add_argument('--bev-range', nargs=4, type=float, metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'), default=None, help='固定 BEV 显示范围；填写后优先级高于 --bev-range-mode')
    parser.add_argument('--bev-heading-style', default='front-edge', choices=['front-edge', 'arrow', 'none'], help='BEV 目标朝向显示方式；front-edge 用前边加粗，arrow 使用中心箭头，none 不画朝向')
    parser.add_argument('--no-bev', action='store_true', help='只保存相机拼图，不拼接 BEV 面板')
    parser.add_argument('--undistort', dest='undistort', action='store_true', default=True, help='画框前先对图片去畸变，并使用 OpenCV 返回的新内参投影；默认开启')
    parser.add_argument('--raw-distorted', dest='undistort', action='store_false', default=argparse.SUPPRESS,
                        help='不去畸变，在原始畸变图上使用畸变投影；用于检查板端原始输入显示效果')
    parser.add_argument('--undistort-alpha', type=float, default=0.0, help='OpenCV 去畸变 alpha；0 裁掉无效区域，1 保留完整视野')
    parser.add_argument('--max-corner-edge-px', type=float, default=0.0, help='原始畸变图中角点连线最大像素长度；0 表示按图像尺寸自动设置。默认去畸变时不使用该限制')
    parser.add_argument('--min-depth', type=float, default=0.1, help='不绘制深度小于该阈值的投影边')
    parser.add_argument('--fps', type=int, default=10, help='输出视频帧率')
    parser.add_argument('--no-video', action='store_true', help='不输出 visualization.mp4 视频')
    parser.add_argument('--video-only', action='store_true', help='只输出 visualization.mp4，不保存 frames 图片；不能和 --no-video 同时使用')
    parser.add_argument('--workers', type=int, default=1, help='并行渲染帧数；视频模式下会按顺序写视频、并行预渲染，建议从 2 或 4 开始')
    parser.add_argument('--image-ext', default='jpg', choices=['jpg', 'png'], help='逐帧可视化图片格式')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    if args.video_only and args.no_video:
        parser.error('--video-only 和 --no-video 不能同时使用')
    return args


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    pkl_path = Path(args.gt_pkl)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infos, metadata = load_fastbev_pkl(pkl_path)
    class_names = get_class_names(metadata, infos)
    pred_results = None
    if args.pred_pkl:
        pred_path = Path(args.pred_pkl)
        pred_results = load_prediction_results(pred_path)
        logger.info('Loaded %d prediction results from %s', len(pred_results), pred_path)
        if len(pred_results) != len(infos):
            logger.warning(
                'Prediction result count %d differs from info count %d; visualization still aligns by index and missing predictions are skipped.',
                len(pred_results), len(infos))
    max_frames = None if args.max_frames is not None and args.max_frames < 0 else args.max_frames
    if args.bev_range:
        bev_range = make_metric_bev_range(tuple(args.bev_range))
    elif args.bev_range_mode == 'fastbev':
        bev_range = make_metric_bev_range((-50.0, -50.0, 50.0, 50.0))
    else:
        bev_range = infer_bev_range(infos)

    draw_pred = pred_results is not None and not args.hide_pred
    if args.box_label_mode == 'auto':
        box_label_mode = 'compact' if draw_pred else 'full'
    else:
        box_label_mode = args.box_label_mode

    logger.info('Loaded %d infos from %s', len(infos), pkl_path)
    logger.info('metadata.coordinate=%s', metadata.get('coordinate', 'unknown'))
    logger.info('data_root=%s', data_root)
    logger.info('class_names=%s', class_names)
    logger.info('bev_range=%s', bev_range)
    logger.info('draw_gt=%s draw_pred=%s score_thr=%.3f max_preds=%d',
                not args.hide_gt, draw_pred,
                args.score_thr, args.max_preds)
    logger.info('box_label_mode=%s', box_label_mode)
    logger.info('camera_width=%d display_aspect=%s', args.camera_width, args.display_aspect)
    logger.info('undistort=%s undistort_alpha=%.3f', args.undistort, args.undistort_alpha)
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
        pred_result = None
        if pred_results is not None and pkl_index < len(pred_results):
            pred_result = pred_results[pkl_index]
        canvas = render_info(
            info=info,
            data_root=data_root,
            class_names=class_names,
            pred_result=pred_result,
            pred_score_thr=args.score_thr,
            max_preds=args.max_preds,
            draw_gt=not args.hide_gt,
            draw_pred=pred_results is not None and not args.hide_pred,
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
            box_label_mode=box_label_mode,
            display_aspect=args.display_aspect,
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

    def write_video_frame(ref_output_dir: Path, canvas: np.ndarray) -> None:
        """按输出目录顺序写视频帧；VideoWriter 只在主线程使用。"""
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
    elif (not args.no_video) and workers > 1:
        logger.info('Parallel rendering with ordered video writing enabled: workers=%d', workers)
        # 视频必须按帧顺序写入，但渲染可以并行预取。窗口限制为 workers*2，
        # 避免长序列时把大量 canvas 堆在内存里。
        pending = deque()
        iterator_iter = iter(iterator)
        window_size = max(workers * 2, 1)

        def submit_next(executor) -> bool:
            try:
                next_item = next(iterator_iter)
            except StopIteration:
                return False
            pending.append(executor.submit(render_and_save_frame, next_item, True))
            return True

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _ in range(window_size):
                if not submit_next(executor):
                    break
            with tqdm(total=len(iterator), desc='visualizing') as progress:
                while pending:
                    future = pending.popleft()
                    frame_dir, ref_output_dir, canvas = future.result()
                    if frame_dir is not None:
                        saved_frame_dirs.add(frame_dir)
                    write_video_frame(ref_output_dir, canvas)
                    rendered += 1
                    progress.update(1)
                    submit_next(executor)
    else:
        for item in tqdm(iterator, total=len(iterator), desc='visualizing'):
            frame_dir, ref_output_dir, canvas = render_and_save_frame(item, need_canvas=not args.no_video)
            if frame_dir is not None:
                saved_frame_dirs.add(frame_dir)

            if not args.no_video:
                write_video_frame(ref_output_dir, canvas)
            rendered += 1

    for writer, _ in writers.values():
        writer.release()
    logger.info('Rendered %d frames under %s', rendered, output_dir)
    for frame_dir in sorted(saved_frame_dirs):
        logger.info('Frame directory: %s', frame_dir)


if __name__ == '__main__':
    main()
