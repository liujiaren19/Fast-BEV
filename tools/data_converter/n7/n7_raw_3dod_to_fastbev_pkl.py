#!/usr/bin/env python3
"""将 N7 自采 3D_OD 数据转换为 Fast-BEV 可训练的 CustomMultiViewDataset pkl。

本脚本只处理 3D 障碍物检测 OD 数据。旧版脚本里为 AVM/车位数据保留的
逻辑已经移除，输出目标是 ``mmdet3d.datasets.CustomMultiViewDataset`` 可以
直接读取的轻量 pkl。

原始数据目录约定
----------------
脚本兼容几种历史 N7 导出目录。最常见的两种结构如下；converter 会自动寻找
匹配的 ``3D_OD/lidar`` 标签目录和 ``frames`` 图像目录。

    <data-root>/<dataset>/<sequence>/output/<clip>/3D_OD/lidar/*.json
    <data-root>/<dataset>/<sequence>/parsed_data/<clip>/frames/<timestamp>/images/<cam_id>/*.jpg

    <data-root>/<dataset>/output/<sequence>/<clip>/3D_OD/lidar/*.json
    <data-root>/<dataset>/parsed_data/<sequence>/<clip>/frames/<timestamp>/images/<cam_id>/*.jpg

如果需要固定 train/val/test 划分，可以放置 manifest 文件，文件名为
``<dataset>_<set>_clips.txt``，支持以下位置：

    <data-root>/<dataset>_<set>_clips.txt
    <data-root>/manifests/<dataset>_<set>_clips.txt
    <data-root>/<dataset>/<dataset>_<set>_clips.txt

manifest 中每个非空、非注释行都必须是 ``dataset/sequence/clip``。如果没有
找到 manifest，脚本会在 ``--data-path`` 下递归发现 clip。

坐标系约定
----------
N7 3D_OD 标签按如下原始 lidar 坐标系理解：

    custom_lidar: x 向左，y 向后，z 向上，原点在 N7 顶部主 lidar。

输出 pkl 中的 3D box、速度、相机外参和帧位姿会被直接转换到 Fast-BEV/MMDet3D
常用 lidar 坐标系：

    mmdet3d_lidar: x 向前，y 向左，z 向上，原点仍在 N7 顶部主 lidar。

直接轴变换关系是：

    x_fastbev = -y_raw
    y_fastbev =  x_raw
    z_fastbev =  z_raw
    yaw_fastbev = normalize(yaw_raw + pi / 2)

这里还没有迁移到“后轴中心在地面投影点”的 ego 坐标系。后轴 ego 迁移需要
统一更新 label、相机外参、BEV range、anchor、伪标签转换和可视化，因此应作为
单独的全几何改动处理。当前输出 metadata 中会明确写入
``rear_axle_ground_ego_applied=False``。

输出 pkl 内容
-------------
pkl 结构是 ``{"infos": infos, "metadata": metadata}``。脚本会同时在同目录写入
``*.summary.json``，只包含 metadata 和统计信息，便于不打开大 pkl 时快速核对。
每个 info 包含：

    - 六目图像路径
    - 相机内参和 camera-to-lidar 外参
    - 3D boxes、类别、速度、track id
    - 当前帧 lidar pose
    - ``prev`` / ``next`` 相邻帧引用

当 ``slamResult/baidu_ins/odom_lidar_reference.txt`` 或 label JSON 中存在
``lidar_pose`` 时，converter 会写入 clip-local 的 ``lidar2global`` 位姿。这里的
“global”不是地图全局坐标，而是该 clip 的参考坐标系，通常以第一帧为起点。
``CustomMultiViewDataset`` 会使用这些位姿把 adjacent camera frame 运动补偿到
key frame lidar 坐标系，从而尽量复刻原 Fast-BEV nuScenes 时序输入语义。

默认情况下，label JSON 和 ``frames/<timestamp>`` 目录使用 lidar 时间戳精确
匹配；只有显式设置 ``--frame-match-mode nearest`` 时才会启用最近邻匹配。

常用命令
--------

    # 704x256 缓存图：data_path 指向离线 resize 后的图片，info_json 仍使用
    # 生成这些图片之前对应的标定尺寸内参，例如当前 N7 高快数据的 1600x900 K。
    python tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py \
        --data-path data/N7_704_256 \
        --datasets 20251031 \
        --sets train val \
        --info-json data/info_json/2025_04_18_2k_byd_info.json \
        --output-dir data/N7_704_256/pkl \
        --extra-tag custom_fastbev \
        --image-size 256 704

    # 原始 1600x900 图：可以不传 --image-size，让脚本读取图片 header；
    # 批量数据为了少做 IO，也可以显式传 --image-size 900 1600。
    python tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py \
        --data-path data/nuscenes \
        --datasets 20251031 \
        --sets train val \
        --info-json data/info_json/2025_04_18_2k_byd_info.json \
        --output-dir data/nuscenes/pkl/od_2k \
        --extra-tag custom_fastbev \
        --image-size 900 1600

    # 转换后建议先做几何校验，再抽帧可视化。
    python tools/data_converter/n7/visualize_n7_fastbev_pkl.py \
        --pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl \
        --data-root data/N7_704_256 \
        --output-dir work_dirs/vis_n7_704_check \
        --no-render \
        --check-geometry \
        --check-temporal-geometry \
        --strict-geometry
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import os
import os.path as osp
import pickle
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def natural_sort_key(value) -> List:
    """按自然顺序排序字符串，避免 clip_10 排在 clip_2 前面。"""

    key = []
    for part in re.split(r'(\d+)', str(value)):
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def sorted_dir_paths(path: Path) -> List[Path]:
    """用 os.scandir 快速列出一级子目录，并按自然顺序排序。"""

    try:
        with os.scandir(path) as entries:
            return sorted(
                (Path(entry.path) for entry in entries if entry.is_dir()),
                key=lambda p: natural_sort_key(p.name),
            )
    except FileNotFoundError:
        return []


def sorted_json_files(path: Path) -> List[Path]:
    """用 os.scandir 快速列出 JSON 文件，并按时间戳/自然顺序排序。"""

    try:
        with os.scandir(path) as entries:
            files = [
                Path(entry.path)
                for entry in entries
                if entry.is_file() and entry.name.lower().endswith('.json')
            ]
    except FileNotFoundError:
        return []
    return sorted(files, key=lambda p: natural_sort_key(p.name))


def has_json_files(path: Path) -> bool:
    """快速判断目录下是否至少有一个 JSON 文件。"""

    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.json'):
                    return True
    except FileNotFoundError:
        return False
    return False


def build_frame_index(frames_dir: Path) -> Dict[int, Path]:
    """把 frames/<timestamp> 目录一次性建成索引，避免逐 label 反复 exists。"""

    frame_index: Dict[int, Path] = {}
    for item in sorted_dir_paths(frames_dir):
        try:
            frame_index[int(item.name)] = item
        except ValueError:
            continue
    return frame_index


class RawDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """保留 epilog 示例换行，同时继续展示 argparse 默认值。"""


CAMERA_ID_TO_SENSOR = {
    'cam0': 'front_wide',
    'cam11': 'right_front',
    'cam9': 'left_front',
    'cam3': 'back',
    'cam8': 'left_back',
    'cam10': 'right_back',
}
CAMERA_ORDER = ['cam0', 'cam11', 'cam9', 'cam3', 'cam8', 'cam10']
# Fast-BEV 原 4D nuScenes pkl 中 [1, 3, 5] 历史帧下标约对应
# 0.5s / 1.0s / 1.5s 的时间差。N7 自采帧率可能更高，默认按时间差
# 选择历史帧，比直接取稠密相邻帧更接近原始时序训练语义。
DEFAULT_ADJACENT_TIME_OFFSETS_US = [500000, 1000000, 1500000]
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}

# 当前业务主线先训练道路车辆两类：乘用车/小车和货车。
# N7 原始标签中 no_motor_bike/person 仍在 CLASS_MAPPING 中保留到
# nuScenes 风格类别名，后续如果要做四类实验，可以通过 --classes
# car truck bicycle pedestrian 显式打开。
FASTBEV_CLASSES = ['car', 'truck']

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

# 原始 N7/自采标签坐标系：x 向左，y 向后，z 向上。
# Fast-BEV/MMDet3D LiDAR 坐标系：x 向前，y 向左，z 向上。
#
# 直接轴向映射的矩阵形式：
#   [x_fastbev]   [ 0 -1  0] [x_raw]
#   [y_fastbev] = [ 1  0  0] [y_raw]
#   [z_fastbev]   [ 0  0  1] [z_raw]
#
# 同一个矩阵会用于 box 中心点、速度和相机外参平移。相机旋转矩阵会左乘
# 该矩阵，使 camera-to-lidar 变换始终表达在输出 pkl 使用的 Fast-BEV
# lidar 坐标系下。
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
    objects_filtered_class: int = 0
    objects_unknown_class: int = 0
    raw_class_counts: Dict[str, int] = field(default_factory=dict)
    mapped_class_counts: Dict[str, int] = field(default_factory=dict)
    kept_class_counts: Dict[str, int] = field(default_factory=dict)
    clips_with_pose_file: int = 0
    labels_with_pose: int = 0
    labels_without_pose: int = 0

    def as_dict(self) -> Dict:
        data = dict(self.__dict__)
        # dict 字段复制一份，避免调用方误改 stats 内部状态。
        for key in ('raw_class_counts', 'mapped_class_counts', 'kept_class_counts'):
            data[key] = dict(data.get(key, {}))
        return data

    def merge(self, other: 'ConversionStats') -> None:
        for key, value in other.as_dict().items():
            current = getattr(self, key)
            if isinstance(current, dict):
                for sub_key, sub_value in value.items():
                    current[sub_key] = current.get(sub_key, 0) + sub_value
            else:
                setattr(self, key, current + value)

    @staticmethod
    def count(mapping: Dict[str, int], key: str) -> None:
        mapping[key] = mapping.get(key, 0) + 1


class TimestampMatcher:
    def __init__(self, frame_root: Path):
        # 最近邻模式也复用同一个 frame 索引，避免逐目录 Path.iterdir 的额外开销。
        frame_index = build_frame_index(frame_root)
        self.items: List[Tuple[int, Path]] = sorted(frame_index.items(), key=lambda x: x[0])
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


def pose_vector_to_fastbev(pose: Sequence[float]) -> Dict:
    """将原始 N7 lidar pose 转换到 Fast-BEV lidar 坐标系。

    odom 文件和 label JSON 中都使用七个数保存 pose：
    ``[tx, ty, tz, qx, qy, qz, qw]``。该 pose 表示当前原始 lidar
    坐标系到 clip 局部参考坐标系的变换：

        p_ref_raw = R_raw @ p_lidar_raw + t_raw

    训练用 box 和相机外参已经通过 ``RAW_TO_FASTBEV`` 转到 Fast-BEV
    lidar 轴向，pose 也必须做同样转换，否则相邻帧运动补偿会混用两个
    不一致的坐标系。
    """
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size != 7:
        raise ValueError(f'expected pose [tx,ty,tz,qx,qy,qz,qw], got shape {arr.shape}')
    tran_raw = arr[:3].astype(np.float32)
    rot_raw = quat_xyzw_to_matrix(arr[3:7])

    rot_fastbev = RAW_TO_FASTBEV @ rot_raw @ RAW_TO_FASTBEV.T
    tran_fastbev = RAW_TO_FASTBEV @ tran_raw
    return {
        'raw_lidar2global_translation': tran_raw.astype(np.float32).tolist(),
        'raw_lidar2global_rotation_xyzw': arr[3:7].astype(np.float32).tolist(),
        'lidar2global_translation': tran_fastbev.astype(np.float32).tolist(),
        'lidar2global_rotation': rotation_to_wxyz(rot_fastbev),
        # 当前 ego 暂时定义为 N7 顶部主 lidar 坐标系。后续如果迁移到后轴中心
        # 地面投影点，需要同步更新 lidar 和 ego 两组字段；现在保持二者别名关系，
        # 可以让 dataset 侧逻辑更接近原始 nuScenes 时序实现。
        'ego2global_translation': tran_fastbev.astype(np.float32).tolist(),
        'ego2global_rotation': rotation_to_wxyz(rot_fastbev),
    }


class OdomPoseIndex:
    """clip 内 SLAM lidar pose 的时间戳索引。

    ``odom_lidar_reference.txt`` 是单个 clip 内的局部轨迹，时间戳已经和
    lidar 帧做过同步。文件首行通常接近单位位姿，因此输出 pkl 里的
    ``global`` 应理解为“clip 局部参考坐标系”，不是地理全局地图坐标。
    Fast-BEV 时序补偿只需要相邻帧和 key frame 之间的相对变换，这个
    局部参考坐标系已经足够。
    """

    def __init__(self, path: Optional[Path]):
        self.path = path
        self.items: List[Tuple[int, Dict]] = []
        if path is None or not path.exists():
            self.timestamps: List[int] = []
            return

        with path.open('r') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    logger.debug('Skip malformed odom line %s:%d: %s', path, line_no, line)
                    continue
                try:
                    timestamp = int(float(parts[0]))
                    pose = [float(x) for x in parts[1:8]]
                    pose_info = pose_vector_to_fastbev(pose)
                except Exception as exc:
                    logger.debug('Skip odom line %s:%d: %s', path, line_no, exc)
                    continue
                pose_info.update({
                    'pose_source': 'odom_lidar_reference',
                    'pose_timestamp': timestamp,
                    'pose_time_diff_us': 0,
                    'pose_file': str(path),
                })
                self.items.append((timestamp, pose_info))

        self.items.sort(key=lambda x: x[0])
        self.timestamps = [x[0] for x in self.items]

    def closest(self, timestamp: int, max_diff_us: int) -> Optional[Dict]:
        if not self.items:
            return None
        pos = np.searchsorted(self.timestamps, timestamp)
        candidates = []
        for idx in (pos - 1, pos):
            if 0 <= idx < len(self.items):
                ts, pose_info = self.items[idx]
                diff = abs(ts - timestamp)
                if diff <= max_diff_us:
                    candidates.append((diff, ts, pose_info))
        if not candidates:
            return None
        diff, _, pose_info = min(candidates, key=lambda x: x[0])
        pose_info = deepcopy(pose_info)
        pose_info['pose_time_diff_us'] = int(diff)
        return pose_info


def label_pose_to_fastbev(raw_pose: Optional[Sequence[float]], timestamp: int) -> Optional[Dict]:
    """将单个 label JSON 中的 ``3d_od.lidar_pose`` 转成 pkl pose 字段。"""
    if raw_pose is None:
        return None
    try:
        pose_info = pose_vector_to_fastbev(raw_pose)
    except Exception as exc:
        logger.debug('Invalid label lidar_pose at %s: %s', timestamp, exc)
        return None
    pose_info.update({
        'pose_source': 'label_lidar_pose',
        'pose_timestamp': int(timestamp),
        'pose_time_diff_us': 0,
        'pose_file': '',
    })
    return pose_info


def select_frame_pose(
    timestamp: int,
    raw_label_pose: Optional[Sequence[float]],
    pose_index: Optional[OdomPoseIndex],
    max_pose_match_us: int,
) -> Optional[Dict]:
    """为时序补偿选择当前帧 pose。

    优先使用 SLAM odom，因为它是 clip 局部轨迹，且时间戳已经和 lidar
    帧同步；当 odom 文件缺失、为空或该帧没有落在阈值内的 pose 时，回退到
    label JSON 自带的 ``lidar_pose``，保证只有标签 pose 时也能生成可用于
    时序对比的 pkl。
    """
    if pose_index is not None:
        pose_info = pose_index.closest(timestamp, max_pose_match_us)
        if pose_info is not None:
            return pose_info
    return label_pose_to_fastbev(raw_label_pose, timestamp)


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
    """保留 N7 lidar_main 混合式 to_ego 约定，方便后续追溯标定来源。"""
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


def build_camera_template(sensor: Dict) -> Dict:
    """预计算单个相机的静态标定模板。

    N7 当前使用一份固定标定 json，同一个 clip 乃至同一次转换中的相机
    内参、畸变参数和 camera-to-lidar 外参都不会随帧变化。这里把这些固定
    计算提前做一次，后续每帧只需要补入当前图片路径即可。

    标定 json 中每个相机的 ``to_lidar_main`` 变换表达在原始 N7/自采 lidar
    坐标系下。训练 pkl 必须自洽，因此写入 ``sensor2lidar_rotation`` 和
    ``sensor2lidar_translation`` 前，会把旋转和平移都转换到 Fast-BEV
    lidar 坐标系。
    """
    raw_ext = sensor.get('extrinsic', {}).get('to_lidar_main')
    if raw_ext is None:
        raise ValueError(f'sensor {sensor.get("name")} lacks to_lidar_main extrinsic')
    sensor_to_raw_rot, sensor_to_raw_tran = parse_sensor_to_lidar(raw_ext)
    sensor_to_fastbev_rot = RAW_TO_FASTBEV @ sensor_to_raw_rot
    sensor_to_fastbev_tran = RAW_TO_FASTBEV @ sensor_to_raw_tran

    intrinsic = np.asarray(sensor.get('intrinsic', {}).get('K', []), dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f'sensor {sensor.get("name")} has invalid K shape {intrinsic.shape}')

    distortion = np.asarray(sensor.get('intrinsic', {}).get('D', []), dtype=np.float32).reshape(-1)
    intrinsic_width = int(sensor.get('width', 0) or 0)
    intrinsic_height = int(sensor.get('height', 0) or 0)
    return {
        'sensor_name': sensor.get('name', ''),
        'cam_intrinsic': intrinsic,
        'sensor2lidar_rotation': sensor_to_fastbev_rot.astype(np.float32),
        'sensor2lidar_translation': sensor_to_fastbev_tran.astype(np.float32),
        'sensor2ego_rotation': rotation_to_wxyz(sensor_to_fastbev_rot),
        'sensor2ego_translation': sensor_to_fastbev_tran.astype(np.float32).tolist(),
        'distortion': distortion,
        # intrinsic_width/height 明确表示 cam_intrinsic 和畸变参数对应的图像坐标系。
        # 旧字段 width/height 保留为兼容别名，含义同样是内参对应尺寸，不是训练图片实际尺寸。
        'intrinsic_width': intrinsic_width,
        'intrinsic_height': intrinsic_height,
        'width': intrinsic_width,
        'height': intrinsic_height,
    }


def build_camera_templates(calib: Dict, camera_ids: Sequence[str]) -> Dict[str, Dict]:
    """按 camera_id 预计算当前转换任务需要的相机模板。"""
    templates: Dict[str, Dict] = {}
    for cam_id in camera_ids:
        sensor_name = CAMERA_ID_TO_SENSOR[cam_id]
        sensor = find_sensor(calib, sensor_name)
        templates[cam_id] = build_camera_template(sensor)
    return templates


def camera_intrinsic_size_summary(calib: Dict, camera_ids: Sequence[str]) -> Dict[str, Dict]:
    """记录每个相机内参对应的图像尺寸，方便 pkl 交接和训练配置核对。"""
    summary: Dict[str, Dict] = {}
    for cam_id in camera_ids:
        sensor = find_sensor(calib, CAMERA_ID_TO_SENSOR[cam_id])
        summary[cam_id] = {
            'sensor_name': sensor.get('name', ''),
            'height': int(sensor.get('height', 0) or 0),
            'width': int(sensor.get('width', 0) or 0),
        }
    return summary


def relative_image_path(image_path: Path, data_root: Path) -> str:
    """把图片路径保存成相对 data_root 的形式，方便 pkl 跨机器迁移。

    常见路径都来自 ``data_root / ...``，优先使用 ``Path.relative_to`` 可以避免
    ``os.path.relpath`` 每次做绝对路径归一化；只有遇到混合绝对/相对路径或
    非同根路径时才回退到兼容逻辑。
    """
    try:
        return image_path.relative_to(data_root).as_posix()
    except ValueError:
        try:
            return osp.relpath(str(image_path), str(data_root))
        except ValueError:
            return str(image_path)


def read_image_size(image_path: Path) -> Tuple[int, int]:
    """读取实际图片尺寸，返回 (height, width)。

    只有在用户没有通过 --image-size 指定缓存图片尺寸时才会调用。PIL 读取
    header 即可获得尺寸，不需要把整张图片解码成数组。
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return int(height), int(width)


def build_camera_info(
    camera_template: Dict,
    image_path: Path,
    data_root: Path,
    image_size_override: Optional[Tuple[int, int]] = None,
) -> Dict:
    """把当前帧图片路径和尺寸填入静态相机模板，生成 pkl 中的相机条目。

    intrinsic_width/height 表示 cam_intrinsic 对应的图像坐标系尺寸；
    image_width/height 表示 data_path 当前指向图片文件的实际尺寸。两者必须
    分开保存，否则 1600x900 标定 + 704x256 缓存图、多车型 2560x1440 标定
    等场景容易在 pipeline 中重复缩放或漏缩放。
    """
    if image_size_override is None:
        image_height, image_width = read_image_size(image_path)
    else:
        image_height, image_width = [int(x) for x in image_size_override]
    return {
        'data_path': relative_image_path(image_path, data_root),
        'sensor_name': camera_template['sensor_name'],
        'cam_intrinsic': camera_template['cam_intrinsic'].copy(),
        'sensor2lidar_rotation': camera_template['sensor2lidar_rotation'].copy(),
        'sensor2lidar_translation': camera_template['sensor2lidar_translation'].copy(),
        'sensor2ego_rotation': list(camera_template['sensor2ego_rotation']),
        'sensor2ego_translation': list(camera_template['sensor2ego_translation']),
        'distortion': camera_template['distortion'].copy(),
        'intrinsic_width': camera_template['intrinsic_width'],
        'intrinsic_height': camera_template['intrinsic_height'],
        'image_width': image_width,
        'image_height': image_height,
        # width/height 作为兼容字段保留，语义是内参对应尺寸。
        'width': camera_template['intrinsic_width'],
        'height': camera_template['intrinsic_height'],
    }


def first_image_file(cam_dir: Path) -> Optional[Path]:
    """在单个相机目录中快速找到一张有效图片。

    N7 每个 ``frames/<lidar_ts>/images/<cam_id>`` 目录通常只有一张同步后的
    图片。旧实现会对 8 种大小写后缀逐个 glob，572 帧六目会触发两万多次
    目录扫描。这里改为一次 ``os.scandir`` 扫完整个目录，并按文件名选择
    第一张非空图片，兼容相机图片时间戳和 lidar 时间戳不完全一致的情况。
    """
    best_name: Optional[str] = None
    try:
        with os.scandir(cam_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                suffix = osp.splitext(entry.name)[1].lower()
                if suffix not in IMAGE_SUFFIXES:
                    continue
                try:
                    if entry.stat().st_size <= 0:
                        continue
                except OSError:
                    continue
                if best_name is None or entry.name < best_name:
                    best_name = entry.name
    except FileNotFoundError:
        return None

    if best_name is None:
        return None
    return cam_dir / best_name


def collect_frame_images(frame_dir: Path, camera_ids: Sequence[str]) -> Optional[Dict[str, Path]]:
    images_root = frame_dir / 'images'
    images: Dict[str, Path] = {}
    for cam_id in camera_ids:
        image_path = first_image_file(images_root / cam_id)
        if image_path is None:
            return None
        images[cam_id] = image_path
    return images


def annotation_payload(label: Dict) -> Dict:
    return label.get('3d_od', label)


def annotation_to_box(anno: Dict) -> Tuple[List[float], List[float]]:
    """将单个原始 3D_OD 标注转换为 Fast-BEV lidar box 格式。

    输出 box 格式为目标 lidar 坐标系下的 ``[x, y, z, l, w, h, yaw]``。
    速度单独返回为 ``[vx, vy]``，后续由 dataset 拼接。yaw 增加 ``+pi/2``
    是轴向映射在角度上的对应关系：原始前向是 ``-Y``，Fast-BEV 前向是
    ``+X``。
    """
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
        stats.count(stats.raw_class_counts, raw_name)
        mapped_name = CLASS_MAPPING.get(raw_name)
        if mapped_name is None:
            stats.objects_unknown_class += 1
            continue
        stats.count(stats.mapped_class_counts, mapped_name)
        if mapped_name not in classes:
            # 已知类别但不在当前训练类别集合中，例如两类车辆实验中过滤
            # no_motor_bike/person。单独统计，避免误认为标签类别未知。
            stats.objects_filtered_class += 1
            continue
        box, velocity = annotation_to_box(anno)
        gt_boxes.append(box)
        gt_names.append(mapped_name)
        stats.count(stats.kept_class_counts, mapped_name)
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
        # 旧版标签中 ``lidar_pose`` 可能不存在。存在时它使用
        # [tx, ty, tz, qx, qy, qz, qw]，表示当前 lidar 坐标系到标注流程使用的
        # clip 局部参考坐标系的变换。
        'raw_lidar_pose': payload.get('lidar_pose'),
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


def resolve_odom_path(data_root: Path, ref: ClipRef, frames_dir: Path) -> Optional[Path]:
    """查找可选的 clip 局部 SLAM lidar pose 文件。

    N7 采集流程通常会把 odom 放在 ``frames`` 同级目录下的
    ``slamResult/baidu_ins/odom_lidar_reference.txt``，其中时间戳已经和
    lidar 帧同步。历史导出目录可能略有差异，所以这里按固定候选路径依次
    查找；找不到 odom 不视为错误，因为 label JSON 中可能仍包含
    ``lidar_pose``。
    """
    base = data_root / ref.dataset
    if not base.exists():
        base = data_root

    candidates = [
        frames_dir.parent / 'slamResult' / 'baidu_ins' / 'odom_lidar_reference.txt',
        base / ref.sequence / 'parsed_data' / ref.clip / 'slamResult' / 'baidu_ins' / 'odom_lidar_reference.txt',
        base / 'parsed_data' / ref.sequence / ref.clip / 'slamResult' / 'baidu_ins' / 'odom_lidar_reference.txt',
        base / 'parsed_data' / ref.clip / 'slamResult' / 'baidu_ins' / 'odom_lidar_reference.txt',
    ]
    for path in candidates:
        if path.exists():
            return path
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

    def add_ref(sequence: str, clip: str, label_dir: Path) -> None:
        if has_json_files(label_dir):
            refs.add(ClipRef(dataset=dataset_name, sequence=sequence, clip=clip))

    # 标准 N7 结构：dataset/sequence/output/clip/3D_OD/lidar。
    for sequence_dir in sorted_dir_paths(base):
        output_root = sequence_dir / 'output'
        for clip_dir in sorted_dir_paths(output_root):
            add_ref(sequence_dir.name, clip_dir.name, clip_dir / '3D_OD' / 'lidar')

    # 历史结构：dataset/output/sequence/clip/3D_OD/lidar。
    output_root = base / 'output'
    for sequence_dir in sorted_dir_paths(output_root):
        for clip_dir in sorted_dir_paths(sequence_dir):
            add_ref(sequence_dir.name, clip_dir.name, clip_dir / '3D_OD' / 'lidar')

    # 历史结构：dataset/output/clip/3D_OD/lidar。
    for clip_dir in sorted_dir_paths(output_root):
        add_ref('', clip_dir.name, clip_dir / '3D_OD' / 'lidar')

    if not refs:
        # 非标准导出路径才退回 os.walk。正常 22W N7 数据不会走这里。
        for dirpath, _, filenames in os.walk(base):
            label_dir = Path(dirpath)
            if label_dir.name != 'lidar' or label_dir.parent.name != '3D_OD':
                continue
            if not any(name.lower().endswith('.json') for name in filenames):
                continue
            try:
                rel_parts = label_dir.relative_to(base).parts
            except ValueError:
                rel_parts = label_dir.parts
            if 'output' not in rel_parts:
                continue
            output_index = rel_parts.index('output')
            sequence = ''
            clip = label_dir.parents[1].name
            if output_index > 0 and output_index + 1 < len(rel_parts):
                sequence = rel_parts[output_index - 1]
                clip = rel_parts[output_index + 1]
            elif output_index + 2 < len(rel_parts) and rel_parts[output_index + 2] != '3D_OD':
                sequence = rel_parts[output_index + 1]
                clip = rel_parts[output_index + 2]
            elif output_index + 1 < len(rel_parts):
                clip = rel_parts[output_index + 1]
            refs.add(ClipRef(dataset=dataset_name, sequence=sequence, clip=clip))

    return sorted(
        refs,
        key=lambda x: (natural_sort_key(x.dataset), natural_sort_key(x.sequence), natural_sort_key(x.clip)),
    )


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


def adjacent_view(
    info: Dict,
    target_time_offset_us: Optional[int] = None,
    actual_time_offset_us: Optional[int] = None,
) -> Dict:
    """返回构建单个相邻帧视图所需的字段子集。

    这里会带上 pose 字段，因为时序 Fast-BEV 需要把相邻帧相机从它自己的
    lidar 坐标系变换到当前 key frame 的 lidar 坐标系。GT boxes 不复制到
    相邻帧视图中，因为检测监督目标始终只属于 key frame。

    ``target_time_offset_us`` 和 ``actual_time_offset_us`` 只用于调试和追溯：
    前者表示本来希望选择的时间差，后者表示实际匹配到的相邻帧时间差。
    """
    view = {
        'token': info['token'],
        'timestamp': info['timestamp'],
        'cams': info['cams'],
    }
    if target_time_offset_us is not None:
        view['target_time_offset_us'] = int(target_time_offset_us)
    if actual_time_offset_us is not None:
        view['actual_time_offset_us'] = int(actual_time_offset_us)
    for key in (
        'lidar2global_rotation', 'lidar2global_translation',
        'ego2global_rotation', 'ego2global_translation',
        'raw_lidar2global_rotation_xyzw', 'raw_lidar2global_translation',
        'pose_source', 'pose_timestamp', 'pose_time_diff_us', 'pose_file'):
        if key in info:
            view[key] = info[key]
    return view


def closest_temporal_info(
    clip_infos: Sequence[Dict],
    timestamps: Sequence[int],
    target_timestamp: int,
    start: int,
    end: int,
) -> Optional[Dict]:
    """从同一 clip 的候选帧中用二分查找选择最接近目标时间的帧。

    ``clip_infos`` 已经按时间戳升序排列，``timestamps`` 是对应的整型时间戳
    列表。``start``/``end`` 用来限定只能从当前 key frame 之前或之后选帧，
    因此 time 模式不需要每次复制候选列表，也不需要线性扫描整个历史窗口。
    """
    if start >= end:
        return None

    target_timestamp = int(target_timestamp)
    pos = bisect.bisect_left(timestamps, target_timestamp, start, end)
    best_idx: Optional[int] = None
    for candidate_idx in (pos - 1, pos):
        if start <= candidate_idx < end:
            if best_idx is None:
                best_idx = candidate_idx
                continue
            candidate_diff = abs(timestamps[candidate_idx] - target_timestamp)
            best_diff = abs(timestamps[best_idx] - target_timestamp)
            if candidate_diff < best_diff:
                best_idx = candidate_idx

    if best_idx is None:
        return None
    return clip_infos[best_idx]


def link_adjacent_infos(
    infos: List[Dict],
    max_adjacent: int,
    adjacent_mode: str,
    adjacent_time_offsets_us: Sequence[int],
) -> None:
    """在每个 clip 内挂载 previous/next 相邻帧引用。

    默认 ``time`` 模式按目标时间差选择历史帧，例如 0.5s / 1.0s / 1.5s。
    这种方式比高帧率数据下直接取稠密上一帧更有意义，也更接近原始
    Fast-BEV 4D nuScenes pkl 的时序间隔。

    ``dense`` 模式保留旧逻辑：prev[0] 是上一帧，prev[1] 是上上帧。它主要
    用于兼容旧实验或排查问题。无论哪种模式，相邻帧只复制相机条目和
    帧级 pose 字段，GT boxes 仍然保留在 key frame 上。
    """
    by_clip: Dict[str, List[Dict]] = {}
    for info in infos:
        by_clip.setdefault(info['clip_id'], []).append(info)

    for clip_infos in by_clip.values():
        clip_infos.sort(key=lambda x: x['timestamp'])
        timestamps = [int(x['timestamp']) for x in clip_infos]
        clip_len = len(clip_infos)
        for idx, info in enumerate(clip_infos):
            if adjacent_mode == 'dense':
                prev_infos = clip_infos[max(0, idx - max_adjacent):idx][::-1]
                next_infos = clip_infos[idx + 1:idx + 1 + max_adjacent]
                info['prev'] = [adjacent_view(x) for x in prev_infos]
                info['next'] = [adjacent_view(x) for x in next_infos]
                continue

            prev_views: List[Dict] = []
            next_views: List[Dict] = []
            key_ts = int(info['timestamp'])
            for offset_us in adjacent_time_offsets_us:
                prev_info = closest_temporal_info(
                    clip_infos, timestamps, key_ts - int(offset_us), start=0, end=idx)
                if prev_info is not None:
                    prev_views.append(adjacent_view(
                        prev_info,
                        target_time_offset_us=offset_us,
                        actual_time_offset_us=key_ts - int(prev_info['timestamp'])))

                next_info = closest_temporal_info(
                    clip_infos, timestamps, key_ts + int(offset_us), start=idx + 1, end=clip_len)
                if next_info is not None:
                    next_views.append(adjacent_view(
                        next_info,
                        target_time_offset_us=offset_us,
                        actual_time_offset_us=int(next_info['timestamp']) - key_ts))
            info['prev'] = prev_views
            info['next'] = next_views


def normalize_adjacent_time_offsets(offsets: Sequence[int]) -> List[int]:
    """清洗用户输入的相邻帧目标时间差，单位是微秒。"""
    values = sorted({int(x) for x in offsets if int(x) > 0})
    if not values:
        raise ValueError('adjacent time offsets must contain at least one positive value')
    return values


def build_info_for_label(
    label_path: Path,
    frame_ts: int,
    frame_dir: Path,
    ref: ClipRef,
    camera_templates: Dict[str, Dict],
    data_root: Path,
    camera_ids: Sequence[str],
    classes: Sequence[str],
    keep_empty: bool,
    image_size_override: Optional[Tuple[int, int]],
    pose_index: Optional[OdomPoseIndex],
    max_pose_match_us: int,
    stats: ConversionStats,
) -> Optional[Dict]:
    """根据单个 label 和匹配到的图像帧构建一个 Fast-BEV info。

    关键原则是坐标自洽：GT boxes、相机 ``sensor2lidar`` 外参和帧级 pose
    全部表达在 Fast-BEV lidar 坐标系下（x 前、y 左、z 上）。这样后续
    时序补偿就只需要做刚体变换组合。
    """
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
        cams[cam_id] = build_camera_info(
            camera_templates[cam_id], image_map[cam_id], data_root, image_size_override)

    label_ts = ann['timestamp']
    pose_info = select_frame_pose(
        timestamp=label_ts,
        raw_label_pose=ann.get('raw_lidar_pose'),
        pose_index=pose_index,
        max_pose_match_us=max_pose_match_us)
    if pose_info is None:
        stats.labels_without_pose += 1
        pose_info = {}
    else:
        stats.labels_with_pose += 1

    # N7 clip 名通常已经包含日期和采集序列，例如 20251031_164821_1。
    # token 使用 clip + lidar 时间戳，clip_id 直接使用 clip 做时序分组边界。
    # 如果以后接入 clip 名不全局唯一的数据源，再把 clip_id 扩展回完整路径。
    token = f'{ref.clip}_{label_ts}'
    clip_uid = ref.clip
    info = {
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
    info.update(pose_info)
    return info


def process_clip(
    data_root: Path,
    ref: ClipRef,
    camera_templates: Dict[str, Dict],
    camera_ids: Sequence[str],
    classes: Sequence[str],
    frame_match_mode: str,
    max_match_us: int,
    max_pose_match_us: int,
    max_adjacent: int,
    adjacent_mode: str,
    adjacent_time_offsets_us: Sequence[int],
    keep_empty: bool,
    image_size_override: Optional[Tuple[int, int]],
) -> Tuple[List[Dict], ConversionStats]:
    stats = ConversionStats(clips_total=1)
    paths = resolve_clip_paths(data_root, ref)
    if paths is None:
        stats.clips_missing_paths += 1
        return [], stats
    label_dir, frames_dir = paths
    # N7 parsed_data/frames 使用 lidar 时间戳作为目录名，且该时间戳已经和
    # label JSON 对齐。因此默认直接精确查找 frames/<label_ts>，避免误把标签
    # 关联到相邻 lidar 帧。最近邻匹配仅作为旧数据导出的兼容选项保留。
    matcher = TimestampMatcher(frames_dir) if frame_match_mode == 'nearest' else None
    frame_index = build_frame_index(frames_dir) if frame_match_mode == 'exact' else None

    odom_path = resolve_odom_path(data_root, ref, frames_dir)
    pose_index = OdomPoseIndex(odom_path) if odom_path is not None else None
    if pose_index is not None and pose_index.items:
        stats.clips_with_pose_file += 1
        logger.debug('Loaded %d odom poses from %s', len(pose_index.items), odom_path)

    infos: List[Dict] = []
    for label_path in sorted_json_files(label_dir):
        stats.labels_total += 1
        try:
            label_ts = timestamp_from_path(label_path)
        except ValueError:
            payload = annotation_payload(load_json(label_path))
            label_ts = int(payload.get('frame_timestamp', 0))
        if frame_match_mode == 'exact':
            frame_dir = frame_index.get(label_ts) if frame_index is not None else None
            if frame_dir is None:
                stats.labels_without_frame += 1
                continue
            frame_ts = label_ts
        else:
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
                camera_templates=camera_templates,
                data_root=data_root,
                camera_ids=camera_ids,
                classes=classes,
                keep_empty=keep_empty,
                image_size_override=image_size_override,
                pose_index=pose_index,
                max_pose_match_us=max_pose_match_us,
                stats=stats,
            )
        except Exception as exc:
            logger.debug('Skip %s: %s', label_path, exc)
            continue
        if info is not None:
            infos.append(info)

    infos.sort(key=lambda x: x['timestamp'])
    link_adjacent_infos(
        infos,
        max_adjacent=max_adjacent,
        adjacent_mode=adjacent_mode,
        adjacent_time_offsets_us=adjacent_time_offsets_us)
    return infos, stats


def make_metadata(
    classes: Sequence[str],
    camera_ids: Sequence[str],
    calib: Dict,
    stats: ConversionStats,
    set_name: str,
    datasets: Sequence[str],
    frame_match_mode: str,
    max_match_us: int,
    max_pose_match_us: int,
    max_adjacent: int,
    adjacent_mode: str,
    adjacent_time_offsets_us: Sequence[int],
    output_scope: str,
    name_date: str,
    image_size_override: Optional[Tuple[int, int]],
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
        'pose_coordinate': 'clip_reference_from_lidar in mmdet3d_lidar axes; clip reference is usually first odom row',
        'pose_sources': ['odom_lidar_reference', 'label_lidar_pose'],
        'temporal_compensation': 'CustomMultiViewDataset composes adjacent lidar2global with key lidar2global',
        'raw_to_fastbev': RAW_TO_FASTBEV.tolist(),
        'rear_axle_ground_ego_applied': False,
        'frame_match_mode': frame_match_mode,
        'max_match_us': max_match_us,
        'max_pose_match_us': max_pose_match_us,
        'max_adjacent': max_adjacent,
        'adjacent_mode': adjacent_mode,
        'adjacent_time_offsets_us': list(adjacent_time_offsets_us),
        'output_scope': output_scope,
        'output_name_date': name_date,
        'image_size_override': list(image_size_override) if image_size_override is not None else None,
        'camera_intrinsic_sizes': camera_intrinsic_size_summary(calib, camera_ids),
        'camera_image_size': (
            {'height': int(image_size_override[0]), 'width': int(image_size_override[1])}
            if image_size_override is not None else None
        ),
        'camera_intrinsic_size_source': 'info_json sensor width/height; 推荐优先使用原始标定尺寸 info_json，当前脚本会把该尺寸写入每个 camera 条目',
        'camera_image_size_source': '--image-size override or actual image file header',
        'stats': stats.as_dict(),
        'lidar_main_mixed_to_ego': lidar_main_metadata(calib),
    }


def sanitize_filename_part(value: str) -> str:
    """把数据集/序列/clip 名清洗成适合放入 pkl 文件名的片段。"""
    value = str(value).strip()
    cleaned = ''.join(ch if (ch.isalnum() or ch in ('-', '_', '.')) else '_' for ch in value)
    return cleaned.strip('_') or 'unknown'


def output_scope_from_refs(refs: Sequence[ClipRef], dataset_names: Sequence[str]) -> str:
    """根据本次转换覆盖的数据范围生成文件名中的 scope。

    规则按业务语义从大到小判断：多个 dataset 时使用 dataset 列表；单 dataset
    内如果只有一个 clip，则使用 clip；多个 clip 但同一个 sequence，则使用
    sequence；多个 sequence 但同一个 dataset，则使用 dataset。
    """
    ref_list = list(refs)
    datasets = sorted({x.dataset for x in ref_list if x.dataset}, key=natural_sort_key)
    if not datasets:
        datasets = sorted({str(x) for x in dataset_names if str(x)}, key=natural_sort_key)
    clips = sorted({x.clip for x in ref_list if x.clip}, key=natural_sort_key)
    sequences = sorted({x.sequence for x in ref_list if x.sequence}, key=natural_sort_key)

    if len(datasets) > 1:
        scope = '-'.join(datasets)
    elif len(clips) == 1:
        scope = clips[0]
    elif len(sequences) == 1:
        scope = sequences[0]
    elif len(datasets) == 1:
        scope = datasets[0]
    else:
        scope = 'multi_dataset'
    return sanitize_filename_part(scope)


def output_pkl_path(out_dir: Path, extra_tag: str, scope: str, set_name: str, name_date: str) -> Path:
    """生成语义化 pkl 文件名：{tag}_{scope}_infos_{set}_{YYYYMMDD}.pkl。"""
    tag = sanitize_filename_part(extra_tag)
    set_part = sanitize_filename_part(set_name)
    date_part = sanitize_filename_part(name_date)
    return out_dir / f'{tag}_{scope}_infos_{set_part}_{date_part}.pkl'


def save_pkl(infos: List[Dict], metadata: Dict, out_path: Path, dry_run: bool) -> None:
    summary_path = out_path.with_suffix('.summary.json')
    summary = {
        'pkl_path': str(out_path),
        'infos': len(infos),
        'metadata': metadata,
    }
    if dry_run:
        logger.info('Dry-run: would write %d infos to %s', len(infos), out_path)
        logger.info('Dry-run: would write summary to %s', summary_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        pickle.dump({'infos': infos, 'metadata': metadata}, f)
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info('Wrote %d infos to %s', len(infos), out_path)
    logger.info('Wrote summary to %s', summary_path)


def convert_one_set(
    data_root: Path,
    dataset_names: Sequence[str],
    set_name: str,
    calib: Dict,
    camera_templates: Dict[str, Dict],
    out_dir: Path,
    extra_tag: str,
    camera_ids: Sequence[str],
    classes: Sequence[str],
    frame_match_mode: str,
    max_match_us: int,
    max_pose_match_us: int,
    max_adjacent: int,
    adjacent_mode: str,
    adjacent_time_offsets_us: Sequence[int],
    name_date: str,
    keep_empty: bool,
    image_size_override: Optional[Tuple[int, int]],
    separate: bool,
    dry_run: bool,
) -> None:
    per_dataset: Dict[str, List[Dict]] = {}
    per_dataset_stats: Dict[str, ConversionStats] = {}
    per_dataset_refs: Dict[str, List[ClipRef]] = {}

    for dataset_name in dataset_names:
        refs = load_refs_for_set(data_root, dataset_name, set_name)
        if not refs:
            logger.warning('No clips found for dataset=%s set=%s', dataset_name, set_name)
            continue
        per_dataset_refs[dataset_name] = refs
        infos: List[Dict] = []
        stats = ConversionStats()
        for ref in tqdm(refs, desc=f'{dataset_name}/{set_name}'):
            clip_infos, clip_stats = process_clip(
                data_root=data_root,
                ref=ref,
                camera_templates=camera_templates,
                camera_ids=camera_ids,
                classes=classes,
                frame_match_mode=frame_match_mode,
                max_match_us=max_match_us,
                max_pose_match_us=max_pose_match_us,
                max_adjacent=max_adjacent,
                adjacent_mode=adjacent_mode,
                adjacent_time_offsets_us=adjacent_time_offsets_us,
                keep_empty=keep_empty,
                image_size_override=image_size_override,
            )
            infos.extend(clip_infos)
            stats.merge(clip_stats)
        infos.sort(key=lambda x: x['timestamp'])
        per_dataset[dataset_name] = infos
        per_dataset_stats[dataset_name] = stats
        logger.info('dataset=%s set=%s infos=%d stats=%s', dataset_name, set_name, len(infos), stats.as_dict())

        if separate:
            output_scope = output_scope_from_refs(refs, [dataset_name])
            metadata = make_metadata(
                classes, camera_ids, calib, stats, set_name, [dataset_name],
                frame_match_mode, max_match_us, max_pose_match_us, max_adjacent,
                adjacent_mode, adjacent_time_offsets_us, output_scope, name_date, image_size_override)
            save_pkl(infos, metadata, output_pkl_path(out_dir, extra_tag, output_scope, set_name, name_date), dry_run)

    if separate:
        return

    merged_infos: List[Dict] = []
    merged_refs: List[ClipRef] = []
    merged_stats = ConversionStats()
    for dataset_name in dataset_names:
        merged_infos.extend(deepcopy(per_dataset.get(dataset_name, [])))
        merged_refs.extend(per_dataset_refs.get(dataset_name, []))
        if dataset_name in per_dataset_stats:
            merged_stats.merge(per_dataset_stats[dataset_name])
    merged_infos.sort(key=lambda x: x['timestamp'])
    output_scope = output_scope_from_refs(merged_refs, dataset_names)
    metadata = make_metadata(
        classes, camera_ids, calib, merged_stats, set_name, dataset_names,
        frame_match_mode, max_match_us, max_pose_match_us, max_adjacent,
        adjacent_mode, adjacent_time_offsets_us, output_scope, name_date, image_size_override)
    if not merged_infos:
        logger.warning('set=%s produced no infos; skipping output', set_name)
        return
    save_pkl(merged_infos, metadata, output_pkl_path(out_dir, extra_tag, output_scope, set_name, name_date), dry_run)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='将 N7 3D_OD 自采数据转换为 Fast-BEV CustomMultiViewDataset 可用的 pkl 文件',
        formatter_class=RawDefaultsHelpFormatter,
        epilog=(
            '示例：\n'
            '  # 704x256 缓存图：K 仍来自 info_json 标定尺寸，图片实际尺寸通过 --image-size 写入 pkl\n'
            '  python tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py '
            '--data-path data/N7_704_256 '
            '--datasets 20251031 '
            '--sets train val '
            '--info-json data/info_json/2025_04_18_2k_byd_info.json '
            '--output-dir data/N7_704_256/pkl '
            '--extra-tag custom_fastbev '
            '--image-size 256 704\n\n'
            '  # 原始 1600x900 图：可省略 --image-size，或为了减少 header IO 显式填写\n'
            '  python tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py '
            '--data-path data/nuscenes '
            '--datasets 20251031 '
            '--sets train val '
            '--info-json data/info_json/2025_04_18_2k_byd_info.json '
            '--output-dir data/nuscenes/pkl/od_2k '
            '--extra-tag custom_fastbev '
            '--image-size 900 1600\n\n'
            '随后使用可视化脚本检查 pkl：\n'
            '  python tools/data_converter/n7/visualize_n7_fastbev_pkl.py '
            '--pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl '
            '--data-root data/N7_704_256 '
            '--output-dir work_dirs/vis_n7_704_check '
            '--no-render --check-geometry --check-temporal-geometry --strict-geometry'
        ))
    parser.add_argument('--data-type', default='od', choices=['od'], help='仅支持 3D 障碍物 od 数据。')
    parser.add_argument('--data-path', required=True, help='N7 数据根目录，通常为 ./data/nuscenes。')
    parser.add_argument('--datasets', nargs='+', required=True, help='数据集名称或 manifest 前缀。')
    parser.add_argument('--sets', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--info-json', required=True, help='N7 传感器标定 json。')
    parser.add_argument('--output-dir', default='./data/nuscenes')
    parser.add_argument('--extra-tag', default='custom_fastbev', help='输出 pkl 前缀，最终文件名为 {tag}_{scope}_infos_{set}_{YYYYMMDD}.pkl。')
    parser.add_argument('--template-pkl', default=None, help='已废弃且会被忽略，仅为兼容旧命令保留。')
    parser.add_argument('--frame-match-mode', choices=['exact', 'nearest'], default='exact', help='label JSON 与 parsed_data/frames 的匹配方式：默认按 lidar 时间戳精确匹配，也可选择最近邻。')
    parser.add_argument('--max-match-us', type=int, default=50000, help='label/frame 最大时间戳差，单位微秒；仅在 --frame-match-mode nearest 时生效。')
    parser.add_argument('--max-pose-match-us', type=int, default=50000, help='label/SLAM pose 最大时间戳差，单位微秒。')
    parser.add_argument('--max-adj', '--max-adjacent', dest='max_adjacent', type=int, default=60, help='dense 模式下每帧最多保留的前后相邻帧数量；time 模式主要由 --adjacent-time-offsets-us 控制。')
    parser.add_argument('--adjacent-mode', choices=['time', 'dense'], default='time', help='时序相邻帧构建方式：time 按目标时间差选帧，dense 按稠密相邻帧下标选帧。')
    parser.add_argument('--adjacent-time-offsets-us', nargs='+', type=int, default=DEFAULT_ADJACENT_TIME_OFFSETS_US, help='time 模式下为每个 key frame 选择历史/未来帧的目标时间差，单位微秒。默认约为 0.5s/1.0s/1.5s。')
    parser.add_argument('--interval', type=int, default=3, help='兼容旧命令参数；当前相邻帧构建由 --adjacent-mode 控制。')
    parser.add_argument('--camera-ids', nargs='+', default=CAMERA_ORDER, choices=CAMERA_ORDER)
    parser.add_argument('--classes', nargs='+', default=FASTBEV_CLASSES)
    parser.add_argument('--keep-empty', action='store_true', help='保留没有有效 3D box 的帧。')
    parser.add_argument('--image-size', nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'), default=None, help='当前 data_path 指向图片的实际尺寸，例如 704 缓存图传 --image-size 256 704。若不传则逐张读取图片 header。')
    parser.add_argument('--separate', '-s', action='store_true', help='每个 dataset/set 单独写一个 pkl，而不是按 set 合并多个数据集。')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--multiprocessing', action='store_true', help='兼容旧命令参数，当前会被忽略。')
    parser.add_argument('--debug', action='store_true', help='兼容旧命令参数，当前不会额外导出 raw dump。')
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
    camera_templates = build_camera_templates(calib, args.camera_ids)
    adjacent_time_offsets_us = normalize_adjacent_time_offsets(args.adjacent_time_offsets_us)
    image_size_override = tuple(args.image_size) if args.image_size is not None else None
    name_date = datetime.now().strftime('%Y%m%d')
    logger.info('data_root=%s', data_root)
    logger.info('datasets=%s sets=%s', args.datasets, args.sets)
    logger.info('camera_ids=%s', args.camera_ids)
    logger.info('coordinate=mmdet3d_lidar:x_front_y_left_z_up origin=N7_top_lidar')
    logger.info('frame matching mode=%s max_match_us=%s', args.frame_match_mode, args.max_match_us)
    logger.info('pose matching max_pose_match_us=%s', args.max_pose_match_us)
    logger.info('adjacent mode=%s time_offsets_us=%s max_adjacent=%s', args.adjacent_mode, adjacent_time_offsets_us, args.max_adjacent)
    logger.info('output name date=%s', name_date)
    logger.info('image_size_override=%s', image_size_override)
    if image_size_override is None:
        logger.warning('未设置 --image-size，converter 会逐帧读取图片 header；22W 帧全量转换建议显式传入图片尺寸。')

    for set_name in args.sets:
        convert_one_set(
            data_root=data_root,
            dataset_names=args.datasets,
            set_name=set_name,
            calib=calib,
            camera_templates=camera_templates,
            out_dir=out_dir,
            extra_tag=args.extra_tag,
            camera_ids=args.camera_ids,
            classes=args.classes,
            frame_match_mode=args.frame_match_mode,
            max_match_us=args.max_match_us,
            max_pose_match_us=args.max_pose_match_us,
            max_adjacent=args.max_adjacent,
            adjacent_mode=args.adjacent_mode,
            adjacent_time_offsets_us=adjacent_time_offsets_us,
            name_date=name_date,
            keep_empty=args.keep_empty,
            image_size_override=image_size_override,
            separate=args.separate,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
