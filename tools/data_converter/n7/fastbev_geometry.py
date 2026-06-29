#!/usr/bin/env python3
"""N7 Fast-BEV pkl 的几何计算和一致性检查工具。

这个模块放置和训练数据几何一致性相关的纯 numpy 逻辑，供
``validate_n7_fastbev_pkl.py`` 和可视化脚本复用。拆出来的目的不是改变行为，
而是把“数据契约校验”和“可视化渲染”解耦：前者用于 converter 后、训练前确认
pkl 字段能被 Fast-BEV dataset 正确解释，后者只负责人工查看投影效果。
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger(__name__)

# converter 默认六目相机的人眼检查顺序：
#   第一行：左前、前视广角、右前
#   第二行：左后、后视、右后
DEFAULT_MOSAIC_ORDER = ['cam9', 'cam0', 'cam11', 'cam8', 'cam3', 'cam10']


def to_numpy(value, dtype=np.float32) -> np.ndarray:
    """把普通列表/ndarray 转成指定 dtype 的 numpy 数组。"""
    return np.asarray(value, dtype=dtype)


def camera_dimension_fields(cam_info: Dict, image_shape: Optional[Tuple[int, int]] = None) -> Dict[str, int]:
    """整理相机内参源尺寸和当前图片尺寸。

    ``intrinsic_width/height`` 表示 ``cam_intrinsic`` 所在的图像坐标尺寸；
    ``image_width/height`` 表示 pkl 中 ``data_path`` 实际图片尺寸。704x256 离线
    缓存图会出现二者不一致的情况，可视化时必须先把 K 从源尺寸缩到实际图片
    尺寸，再投影到图上。
    """
    required_size_keys = (
        'intrinsic_width', 'intrinsic_height', 'image_width', 'image_height')
    missing_size_keys = [key for key in required_size_keys if key not in cam_info]
    if missing_size_keys:
        raise KeyError(
            'N7 camera info missing required size fields: {}'.format(
                ', '.join(missing_size_keys)))

    intrinsic_width = int(cam_info['intrinsic_width'])
    intrinsic_height = int(cam_info['intrinsic_height'])
    image_width = int(cam_info['image_width'])
    image_height = int(cam_info['image_height'])
    if min(intrinsic_width, intrinsic_height, image_width, image_height) <= 0:
        raise ValueError(
            'N7 camera size fields must be positive: intrinsic={}x{}, image={}x{}'.format(
                intrinsic_width, intrinsic_height, image_width, image_height))

    if image_shape is not None:
        loaded_height, loaded_width = [int(v) for v in image_shape[:2]]
        if image_width != loaded_width or image_height != loaded_height:
            raise ValueError(
                'pkl image size {}x{} differs from loaded image size {}x{}'.format(
                    image_width, image_height, loaded_width, loaded_height))

    return dict(
        intrinsic_width=intrinsic_width,
        intrinsic_height=intrinsic_height,
        image_width=image_width,
        image_height=image_height,
    )


def choose_camera_order(info: Dict, requested: Optional[Sequence[str]]) -> List[str]:
    """按默认六目顺序返回当前 info 中实际存在的相机。"""
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


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    """把 pkl 中 nuScenes 风格的 ``[w, x, y, z]`` 四元数转成旋转矩阵。"""
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
    的方式做相邻帧运动补偿。新版 N7 pkl 只读取 ``lidar2global_*`` 字段。
    """
    rot = info.get('lidar2global_rotation')
    tran = info.get('lidar2global_translation')
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
    """返回两个矩阵或向量的最大绝对误差。"""
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
    """检查单个相机在 key frame 或相邻帧中的几何一致性。"""
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
