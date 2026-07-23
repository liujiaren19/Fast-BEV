#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 N7 Fast-BEV pkl + config 离线生成 2D->3D LUT 调试数据。

这个脚本只复现测试流程中的确定性几何，不加载模型权重，也不跑 backbone。
用途是给板端构建固定 LUT 索引表，或用确定性 debug feature 验证板端
gather/scatter 和 PC 端 ``backproject_inplace`` 是否一致。

典型用法：

    python tools/data_converter/n7/build_fastbev_lut.py \
        configs/fastbev/custom/custom_fastbev_6v_r18_n7_704x256.py \
        --pkl data/N7_704_256/pkl/custom_fastbev_xxx_infos_test_20260624.pkl \
        --sample-index 0 \
        --seq-id 0 \
        --out-dir output/n7_lut_debug/seq0 \
        --dump-debug-volume

输出目录中会包含旧调试脚本兼容的：
``x.npy``、``y.npy``、``valid.npy``、``projection.npy``、``features.npy``、
``volume.npy``，以及更适合板端构建索引表的
``gather_index_dense.npy``、``fused_gather_index.npy``、
``fused_scatter_index.npy``、``fused_camera_index.npy``。同时会直接输出
已验证板端工具使用的 ``LUT/gather_new_i.bin``、
``LUT/scatter_nd_new_i.bin`` 和 ``LUT/featurePointLength.bin``。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fastbev_geometry import (
    dataset_lidar2img_from_cam_info,
    sensor2reference_lidar_for_dataset,
)


def deep_merge(base: Any, override: Any) -> Any:
    """按 mmcv config 的常用规则递归合并 dict。"""
    if isinstance(base, dict) and isinstance(override, dict):
        if override.get('_delete_', False):
            return {k: copy.deepcopy(v) for k, v in override.items() if k != '_delete_'}
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key == '_delete_':
                continue
            merged[key] = deep_merge(merged.get(key), value)
        return merged
    return copy.deepcopy(override)


def load_py_config(config_path: Path) -> Dict[str, Any]:
    """轻量读取 OpenMMLab Python config，避免脚本依赖 mmcv。"""
    config_path = config_path.resolve()
    namespace: Dict[str, Any] = {'__file__': str(config_path)}
    code = compile(config_path.read_text(encoding='utf-8'), str(config_path), 'exec')
    exec(code, namespace)

    base_cfg: Dict[str, Any] = {}
    base_item = namespace.get('_base_')
    if base_item:
        base_paths = base_item if isinstance(base_item, (list, tuple)) else [base_item]
        for rel_path in base_paths:
            base_cfg = deep_merge(base_cfg, load_py_config((config_path.parent / rel_path).resolve()))

    current = {
        key: value
        for key, value in namespace.items()
        if not key.startswith('__') and key != '_base_'
    }
    return deep_merge(base_cfg, current)


def to_jsonable(value: Any) -> Any:
    """把 numpy/Path/tuple 等对象转成 metadata.json 可写的普通类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def get_nested(mapping: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_infos(pkl_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with pkl_path.open('rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        infos = data.get('infos', data.get('data_list'))
        metadata = data.get('metadata', {})
    else:
        infos = data
        metadata = {}
    if infos is None:
        raise KeyError(f'{pkl_path} 中找不到 infos/data_list')
    infos = list(sorted(infos, key=lambda item: item.get('timestamp', 0)))
    return infos, metadata


def find_pipeline_step(pipeline: Sequence[Dict[str, Any]], step_type: str) -> Optional[Dict[str, Any]]:
    for step in pipeline:
        if isinstance(step, dict) and step.get('type') == step_type:
            return step
    return None


def normalize_first_item(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return value[0]
    return value


def parse_index_list(text: Optional[str], expected_len: int) -> List[int]:
    if text is None or text == '':
        return list(range(expected_len))
    values = [int(item) for item in text.split(',') if item.strip() != '']
    if sorted(values) != list(range(expected_len)):
        raise ValueError(
            '--camera-overwrite-order 必须是 0..{} 的一个排列，当前为 {}'.format(
                expected_len - 1, values))
    return values


def get_points(n_voxels: Sequence[int], voxel_size: Sequence[float], origin: Sequence[float]) -> np.ndarray:
    """复刻 FastBEV.get_points，返回 [3, X, Y, Z] 的 BEV 体素中心。"""
    n_voxels_arr = np.asarray(n_voxels, dtype=np.int64)
    voxel_size_arr = np.asarray(voxel_size, dtype=np.float32)
    origin_arr = np.asarray(origin, dtype=np.float32)
    grid = np.stack(np.meshgrid(
        np.arange(n_voxels_arr[0], dtype=np.float32),
        np.arange(n_voxels_arr[1], dtype=np.float32),
        np.arange(n_voxels_arr[2], dtype=np.float32),
        indexing='ij'))
    new_origin = origin_arr - n_voxels_arr.astype(np.float32) / 2.0 * voxel_size_arr
    return grid * voxel_size_arr.reshape(3, 1, 1, 1) + new_origin.reshape(3, 1, 1, 1)


def test_image_aug_params(cam_aug: Dict[str, Any], data_config: Dict[str, Any], force_resize: bool) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """复现 RandomAugImageMultiViewImage 在 is_train=False 下的 post_rot/tran。"""
    if force_resize:
        target_h, target_w = [int(v) for v in data_config['test_input_size']]
        source_h = int(cam_aug['intrinsic_height'])
        source_w = int(cam_aug['intrinsic_width'])
        sx = float(target_w) / float(source_w)
        sy = float(target_h) / float(source_h)
        post_rot = np.diag([sx, sy]).astype(np.float32)
        post_tran = np.zeros(2, dtype=np.float32)
        resize = 1.0
        crop = (0, 0, target_w, target_h)
        flip = False
        rotate = 0.0
    else:
        h = int(cam_aug['image_height'])
        w = int(cam_aug['image_width'])
        target_h, target_w = [int(v) for v in data_config['test_input_size']]
        resize = float(target_w) / float(w) + float(data_config.get('test_resize', 0.0))
        resize_dims = (int(w * resize), int(h * resize))
        new_w, new_h = resize_dims
        crop_h_start = (new_h - target_h) // 2
        crop_w_start = (new_w - target_w) // 2
        crop = (crop_w_start, crop_h_start, crop_w_start + target_w, crop_h_start + target_h)
        flip = bool(data_config.get('test_flip', False))
        rotate = float(data_config.get('test_rotate', 0.0))
        post_rot = np.eye(2, dtype=np.float32)
        post_tran = np.zeros(2, dtype=np.float32)

    post_rot = post_rot * float(resize)
    post_tran = post_tran - np.asarray(crop[:2], dtype=np.float32)
    if flip:
        mat = np.asarray([[-1, 0], [0, 1]], dtype=np.float32)
        bias = np.asarray([crop[2] - crop[0], 0], dtype=np.float32)
        post_rot = mat @ post_rot
        post_tran = mat @ post_tran + bias

    theta = rotate / 180.0 * math.pi
    rot_mat = np.asarray([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)],
    ], dtype=np.float32)
    center = np.asarray([crop[2] - crop[0], crop[3] - crop[1]], dtype=np.float32) / 2.0
    bias = rot_mat @ (-center) + center
    post_rot = rot_mat @ post_rot
    post_tran = rot_mat @ post_tran + bias

    pad_data = data_config.get('pad', (0, 0, 0, 0))
    top, right, bottom, left = [int(v) for v in pad_data]
    post_tran[0] += left
    post_tran[1] += top

    ret_post_rot = np.eye(3, dtype=np.float32)
    ret_post_tran = np.zeros(3, dtype=np.float32)
    ret_post_rot[:2, :2] = post_rot
    ret_post_tran[:2] = post_tran
    image_shape = (target_h + top + bottom, target_w + left + right, 3)
    return ret_post_rot, ret_post_tran, image_shape


def rts2proj(cam_aug: Dict[str, Any], post_rot: np.ndarray, post_tran: np.ndarray) -> np.ndarray:
    """复刻 RandomAugImageMultiViewImage.rts2proj。"""
    rot = np.asarray(cam_aug['rot'], dtype=np.float32)
    tran = np.asarray(cam_aug['tran'], dtype=np.float32).reshape(3)
    intrinsic = np.asarray(cam_aug['intrin'], dtype=np.float32)

    lidar2cam_r = np.linalg.inv(rot)
    lidar2cam_t = tran @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = post_rot @ intrinsic
    viewpad[:3, 2] += post_tran
    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


def projection_for_stride(lidar2img_rt: np.ndarray, stride: int) -> np.ndarray:
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[:2] /= float(stride)
    return (intrinsic @ lidar2img_rt[:3]).astype(np.float32)


def flatten_distortion(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    coeffs = np.asarray(value, dtype=np.float32).reshape(-1)
    if coeffs.size == 0:
        return None
    return coeffs


def camera_distortion(img_meta: Dict[str, Any], cam_id: int) -> Optional[np.ndarray]:
    lidar2img = img_meta.get('lidar2img', {})
    for key in ('lidar2img_extra', 'lidar2img_aug'):
        values = lidar2img.get(key, [])
        if isinstance(values, list) and cam_id < len(values) and isinstance(values[cam_id], dict):
            coeffs = flatten_distortion(values[cam_id].get('distortion'))
            if coeffs is not None:
                return coeffs
    return None


def project_points_with_distortion(points_homo: np.ndarray, projection: np.ndarray, img_meta: Dict[str, Any], stride: int) -> np.ndarray:
    """复刻 fastbev.py 的 batched distortion 投影；缺字段时退回 pinhole。"""
    aug_infos = img_meta.get('lidar2img', {}).get('lidar2img_aug', [])
    n_images = points_homo.shape[0]
    if not isinstance(aug_infos, list) or len(aug_infos) < n_images:
        return projection @ points_homo

    outputs = []
    for cam_id in range(n_images):
        aug = aug_infos[cam_id]
        distortion = camera_distortion(img_meta, cam_id)
        if not isinstance(aug, dict) or distortion is None:
            outputs.append(projection[cam_id] @ points_homo[cam_id])
            continue

        sensor2lidar_r = np.asarray(aug['rot'], dtype=np.float32).reshape(3, 3)
        sensor2lidar_t = np.asarray(aug['tran'], dtype=np.float32).reshape(3, 1)
        intrinsic = np.asarray(aug['intrin'], dtype=np.float32).reshape(3, 3)
        post_rot = np.asarray(aug.get('post_rot', np.eye(3)), dtype=np.float32).reshape(3, 3)
        post_tran = np.asarray(aug.get('post_tran', np.zeros(3)), dtype=np.float32).reshape(3, 1)

        padded = np.zeros(8, dtype=np.float32)
        if distortion.size >= 8:
            padded[:] = distortion[:8]
        else:
            padded[:min(distortion.size, 5)] = distortion[:min(distortion.size, 5)]
        k1, k2, p1, p2, k3, k4, k5, k6 = padded

        lidar_points = points_homo[cam_id, :3]
        lidar2cam_r = np.linalg.inv(sensor2lidar_r)
        cam_points = lidar2cam_r @ (lidar_points - sensor2lidar_t)
        z_depth = cam_points[2]
        z = np.maximum(z_depth, 1e-5)
        x_c = cam_points[0] / z
        y_c = cam_points[1] / z
        # ROI 中相机背后的点会因 z clamp 产生极大的归一化坐标，后续 valid
        # 会将这些点全部排除。这里仅屏蔽预期的浮点告警，不改变投影或筛选语义。
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            r2 = x_c * x_c + y_c * y_c
            radial_num = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            radial_den = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
            x_distorted = x_c * radial_num / radial_den + (
                2 * p1 * x_c * y_c + p2 * (r2 + 2 * x_c ** 2))
            y_distorted = y_c * radial_num / radial_den + (
                p1 * (r2 + 2 * y_c ** 2) + 2 * p2 * x_c * y_c)

            distorted = np.stack(
                (x_distorted, y_distorted, np.ones_like(x_distorted)), axis=0)
            pixel = intrinsic @ distorted
            pixel = pixel / np.maximum(pixel[2:3], 1e-5)
        pixel = post_rot @ pixel + post_tran
        pixel_xy = pixel[:2] / float(stride)
        outputs.append(np.stack((pixel_xy[0] * z_depth, pixel_xy[1] * z_depth, z_depth), axis=0))
    return np.stack(outputs, axis=0).astype(np.float32)


def project_points_with_distortion_torch(
    points_homo: np.ndarray,
    projection: np.ndarray,
    img_meta: Dict[str, Any],
    stride: int,
    device_text: str,
):
    """使用和 ``fastbev.py`` 动态投影相同的 Torch 运算顺序。"""
    import torch

    device = torch.device(device_text)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            f'--projection-backend=torch 请求 {device}，但 torch.cuda.is_available()=False')

    points = torch.as_tensor(points_homo, device=device, dtype=torch.float32)
    projection_tensor = torch.as_tensor(
        projection, device=device, dtype=torch.float32)
    aug_infos = img_meta.get('lidar2img', {}).get('lidar2img_aug', [])
    n_images = points.shape[0]
    if not isinstance(aug_infos, list) or len(aug_infos) < n_images:
        raise KeyError(
            'Torch distortion LUT 要求每个相机都包含 lidar2img_aug；'
            f'n_images={n_images}, aug_count={len(aug_infos) if isinstance(aug_infos, list) else None}')

    eye3 = torch.eye(3, device=device, dtype=points.dtype)
    zero3 = torch.zeros(3, device=device, dtype=points.dtype)
    sensor2lidar_rs = []
    sensor2lidar_ts = []
    intrinsics = []
    post_rots = []
    post_trans = []
    distortions = []
    for cam_id in range(n_images):
        aug = aug_infos[cam_id]
        distortion = camera_distortion(img_meta, cam_id)
        if (not isinstance(aug, dict) or
                any(key not in aug for key in ('rot', 'tran', 'intrin')) or
                distortion is None):
            # 精确对齐模式不能静默回退，否则会生成看似有效但与动态畸变路径
            # 不同的 LUT。N7 正式资产必须补齐这些字段。
            raise KeyError(
                f'Torch distortion LUT 的 cam{cam_id} 缺少 rot/tran/intrin/distortion')

        sensor2lidar_rs.append(torch.as_tensor(
            aug['rot'], device=device, dtype=points.dtype).reshape(3, 3))
        sensor2lidar_ts.append(torch.as_tensor(
            aug['tran'], device=device, dtype=points.dtype).reshape(3, 1))
        intrinsics.append(torch.as_tensor(
            aug['intrin'], device=device, dtype=points.dtype).reshape(3, 3))
        post_rots.append(torch.as_tensor(
            aug.get('post_rot', eye3), device=device, dtype=points.dtype).reshape(3, 3))
        post_trans.append(torch.as_tensor(
            aug.get('post_tran', zero3), device=device, dtype=points.dtype).reshape(3, 1))

        distortion_tensor = torch.as_tensor(
            distortion, device=device, dtype=points.dtype).reshape(-1)
        padded = torch.zeros(8, device=device, dtype=points.dtype)
        if distortion_tensor.numel() >= 8:
            padded[:] = distortion_tensor[:8]
        else:
            padded[:min(distortion_tensor.numel(), 5)] = distortion_tensor[
                :min(distortion_tensor.numel(), 5)]
        distortions.append(padded)

    sensor2lidar_r = torch.stack(sensor2lidar_rs)
    sensor2lidar_t = torch.stack(sensor2lidar_ts)
    intrinsic = torch.stack(intrinsics)
    post_rot = torch.stack(post_rots)
    post_tran = torch.stack(post_trans)
    distortion = torch.stack(distortions)

    lidar2cam_r = torch.inverse(sensor2lidar_r)
    cam_points = torch.bmm(
        lidar2cam_r, points[:, :3] - sensor2lidar_t)
    z_depth = cam_points[:, 2]
    z = z_depth.clamp(min=1e-5)
    x_c = cam_points[:, 0] / z
    y_c = cam_points[:, 1] / z
    r2 = x_c * x_c + y_c * y_c

    k1 = distortion[:, 0:1]
    k2 = distortion[:, 1:2]
    p1 = distortion[:, 2:3]
    p2 = distortion[:, 3:4]
    k3 = distortion[:, 4:5]
    k4 = distortion[:, 5:6]
    k5 = distortion[:, 6:7]
    k6 = distortion[:, 7:8]
    radial_num = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    radial_den = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
    x_distorted = x_c * radial_num / radial_den + (
        2 * p1 * x_c * y_c + p2 * (r2 + 2 * x_c ** 2))
    y_distorted = y_c * radial_num / radial_den + (
        p1 * (r2 + 2 * y_c ** 2) + 2 * p2 * x_c * y_c)

    distorted = torch.stack(
        (x_distorted, y_distorted, torch.ones_like(x_distorted)), dim=1)
    pixel = torch.bmm(intrinsic, distorted)
    pixel = pixel / pixel[:, 2:3].clamp(min=1e-5)
    pixel = torch.bmm(post_rot, pixel) + post_tran
    pixel_xy = pixel[:, :2] / float(stride)
    return torch.stack(
        (pixel_xy[:, 0] * z_depth, pixel_xy[:, 1] * z_depth, z_depth),
        dim=1)


def compute_lut_indices_torch(
    flat_points: np.ndarray,
    projection: np.ndarray,
    img_meta: Dict[str, Any],
    stride: int,
    use_distortion: bool,
    height: int,
    width: int,
    device_text: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用 Torch 完成投影、round 和 valid，避免跨后端像素边界差异。"""
    import torch

    device = torch.device(device_text)
    points = torch.as_tensor(flat_points, device=device, dtype=torch.float32)
    projection_tensor = torch.as_tensor(
        projection, device=device, dtype=torch.float32)
    if use_distortion:
        points_homo = torch.cat(
            (points, torch.ones_like(points[:, :1])), dim=1)
        points_2d_3 = project_points_with_distortion_torch(
            points_homo.detach().cpu().numpy(), projection, img_meta, stride,
            device_text=device_text)
    else:
        points_2d_3 = torch.bmm(
            projection_tensor[:, :, :3], points) + projection_tensor[:, :, 3:4]

    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (
        (x >= 0) & (y >= 0) &
        (x < int(width)) & (y < int(height)) & (z > 0))
    return (
        points_2d_3.detach().cpu().numpy().astype(np.float32),
        x.detach().cpu().numpy().astype(np.int64),
        y.detach().cpu().numpy().astype(np.int64),
        z.detach().cpu().numpy().astype(np.float32),
        valid.detach().cpu().numpy().astype(bool),
    )


def compute_lut(
    features_shape: Tuple[int, int, int, int],
    points: np.ndarray,
    projection: np.ndarray,
    img_meta: Dict[str, Any],
    stride: int,
    use_distortion: bool,
    camera_overwrite_order: Sequence[int],
    dump_debug_volume: bool,
    projection_backend: str = 'numpy',
    torch_device: str = 'cuda:0',
) -> Dict[str, np.ndarray]:
    n_images, n_channels, height, width = features_shape
    flat_points = points.reshape(1, 3, -1).repeat(n_images, axis=0)
    if projection_backend == 'torch':
        points_2d_3, x, y, z, valid = compute_lut_indices_torch(
            flat_points=flat_points,
            projection=projection,
            img_meta=img_meta,
            stride=stride,
            use_distortion=use_distortion,
            height=height,
            width=width,
            device_text=torch_device,
        )
    else:
        if use_distortion:
            points_homo = np.concatenate(
                [flat_points, np.ones(
                    (n_images, 1, flat_points.shape[-1]), dtype=np.float32)],
                axis=1)
            points_2d_3 = project_points_with_distortion(
                points_homo, projection, img_meta, stride)
        else:
            points_2d_3 = (
                projection[:, :, :3] @ flat_points + projection[:, :, 3:4])

        z = points_2d_3[:, 2].astype(np.float32)
        denom = np.where(np.abs(z) > 1e-8, z, np.nan)
        x_float = np.rint(points_2d_3[:, 0] / denom)
        y_float = np.rint(points_2d_3[:, 1] / denom)
        x = np.where(np.isfinite(x_float), x_float, -1).astype(np.int64)
        y = np.where(np.isfinite(y_float), y_float, -1).astype(np.int64)
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)

    gather_index_dense = np.full_like(x, -1, dtype=np.int64)
    gather_index_dense[valid] = y[valid] * width + x[valid]
    scatter_index_dense = np.broadcast_to(
        np.arange(flat_points.shape[-1], dtype=np.int64).reshape(1, -1),
        valid.shape).copy()

    camera_choice = np.full(flat_points.shape[-1], -1, dtype=np.int64)
    for cam_id in camera_overwrite_order:
        camera_choice[valid[cam_id]] = int(cam_id)
    assigned = camera_choice >= 0
    fused_scatter_index = np.nonzero(assigned)[0].astype(np.int64)
    fused_camera_index = camera_choice[fused_scatter_index].astype(np.int64)
    fused_gather_index = gather_index_dense[fused_camera_index, fused_scatter_index].astype(np.int64)

    outputs = dict(
        x=x,
        y=y,
        z=z,
        valid=valid,
        points_2d_3=points_2d_3,
        gather_index_dense=gather_index_dense,
        scatter_index_dense=scatter_index_dense,
        camera_choice=camera_choice,
        assigned=assigned,
        fused_scatter_index=fused_scatter_index,
        fused_camera_index=fused_camera_index,
        fused_gather_index=fused_gather_index,
    )

    if dump_debug_volume:
        features = np.arange(np.prod(features_shape), dtype=np.float32).reshape(features_shape)
        volume = np.zeros((n_channels, flat_points.shape[-1]), dtype=np.float32)
        for cam_id in camera_overwrite_order:
            cam_valid = valid[cam_id]
            if cam_valid.any():
                volume[:, cam_valid] = features[cam_id][:, y[cam_id, cam_valid], x[cam_id, cam_valid]]
        outputs['features'] = features
        outputs['volume'] = volume
    return outputs


def save_board_bin_outputs(out_dir: Path, arrays: Dict[str, np.ndarray]) -> List[int]:
    """保存和已验证板端 ``get_lut.py`` 兼容的 bin LUT 文件。

    ``get_lut.py`` 先按相机保存 gather/scatter，再删除会被后续相机覆盖的
    scatter 点。这里已经有最终 ``camera_choice``，因此直接按最终归属导出：
    - ``LUT/gather_new_i.bin``: 第 i 个相机的 feature 平面线性索引；
    - ``LUT/scatter_nd_new_i.bin``: 对应 BEV flatten 后的点索引；
    - ``LUT/featurePointLength.bin``: 每个相机保留下来的点数。
    """
    board_dir = out_dir / 'LUT'
    board_dir.mkdir(parents=True, exist_ok=True)
    lut_arr_dir = out_dir / 'LUT_arr'
    lut_arr_dir.mkdir(parents=True, exist_ok=True)

    camera_choice = arrays['camera_choice']
    gather_dense = arrays['gather_index_dense']
    n_images = gather_dense.shape[0]
    lengths: List[int] = []
    for cam_id in range(n_images):
        scatter_index = np.nonzero(camera_choice == cam_id)[0].astype(np.int32)
        gather_index = gather_dense[cam_id, scatter_index].astype(np.int32)
        if np.any(gather_index < 0):
            raise ValueError(f'cam{cam_id} 存在无效 gather index')

        gather_index.tofile(board_dir / f'gather_new_{cam_id}.bin')
        scatter_index.tofile(board_dir / f'scatter_nd_new_{cam_id}.bin')
        lengths.append(int(gather_index.shape[0]))

        # 同时保存 get_lut.py 中间 npy 形态，便于和旧工具做逐项对照。
        np.save(lut_arr_dir / f'gather_{cam_id}.npy', gather_index)
        np.save(lut_arr_dir / f'scatter_nd_{cam_id}.npy', scatter_index.reshape(1, -1, 1))

    np.asarray(lengths, dtype=np.int32).tofile(board_dir / 'featurePointLength.bin')
    return lengths


def save_lut_outputs(
    out_dir: Path,
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    compact_output: bool = False,
) -> None:
    """保存 LUT 输出。

    ``compact_output=False`` 保持原诊断工具的完整输出，包含 projection、
    valid、camera_choice 等中间数组。正式板端运行只需要 ``LUT``、
    ``LUT_arr`` 和 ``metadata.json``，因此可显式传 ``compact_output=True``
    避免为每辆车重复保存较大的稠密调试数组。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not compact_output:
        for name, value in arrays.items():
            np.save(out_dir / f'{name}.npy', value)

        # 兼容旧脚本命名：旧逻辑直接读取 features.npy 和 volume.npy。
        if 'features' not in arrays:
            (out_dir / 'features.npy').unlink(missing_ok=True)
        if 'volume' not in arrays:
            (out_dir / 'volume.npy').unlink(missing_ok=True)

        per_camera_dir = out_dir / 'per_camera'
        per_camera_dir.mkdir(exist_ok=True)
        valid = arrays['valid']
        gather_dense = arrays['gather_index_dense']
        scatter_dense = arrays['scatter_index_dense']
        for cam_id in range(valid.shape[0]):
            cam_valid = valid[cam_id]
            np.save(
                per_camera_dir / f'cam{cam_id}_gather_index.npy',
                gather_dense[cam_id, cam_valid])
            np.save(
                per_camera_dir / f'cam{cam_id}_scatter_index.npy',
                scatter_dense[cam_id, cam_valid])

    board_lengths = save_board_bin_outputs(out_dir, arrays)
    metadata['board_bin_dir'] = str(out_dir / 'LUT')
    metadata['board_feature_point_lengths'] = board_lengths
    with (out_dir / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump(to_jsonable(metadata), f, ensure_ascii=False, indent=2)


def select_temporal_frames(
    info: Dict[str, Any],
    n_times: int,
    sequential: bool,
    test_adj_ids: Optional[Sequence[int]],
    min_interval: int,
) -> List[Dict[str, Any]]:
    frames = [info]
    if not sequential:
        return frames
    for time_id in range(1, n_times):
        prev_infos = info.get('prev', [])
        if not prev_infos:
            frames.append(info)
            continue
        if test_adj_ids is not None:
            select_id = min(abs(int(test_adj_ids[time_id - 1])), len(prev_infos) - 1)
        else:
            select_id = min(int(min_interval) + time_id - 1, len(prev_infos) - 1)
        frames.append(prev_infos[max(select_id, 0)])
    return frames


def build_img_meta_for_sample(
    info: Dict[str, Any],
    camera_types: Sequence[str],
    n_times: int,
    sequential: bool,
    test_adj_ids: Optional[Sequence[int]],
    min_interval: int,
    temporal_compensate: bool,
    data_config: Dict[str, Any],
    force_resize: bool,
) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
    frames = select_temporal_frames(info, n_times, sequential, test_adj_ids, min_interval)
    extrinsics = []
    aug_infos = []
    extra_infos = []
    image_shapes = []
    img_info = []
    frame_records = []

    for time_id, frame_info in enumerate(frames):
        frame_records.append(dict(
            time_id=time_id,
            token=frame_info.get('token'),
            timestamp=frame_info.get('timestamp')))
        for cam_id in camera_types:
            if cam_id not in frame_info.get('cams', {}):
                raise KeyError(f'token={frame_info.get("token")} 缺少相机 {cam_id}')
            cam_info = frame_info['cams'][cam_id]
            sensor_r, sensor_t, compensated, reason = sensor2reference_lidar_for_dataset(
                cam_info,
                frame_info=frame_info,
                ref_info=info,
                temporal_compensate=temporal_compensate)
            _, cam_aug, cam_extra = dataset_lidar2img_from_cam_info(
                cam_info,
                sensor2lidar_r=sensor_r,
                sensor2lidar_t=sensor_t,
                temporal_compensated=compensated)
            post_rot, post_tran, image_shape = test_image_aug_params(
                cam_aug, data_config=data_config, force_resize=force_resize)
            cam_aug['post_rot'] = post_rot
            cam_aug['post_tran'] = post_tran
            extrinsics.append(rts2proj(cam_aug, post_rot, post_tran))
            aug_infos.append(cam_aug)
            cam_extra = dict(cam_extra)
            cam_extra['projection_reason'] = reason
            extra_infos.append(cam_extra)
            image_shapes.append(image_shape)
            img_info.append(dict(filename=cam_info.get('data_path', ''), camera_id=cam_id, time_id=time_id))

    img_meta = dict(
        sample_idx=info.get('token'),
        timestamp=info.get('timestamp', 0) / 1e6,
        img_shape=image_shapes,
        img_info=img_info,
        lidar2img=dict(
            extrinsic=[item.astype(np.float32) for item in extrinsics],
            intrinsic=np.eye(4, dtype=np.float32),
            lidar2img_aug=aug_infos,
            lidar2img_extra=extra_infos,
        ))
    return img_meta, [item['camera_id'] for item in img_info], frame_records


def export_one_sequence(
    out_dir: Path,
    img_meta: Dict[str, Any],
    seq_id: int,
    n_images: int,
    n_voxels: Sequence[int],
    voxel_size: Sequence[float],
    origin: Sequence[float],
    stride: int,
    use_distortion: bool,
    camera_overwrite_order: Sequence[int],
    feature_channels: int,
    dump_debug_volume: bool,
    projection_backend: str,
    torch_device: str,
    common_metadata: Dict[str, Any],
    compact_output: bool = False,
) -> None:
    start = seq_id * n_images
    end = (seq_id + 1) * n_images
    seq_meta = copy.deepcopy(img_meta)
    lidar2img = seq_meta['lidar2img']
    for key in ('extrinsic', 'lidar2img_aug', 'lidar2img_extra'):
        lidar2img[key] = lidar2img[key][start:end]
    seq_meta['img_shape'] = seq_meta['img_shape'][start:end]
    seq_meta['img_info'] = seq_meta['img_info'][start:end]

    image_h, image_w = [int(v) for v in seq_meta['img_shape'][0][:2]]
    feat_h = int(math.ceil(image_h / float(stride)))
    feat_w = int(math.ceil(image_w / float(stride)))
    projection = np.stack(
        [projection_for_stride(item, stride) for item in lidar2img['extrinsic']],
        axis=0)
    points = get_points(n_voxels, voxel_size, origin)
    arrays = compute_lut(
        features_shape=(n_images, feature_channels, feat_h, feat_w),
        points=points,
        projection=projection,
        img_meta=seq_meta,
        stride=stride,
        use_distortion=use_distortion,
        camera_overwrite_order=camera_overwrite_order,
        dump_debug_volume=dump_debug_volume,
        projection_backend=projection_backend,
        torch_device=torch_device)
    arrays['points'] = points.astype(np.float32)
    arrays['projection'] = projection.astype(np.float32)
    arrays['camera_overwrite_order'] = np.asarray(camera_overwrite_order, dtype=np.int64)
    arrays['valid_5d'] = arrays['valid'].reshape(n_images, *[int(v) for v in n_voxels])
    arrays['camera_choice_3d'] = arrays['camera_choice'].reshape(*[int(v) for v in n_voxels])

    metadata = copy.deepcopy(common_metadata)
    metadata.update(dict(
        seq_id=seq_id,
        seq_camera_ids=[item.get('camera_id') for item in seq_meta['img_info']],
        seq_img_shapes=seq_meta['img_shape'],
        feature_shape=[n_images, feature_channels, feat_h, feat_w],
        projection_shape=list(projection.shape),
        points_shape=list(points.shape),
        x_shape=list(arrays['x'].shape),
        valid_shape=list(arrays['valid'].shape),
        volume_shape=list(arrays['volume'].shape) if 'volume' in arrays else None,
        valid_count_per_camera=arrays['valid'].sum(axis=1).astype(int).tolist(),
        assigned_bev_points=int(arrays['assigned'].sum()),
        projection_backend=projection_backend,
        torch_device=torch_device if projection_backend == 'torch' else None,
    ))
    save_lut_outputs(
        out_dir,
        arrays,
        metadata,
        compact_output=compact_output)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='从 N7 Fast-BEV pkl + config 生成 2D->3D LUT npy/bin 文件')
    parser.add_argument('config', help='Fast-BEV config 路径')
    parser.add_argument('--pkl', default=None, help='输入 pkl；默认使用 config.data.<split>.ann_file')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='从 config.data 哪个 split 取数据配置')
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument('--sample-index', type=int, default=None,
                              help='按 timestamp 排序后的样本下标；未传 --token 时默认 0')
    sample_group.add_argument('--token', default=None, help='按 token 精确选择样本，推荐真实数值对齐使用')
    parser.add_argument('--seq-id', type=int, default=0, help='导出第几个时序片段，默认 0')
    parser.add_argument('--all-seqs', action='store_true', help='导出样本内所有时序片段到 seq_0/seq_1/... 子目录')
    parser.add_argument('--out-dir', required=True, help='输出目录')
    parser.add_argument('--stride', type=int, default=4, help='2D feature 相对输入图像的 stride，R18 m0 默认 4')
    parser.add_argument('--feature-channels', type=int, default=64, help='debug features 的通道数，R18 m0 neck_fuse 输出默认 64')
    parser.add_argument('--camera-overwrite-order', default=None, help='相机覆盖顺序，如 1,2,0,4,5,3；默认自然顺序 0..n-1')
    parser.add_argument('--dump-debug-volume', action='store_true', help='额外生成 features.npy 和 volume.npy')
    parser.add_argument('--use-distortion', dest='use_distortion', action='store_true', default=None, help='强制使用动态畸变投影')
    parser.add_argument('--no-use-distortion', dest='use_distortion', action='store_false', help='强制使用 pinhole 投影')
    parser.add_argument(
        '--projection-backend', choices=('numpy', 'torch'), default='numpy',
        help='LUT 投影/round 后端；严格复现 GPU PTH 动态 backproject 时使用 torch')
    parser.add_argument(
        '--torch-device', default='cuda:0',
        help='--projection-backend=torch 使用的设备；应与黄金 PTH 推理设备一致')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config_path = Path(args.config)
    cfg = load_py_config(config_path)
    dataset_cfg = get_nested(cfg, ['data', args.split], {})
    if not isinstance(dataset_cfg, dict):
        raise TypeError(f'config.data.{args.split} 不是 dict')

    pkl_text = args.pkl or dataset_cfg.get('ann_file')
    if not pkl_text:
        raise ValueError('请通过 --pkl 或 config.data.<split>.ann_file 指定 pkl')
    pkl_path = Path(pkl_text)
    infos, pkl_metadata = load_infos(pkl_path)
    if args.token is not None:
        matches = [index for index, item in enumerate(infos)
                   if str(item.get('token')) == str(args.token)]
        if len(matches) != 1:
            raise ValueError(f'--token={args.token!r} 匹配数量应为 1，实际 {len(matches)}')
        sample_index = matches[0]
    else:
        sample_index = 0 if args.sample_index is None else int(args.sample_index)
        if not (0 <= sample_index < len(infos)):
            raise IndexError(f'--sample-index={sample_index} 超出 infos 数量 {len(infos)}')
    info = infos[sample_index]

    pipeline = dataset_cfg.get('pipeline') or cfg.get('test_pipeline') or []
    aug_step = find_pipeline_step(pipeline, 'RandomAugImageMultiViewImage')
    if aug_step is None:
        raise KeyError('config test pipeline 中找不到 RandomAugImageMultiViewImage')
    origin_step = find_pipeline_step(pipeline, 'KittiSetOrigin')
    point_cloud_range = (
        origin_step.get('point_cloud_range') if origin_step else cfg.get('point_cloud_range'))
    if point_cloud_range is None:
        raise KeyError('config 中找不到 KittiSetOrigin.point_cloud_range 或 point_cloud_range')
    origin = ((np.asarray(point_cloud_range[:3], dtype=np.float32) +
               np.asarray(point_cloud_range[3:], dtype=np.float32)) / 2.0)

    model_cfg = cfg.get('model', {})
    n_images = int(model_cfg.get('n_images', dataset_cfg.get('n_images', 6)))
    n_voxels = normalize_first_item(model_cfg.get('n_voxels', [[200, 200, 4]]))
    voxel_size = normalize_first_item(model_cfg.get('voxel_size', [[0.5, 0.5, 1.5]]))
    use_distortion = bool(model_cfg.get('use_distortion', False))
    if args.use_distortion is not None:
        use_distortion = bool(args.use_distortion)

    camera_types = dataset_cfg.get('camera_types') or cfg.get('camera_types')
    if camera_types is None:
        camera_types = list(info.get('cams', {}).keys())[:n_images]
    if len(camera_types) != n_images:
        raise ValueError(f'camera_types 数量 {len(camera_types)} 和 n_images={n_images} 不一致')

    n_times = int(dataset_cfg.get('n_times', 1))
    sequential = bool(dataset_cfg.get('sequential', False))
    test_adj_ids = dataset_cfg.get('test_adj_ids')
    min_interval = int(dataset_cfg.get('min_interval', 0))
    temporal_compensate = bool(dataset_cfg.get('temporal_compensate', True))
    data_config = aug_step['data_config']
    force_resize = bool(aug_step.get('force_resize', False))
    camera_overwrite_order = parse_index_list(args.camera_overwrite_order, n_images)

    img_meta, camera_id_sequence, frame_records = build_img_meta_for_sample(
        info=info,
        camera_types=camera_types,
        n_times=n_times,
        sequential=sequential,
        test_adj_ids=test_adj_ids,
        min_interval=min_interval,
        temporal_compensate=temporal_compensate,
        data_config=data_config,
        force_resize=force_resize)

    common_metadata = dict(
        config=str(config_path),
        pkl=str(pkl_path),
        pkl_metadata=pkl_metadata,
        split=args.split,
        sample_index=sample_index,
        sample_selection='token' if args.token is not None else 'sample_index',
        sample_token=info.get('token'),
        sample_timestamp=info.get('timestamp'),
        frame_records=frame_records,
        camera_types=list(camera_types),
        camera_id_sequence=camera_id_sequence,
        n_images=n_images,
        n_times=n_times,
        n_voxels=[int(v) for v in n_voxels],
        voxel_size=[float(v) for v in voxel_size],
        origin=[float(v) for v in origin],
        stride=args.stride,
        use_distortion=use_distortion,
        force_resize=force_resize,
        data_config=data_config,
        camera_overwrite_order=camera_overwrite_order,
        dump_debug_volume=bool(args.dump_debug_volume),
        projection_backend=args.projection_backend,
        torch_device=args.torch_device if args.projection_backend == 'torch' else None,
    )

    out_dir = Path(args.out_dir)
    seq_count = len(camera_id_sequence) // n_images
    if args.all_seqs:
        for seq_id in range(seq_count):
            export_one_sequence(
                out_dir=out_dir / f'seq_{seq_id}',
                img_meta=img_meta,
                seq_id=seq_id,
                n_images=n_images,
                n_voxels=n_voxels,
                voxel_size=voxel_size,
                origin=origin,
                stride=args.stride,
                use_distortion=use_distortion,
                camera_overwrite_order=camera_overwrite_order,
                feature_channels=args.feature_channels,
                dump_debug_volume=args.dump_debug_volume,
                projection_backend=args.projection_backend,
                torch_device=args.torch_device,
                common_metadata=common_metadata)
    else:
        if not (0 <= args.seq_id < seq_count):
            raise IndexError(f'--seq-id={args.seq_id} 超出样本时序片段数量 {seq_count}')
        export_one_sequence(
            out_dir=out_dir,
            img_meta=img_meta,
            seq_id=args.seq_id,
            n_images=n_images,
            n_voxels=n_voxels,
            voxel_size=voxel_size,
            origin=origin,
            stride=args.stride,
            use_distortion=use_distortion,
            camera_overwrite_order=camera_overwrite_order,
            feature_channels=args.feature_channels,
            dump_debug_volume=args.dump_debug_volume,
            projection_backend=args.projection_backend,
            torch_device=args.torch_device,
            common_metadata=common_metadata)

    print(f'LUT npy 已输出到: {out_dir}')


if __name__ == '__main__':
    main()
