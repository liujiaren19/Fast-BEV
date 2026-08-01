#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较 N7 mono-front direct stretch 与等比缩放/垂直裁剪几何。

工具复现当前 dataset eval 的原图 pinhole 可见性，以及训练 pipeline 在图像
变换后的第二次 ``FrontCameraVisibleObjectFilter``。目标像素统计则沿用模型的
distortion-aware 投影，再施加对应 post affine。全量帧用于精确 keep mask 和
yaw 计数；像素分位数按固定帧步长抽样，避免数百万 GT 乘多个 crop offset 后
产生数十 GB Python 记录。该工具只读，不修改 PKL、图片或训练/eval 过滤逻辑。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from tools.data_converter.n7.visualize_n7_fastbev_pkl import (
        camera_info_for_pkl_image_size,
        compute_lidar2img,
        load_fastbev_pkl,
        project_lidar_points_distorted,
        project_lidar_points_pinhole,
    )
except ModuleNotFoundError:
    from visualize_n7_fastbev_pkl import (
        camera_info_for_pkl_image_size,
        compute_lidar2img,
        load_fastbev_pkl,
        project_lidar_points_distorted,
        project_lidar_points_pinhole,
    )


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_box_corners(boxes: np.ndarray) -> np.ndarray:
    """复现 ``CustomMultiViewDataset._box_corners_from_boxes``。"""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    boxes = boxes.reshape(-1, boxes.shape[-1])[:, :7]
    corners = np.zeros((boxes.shape[0], 8, 3), dtype=np.float32)
    template = np.asarray([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ], dtype=np.float32) / 2.0
    for index, box in enumerate(boxes):
        x, y, z, length, width, height, yaw = box[:7]
        local = template * np.asarray([length, width, height], dtype=np.float32)
        local_xy = local[:, :2].copy()
        c, s = np.cos(yaw), np.sin(yaw)
        local[:, 0] = local_xy[:, 0] * c + local_xy[:, 1] * s
        local[:, 1] = -local_xy[:, 0] * s + local_xy[:, 1] * c
        corners[index] = local + np.asarray([x, y, z], dtype=np.float32)
    return corners


def sample_box_edges(corners: np.ndarray, samples_per_edge: int) -> np.ndarray:
    weights = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float32)
    return np.concatenate([
        corners[start][None] * (1.0 - weights[:, None]) +
        corners[end][None] * weights[:, None]
        for start, end in BOX_EDGES
    ], axis=0)


def strategy_contracts(
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    crop_top_offsets: Sequence[int],
) -> List[Dict]:
    source_h, source_w = source_size
    target_h, target_w = target_size
    scale = target_w / float(source_w)
    scaled_h = int(source_h * scale)
    contracts = [dict(
        name='direct_stretch',
        resize_x=target_w / float(source_w),
        resize_y=target_h / float(source_h),
        resize_dims=[target_w, target_h],
        crop=[0, 0, target_w, target_h],
        post_rot=[[target_w / float(source_w), 0.0],
                  [0.0, target_h / float(source_h)]],
        post_tran=[0.0, 0.0],
    )]
    for offset in crop_top_offsets:
        if offset < 0 or offset + target_h > scaled_h:
            raise ValueError(
                'crop top {} is outside scaled height {} for target height {}'.format(
                    offset, scaled_h, target_h))
        contracts.append(dict(
            name='scale_crop_top_{}'.format(offset),
            resize_x=scale,
            resize_y=scale,
            resize_dims=[target_w, scaled_h],
            crop=[0, offset, target_w, offset + target_h],
            post_rot=[[scale, 0.0], [0.0, scale]],
            post_tran=[0.0, -float(offset)],
        ))
    return contracts


def affine_uv(uv: np.ndarray, contract: Mapping) -> np.ndarray:
    matrix = np.asarray(contract['post_rot'], dtype=np.float32)
    translation = np.asarray(contract['post_tran'], dtype=np.float32)
    return uv @ matrix.T + translation


def points_visible(
    points: np.ndarray,
    lidar2img: np.ndarray,
    image_shape: Tuple[int, int],
    min_depth: float,
    contract: Mapping = None,
) -> np.ndarray:
    uv, depth = project_lidar_points_pinhole(points, lidar2img)
    if contract is not None:
        uv = affine_uv(uv, contract)
    height, width = image_shape
    return (
        (depth > min_depth) & np.isfinite(uv).all(axis=1) &
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )


def projected_points_visible(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: Tuple[int, int],
    min_depth: float,
    contract: Mapping = None,
) -> np.ndarray:
    """复用已经完成的 pinhole 投影，避免每个 crop offset 重复矩阵乘法。"""
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    if contract is not None:
        uv = affine_uv(uv, contract)
    height, width = image_shape
    return (
        (depth > min_depth) & np.isfinite(uv).all(axis=1) &
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )


def box_visible_masks(
    boxes: np.ndarray,
    lidar2img: np.ndarray,
    source_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    min_depth: float,
    contract: Mapping,
) -> Tuple[np.ndarray, np.ndarray]:
    centers = boxes[:, :3]
    corners = dataset_box_corners(boxes)
    raw_keep = (
        points_visible(centers, lidar2img, source_shape, min_depth) |
        points_visible(
            corners.reshape(-1, 3), lidar2img, source_shape, min_depth
        ).reshape(len(boxes), -1).any(axis=1)
    )
    transformed_keep = (
        points_visible(
            centers, lidar2img, target_shape, min_depth, contract) |
        points_visible(
            corners.reshape(-1, 3), lidar2img, target_shape, min_depth, contract
        ).reshape(len(boxes), -1).any(axis=1)
    )
    return raw_keep, transformed_keep


def roi_mask(boxes: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    x0, y0, z0, x1, y1, z1 = [float(value) for value in roi]
    return (
        (boxes[:, 0] >= x0) & (boxes[:, 0] <= x1) &
        (boxes[:, 1] >= y0) & (boxes[:, 1] <= y1) &
        (boxes[:, 2] >= z0) & (boxes[:, 2] <= z1)
    )


def interval_name(value: float, bins: Sequence[float], suffix: str) -> str:
    for left, right in zip(bins[:-1], bins[1:]):
        if left <= value < right:
            return '{:g}-{:g}{}'.format(left, right, suffix)
    return 'outside'


def normalize_yaw_deg(yaw: float) -> float:
    return math.degrees(math.atan2(math.sin(float(yaw)), math.cos(float(yaw))))


def projection_record(
    box: np.ndarray,
    cam_info: Mapping,
    contract: Mapping,
    target_shape: Tuple[int, int],
    samples_per_edge: int,
    min_depth: float,
) -> Dict:
    scaled_cam, _ = camera_info_for_pkl_image_size(dict(cam_info))
    intrinsic = np.asarray(scaled_cam['cam_intrinsic'], dtype=np.float32)
    corners = dataset_box_corners(box.reshape(1, -1))[0]
    points = sample_box_edges(corners, samples_per_edge)
    uv, depth = project_lidar_points_distorted(points, scaled_cam, intrinsic)
    valid = (depth > min_depth) & np.isfinite(uv).all(axis=1)
    if not np.any(valid):
        raise ValueError('no finite projected edge point in front')
    uv = affine_uv(uv[valid], contract).astype(np.float64)
    x1, y1 = np.min(uv, axis=0)
    x2, y2 = np.max(uv, axis=0)
    height, width = target_shape
    full_width = max(0.0, float(x2 - x1))
    full_height = max(0.0, float(y2 - y1))
    full_area = full_width * full_height
    clipped_x1 = float(np.clip(x1, 0.0, width))
    clipped_y1 = float(np.clip(y1, 0.0, height))
    clipped_x2 = float(np.clip(x2, 0.0, width))
    clipped_y2 = float(np.clip(y2, 0.0, height))
    clipped_width = max(0.0, clipped_x2 - clipped_x1)
    clipped_height = max(0.0, clipped_y2 - clipped_y1)
    clipped_area = clipped_width * clipped_height
    visible_ratio = clipped_area / full_area if full_area > 0 else 0.0
    margins = dict(
        margin_left_px=float(x1),
        margin_top_px=float(y1),
        margin_right_px=float(width - x2),
        margin_bottom_px=float(height - y2),
    )
    return dict(
        bbox_x1=float(x1), bbox_y1=float(y1),
        bbox_x2=float(x2), bbox_y2=float(y2),
        input_width_px=full_width,
        input_height_px=full_height,
        stride4_width_px=full_width / 4.0,
        stride4_height_px=full_height / 4.0,
        clipped_width_px=clipped_width,
        clipped_height_px=clipped_height,
        visible_area_ratio=visible_ratio,
        cropped_area_ratio=1.0 - visible_ratio,
        min_edge_margin_px=min(margins.values()),
        **margins,
    )


def project_box_edges_distorted(
    boxes: np.ndarray,
    cam_info: Mapping,
    samples_per_edge: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """一次投影一帧内所有目标边采样点，供多个 crop contract 复用。"""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return (
            np.zeros((0, len(BOX_EDGES) * samples_per_edge, 2), dtype=np.float32),
            np.zeros((0, len(BOX_EDGES) * samples_per_edge), dtype=np.float32),
        )
    boxes = boxes.reshape(-1, boxes.shape[-1])[:, :7]
    corners = dataset_box_corners(boxes)
    weights = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float32)
    edge_points = np.concatenate([
        corners[:, start, None, :] * (1.0 - weights[None, :, None]) +
        corners[:, end, None, :] * weights[None, :, None]
        for start, end in BOX_EDGES
    ], axis=1)
    scaled_cam, _ = camera_info_for_pkl_image_size(dict(cam_info))
    intrinsic = np.asarray(scaled_cam['cam_intrinsic'], dtype=np.float32)
    uv, depth = project_lidar_points_distorted(
        edge_points.reshape(-1, 3), scaled_cam, intrinsic)
    return (
        uv.reshape(len(boxes), -1, 2),
        depth.reshape(len(boxes), -1),
    )


def projection_record_from_edges(
    uv: np.ndarray,
    depth: np.ndarray,
    contract: Mapping,
    target_shape: Tuple[int, int],
    min_depth: float,
) -> Dict:
    """从已投影边采样点计算一个目标的像素尺寸、裁边和边缘余量。"""
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    valid = (depth > min_depth) & np.isfinite(uv).all(axis=1)
    if not np.any(valid):
        raise ValueError('no finite projected edge point in front')
    transformed = affine_uv(uv[valid], contract).astype(np.float64)
    x1, y1 = np.min(transformed, axis=0)
    x2, y2 = np.max(transformed, axis=0)
    height, width = target_shape
    full_width = max(0.0, float(x2 - x1))
    full_height = max(0.0, float(y2 - y1))
    full_area = full_width * full_height
    clipped_x1 = float(np.clip(x1, 0.0, width))
    clipped_y1 = float(np.clip(y1, 0.0, height))
    clipped_x2 = float(np.clip(x2, 0.0, width))
    clipped_y2 = float(np.clip(y2, 0.0, height))
    clipped_width = max(0.0, clipped_x2 - clipped_x1)
    clipped_height = max(0.0, clipped_y2 - clipped_y1)
    clipped_area = clipped_width * clipped_height
    visible_ratio = clipped_area / full_area if full_area > 0 else 0.0
    margins = dict(
        margin_left_px=float(x1),
        margin_top_px=float(y1),
        margin_right_px=float(width - x2),
        margin_bottom_px=float(height - y2),
    )
    return dict(
        bbox_x1=float(x1), bbox_y1=float(y1),
        bbox_x2=float(x2), bbox_y2=float(y2),
        input_width_px=full_width,
        input_height_px=full_height,
        stride4_width_px=full_width / 4.0,
        stride4_height_px=full_height / 4.0,
        clipped_width_px=clipped_width,
        clipped_height_px=clipped_height,
        visible_area_ratio=visible_ratio,
        cropped_area_ratio=1.0 - visible_ratio,
        min_edge_margin_px=min(margins.values()),
        **margins,
    )


def quantiles(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {}
    return {
        'p{:02d}'.format(round(level * 100)): float(np.quantile(array, level))
        for level in QUANTILES
    }


def summarize_records(
    records: Sequence[Mapping],
    population_count: int = None,
) -> Dict:
    fields = (
        'input_width_px', 'input_height_px',
        'stride4_width_px', 'stride4_height_px',
        'cropped_area_ratio', 'min_edge_margin_px',
        'margin_top_px', 'margin_bottom_px',
    )
    result = {
        'count': int(len(records) if population_count is None else population_count),
        'sample_count': len(records),
    }
    for field in fields:
        result[field] = quantiles(row[field] for row in records)
    return result


def analyze(args: argparse.Namespace) -> Tuple[List[Dict], Dict, Dict]:
    source_shape = tuple(int(value) for value in args.source_image_size)
    target_shape = tuple(int(value) for value in args.target_image_size)
    contracts = strategy_contracts(
        source_shape, target_shape, args.crop_top_offsets)
    contract_names = {contract['name'] for contract in contracts}
    gate_strategy = str(getattr(args, 'gate_strategy', 'scale_crop_top_70'))
    if gate_strategy not in contract_names:
        raise ValueError(
            '--gate-strategy {} is not among configured contracts {}'.format(
                gate_strategy, sorted(contract_names)))
    records: List[Dict] = []
    visibility = defaultdict(Counter)
    group_population = Counter()
    yaw_counts = Counter()
    pkl_assets = []
    visibility_frames = 0
    metric_frames = 0
    metric_stride = int(getattr(args, 'metric_stride', 1))
    log_every = int(getattr(args, 'log_every', 0))
    started_at = time.monotonic()

    for pkl_path in args.pkl:
        infos, metadata = load_fastbev_pkl(pkl_path)
        split = str(metadata.get('set', pkl_path.stem))
        pkl_assets.append(dict(
            path=str(pkl_path), sha256=file_sha256(pkl_path),
            split=split, infos=len(infos)))
        selected_infos = infos[args.start_index::args.stride]
        if args.max_frames >= 0:
            selected_infos = selected_infos[:args.max_frames]
        for selected_index, info in enumerate(selected_infos):
            visibility_frames += 1
            info_index = args.start_index + selected_index * args.stride
            collect_metrics = selected_index % metric_stride == 0
            if collect_metrics:
                metric_frames += 1
            boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32)
            names = np.asarray(info.get('gt_names', []), dtype=object).reshape(-1)
            if boxes.size == 0:
                continue
            boxes = boxes.reshape(-1, boxes.shape[-1])[:, :7]
            count = min(len(boxes), len(names))
            boxes, names = boxes[:count], names[:count]
            cam_info = (info.get('cams', {}) or {}).get(args.camera_id)
            if cam_info is None:
                raise KeyError('token={} missing {}'.format(info.get('token'), args.camera_id))
            scaled_cam, actual_source_shape = camera_info_for_pkl_image_size(dict(cam_info))
            if tuple(actual_source_shape) != source_shape:
                raise ValueError(
                    'token={} image shape {} expected {}'.format(
                        info.get('token'), actual_source_shape, source_shape))
            lidar2img = compute_lidar2img(scaled_cam)
            class_keep = np.asarray([
                str(name) in args.classes for name in names
            ], dtype=np.bool_)
            base_keep = class_keep & roi_mask(boxes, args.roi)

            # Dataset eval 的原图 pinhole 可见性与 contract 无关，只投影一次。
            centers = boxes[:, :3]
            corners = dataset_box_corners(boxes)
            center_uv, center_depth = project_lidar_points_pinhole(
                centers, lidar2img)
            corner_uv, corner_depth = project_lidar_points_pinhole(
                corners.reshape(-1, 3), lidar2img)
            raw_visible = (
                projected_points_visible(
                    center_uv, center_depth, source_shape, args.min_depth) |
                projected_points_visible(
                    corner_uv, corner_depth, source_shape, args.min_depth
                ).reshape(len(boxes), -1).any(axis=1)
            )
            eval_keep = base_keep & raw_visible
            eval_indices = np.flatnonzero(eval_keep)
            class_labels = [str(name) for name in names]
            distance_labels = [
                interval_name(float(box[0]), args.distance_bins, 'm')
                for box in boxes
            ]
            yaw_degrees = [normalize_yaw_deg(float(box[6])) for box in boxes]
            yaw_labels = [
                interval_name(value, args.yaw_bins, 'deg')
                for value in yaw_degrees
            ]

            # distortion-aware 像素框只在固定帧子样本上计算；同一帧的边投影被
            # 所有 crop offset 复用。严格 keep mask 仍覆盖上面的全部帧。
            metric_gt_indices = eval_indices if collect_metrics else np.zeros(0, dtype=np.int64)
            if len(metric_gt_indices):
                metric_edge_uv, metric_edge_depth = project_box_edges_distorted(
                    boxes[metric_gt_indices], cam_info, args.samples_per_edge)
            else:
                metric_edge_uv = np.zeros(
                    (0, len(BOX_EDGES) * args.samples_per_edge, 2), dtype=np.float32)
                metric_edge_depth = np.zeros(
                    (0, len(BOX_EDGES) * args.samples_per_edge), dtype=np.float32)

            for contract in contracts:
                transformed_visible = (
                    projected_points_visible(
                        center_uv, center_depth, target_shape,
                        args.min_depth, contract) |
                    projected_points_visible(
                        corner_uv, corner_depth, target_shape,
                        args.min_depth, contract
                    ).reshape(len(boxes), -1).any(axis=1)
                )
                train_keep = eval_keep & transformed_visible
                counter = visibility[(split, contract['name'])]
                counter['eval_keep'] += int(eval_keep.sum())
                counter['train_keep'] += int(train_keep.sum())
                counter['eval_only'] += int((eval_keep & ~train_keep).sum())
                counter['train_only'] += int((train_keep & ~eval_keep).sum())

                for gt_index in eval_indices:
                    class_name = class_labels[gt_index]
                    distance_bin = distance_labels[gt_index]
                    yaw_bin = yaw_labels[gt_index]
                    keep_after_crop = bool(train_keep[gt_index])
                    group_population['strategy={}'.format(contract['name'])] += 1
                    group_population[
                        'strategy={}|class={}'.format(
                            contract['name'], class_name)] += 1
                    group_population[
                        'strategy={}|distance={}'.format(
                            contract['name'], distance_bin)] += 1
                    group_population[
                        'strategy={}|class={}|distance={}'.format(
                            contract['name'], class_name, distance_bin)] += 1
                    yaw_counts[(
                        contract['name'], class_name, distance_bin,
                        yaw_bin, keep_after_crop)] += 1

                for metric_index, gt_index in enumerate(metric_gt_indices):
                    box = boxes[gt_index]
                    try:
                        pixel = projection_record_from_edges(
                            metric_edge_uv[metric_index],
                            metric_edge_depth[metric_index],
                            contract, target_shape, args.min_depth)
                    except ValueError:
                        counter['projection_without_depth'] += 1
                        continue
                    center_x = float(box[0])
                    yaw_deg = normalize_yaw_deg(float(box[6]))
                    records.append(dict(
                        split=split,
                        strategy=contract['name'],
                        token=str(info.get('token', info_index)),
                        info_index=int(info_index),
                        gt_index=int(gt_index),
                        class_name=str(names[gt_index]),
                        center_x_m=center_x,
                        center_y_m=float(box[1]),
                        distance_bin=distance_labels[gt_index],
                        yaw_deg=yaw_deg,
                        yaw_bin=yaw_labels[gt_index],
                        eval_keep=True,
                        train_keep=bool(train_keep[gt_index]),
                        **pixel))

            if log_every > 0 and visibility_frames % log_every == 0:
                elapsed = max(time.monotonic() - started_at, 1e-6)
                print(
                    'F4_PROGRESS frames={} split={} split_frame={}/{} '
                    'metric_frames={} records={} rate={:.2f}_frames_s'.format(
                        visibility_frames, split, selected_index + 1,
                        len(selected_infos), metric_frames, len(records),
                        visibility_frames / elapsed),
                    flush=True)

    groups = defaultdict(list)
    for row in records:
        keys = (
            'strategy={}'.format(row['strategy']),
            'strategy={}|class={}'.format(row['strategy'], row['class_name']),
            'strategy={}|distance={}'.format(row['strategy'], row['distance_bin']),
            'strategy={}|class={}|distance={}'.format(
                row['strategy'], row['class_name'], row['distance_bin']),
        )
        for key in keys:
            groups[key].append(row)

    visibility_rows = []
    f4_required = False
    for (split, strategy), counter in sorted(visibility.items()):
        eval_keep = int(counter['eval_keep'])
        eval_only = int(counter['eval_only'])
        ratio = eval_only / float(eval_keep) if eval_keep else 0.0
        gated = strategy == gate_strategy
        exceeds_limit = ratio > float(args.max_eval_only_ratio)
        requires_review = gated and exceeds_limit
        f4_required |= requires_review
        visibility_rows.append(dict(
            split=split,
            strategy=strategy,
            gated=gated,
            eval_keep=eval_keep,
            train_keep=int(counter['train_keep']),
            eval_only=eval_only,
            train_only=int(counter['train_only']),
            eval_only_ratio=ratio,
            max_eval_only_ratio=float(args.max_eval_only_ratio),
            status=(
                'F4_REVIEW_REQUIRED' if requires_review else
                'INFO_DIFFERENCE' if exceeds_limit else 'PASS')))

    summary = dict(
        schema_version=1,
        status='F4_REVIEW_REQUIRED' if f4_required else 'PASS',
        source_image_size=list(source_shape),
        target_image_size=list(target_shape),
        gate_strategy=gate_strategy,
        contracts=contracts,
        pkl_assets=pkl_assets,
        visibility_frames=visibility_frames,
        metric_frame_stride=metric_stride,
        metric_frames=metric_frames,
        metric_records=len(records),
        population_records=sum(
            count for key, count in group_population.items()
            if '|' not in key),
        pixel_statistics='deterministic_frame_stride_sample',
        visibility=visibility_rows,
        groups={
            key: summarize_records(groups.get(key, []), population_count=count)
            for key, count in sorted(group_population.items())
        },
        yaw_counts=[
            dict(
                strategy=key[0], class_name=key[1], distance_bin=key[2],
                yaw_bin=key[3], train_keep=key[4], count=count)
            for key, count in sorted(yaw_counts.items())
        ],
    )
    selection = build_sample_selection(records, args.selection_count)
    return records, summary, selection


def build_sample_selection(records: Sequence[Mapping], count: int) -> Dict:
    center_records = [row for row in records if row['strategy'] == 'scale_crop_top_70']
    categories = dict(
        far=sorted(
            [row for row in center_records if row['center_x_m'] >= 40.0],
            key=lambda row: (-row['center_x_m'], row['token'])),
        image_edge=sorted(
            center_records,
            key=lambda row: (row['min_edge_margin_px'], row['token'])),
        nonzero_yaw=sorted(
            [row for row in center_records if abs(row['yaw_deg']) >= 30.0],
            key=lambda row: (-abs(row['yaw_deg']), row['token'])),
        crop_dropped=sorted(
            [row for row in center_records if not row['train_keep']],
            key=lambda row: (row['min_edge_margin_px'], row['token'])),
    )
    fields = (
        'split', 'token', 'info_index', 'gt_index', 'class_name',
        'center_x_m', 'center_y_m', 'yaw_deg', 'min_edge_margin_px',
        'train_keep')
    return {
        name: [
            {field: row[field] for field in fields}
            for row in values[:count]
        ]
        for name, values in categories.items()
    }


def write_outputs(
    records: Sequence[Mapping],
    summary: Mapping,
    selection: Mapping,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if records:
        with (output_dir / 'strategy_records.csv').open(
                'w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output_dir / 'scale_crop_geometry.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    (output_dir / 'sample_selection.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')

    lines = [
        '# N7 GEOM1-A resize/crop 几何统计',
        '',
        '- status: `{}`'.format(summary['status']),
        '- gate strategy: `{}`'.format(summary['gate_strategy']),
        '- input: `1600x900 -> 704x396 -> 704x256`',
        '- visibility frames: `{}`（全量 keep/yaw 统计）'.format(
            summary['visibility_frames']),
        '- pixel metric frames: `{}`（固定步长 `{}`，records `{}`）'.format(
            summary['metric_frames'], summary['metric_frame_stride'],
            summary['metric_records']),
        '',
        '## Train/Eval GT keep mask',
        '',
        '| split | strategy | gated | eval keep | train keep | eval-only | ratio | status |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in summary['visibility']:
        lines.append(
            '| {split} | {strategy} | {gated} | {eval_keep} | {train_keep} | '
            '{eval_only} | {eval_only_ratio:.4%} | {status} |'.format(**row))
    lines.extend([
        '',
        '## 分桶像素统计',
        '',
        '| group | population/sample | input w p10/p50 | input h p10/p50 | '
        'stride4 w/h p50 | crop ratio p90 | edge margin p10 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for name, group in summary['groups'].items():
        def value(field, quantile):
            return group.get(field, {}).get(quantile, float('nan'))
        lines.append(
            '| {} | {}/{} | {:.2f}/{:.2f} | {:.2f}/{:.2f} | '
            '{:.2f}/{:.2f} | {:.2%} | {:.2f} |'.format(
                name, group['count'], group['sample_count'],
                value('input_width_px', 'p10'), value('input_width_px', 'p50'),
                value('input_height_px', 'p10'), value('input_height_px', 'p50'),
                value('stride4_width_px', 'p50'), value('stride4_height_px', 'p50'),
                value('cropped_area_ratio', 'p90'),
                value('min_edge_margin_px', 'p10')))
    (output_dir / 'scale_crop_geometry.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pkl', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera-id', default='cam0')
    parser.add_argument('--classes', nargs='+', default=['car', 'truck'])
    parser.add_argument('--roi', type=float, nargs=6,
                        default=[0, -35, -5, 80, 35, 3])
    parser.add_argument('--source-image-size', type=int, nargs=2,
                        metavar=('HEIGHT', 'WIDTH'), default=(900, 1600))
    parser.add_argument('--target-image-size', type=int, nargs=2,
                        metavar=('HEIGHT', 'WIDTH'), default=(256, 704))
    parser.add_argument('--crop-top-offsets', type=int, nargs='+',
                        default=[50, 60, 70, 80, 90])
    parser.add_argument(
        '--gate-strategy', default='scale_crop_top_70',
        help='决定全局 F4 严格状态的主候选；其他 offset 仅作统计对照')
    parser.add_argument('--distance-bins', type=float, nargs='+',
                        default=[0, 20, 40, 60, 80])
    parser.add_argument('--yaw-bins', type=float, nargs='+',
                        default=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
    parser.add_argument('--samples-per-edge', type=int, default=17)
    parser.add_argument('--min-depth', type=float, default=0.1)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=-1)
    parser.add_argument(
        '--metric-stride', type=int, default=100,
        help='像素尺寸/裁边分位数每多少个已分析帧取一帧；keep mask 和 yaw 始终全量')
    parser.add_argument(
        '--log-every', type=int, default=5000,
        help='每处理多少帧打印一次进度；0 表示关闭')
    parser.add_argument('--selection-count', type=int, default=20)
    parser.add_argument(
        '--max-eval-only-ratio', type=float, default=0.0,
        help='超过该比例标记 F4_REVIEW_REQUIRED；默认要求 train/eval 精确一致')
    parser.add_argument('--strict-visibility', action='store_true')
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if (args.start_index < 0 or args.stride <= 0 or
            args.metric_stride <= 0 or args.log_every < 0):
        raise ValueError(
            'start-index/log-every must be >=0; stride and metric-stride must be positive')
    if args.samples_per_edge < 2:
        raise ValueError('--samples-per-edge must be >=2')
    if not 0.0 <= args.max_eval_only_ratio <= 1.0:
        raise ValueError('--max-eval-only-ratio must be in [0,1]')
    for values, name in (
            (args.distance_bins, 'distance-bins'),
            (args.yaw_bins, 'yaw-bins')):
        if len(values) < 2 or any(b <= a for a, b in zip(values[:-1], values[1:])):
            raise ValueError('--{} must be strictly increasing'.format(name))
    for path in args.pkl:
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    records, summary, selection = analyze(args)
    write_outputs(records, summary, selection, args.output_dir)
    print(
        'N7_SCALE_CROP_GEOMETRY={} visibility_frames={} '
        'metric_frames={} records={}'.format(
            summary['status'], summary['visibility_frames'],
            summary['metric_frames'], len(records)))
    if args.strict_visibility and summary['status'] != 'PASS':
        sys.exit(2)


if __name__ == '__main__':
    main()
