#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 N7 mono-front GT 在 704x256 网络输入上的尺寸和边缘分布。

该工具只做只读诊断，不修改 pkl 或训练过滤逻辑。目标筛选先复现当前
S0/B0 的类别、前视 ROI 和 cam0 pinhole-visible 口径；对筛选后的目标再用
pkl distortion 投影 3D box 边采样点，从而观察模型实际畸变输入坐标系中的
目标像素宽高、裁边比例和边缘余量。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from tools.data_converter.n7.visualize_n7_fastbev_pkl import (
        camera_info_for_pkl_image_size,
        corners_from_boxes,
        gt_used_mask,
        load_fastbev_pkl,
        project_lidar_points_distorted,
    )
except ModuleNotFoundError:
    from visualize_n7_fastbev_pkl import (
        camera_info_for_pkl_image_size,
        corners_from_boxes,
        gt_used_mask,
        load_fastbev_pkl,
        project_lidar_points_distorted,
    )


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
RECORD_FIELDS = (
    'info_index', 'token', 'class_name', 'gt_index', 'center_x_m',
    'center_y_m', 'distance_xy_m', 'distance_bin', 'input_width',
    'input_height', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
    'bbox_width_px', 'bbox_height_px', 'bbox_area_px2',
    'clipped_width_px', 'clipped_height_px', 'clipped_area_px2',
    'visible_area_ratio', 'edge_margin_px', 'crosses_image_edge',
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def distance_bin_name(value: float, bins: Sequence[float]) -> str:
    if value < bins[0]:
        return '<{}m'.format(int(bins[0]))
    for left, right in zip(bins[:-1], bins[1:]):
        if left <= value < right:
            return '{}-{}m'.format(int(left), int(right))
    return '>={}m'.format(int(bins[-1]))


def sample_box_edges(corners: np.ndarray, samples_per_edge: int) -> np.ndarray:
    weights = np.linspace(
        0.0, 1.0, int(samples_per_edge), dtype=np.float32)
    points = []
    for start, end in BOX_EDGES:
        points.append(
            corners[start][None] * (1.0 - weights[:, None]) +
            corners[end][None] * weights[:, None])
    return np.concatenate(points, axis=0)


def projected_box_record(
    box: np.ndarray,
    cam_info: Mapping,
    samples_per_edge: int,
    min_depth: float,
) -> Dict[str, float]:
    scaled_cam, image_shape = camera_info_for_pkl_image_size(dict(cam_info))
    intrinsic = np.asarray(scaled_cam['cam_intrinsic'], dtype=np.float32)
    corners = corners_from_boxes(
        np.asarray(box, dtype=np.float32).reshape(1, -1))[0]
    points = sample_box_edges(corners, samples_per_edge)
    uv, depth = project_lidar_points_distorted(
        points, scaled_cam, intrinsic)
    valid = (
        (depth > float(min_depth)) &
        np.isfinite(uv[:, 0]) &
        np.isfinite(uv[:, 1])
    )
    if not np.any(valid):
        raise ValueError('box has no finite projected edge point in front')

    uv = uv[valid].astype(np.float64)
    x1, y1 = np.min(uv, axis=0)
    x2, y2 = np.max(uv, axis=0)
    height, width = [int(value) for value in image_shape]
    box_width = max(0.0, float(x2 - x1))
    box_height = max(0.0, float(y2 - y1))
    full_area = box_width * box_height
    clipped_x1 = float(np.clip(x1, 0.0, float(width)))
    clipped_y1 = float(np.clip(y1, 0.0, float(height)))
    clipped_x2 = float(np.clip(x2, 0.0, float(width)))
    clipped_y2 = float(np.clip(y2, 0.0, float(height)))
    clipped_width = max(0.0, clipped_x2 - clipped_x1)
    clipped_height = max(0.0, clipped_y2 - clipped_y1)
    clipped_area = clipped_width * clipped_height
    edge_margin = min(
        float(x1),
        float(y1),
        float(width) - float(x2),
        float(height) - float(y2),
    )
    return dict(
        input_width=width,
        input_height=height,
        bbox_x1=float(x1),
        bbox_y1=float(y1),
        bbox_x2=float(x2),
        bbox_y2=float(y2),
        bbox_width_px=box_width,
        bbox_height_px=box_height,
        bbox_area_px2=full_area,
        clipped_width_px=clipped_width,
        clipped_height_px=clipped_height,
        clipped_area_px2=clipped_area,
        visible_area_ratio=(
            clipped_area / full_area if full_area > 0.0 else 0.0),
        edge_margin_px=edge_margin,
        crosses_image_edge=bool(edge_margin < 0.0),
    )


def quantile_dict(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        'p{:02d}'.format(int(round(level * 100))): float(
            np.quantile(array, level))
        for level in QUANTILES
    }


def summarize_group(
    records: Sequence[Mapping],
    edge_thresholds: Sequence[float],
) -> Dict:
    count = len(records)
    summary = {'count': count}
    if count == 0:
        return summary
    for field in (
        'bbox_width_px',
        'bbox_height_px',
        'bbox_area_px2',
        'visible_area_ratio',
        'edge_margin_px',
    ):
        summary[field] = quantile_dict(row[field] for row in records)
    summary['crosses_image_edge_ratio'] = float(np.mean([
        bool(row['crosses_image_edge']) for row in records]))
    summary['small_target_ratio'] = {
        'width_lt_8px': float(np.mean([
            row['bbox_width_px'] < 8.0 for row in records])),
        'width_lt_16px': float(np.mean([
            row['bbox_width_px'] < 16.0 for row in records])),
        'width_lt_24px': float(np.mean([
            row['bbox_width_px'] < 24.0 for row in records])),
        'height_lt_4px': float(np.mean([
            row['bbox_height_px'] < 4.0 for row in records])),
        'height_lt_8px': float(np.mean([
            row['bbox_height_px'] < 8.0 for row in records])),
        'height_lt_12px': float(np.mean([
            row['bbox_height_px'] < 12.0 for row in records])),
    }
    summary['edge_margin_le_ratio'] = {
        '{:g}px'.format(float(threshold)): float(np.mean([
            row['edge_margin_px'] <= float(threshold)
            for row in records
        ]))
        for threshold in edge_thresholds
    }
    return summary


def build_summary(
    records: Sequence[Mapping],
    counters: Mapping[str, int],
    args: argparse.Namespace,
) -> Dict:
    groups = defaultdict(list)
    for row in records:
        groups['all'].append(row)
        groups['class={}'.format(row['class_name'])].append(row)
        groups['distance={}'.format(row['distance_bin'])].append(row)
        groups[
            'class={}|distance={}'.format(
                row['class_name'], row['distance_bin'])
        ].append(row)
    return dict(
        schema_version=1,
        pkl=str(Path(args.pkl)),
        pkl_sha256=file_sha256(Path(args.pkl)),
        camera_id=args.camera_id,
        class_names=list(args.classes),
        roi=[float(value) for value in args.roi],
        distance_bins=[float(value) for value in args.distance_bins],
        edge_thresholds=[float(value) for value in args.edge_thresholds],
        samples_per_edge=int(args.samples_per_edge),
        projection=(
            'current pinhole-visible GT filter followed by distortion-aware '
            '3D box edge projection in pkl image/input coordinates'),
        counters={key: int(value) for key, value in counters.items()},
        groups={
            key: summarize_group(value, args.edge_thresholds)
            for key, value in sorted(groups.items())
        },
    )


def write_outputs(
    records: Sequence[Mapping],
    summary: Mapping,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'target_records.csv').open(
            'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    lines = [
        '# N7 输入图目标尺寸与边缘分布',
        '',
        '- pkl: `{}`'.format(summary['pkl']),
        '- camera: `{}`'.format(summary['camera_id']),
        '- projected targets: `{}`'.format(
            summary['counters'].get('projected_targets', 0)),
        '',
        '| group | count | width p10/p50 | height p10/p50 | '
        'edge margin p10/p50 | crosses edge |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, group in summary['groups'].items():
        width = group.get('bbox_width_px', {})
        height = group.get('bbox_height_px', {})
        margin = group.get('edge_margin_px', {})
        lines.append(
            '| {} | {} | {:.2f}/{:.2f} | {:.2f}/{:.2f} | '
            '{:.2f}/{:.2f} | {:.2%} |'.format(
                name,
                group['count'],
                width.get('p10', float('nan')),
                width.get('p50', float('nan')),
                height.get('p10', float('nan')),
                height.get('p50', float('nan')),
                margin.get('p10', float('nan')),
                margin.get('p50', float('nan')),
                group.get('crosses_image_edge_ratio', float('nan')),
            ))
    (output_dir / 'summary.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')


def analyze(args: argparse.Namespace) -> Tuple[List[Dict], Dict]:
    infos, _ = load_fastbev_pkl(Path(args.pkl))
    records = []
    counters = defaultdict(int)
    selected_infos = infos[args.start_index::args.stride]
    if args.max_frames >= 0:
        selected_infos = selected_infos[:args.max_frames]
    for local_index, info in enumerate(selected_infos):
        info_index = args.start_index + local_index * args.stride
        counters['frames'] += 1
        boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32)
        names = np.asarray(info.get('gt_names', []), dtype=object).reshape(-1)
        if boxes.size == 0:
            continue
        boxes = boxes.reshape(-1, boxes.shape[-1])
        count = min(len(boxes), len(names))
        boxes = boxes[:count, :7]
        names = names[:count]
        counters['raw_targets'] += count
        cams = info.get('cams', {}) or {}
        if args.camera_id not in cams:
            counters['frames_missing_camera'] += 1
            continue
        used = gt_used_mask(
            info=info,
            boxes=boxes,
            names=names,
            class_names=args.classes,
            gt_filter_range=args.roi,
            visible_camera_ids=[args.camera_id],
            min_depth=args.min_depth,
        )
        counters['used_targets'] += int(np.sum(used))
        for gt_index in np.flatnonzero(used):
            box = boxes[gt_index]
            try:
                projection = projected_box_record(
                    box=box,
                    cam_info=cams[args.camera_id],
                    samples_per_edge=args.samples_per_edge,
                    min_depth=args.min_depth,
                )
            except ValueError:
                counters['projection_without_valid_depth'] += 1
                continue
            center_x = float(box[0])
            center_y = float(box[1])
            distance_xy = float(np.hypot(center_x, center_y))
            record = dict(
                info_index=info_index,
                token=str(info.get('token', info_index)),
                class_name=str(names[gt_index]),
                gt_index=int(gt_index),
                center_x_m=center_x,
                center_y_m=center_y,
                distance_xy_m=distance_xy,
                distance_bin=distance_bin_name(
                    center_x, args.distance_bins),
                **projection,
            )
            records.append(record)
            counters['projected_targets'] += 1
    summary = build_summary(records, counters, args)
    return records, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--pkl', required=True, help='N7 Fast-BEV train/val pkl')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='写入 target_records.csv、summary.json 和 summary.md',
    )
    parser.add_argument('--camera-id', default='cam0')
    parser.add_argument('--classes', nargs='+', default=['car', 'truck'])
    parser.add_argument(
        '--roi',
        nargs=6,
        type=float,
        default=[0, -35, -5, 80, 35, 3],
    )
    parser.add_argument(
        '--distance-bins',
        nargs='+',
        type=float,
        default=[0, 20, 40, 60, 80],
        help='按 GT center x 分桶，与当前 eval 距离表一致',
    )
    parser.add_argument(
        '--edge-thresholds',
        nargs='+',
        type=float,
        default=[0, 4, 8, 16, 32],
    )
    parser.add_argument('--samples-per-edge', type=int, default=17)
    parser.add_argument('--min-depth', type=float, default=0.1)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=-1)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise ValueError('--start-index must be >= 0')
    if args.stride <= 0:
        raise ValueError('--stride must be positive')
    if args.samples_per_edge < 2:
        raise ValueError('--samples-per-edge must be >= 2')
    if len(args.distance_bins) < 2:
        raise ValueError('--distance-bins needs at least two values')
    if any(
            right <= left
            for left, right in zip(
                args.distance_bins[:-1], args.distance_bins[1:])):
        raise ValueError('--distance-bins must be strictly increasing')


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    records, summary = analyze(args)
    write_outputs(records, summary, Path(args.output_dir))
    print(json.dumps(summary['counters'], ensure_ascii=False, sort_keys=True))
    print('N7_IMAGE_AUG_TARGET_STATS=PASS records={}'.format(len(records)))


if __name__ == '__main__':
    main()
