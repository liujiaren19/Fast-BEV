#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用真实 RandomAugImageMultiViewImage 渲染 GEOM1-A 定向样本。

输入 ``analyze_n7_scale_crop_geometry.py`` 生成的 sample_selection.json，
读取原生 1600x900 图片，通过 resolved config 中的真实训练/测试图像变换，
再用同一 post_rot/post_tran 叠加 distortion-aware GT 边线。该工具不会写回
PKL 或源图片。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import cv2
import numpy as np

try:
    from tools.data_converter.n7.analyze_n7_scale_crop_geometry import (
        BOX_EDGES,
        dataset_box_corners,
        sample_box_edges,
    )
    from tools.data_converter.n7.visualize_n7_fastbev_pkl import (
        camera_info_for_pkl_image_size,
        load_fastbev_pkl,
        project_lidar_points_distorted,
    )
except ModuleNotFoundError:
    from analyze_n7_scale_crop_geometry import (
        BOX_EDGES,
        dataset_box_corners,
        sample_box_edges,
    )
    from visualize_n7_fastbev_pkl import (
        camera_info_for_pkl_image_size,
        load_fastbev_pkl,
        project_lidar_points_distorted,
    )


CATEGORY_COLORS = {
    'far': (0, 220, 255),
    'image_edge': (255, 180, 0),
    'nonzero_yaw': (80, 255, 80),
    'crop_dropped': (60, 60, 255),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_step(pipeline: Sequence[Mapping], step_type: str) -> Mapping:
    for step in pipeline:
        if isinstance(step, Mapping) and step.get('type') == step_type:
            return step
    raise KeyError('{} not found in pipeline'.format(step_type))


def camera_aug_from_info(cam_info: Mapping) -> Dict:
    return dict(
        intrin=np.asarray(cam_info['cam_intrinsic'], dtype=np.float32),
        rot=np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float32),
        tran=np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float32),
        post_rot=np.eye(3, dtype=np.float32),
        post_tran=np.zeros(3, dtype=np.float32),
        distortion=np.asarray(cam_info.get('distortion', []), dtype=np.float32),
        intrinsic_width=int(cam_info['intrinsic_width']),
        intrinsic_height=int(cam_info['intrinsic_height']),
        image_width=int(cam_info['image_width']),
        image_height=int(cam_info['image_height']))


def transformed_edge_uv(
    box: np.ndarray,
    cam_info: Mapping,
    post_rot: np.ndarray,
    post_tran: np.ndarray,
    samples_per_edge: int,
) -> np.ndarray:
    scaled_cam, _ = camera_info_for_pkl_image_size(dict(cam_info))
    intrinsic = np.asarray(scaled_cam['cam_intrinsic'], dtype=np.float32)
    corners = dataset_box_corners(box.reshape(1, -1))[0]
    points = sample_box_edges(corners, samples_per_edge)
    uv, depth = project_lidar_points_distorted(points, scaled_cam, intrinsic)
    uv_h = np.concatenate([
        uv, np.ones((len(uv), 1), dtype=np.float32)
    ], axis=1)
    transformed = (np.asarray(post_rot, dtype=np.float32) @ uv_h.T).T
    transformed += np.asarray(post_tran, dtype=np.float32).reshape(1, 3)
    transformed[depth <= 0.1] = np.nan
    return transformed[:, :2].reshape(len(BOX_EDGES), samples_per_edge, 2)


def draw_edges(image: np.ndarray, edge_uv: np.ndarray, color) -> None:
    height, width = image.shape[:2]
    for samples in edge_uv:
        for left, right in zip(samples[:-1], samples[1:]):
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                continue
            p0 = tuple(np.round(np.clip(left, -100000, 100000)).astype(int))
            p1 = tuple(np.round(np.clip(right, -100000, 100000)).astype(int))
            ok, q0, q1 = cv2.clipLine((0, 0, width, height), p0, p1)
            if ok:
                cv2.line(image, q0, q1, color, 1, cv2.LINE_AA)


def resolve_image_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def run(args: argparse.Namespace) -> Dict:
    # legacy 依赖延迟导入，使本地仍可执行 --help/py_compile。
    from mmcv import Config
    from mmdet3d.datasets.pipelines.transforms_3d import (
        RandomAugImageMultiViewImage,
    )

    config = Config.fromfile(str(args.config))
    pipeline = config.data[args.split].pipeline
    step = dict(find_step(pipeline, 'RandomAugImageMultiViewImage'))
    transform = RandomAugImageMultiViewImage(
        data_config=dict(step['data_config']),
        is_train=bool(step.get('is_train', args.split == 'train')),
        force_resize=bool(step.get('force_resize', False)),
        enable_random_aug=bool(step.get('enable_random_aug', False)),
        n_images=int(step.get('n_images', 1)))
    if transform.force_resize or transform.enable_random_aug:
        raise ValueError(
            'GEOM1-A requires force_resize=False and enable_random_aug=False')

    infos, metadata = load_fastbev_pkl(args.pkl)
    by_token = {str(info.get('token')): info for info in infos}
    selection = json.loads(args.selection_json.read_text(encoding='utf-8'))
    data_root = args.data_root or Path(config.data[args.split].data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    for category in args.categories:
        for row in selection.get(category, [])[:args.max_per_category]:
            token = str(row['token'])
            gt_index = int(row['gt_index'])
            identity = (category, token, gt_index)
            if identity in seen:
                continue
            seen.add(identity)
            info = by_token.get(token)
            if info is None:
                raise KeyError('selection token missing in pkl: {}'.format(token))
            cam_info = info['cams'][args.camera_id]
            image_path = resolve_image_path(data_root, str(cam_info['data_path']))
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            results = dict(
                img=[image],
                view_layout=dict(n_images=1, n_times=1, sequential=False),
                lidar2img=dict(
                    lidar2img_aug=[camera_aug_from_info(cam_info)],
                    extrinsic=[np.eye(4, dtype=np.float32)]))
            output = transform(results)
            rendered = np.asarray(output['img'][0]).copy()
            boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32)
            if gt_index >= len(boxes):
                raise IndexError('token={} gt_index={}'.format(token, gt_index))
            camera_aug = output['lidar2img']['lidar2img_aug'][0]
            edge_uv = transformed_edge_uv(
                boxes[gt_index, :7], cam_info,
                camera_aug['post_rot'], camera_aug['post_tran'],
                args.samples_per_edge)
            color = CATEGORY_COLORS.get(category, (255, 255, 255))
            draw_edges(rendered, edge_uv, color)
            label = '{} {} x={:.1f}m yaw={:.1f}deg keep={}'.format(
                category, row['class_name'], row['center_x_m'],
                row['yaw_deg'], row['train_keep'])
            cv2.putText(
                rendered, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
            output_path = (
                args.output_dir / category /
                '{}_gt{}.jpg'.format(token.replace('/', '_'), gt_index))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), rendered):
                raise IOError('failed to write {}'.format(output_path))
            records.append(dict(
                category=category,
                token=token,
                gt_index=gt_index,
                source_path=str(image_path),
                output_path=str(output_path),
                output_sha256=file_sha256(output_path),
                image_shape=list(rendered.shape),
                post_rot=np.asarray(camera_aug['post_rot']).tolist(),
                post_tran=np.asarray(camera_aug['post_tran']).tolist()))
    report = dict(
        schema_version=1,
        status='PASS',
        config=str(args.config),
        pkl=str(args.pkl),
        pkl_sha256=file_sha256(args.pkl),
        pkl_split=metadata.get('set'),
        transform=dict(force_resize=False, enable_random_aug=False),
        rendered=len(records),
        records=records)
    (args.output_dir / 'pipeline_visualization.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    parser.add_argument('--selection-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--camera-id', default='cam0')
    parser.add_argument('--categories', nargs='+',
                        default=['far', 'image_edge', 'nonzero_yaw', 'crop_dropped'])
    parser.add_argument('--max-per-category', type=int, default=8)
    parser.add_argument('--samples-per-edge', type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in (args.config, args.pkl, args.selection_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = run(args)
    print('N7_SCALE_CROP_PIPELINE_VIS=PASS rendered={}'.format(
        report['rendered']))


if __name__ == '__main__':
    main()
