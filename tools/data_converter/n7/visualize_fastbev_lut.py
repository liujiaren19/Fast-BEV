#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化 Fast-BEV 2D->3D LUT npy/bin 文件。

输入通常是 ``build_fastbev_lut.py`` 的单个 seq 输出目录。脚本会：
- 用不同颜色显示 6 个相机在 BEV 网格上的最终覆盖关系；
- 读取板端 ``LUT/*.bin`` 后重建相机覆盖关系，并和 npy 结果对比；
- 按 bin 中的 gather index 反查 2D feature 平面，显示每个相机哪些 feature
  像素会被当前 Z 层使用；
- 如果存在 ``features.npy`` 和 ``volume.npy``，用 bin 重建 volume 并检查误差。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = np.asarray([
    [230, 57, 70],    # cam0 red
    [42, 157, 143],   # cam1 teal
    [69, 123, 157],   # cam2 blue
    [244, 162, 97],   # cam3 orange
    [156, 92, 200],   # cam4 purple
    [233, 196, 106],  # cam5 yellow
    [38, 70, 83],
    [138, 201, 38],
], dtype=np.uint8)


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def parse_z_slices(text: str, z_count: int) -> List[int]:
    if text == 'all':
        return list(range(z_count))
    values = [int(item) for item in text.split(',') if item.strip()]
    for value in values:
        if not (0 <= value < z_count):
            raise ValueError(f'z slice {value} 超出范围 0..{z_count - 1}')
    return values


def camera_names_from_metadata(metadata: Dict, n_images: int) -> List[str]:
    seq_names = metadata.get('seq_camera_ids') or metadata.get('camera_types') or []
    names = [str(item) for item in seq_names[:n_images]]
    names.extend([f'cam{i}' for i in range(len(names), n_images)])
    return names


def load_choice_3d(lut_dir: Path, metadata: Dict) -> np.ndarray:
    path = lut_dir / 'camera_choice_3d.npy'
    if path.exists():
        return np.load(path)
    flat = np.load(lut_dir / 'camera_choice.npy')
    n_voxels = metadata.get('n_voxels')
    if not n_voxels:
        raise KeyError('metadata.json 缺少 n_voxels，无法 reshape camera_choice.npy')
    return flat.reshape([int(v) for v in n_voxels])


def load_valid_5d(lut_dir: Path, metadata: Dict) -> np.ndarray:
    path = lut_dir / 'valid_5d.npy'
    if path.exists():
        return np.load(path)
    valid = np.load(lut_dir / 'valid.npy')
    n_voxels = metadata.get('n_voxels')
    if not n_voxels:
        raise KeyError('metadata.json 缺少 n_voxels，无法 reshape valid.npy')
    return valid.reshape(valid.shape[0], *[int(v) for v in n_voxels])


def colorize_choice(choice_xy: np.ndarray, palette: np.ndarray) -> np.ndarray:
    display = choice_xy.T
    image = np.full((display.shape[0], display.shape[1], 3), 18, dtype=np.uint8)
    for cam_id in range(palette.shape[0]):
        image[display == cam_id] = palette[cam_id]
    image[display < 0] = np.asarray([35, 35, 35], dtype=np.uint8)
    return image


def colorize_mask(mask_xy: np.ndarray, color: np.ndarray) -> np.ndarray:
    display = mask_xy.T
    image = np.full((display.shape[0], display.shape[1], 3), 25, dtype=np.uint8)
    image[display] = color
    return image


def upscale(image: np.ndarray, cell_size: int) -> Image.Image:
    pil = Image.fromarray(image)
    if cell_size <= 1:
        return pil
    return pil.resize((pil.width * cell_size, pil.height * cell_size), Image.Resampling.NEAREST)


def draw_title_and_legend(
    image: Image.Image,
    title: str,
    camera_names: Sequence[str],
    palette: np.ndarray,
) -> Image.Image:
    font = ImageFont.load_default()
    top = 24
    legend_w = 190
    canvas = Image.new('RGB', (image.width + legend_w, image.height + top), (245, 245, 245))
    canvas.paste(image, (0, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(0, 0, 0), font=font)
    x0 = image.width + 10
    y0 = top + 8
    for cam_id, name in enumerate(camera_names):
        color = tuple(int(v) for v in palette[cam_id % len(palette)])
        draw.rectangle((x0, y0 + cam_id * 18, x0 + 12, y0 + cam_id * 18 + 12), fill=color)
        draw.text((x0 + 18, y0 + cam_id * 18), f'{cam_id}: {name}', fill=(0, 0, 0), font=font)
    draw.rectangle((x0, y0 + len(camera_names) * 18, x0 + 12, y0 + len(camera_names) * 18 + 12), fill=(35, 35, 35))
    draw.text((x0 + 18, y0 + len(camera_names) * 18), '-1: unassigned', fill=(0, 0, 0), font=font)
    return canvas


def save_choice_image(
    choice_3d: np.ndarray,
    z_id: int,
    path: Path,
    title: str,
    camera_names: Sequence[str],
    cell_size: int,
) -> None:
    image = upscale(colorize_choice(choice_3d[:, :, z_id], PALETTE), cell_size)
    draw_title_and_legend(image, title, camera_names, PALETTE).save(path)


def save_diff_image(npy_choice: np.ndarray, bin_choice: np.ndarray, z_id: int, path: Path, cell_size: int) -> int:
    npy_slice = npy_choice[:, :, z_id].T
    bin_slice = bin_choice[:, :, z_id].T
    image = np.full((npy_slice.shape[0], npy_slice.shape[1], 3), 20, dtype=np.uint8)
    both_empty = (npy_slice < 0) & (bin_slice < 0)
    same = (npy_slice == bin_slice) & ~both_empty
    only_npy = (npy_slice >= 0) & (bin_slice < 0)
    only_bin = (npy_slice < 0) & (bin_slice >= 0)
    mismatch = (npy_slice != bin_slice) & ~both_empty & ~only_npy & ~only_bin
    image[both_empty] = [35, 35, 35]
    image[same] = [65, 170, 95]
    image[only_npy] = [45, 120, 220]
    image[only_bin] = [240, 170, 50]
    image[mismatch] = [220, 40, 40]
    canvas = upscale(image, cell_size)
    font = ImageFont.load_default()
    top = 24
    legend_w = 230
    out = Image.new('RGB', (canvas.width + legend_w, canvas.height + top), (245, 245, 245))
    out.paste(canvas, (0, top))
    draw = ImageDraw.Draw(out)
    draw.text((6, 5), f'bin vs npy diff z={z_id}', fill=(0, 0, 0), font=font)
    legend = [
        ('same', [65, 170, 95]),
        ('both unassigned', [35, 35, 35]),
        ('npy only', [45, 120, 220]),
        ('bin only', [240, 170, 50]),
        ('mismatch', [220, 40, 40]),
    ]
    for idx, (name, color) in enumerate(legend):
        y = top + 8 + idx * 18
        x = canvas.width + 10
        draw.rectangle((x, y, x + 12, y + 12), fill=tuple(color))
        draw.text((x + 18, y), name, fill=(0, 0, 0), font=font)
    out.save(path)
    return int(np.count_nonzero(only_npy | only_bin | mismatch))


def save_per_camera_valid_mosaic(
    valid_5d: np.ndarray,
    z_id: int,
    path: Path,
    camera_names: Sequence[str],
    cell_size: int,
) -> None:
    n_images = valid_5d.shape[0]
    panels = []
    font = ImageFont.load_default()
    for cam_id in range(n_images):
        panel = upscale(colorize_mask(valid_5d[cam_id, :, :, z_id], PALETTE[cam_id % len(PALETTE)]), cell_size)
        canvas = Image.new('RGB', (panel.width, panel.height + 18), (245, 245, 245))
        canvas.paste(panel, (0, 18))
        ImageDraw.Draw(canvas).text((4, 3), f'{cam_id}: {camera_names[cam_id]}', fill=(0, 0, 0), font=font)
        panels.append(canvas)
    cols = min(3, n_images)
    rows = int(np.ceil(n_images / cols))
    w = max(panel.width for panel in panels)
    h = max(panel.height for panel in panels)
    mosaic = Image.new('RGB', (cols * w, rows * h), (245, 245, 245))
    for idx, panel in enumerate(panels):
        mosaic.paste(panel, ((idx % cols) * w, (idx // cols) * h))
    mosaic.save(path)


def load_board_bins(bin_dir: Path, n_images: int, n_points: int) -> Optional[Dict[str, object]]:
    if not bin_dir.exists():
        return None
    scatter_by_cam = []
    gather_by_cam = []
    lengths_path = bin_dir / 'featurePointLength.bin'
    lengths = np.fromfile(lengths_path, dtype=np.int32) if lengths_path.exists() else None
    for cam_id in range(n_images):
        scatter_path = bin_dir / f'scatter_nd_new_{cam_id}.bin'
        gather_path = bin_dir / f'gather_new_{cam_id}.bin'
        if not scatter_path.exists() or not gather_path.exists():
            return None
        scatter = np.fromfile(scatter_path, dtype=np.int32).astype(np.int64)
        gather = np.fromfile(gather_path, dtype=np.int32).astype(np.int64)
        if scatter.shape[0] != gather.shape[0]:
            raise ValueError(f'cam{cam_id} scatter/gather 长度不一致')
        scatter_by_cam.append(scatter)
        gather_by_cam.append(gather)

    choice = np.full(n_points, -1, dtype=np.int64)
    gather_dense = np.full(n_points, -1, dtype=np.int64)
    duplicate_count = 0
    for cam_id, (scatter, gather) in enumerate(zip(scatter_by_cam, gather_by_cam)):
        if np.any(scatter < 0) or np.any(scatter >= n_points):
            raise ValueError(f'cam{cam_id} scatter index 超出 BEV flatten 范围')
        duplicate_count += int(np.count_nonzero(choice[scatter] >= 0))
        choice[scatter] = cam_id
        gather_dense[scatter] = gather
    return dict(
        scatter_by_cam=scatter_by_cam,
        gather_by_cam=gather_by_cam,
        lengths=lengths,
        choice=choice,
        gather_dense=gather_dense,
        duplicate_count=duplicate_count,
    )


def save_feature_hit_mosaic(
    scatter_by_cam: Sequence[np.ndarray],
    gather_by_cam: Sequence[np.ndarray],
    n_voxels: Sequence[int],
    feature_shape: Sequence[int],
    z_id: int,
    path: Path,
    camera_names: Sequence[str],
    cell_size: int,
) -> None:
    n_images, _, feat_h, feat_w = [int(v) for v in feature_shape]
    _, y_count, z_count = [int(v) for v in n_voxels]
    panels = []
    font = ImageFont.load_default()
    for cam_id in range(n_images):
        scatter = scatter_by_cam[cam_id]
        gather = gather_by_cam[cam_id]
        keep = (scatter % z_count) == z_id
        hit = np.zeros((feat_h, feat_w), dtype=np.bool_)
        if keep.any():
            gather_z = gather[keep]
            gy = gather_z // feat_w
            gx = gather_z % feat_w
            valid = (gx >= 0) & (gx < feat_w) & (gy >= 0) & (gy < feat_h)
            hit[gy[valid], gx[valid]] = True
        image = np.full((feat_h, feat_w, 3), 25, dtype=np.uint8)
        image[hit] = PALETTE[cam_id % len(PALETTE)]
        panel = upscale(image, max(1, cell_size))
        canvas = Image.new('RGB', (panel.width, panel.height + 18), (245, 245, 245))
        canvas.paste(panel, (0, 18))
        ImageDraw.Draw(canvas).text(
            (4, 3), f'{cam_id}: {camera_names[cam_id]} hits={int(hit.sum())}',
            fill=(0, 0, 0), font=font)
        panels.append(canvas)

    cols = min(3, n_images)
    rows = int(np.ceil(n_images / cols))
    w = max(panel.width for panel in panels)
    h = max(panel.height for panel in panels)
    mosaic = Image.new('RGB', (cols * w, rows * h), (245, 245, 245))
    for idx, panel in enumerate(panels):
        mosaic.paste(panel, ((idx % cols) * w, (idx // cols) * h))
    mosaic.save(path)


def reconstruct_volume_from_bin(
    features: np.ndarray,
    scatter_by_cam: Sequence[np.ndarray],
    gather_by_cam: Sequence[np.ndarray],
    n_points: int,
) -> np.ndarray:
    n_images, channels, _, _ = features.shape
    volume = np.zeros((channels, n_points), dtype=features.dtype)
    for cam_id in range(n_images):
        scatter = scatter_by_cam[cam_id]
        gather = gather_by_cam[cam_id]
        if scatter.size == 0:
            continue
        feature_flat = features[cam_id].transpose(1, 2, 0).reshape(-1, channels)
        volume[:, scatter] = feature_flat[gather].T
    return volume


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='可视化 Fast-BEV LUT npy/bin 文件')
    parser.add_argument('lut_dir', help='build_fastbev_lut.py 输出的单个 seq 目录')
    parser.add_argument('--bin-dir', default=None, help='板端 bin 目录，默认 lut_dir/LUT')
    parser.add_argument('--out-dir', default=None, help='可视化输出目录，默认 lut_dir/vis')
    parser.add_argument('--z-slices', default='all', help='要绘制的 z 层，如 all 或 0,1,2,3')
    parser.add_argument('--cell-size', type=int, default=3, help='BEV 像素放大倍数')
    parser.add_argument('--feature-cell-size', type=int, default=4, help='feature hitmap 放大倍数')
    parser.add_argument('--no-volume-check', action='store_true', help='不使用 features.npy/volume.npy 做 bin 重建校验')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    lut_dir = Path(args.lut_dir)
    out_dir = Path(args.out_dir) if args.out_dir else lut_dir / 'vis'
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(lut_dir / 'metadata.json')
    choice_3d = load_choice_3d(lut_dir, metadata)
    valid_5d = load_valid_5d(lut_dir, metadata)
    n_voxels = [int(v) for v in metadata.get('n_voxels', list(choice_3d.shape))]
    n_images = int(metadata.get('n_images', valid_5d.shape[0]))
    n_points = int(np.prod(n_voxels))
    camera_names = camera_names_from_metadata(metadata, n_images)
    z_slices = parse_z_slices(args.z_slices, n_voxels[2])

    summary: Dict[str, object] = dict(
        lut_dir=str(lut_dir),
        n_voxels=n_voxels,
        n_images=n_images,
        z_slices=z_slices,
        npy_assigned_points=int(np.count_nonzero(choice_3d.reshape(-1) >= 0)),
    )

    for z_id in z_slices:
        save_choice_image(
            choice_3d, z_id, out_dir / f'npy_camera_choice_z{z_id}.png',
            f'npy final camera choice z={z_id}', camera_names, args.cell_size)
        save_per_camera_valid_mosaic(
            valid_5d, z_id, out_dir / f'npy_per_camera_valid_z{z_id}.png',
            camera_names, args.cell_size)

    bin_dir = Path(args.bin_dir) if args.bin_dir else lut_dir / 'LUT'
    board = load_board_bins(bin_dir, n_images=n_images, n_points=n_points)
    if board is not None:
        bin_choice_3d = board['choice'].reshape(n_voxels)
        summary['bin_dir'] = str(bin_dir)
        summary['bin_duplicate_scatter_count'] = int(board['duplicate_count'])
        summary['bin_feature_point_lengths'] = (
            board['lengths'].astype(int).tolist()
            if isinstance(board.get('lengths'), np.ndarray)
            else [int(x.shape[0]) for x in board['gather_by_cam']])
        summary['bin_assigned_points'] = int(np.count_nonzero(board['choice'] >= 0))
        summary['bin_vs_npy_choice_mismatch'] = int(np.count_nonzero(bin_choice_3d != choice_3d))

        gather_dense_npy = np.load(lut_dir / 'gather_index_dense.npy')
        final_gather = np.full(n_points, -1, dtype=np.int64)
        final_choice = choice_3d.reshape(-1)
        assigned = final_choice >= 0
        final_gather[assigned] = gather_dense_npy[final_choice[assigned], np.nonzero(assigned)[0]]
        gather_compare_mask = (final_gather >= 0) | (board['gather_dense'] >= 0)
        summary['bin_vs_npy_gather_mismatch'] = int(np.count_nonzero(
            final_gather[gather_compare_mask] != board['gather_dense'][gather_compare_mask]))

        feature_shape = metadata.get('feature_shape')
        for z_id in z_slices:
            save_choice_image(
                bin_choice_3d, z_id, out_dir / f'bin_camera_choice_z{z_id}.png',
                f'board bin camera choice z={z_id}', camera_names, args.cell_size)
            diff_count = save_diff_image(
                choice_3d, bin_choice_3d, z_id,
                out_dir / f'bin_vs_npy_diff_z{z_id}.png',
                args.cell_size)
            summary[f'bin_vs_npy_diff_z{z_id}'] = diff_count
            if feature_shape:
                save_feature_hit_mosaic(
                    board['scatter_by_cam'], board['gather_by_cam'],
                    n_voxels=n_voxels,
                    feature_shape=feature_shape,
                    z_id=z_id,
                    path=out_dir / f'bin_feature_hits_z{z_id}.png',
                    camera_names=camera_names,
                    cell_size=args.feature_cell_size)

        if not args.no_volume_check and (lut_dir / 'features.npy').exists() and (lut_dir / 'volume.npy').exists():
            features = np.load(lut_dir / 'features.npy')
            volume = np.load(lut_dir / 'volume.npy')
            reconstructed = reconstruct_volume_from_bin(
                features, board['scatter_by_cam'], board['gather_by_cam'], n_points)
            np.save(out_dir / 'bin_reconstructed_volume.npy', reconstructed)
            diff = np.abs(reconstructed.astype(np.float64) - volume.astype(np.float64))
            summary['bin_volume_max_abs_diff'] = float(diff.max()) if diff.size else 0.0
            summary['bin_volume_nonzero_diff'] = int(np.count_nonzero(diff))
    else:
        summary['bin_dir'] = str(bin_dir)
        summary['bin_status'] = 'not_found'

    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'LUT 可视化已输出到: {out_dir}')


if __name__ == '__main__':
    main()
