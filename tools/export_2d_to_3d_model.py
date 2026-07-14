#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把固定 LUT 的 2D->3D gather/scatter 导出成 ONNX。

推荐输入是 ``tools/data_converter/n7/build_fastbev_lut.py`` 的输出目录。
脚本优先读取板端一致的 ``LUT/gather_new_i.bin``、
``LUT/scatter_nd_new_i.bin`` 和 ``LUT/featurePointLength.bin``。这些 bin
已经按最终相机覆盖关系裁剪过重复 BEV 点，和当前主线
``backproject_inplace`` 的 fused final-camera scatter 语义一致。

输出 ONNX 接收单个样本内同一时序片段的 2D features：
- NCHW: ``[n_images, channels, feat_h, feat_w]``，默认用于 PC 模拟；
- NHWC: ``[n_images, feat_h, feat_w, channels]``，用于对接 NHWC 2D ONNX。

输出为 3D head 的单时序输入 ``[1, channels * z, x, y]``。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


class FixedLUT2Dto3D(nn.Module):
    """固定 LUT 的 2D feature gather/scatter。"""

    def __init__(
        self,
        gather_indices: Sequence[np.ndarray],
        scatter_indices: Sequence[np.ndarray],
        n_voxels: Sequence[int],
        channels: int,
        input_layout: str,
    ) -> None:
        super().__init__()
        self.n_images = len(gather_indices)
        self.channels = int(channels)
        self.n_x, self.n_y, self.n_z = [int(v) for v in n_voxels]
        self.input_layout = input_layout
        self.total_points = self.n_x * self.n_y * self.n_z

        for cam_id, (gather, scatter) in enumerate(zip(gather_indices, scatter_indices)):
            self.register_buffer(f'gather_{cam_id}', torch.as_tensor(gather, dtype=torch.long))
            self.register_buffer(f'scatter_{cam_id}', torch.as_tensor(scatter, dtype=torch.long))

    def forward(self, features):
        if self.input_layout == 'nhwc':
            features = features.permute(0, 3, 1, 2)

        flat_features = features.reshape(self.n_images, self.channels, -1)
        volume = features.new_zeros((self.channels, self.total_points))
        for cam_id in range(self.n_images):
            gather = getattr(self, f'gather_{cam_id}')
            scatter = getattr(self, f'scatter_{cam_id}')
            volume[:, scatter] = flat_features[cam_id, :, gather]

        return volume.reshape(
            1, self.channels, self.n_x, self.n_y, self.n_z
        ).permute(0, 4, 1, 2, 3).reshape(
            1, self.channels * self.n_z, self.n_x, self.n_y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='从固定 LUT 导出 Fast-BEV 2D->3D gather/scatter ONNX')
    parser.add_argument('--lut-dir', required=True, help='build_fastbev_lut.py 输出目录或其中的 LUT 目录')
    parser.add_argument('--out', default='./2d_to_3d.onnx', help='原始 ONNX 输出路径')
    parser.add_argument('--simplified-out', default=None, help='simplify 后 ONNX 输出路径')
    parser.add_argument('--opset-version', type=int, default=13, help='ONNX opset')
    parser.add_argument('--no-simplify', action='store_true', help='不运行 onnxsim')
    parser.add_argument('--verify', action='store_true', help='用 ONNXRuntime 对齐 PyTorch 输出')
    parser.add_argument('--input-name', default='features', help='ONNX 输入名')
    parser.add_argument(
        '--input-layout',
        choices=['nchw', 'nhwc'],
        default='nchw',
        help='输入 feature layout；默认 nchw 兼容 PC 侧 test_onnx')
    parser.add_argument(
        '--feature-shape',
        type=int,
        nargs=4,
        metavar=('N', 'C', 'H', 'W'),
        default=None,
        help='NCHW feature shape；默认从 metadata.json 或 features.npy 推导')
    parser.add_argument('--n-voxels', type=int, nargs=3, metavar=('X', 'Y', 'Z'), default=None)
    parser.add_argument('--n-images', type=int, default=None, help='相机数；默认从 LUT 文件数量推导')
    parser.add_argument(
        '--legacy-overwrite-order',
        default=None,
        help='仅 legacy x/y/valid.npy 兜底使用，例如 0,1,2,3,4,5')
    return parser.parse_args()


def read_metadata(lut_root: Path) -> Dict:
    for candidate in (lut_root / 'metadata.json', lut_root.parent / 'metadata.json'):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding='utf-8'))
    return {}


def resolve_lut_root(path: Path) -> Path:
    path = path.resolve()
    if (path / 'LUT').is_dir():
        return path
    if path.name == 'LUT' and path.is_dir():
        return path.parent
    return path


def sorted_lut_files(board_dir: Path, prefix: str) -> List[Path]:
    files = []
    idx = 0
    while True:
        candidate = board_dir / f'{prefix}_{idx}.bin'
        if not candidate.exists():
            break
        files.append(candidate)
        idx += 1
    return files


def infer_n_images(args: argparse.Namespace, metadata: Dict, lut_root: Path) -> int:
    if args.n_images is not None:
        return int(args.n_images)
    if metadata.get('n_images') is not None:
        return int(metadata['n_images'])
    board_dir = lut_root / 'LUT'
    gather_files = sorted_lut_files(board_dir, 'gather_new')
    if gather_files:
        return len(gather_files)
    lut_arr_dir = lut_root / 'LUT_arr'
    arr_files = sorted(lut_arr_dir.glob('gather_*.npy')) if lut_arr_dir.exists() else []
    if arr_files:
        return len(arr_files)
    valid_path = lut_root / 'valid.npy'
    if valid_path.exists():
        return int(np.load(valid_path, mmap_mode='r').shape[0])
    raise FileNotFoundError('无法从 LUT 目录推导 n_images，请传 --n-images')


def infer_feature_shape(args: argparse.Namespace, metadata: Dict, lut_root: Path) -> Tuple[int, int, int, int]:
    if args.feature_shape is not None:
        return tuple(int(v) for v in args.feature_shape)
    if metadata.get('feature_shape') is not None:
        return tuple(int(v) for v in metadata['feature_shape'])
    features_path = lut_root / 'features.npy'
    if features_path.exists():
        return tuple(int(v) for v in np.load(features_path, mmap_mode='r').shape)
    raise ValueError('无法推导 feature shape，请传 --feature-shape N C H W')


def infer_n_voxels(args: argparse.Namespace, metadata: Dict, lut_root: Path) -> Tuple[int, int, int]:
    if args.n_voxels is not None:
        return tuple(int(v) for v in args.n_voxels)
    if metadata.get('n_voxels') is not None:
        return tuple(int(v) for v in metadata['n_voxels'])
    points_path = lut_root / 'points.npy'
    if points_path.exists():
        return tuple(int(v) for v in np.load(points_path, mmap_mode='r').shape[1:])
    volume_path = lut_root / 'volume.npy'
    if volume_path.exists():
        total_points = int(np.load(volume_path, mmap_mode='r').shape[-1])
        raise ValueError(
            f'volume.npy 只能推导 total_points={total_points}，请显式传 --n-voxels X Y Z')
    raise ValueError('无法推导 n_voxels，请传 --n-voxels X Y Z')


def load_board_bin_lut(lut_root: Path, n_images: int) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], str]]:
    board_dir = lut_root / 'LUT'
    if not board_dir.exists():
        return None
    length_path = board_dir / 'featurePointLength.bin'
    lengths = None
    if length_path.exists():
        lengths = np.fromfile(length_path, dtype=np.int32).astype(np.int64)
        if lengths.shape[0] < n_images:
            raise ValueError(
                f'{length_path} 只包含 {lengths.shape[0]} 个长度，少于 n_images={n_images}')
    gather_indices: List[np.ndarray] = []
    scatter_indices: List[np.ndarray] = []
    for cam_id in range(n_images):
        gather_path = board_dir / f'gather_new_{cam_id}.bin'
        scatter_path = board_dir / f'scatter_nd_new_{cam_id}.bin'
        if not gather_path.exists() or not scatter_path.exists():
            return None
        gather = np.fromfile(gather_path, dtype=np.int32).astype(np.int64)
        scatter = np.fromfile(scatter_path, dtype=np.int32).astype(np.int64)
        if lengths is not None and int(lengths[cam_id]) != gather.shape[0]:
            raise ValueError(
                f'cam{cam_id} featurePointLength={int(lengths[cam_id])} '
                f'但 gather 长度为 {gather.shape[0]}')
        gather_indices.append(gather)
        scatter_indices.append(scatter)
    return gather_indices, scatter_indices, 'board_bin'


def load_lut_arr(lut_root: Path, n_images: int) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], str]]:
    lut_arr_dir = lut_root / 'LUT_arr'
    if not lut_arr_dir.exists():
        return None
    gather_indices: List[np.ndarray] = []
    scatter_indices: List[np.ndarray] = []
    for cam_id in range(n_images):
        gather_path = lut_arr_dir / f'gather_{cam_id}.npy'
        scatter_path = lut_arr_dir / f'scatter_nd_{cam_id}.npy'
        if not gather_path.exists() or not scatter_path.exists():
            return None
        gather_indices.append(np.load(gather_path).reshape(-1).astype(np.int64))
        scatter_indices.append(np.load(scatter_path).reshape(-1).astype(np.int64))
    return gather_indices, scatter_indices, 'lut_arr'


def load_dense_lut(lut_root: Path, n_images: int) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], str]]:
    gather_dense_path = lut_root / 'gather_index_dense.npy'
    camera_choice_path = lut_root / 'camera_choice.npy'
    if not gather_dense_path.exists() or not camera_choice_path.exists():
        return None
    gather_dense = np.load(gather_dense_path)
    camera_choice = np.load(camera_choice_path).reshape(-1)
    gather_indices: List[np.ndarray] = []
    scatter_indices: List[np.ndarray] = []
    for cam_id in range(n_images):
        scatter = np.nonzero(camera_choice == cam_id)[0].astype(np.int64)
        gather = gather_dense[cam_id, scatter].astype(np.int64)
        gather_indices.append(gather)
        scatter_indices.append(scatter)
    return gather_indices, scatter_indices, 'dense_final_choice'


def parse_order(text: Optional[str], n_images: int) -> List[int]:
    if not text:
        return list(range(n_images))
    order = [int(item) for item in text.split(',') if item.strip()]
    if sorted(order) != list(range(n_images)):
        raise ValueError(f'legacy overwrite order 必须是 0..{n_images - 1} 的排列，当前 {order}')
    return order


def load_legacy_xy_valid_lut(
    lut_root: Path,
    n_images: int,
    feature_w: int,
    n_voxels: Sequence[int],
    order: Sequence[int],
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], str]]:
    x_path = lut_root / 'x.npy'
    y_path = lut_root / 'y.npy'
    valid_path = lut_root / 'valid.npy'
    if not x_path.exists() or not y_path.exists() or not valid_path.exists():
        return None
    x = np.load(x_path)
    y = np.load(y_path)
    valid = np.load(valid_path).astype(bool)
    total_points = int(np.prod(n_voxels))
    camera_choice = np.full(total_points, -1, dtype=np.int64)
    for cam_id in order:
        camera_choice[valid[cam_id]] = int(cam_id)

    gather_indices: List[np.ndarray] = []
    scatter_indices: List[np.ndarray] = []
    for cam_id in range(n_images):
        scatter = np.nonzero(camera_choice == cam_id)[0].astype(np.int64)
        gather = (y[cam_id, scatter] * int(feature_w) + x[cam_id, scatter]).astype(np.int64)
        gather_indices.append(gather)
        scatter_indices.append(scatter)
    return gather_indices, scatter_indices, 'legacy_xy_valid'


def load_lut_arrays(
    args: argparse.Namespace,
    lut_root: Path,
    n_images: int,
    feature_shape: Sequence[int],
    n_voxels: Sequence[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
    for loader in (load_board_bin_lut, load_lut_arr, load_dense_lut):
        loaded = loader(lut_root, n_images)
        if loaded is not None:
            return loaded

    order = parse_order(args.legacy_overwrite_order, n_images)
    loaded = load_legacy_xy_valid_lut(
        lut_root=lut_root,
        n_images=n_images,
        feature_w=int(feature_shape[3]),
        n_voxels=n_voxels,
        order=order)
    if loaded is not None:
        return loaded
    raise FileNotFoundError(
        'LUT 目录中未找到 board bin、LUT_arr、dense final-choice 或 legacy x/y/valid 文件')


def validate_indices(
    gather_indices: Sequence[np.ndarray],
    scatter_indices: Sequence[np.ndarray],
    feature_shape: Sequence[int],
    n_voxels: Sequence[int],
) -> None:
    _, channels, feat_h, feat_w = [int(v) for v in feature_shape]
    del channels
    feature_points = feat_h * feat_w
    total_points = int(np.prod(n_voxels))
    for cam_id, (gather, scatter) in enumerate(zip(gather_indices, scatter_indices)):
        if gather.shape[0] != scatter.shape[0]:
            raise ValueError(
                f'cam{cam_id} gather/scatter 长度不一致: {gather.shape[0]} vs {scatter.shape[0]}')
        if gather.size and (gather.min() < 0 or gather.max() >= feature_points):
            raise ValueError(
                f'cam{cam_id} gather index 越界，feature_points={feature_points}')
        if scatter.size and (scatter.min() < 0 or scatter.max() >= total_points):
            raise ValueError(
                f'cam{cam_id} scatter index 越界，total_points={total_points}')


def load_onnx_modules():
    import onnx
    try:
        from onnxsim import simplify
    except Exception:
        simplify = None
    return onnx, simplify


def check_and_simplify(onnx_path: Path, simplified_path: Path, do_simplify: bool) -> Path:
    onnx, simplify = load_onnx_modules()
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    if not do_simplify:
        return onnx_path
    if simplify is None:
        print('未安装 onnxsim，跳过 simplify')
        return onnx_path
    simplified_model, check_ok = simplify(model)
    if not check_ok:
        raise RuntimeError(f'onnxsim 校验失败: {onnx_path}')
    onnx.save_model(simplified_model, str(simplified_path))
    onnx.checker.check_model(onnx.load(str(simplified_path)))
    return simplified_path


def verify_onnx(onnx_path: Path, input_name: str, dummy: torch.Tensor, expected: torch.Tensor) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    output = sess.run(None, {input_name: dummy.detach().cpu().numpy()})[0]
    return float(np.max(np.abs(output - expected.detach().cpu().numpy())))


def main() -> None:
    args = parse_args()
    lut_root = resolve_lut_root(Path(args.lut_dir))
    metadata = read_metadata(lut_root)
    n_images = infer_n_images(args, metadata, lut_root)
    feature_shape = infer_feature_shape(args, metadata, lut_root)
    n_voxels = infer_n_voxels(args, metadata, lut_root)

    if feature_shape[0] != n_images:
        raise ValueError(f'feature_shape[0]={feature_shape[0]} 与 n_images={n_images} 不一致')

    gather_indices, scatter_indices, lut_source = load_lut_arrays(
        args=args,
        lut_root=lut_root,
        n_images=n_images,
        feature_shape=feature_shape,
        n_voxels=n_voxels)
    validate_indices(gather_indices, scatter_indices, feature_shape, n_voxels)

    _, channels, feat_h, feat_w = [int(v) for v in feature_shape]
    if args.input_layout == 'nhwc':
        dummy_shape = (n_images, feat_h, feat_w, channels)
    else:
        dummy_shape = (n_images, channels, feat_h, feat_w)
    dummy = torch.randn(dummy_shape, dtype=torch.float32)

    model = FixedLUT2Dto3D(
        gather_indices=gather_indices,
        scatter_indices=scatter_indices,
        n_voxels=n_voxels,
        channels=channels,
        input_layout=args.input_layout).eval()
    with torch.no_grad():
        expected = model(dummy)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    simplified_path = (
        Path(args.simplified_out) if args.simplified_out
        else out_path.with_name(f'simplified_{out_path.name}'))
    simplified_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        opset_version=args.opset_version,
        input_names=[args.input_name],
        output_names=['bev_feature'],
    )
    final_path = check_and_simplify(out_path, simplified_path, not args.no_simplify)
    verify_diff = verify_onnx(final_path, args.input_name, dummy, expected) if args.verify else None

    out_metadata = dict(
        lut_dir=str(lut_root),
        lut_source=lut_source,
        input_layout=args.input_layout,
        input_name=args.input_name,
        input_shape=list(dummy_shape),
        output_name='bev_feature',
        output_shape=list(expected.shape),
        feature_shape_nchw=[int(v) for v in feature_shape],
        n_voxels=[int(v) for v in n_voxels],
        n_images=int(n_images),
        channels=int(channels),
        gather_lengths=[int(item.shape[0]) for item in gather_indices],
        scatter_lengths=[int(item.shape[0]) for item in scatter_indices],
        raw_onnx=str(out_path),
        simplified_onnx=str(final_path),
        verify_max_abs_diff=verify_diff,
    )
    metadata_path = out_path.with_suffix('.metadata.json')
    metadata_path.write_text(json.dumps(out_metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'2D->3D ONNX: {final_path}')
    print(f'metadata: {metadata_path}')


if __name__ == '__main__':
    main()
