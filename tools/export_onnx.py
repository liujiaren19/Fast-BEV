#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 Fast-BEV 板端/ONNXRuntime 测试用 2D 与 3D ONNX。

当前 N7 板端主线支持 v1/R18 的原生单帧和 4 时序 BEV 输入：
- 2D ONNX 只包含 backbone + FPN + neck_fuse，并固定使用 nearest resize；
- 3D ONNX 接收每个时序的 ``[bs, z*c, x, y]`` BEV 输入，输出 bbox head
  原始 logits，sigmoid/topk/NMS 留在图外后处理；
- PC 侧 ``test_onnx`` 仍期望 2D 输出为 NCHW，所以本脚本默认导出 NCHW。
  如果芯片编译工具需要 NHWC，可显式传 ``--2d-output-layout nhwc``。
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class FastBEV2DExportWrapper(nn.Module):
    """显式进入 FastBEV 2D ONNX 导出分支。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, img):
        # 直接调用明确的导出函数，避免 eager reference 阶段因
        # torch.onnx.is_in_onnx_export()==False 误入普通 forward_test。
        return self.model.onnx_export_2d(img, None)


class FastBEV3DExportWrapper(nn.Module):
    """显式进入 FastBEV 3D ONNX 导出分支。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model.onnx_export_3d(tuple(inputs), None)


class ScopedEnv:
    """临时设置环境变量，退出后恢复。"""

    def __init__(self, **updates):
        self.updates = updates
        self.previous: Dict[str, Optional[str]] = {}

    def __enter__(self):
        for key, value in self.updates.items():
            self.previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """流式计算导出输入文件哈希，避免大 checkpoint 一次性读入内存。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='导出 Fast-BEV 2D/3D ONNX，并检查当前板端约束')
    parser.add_argument('--config', required=True, help='Fast-BEV config 路径')
    parser.add_argument('--checkpoint', required=True, help='pth checkpoint 路径')
    parser.add_argument('--out-dir', default='./output/onnx', help='ONNX 输出目录')
    parser.add_argument(
        '--export-parts',
        nargs='+',
        choices=['2d', '3d'],
        default=['2d', '3d'],
        help='选择导出哪些部分，默认同时导出 2d 和 3d')
    parser.add_argument('--opset-version', type=int, default=13, help='ONNX opset')
    parser.add_argument('--no-simplify', action='store_true', help='不运行 onnxsim')
    parser.add_argument('--verify', action='store_true', help='导出后用 ONNXRuntime 跑随机输入对齐 PyTorch 输出')
    parser.add_argument('--verbose', action='store_true', help='torch.onnx.export verbose')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='导出前 fuse conv/bn')
    parser.add_argument('--device', default=None, help='导出设备，默认优先 cuda:0，否则 cpu')
    parser.add_argument('--batch-size', type=int, default=1, help='dummy input batch')
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        default=None,
        help='2D ONNX 输入尺寸；默认从 config.data_config/test pipeline 推导')
    parser.add_argument(
        '--2d-output-layout',
        dest='output_layout_2d',
        choices=['nchw', 'nhwc'],
        default='nchw',
        help='2D ONNX 输出布局；test_onnx/PC 模拟默认需要 nchw，芯片工具需要时可选 nhwc')
    parser.add_argument('--n-images', type=int, default=None, help='覆盖每个时序的相机数')
    parser.add_argument('--n-times', type=int, default=None, help='覆盖时序数')
    parser.add_argument('--n-voxels', type=int, nargs=3, metavar=('X', 'Y', 'Z'), default=None)
    parser.add_argument('--feature-channels', type=int, default=None, help='neck_fuse 输出通道数')
    parser.add_argument(
        '--allow-feature-resize-mismatch',
        action='store_true',
        help='允许 config.model.feature_resize_mode 不是 nearest；仅做实验时使用')
    parser.add_argument(
        '--allow-non-v1',
        action='store_true',
        help='允许导出非 v1 style；当前板端只审计过 v1/R18')
    parser.add_argument(
        '--allow-non-4-times',
        action='store_true',
        help='允许导出 n_times 不为 1/4 的实验 head；产品单帧和四时序无需此参数')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        default=None,
        help='覆盖 config，格式 key=value；支持 model.xxx=... 形式')
    return parser.parse_args()


def parse_cfg_value(text: str) -> Any:
    lower = text.lower()
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    if lower in ('none', 'null'):
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_cfg_options(options: Optional[Sequence[str]]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    if not options:
        return parsed
    for item in options:
        if '=' not in item:
            raise ValueError(f'--cfg-options 项缺少 "=": {item}')
        key, value = item.split('=', 1)
        parsed[key] = parse_cfg_value(value)
    return parsed


def flatten_tensors(outputs: Any) -> List[torch.Tensor]:
    if torch.is_tensor(outputs):
        return [outputs]
    if isinstance(outputs, (list, tuple)):
        flat: List[torch.Tensor] = []
        for item in outputs:
            flat.extend(flatten_tensors(item))
        return flat
    raise TypeError(f'ONNX 输出包含非 Tensor 类型: {type(outputs)}')


def first_scalar_or_list_item(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def normalize_first_item(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return value[0]
    return value


def cfg_get(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def iter_dataset_cfgs(data_cfg: Any) -> Iterable[Any]:
    if data_cfg is None:
        return []
    items = []
    for split in ('test', 'val', 'train'):
        split_cfg = cfg_get(data_cfg, split)
        if split_cfg is None:
            continue
        if isinstance(split_cfg, (list, tuple)):
            items.extend(split_cfg)
        else:
            items.append(split_cfg)
    return items


def find_pipeline_step(pipeline: Any, step_type: str) -> Optional[Any]:
    if not isinstance(pipeline, (list, tuple)):
        return None
    for step in pipeline:
        if isinstance(step, dict) and step.get('type') == step_type:
            return step
    return None


def infer_data_config(cfg: Any) -> Dict[str, Any]:
    direct = cfg_get(cfg, 'data_config')
    if isinstance(direct, dict):
        return direct
    for dataset_cfg in iter_dataset_cfgs(cfg_get(cfg, 'data')):
        step = find_pipeline_step(cfg_get(dataset_cfg, 'pipeline'), 'RandomAugImageMultiViewImage')
        data_config = cfg_get(step, 'data_config') if step is not None else None
        if isinstance(data_config, dict):
            return data_config
    return {}


def infer_pipeline_value(cfg: Any, key: str, default: Any = None) -> Any:
    for dataset_cfg in iter_dataset_cfgs(cfg_get(cfg, 'data')):
        if cfg_get(dataset_cfg, key) is not None:
            return cfg_get(dataset_cfg, key)
        step = find_pipeline_step(cfg_get(dataset_cfg, 'pipeline'), 'MultiViewPipeline')
        if step is not None and step.get(key) is not None:
            return step.get(key)
    return default


def infer_neck_fuse_channels(model_cfg: Any) -> int:
    neck_fuse = cfg_get(model_cfg, 'neck_fuse', {})
    out_channels = cfg_get(neck_fuse, 'out_channels')
    if out_channels is None:
        out_channels = cfg_get(cfg_get(model_cfg, 'neck', {}), 'out_channels', 64)
    return int(first_scalar_or_list_item(out_channels))


def infer_export_spec(cfg: Any, args: argparse.Namespace) -> Dict[str, Any]:
    model_cfg = cfg_get(cfg, 'model', {})
    data_config = infer_data_config(cfg)
    input_size = args.input_size
    if input_size is None:
        input_size = data_config.get('test_input_size') or data_config.get('input_size') or (256, 704)
    input_h, input_w = [int(v) for v in input_size]

    n_images = args.n_images
    if n_images is None:
        n_images = cfg_get(model_cfg, 'n_images', infer_pipeline_value(cfg, 'n_images', 6))
    n_times = args.n_times
    if n_times is None:
        n_times = infer_pipeline_value(cfg, 'n_times', 4)

    n_voxels = args.n_voxels or normalize_first_item(cfg_get(model_cfg, 'n_voxels', [[200, 200, 4]]))
    n_voxels = [int(v) for v in n_voxels]
    if len(n_voxels) != 3:
        raise ValueError(f'n_voxels 需要 3 个值，当前为 {n_voxels}')

    feature_channels = args.feature_channels or infer_neck_fuse_channels(model_cfg)
    feature_channels = int(feature_channels)
    per_time_channels = feature_channels * int(n_voxels[2])

    return dict(
        input_size=[input_h, input_w],
        n_images=int(n_images),
        n_times=int(n_times),
        temporal_order=(
            ['key'] if int(n_times) == 1 else
            ['key'] + [f'history_{idx}' for idx in range(1, int(n_times))]),
        channel_layout='ZC' if int(n_times) == 1 else 'TZC',
        n_voxels=n_voxels,
        feature_channels=feature_channels,
        per_time_channels=per_time_channels,
        style=cfg_get(model_cfg, 'style', None),
        feature_resize_mode=cfg_get(model_cfg, 'feature_resize_mode', 'bilinear'),
        neck_3d=copy.deepcopy(cfg_get(model_cfg, 'neck_3d', {})),
        multi_scale_id=copy.deepcopy(cfg_get(model_cfg, 'multi_scale_id', None)),
    )


def validate_board_spec(spec: Dict[str, Any], args: argparse.Namespace) -> None:
    errors = []
    if spec['style'] != 'v1' and not args.allow_non_v1:
        errors.append(
            f"当前板端主线只审计过 model.style='v1'，config 为 {spec['style']!r}；"
            "如确认需要实验导出，请加 --allow-non-v1")
    if spec['n_times'] not in (1, 4) and not args.allow_non_4_times:
        errors.append(
            f"当前产品板端只审计原生单帧和 4 时序，config 推导 n_times={spec['n_times']}；"
            "如确认工具链已支持，请加 --allow-non-4-times")
    if spec['feature_resize_mode'] != 'nearest' and not args.allow_feature_resize_mismatch:
        errors.append(
            "2D ONNX 导出固定使用 nearest resize，但 config.model.feature_resize_mode="
            f"{spec['feature_resize_mode']!r}。这会造成 pth 训练/推理和 ONNX 特征不一致；"
            "请把 config 设为 nearest，或显式加 --allow-feature-resize-mismatch")

    neck_3d = spec['neck_3d']
    fuse_cfg = cfg_get(neck_3d, 'fuse')
    concat_channels = spec['per_time_channels'] * spec['n_times']
    if isinstance(fuse_cfg, dict):
        fuse_in = int(cfg_get(fuse_cfg, 'in_channels', concat_channels))
        fuse_out = int(cfg_get(fuse_cfg, 'out_channels', spec['per_time_channels']))
        neck_in = int(cfg_get(neck_3d, 'in_channels', fuse_out))
        if fuse_in != concat_channels:
            errors.append(
                f"neck_3d.fuse.in_channels={fuse_in} 与 3D ONNX concat 输入通道 "
                f"{concat_channels} 不一致")
        if neck_in != fuse_out:
            errors.append(
                f"neck_3d.in_channels={neck_in} 与 neck_3d.fuse.out_channels={fuse_out} 不一致")
    else:
        neck_in = int(cfg_get(neck_3d, 'in_channels', concat_channels))
        if neck_in != concat_channels:
            errors.append(
                f"neck_3d.in_channels={neck_in} 与 3D ONNX concat 输入通道 "
                f"{concat_channels} 不一致")

    if errors:
        raise ValueError('导出前约束检查失败：\n- ' + '\n- '.join(errors))


def add_mmdet3d_root_to_path() -> None:
    mmdet3d_root = os.environ.get('MMDET3D')
    if mmdet3d_root and Path(mmdet3d_root).exists():
        sys.path.insert(0, mmdet3d_root)
        print(f'using mmdet3d: {mmdet3d_root}')


def build_and_load_model(args: argparse.Namespace):
    add_mmdet3d_root_to_path()
    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    cfg = Config.fromfile(args.config)
    cfg_options = parse_cfg_options(args.cfg_options)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    return cfg, model


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
        warnings.warn('未安装 onnxsim，跳过 simplify')
        return onnx_path
    simplified_model, check_ok = simplify(model)
    if not check_ok:
        raise RuntimeError(f'onnxsim 校验失败: {onnx_path}')
    onnx.save_model(simplified_model, str(simplified_path))
    onnx.checker.check_model(onnx.load(str(simplified_path)))
    return simplified_path


def verify_with_onnxruntime(onnx_path: Path, input_names: Sequence[str], inputs: Sequence[torch.Tensor],
                            expected_outputs: Sequence[torch.Tensor]) -> List[float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_dict = {
        name: tensor.detach().cpu().float().numpy()
        for name, tensor in zip(input_names, inputs)
    }
    actual_outputs = sess.run(None, input_dict)
    if len(actual_outputs) != len(expected_outputs):
        raise RuntimeError(
            f'ONNXRuntime 输出数量 {len(actual_outputs)} 与 PyTorch {len(expected_outputs)} 不一致')
    max_diffs = []
    for actual, expected in zip(actual_outputs, expected_outputs):
        expected_np = expected.detach().cpu().float().numpy()
        max_diffs.append(float(abs(actual - expected_np).max()))
    return max_diffs


def export_2d(model: nn.Module, spec: Dict[str, Any], args: argparse.Namespace,
              out_dir: Path, device: torch.device) -> Dict[str, Any]:
    wrapper = FastBEV2DExportWrapper(model).eval()
    input_h, input_w = spec['input_size']
    dummy = torch.randn((args.batch_size, 3, input_h, input_w), device=device)
    env = {'DEPLOY': '1'} if args.output_layout_2d == 'nhwc' else {'DEPLOY': None}
    with torch.no_grad(), ScopedEnv(**env):
        expected = flatten_tensors(wrapper(dummy))

    output_names = ['features'] if len(expected) == 1 else [f'features_{idx}' for idx in range(len(expected))]
    onnx_path = out_dir / 'export_2d_model.onnx'
    simplified_path = out_dir / 'simplified_export_2d_model.onnx'
    with ScopedEnv(**env):
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            verbose=args.verbose,
            opset_version=args.opset_version,
            input_names=['prep_input_input.1'],
            output_names=output_names,
        )
    final_path = check_and_simplify(onnx_path, simplified_path, not args.no_simplify)
    verify_diffs = None
    if args.verify:
        verify_diffs = verify_with_onnxruntime(
            final_path, ['prep_input_input.1'], [dummy], expected)
    return dict(
        raw=str(onnx_path),
        simplified=str(final_path),
        input_names=['prep_input_input.1'],
        output_names=output_names,
        input_shape=list(dummy.shape),
        input_dtype=str(dummy.dtype).replace('torch.', ''),
        output_shapes=[list(item.shape) for item in expected],
        output_dtypes=[str(item.dtype).replace('torch.', '') for item in expected],
        output_layout=args.output_layout_2d,
        verify_max_abs_diff=verify_diffs,
    )


def export_3d(model: nn.Module, spec: Dict[str, Any], args: argparse.Namespace,
              out_dir: Path, device: torch.device) -> Dict[str, Any]:
    wrapper = FastBEV3DExportWrapper(model).eval()
    n_x, n_y, _ = spec['n_voxels']
    input_shape = (args.batch_size, spec['per_time_channels'], n_x, n_y)
    inputs = [torch.randn(input_shape, device=device) for _ in range(spec['n_times'])]
    # 纯数字输入名（例如 ``"0"``）会被部分 PyTorch/ONNX 版本重命名成
    # ``onnx::...``，导致 metadata/ORT feed 名称和真实图输入不一致。
    input_names = [f'bev_{idx}' for idx in range(spec['n_times'])]
    with torch.no_grad(), ScopedEnv(DEPLOY='1'):
        expected = flatten_tensors(wrapper(*inputs))

    default_output_names = ['cls_score', 'bbox_pred', 'dir_cls_preds']
    output_names = default_output_names[:len(expected)]
    if len(output_names) < len(expected):
        output_names.extend([f'output_{idx}' for idx in range(len(output_names), len(expected))])

    onnx_path = out_dir / 'export_3d_model.onnx'
    simplified_path = out_dir / 'simplified_export_3d_model.onnx'
    with ScopedEnv(DEPLOY='1'):
        torch.onnx.export(
            wrapper,
            tuple(inputs),
            str(onnx_path),
            verbose=args.verbose,
            opset_version=args.opset_version,
            input_names=input_names,
            output_names=output_names,
        )
    final_path = check_and_simplify(onnx_path, simplified_path, not args.no_simplify)
    verify_diffs = None
    if args.verify:
        verify_diffs = verify_with_onnxruntime(final_path, input_names, inputs, expected)
    return dict(
        raw=str(onnx_path),
        simplified=str(final_path),
        input_names=input_names,
        output_names=output_names,
        input_shapes=[list(item.shape) for item in inputs],
        input_dtypes=[str(item.dtype).replace('torch.', '') for item in inputs],
        output_shapes=[list(item.shape) for item in expected],
        output_dtypes=[str(item.dtype).replace('torch.', '') for item in expected],
        outputs_are_raw_logits=True,
        verify_max_abs_diff=verify_diffs,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, model = build_and_load_model(args)
    spec = infer_export_spec(cfg, args)
    validate_board_spec(spec, args)

    model = copy.deepcopy(model).to(device).eval()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    metadata: Dict[str, Any] = dict(
        config=str(config_path),
        config_sha256=sha256_file(config_path),
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        device=str(device),
        opset_version=args.opset_version,
        simplify=not args.no_simplify,
        board_spec=spec,
        exports={},
    )

    print('导出配置:')
    print(json.dumps(metadata['board_spec'], ensure_ascii=False, indent=2))

    if '2d' in args.export_parts:
        metadata['exports']['2d'] = export_2d(model, spec, args, out_dir, device)
        print(f"2D ONNX: {metadata['exports']['2d']['simplified']}")
    if '3d' in args.export_parts:
        metadata['exports']['3d'] = export_3d(model, spec, args, out_dir, device)
        print(f"3D ONNX: {metadata['exports']['3d']['simplified']}")

    metadata_path = out_dir / 'export_metadata.json'
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'导出 metadata: {metadata_path}')


if __name__ == '__main__':
    main()
