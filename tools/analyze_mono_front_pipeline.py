#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一分析 N7 mono-front S0 的 PTH/ONNX、几何和前后处理链路。

这是唯一允许保存逐阶段 tensor、执行 first-divergence trace 和区分量化/图
runtime/LUT/后处理离散边界差异的公开入口。生产推理入口不承担这些职责。

首次运行前先执行 tools/build_mono_front_board_assets.py 初始化模型和车辆资产。

直接执行会读取同一个 ``board_model_spec.json``，并覆盖 PTH/ONNX ×
dynamic/fixed × PC/board 前后处理组合；PTH 和 dynamic canonical 几何需要
legacy MMDetection3D 环境。INT8 外部 raw tensor 会同时保存 ``*_raw.npy`` 与
``*_dequant.npy``。``--stage-input-dir`` 仅用于复核既有黄金阶段目录或板端
dump。公开 CLI 不内嵌 synthetic 资产生成逻辑。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.run_mono_front_board_inference as board


ASSET_INIT_HINT = (
    "首次运行前先执行 tools/build_mono_front_board_assets.py 初始化模型和车辆资产。")
LOGGER = logging.getLogger("analyze_mono_front_pipeline")
STAGE_NAMES = (
    "raw_image", "input_tensor", "feature_2d", "projection_indices",
    "bev_input", "head_cls", "head_bbox", "head_dir", "decoded_anchors",
    "center_boxes", "business_output",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一逐阶段分析 mono-front PTH/ONNX/INT8/LUT/后处理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=ASSET_INIT_HINT,
    )
    parser.add_argument(
        "--backend", choices=("pth", "onnx-fp", "onnx-int8"), required=True)
    parser.add_argument(
        "--geometry", choices=("dynamic", "fixed"), required=True)
    parser.add_argument(
        "--preprocess", choices=("pc", "board"), required=True)
    parser.add_argument(
        "--postprocess", choices=("pc", "board"), required=True)
    parser.add_argument(
        "--weights", required=True, help="board_model_spec.json 所在目录")
    parser.add_argument(
        "--asset-root", default="data/board_lut",
        help="公共 anchors/points 和车辆 LUT 的根目录")
    parser.add_argument("--lut-dir", help="geometry=fixed 使用的车辆 LUT 目录")
    parser.add_argument("--image", help="直接执行时的原始图片")
    parser.add_argument("--info-json", help="直接执行 canonical/dynamic 所需 N7 标定")
    parser.add_argument("--sensor-name", default="front_wide")
    parser.add_argument("--camera-id", default="cam0")
    parser.add_argument(
        "--intrinsic-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--intrinsic-size-source", choices=("auto", "info", "image"),
        default="auto")
    parser.add_argument(
        "--info-extrinsic-coordinate", choices=("raw_n7", "fastbev"),
        default="raw_n7")
    parser.add_argument("--aspect-tolerance", type=float, default=0.02)
    parser.add_argument("--allow-aspect-mismatch", action="store_true")
    parser.add_argument("--config", help="Fast-BEV config；直接 PTH 或 PC 后处理需要")
    parser.add_argument("--checkpoint", help="backend=pth 直接执行时的 checkpoint")
    parser.add_argument("--onnx-custom-op-path")
    parser.add_argument("--device", default="cpu", help="canonical PC NMS device")
    parser.add_argument(
        "--provider", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--stage-input-dir",
        help="PTH/dynamic 或外部板端 dump 的阶段目录；不会在此生成 synthetic 资产")
    parser.add_argument(
        "--pc-input-tensor",
        help="preprocess=pc 时由 canonical pipeline 保存的 normalized NCHW .npy")
    parser.add_argument("--compare-dir", help="可选参考阶段目录")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-classes", nargs="+", default=("car",))
    parser.add_argument("--score-thr", type=float, default=0.2)
    parser.add_argument("--max-preds", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--strict", action="store_true",
        help="提供 --compare-dir 时，差异或参考阶段缺失返回非零")
    args = parser.parse_args()
    if args.geometry == "fixed" and not args.lut_dir:
        parser.error(f"--geometry fixed 必须传 --lut-dir。{ASSET_INIT_HINT}")
    direct = not bool(args.stage_input_dir)
    if direct and not args.image:
        parser.error("直接执行需要 --image；复核既有阶段可传 --stage-input-dir")
    if direct and args.backend == "pth" and not args.checkpoint:
        parser.error("backend=pth 直接执行需要 --checkpoint")
    if direct and (
        args.backend == "pth" or args.geometry == "dynamic" or
        (args.preprocess == "pc" and not args.pc_input_tensor)
    ) and not args.info_json:
        parser.error("直接 canonical/dynamic 执行需要 --info-json")
    if direct and (
        args.backend == "pth" or args.geometry == "dynamic" or
        (args.preprocess == "pc" and not args.pc_input_tensor)
    ) and not args.config:
        parser.error("直接 PTH/dynamic 执行需要 --config")
    if args.postprocess == "pc" and not args.config:
        parser.error("--postprocess pc 必须传 --config")
    return args


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8")


def array_summary(value: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(value)
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.astype(np.float64).mean()) if finite.size else None,
    }


def save_tensor(
    output_dir: Path,
    name: str,
    raw: np.ndarray,
    dequant: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_array = np.asarray(raw)
    np.save(output_dir / f"{name}_raw.npy", raw_array)
    record = {"raw": array_summary(raw_array)}
    if dequant is None:
        dequant = raw_array.astype(np.float32, copy=False)
    dequant_array = np.asarray(dequant, dtype=np.float32)
    np.save(output_dir / f"{name}_dequant.npy", dequant_array)
    record["dequant"] = array_summary(dequant_array)
    return record


def create_sessions(
    model_2d: Path,
    model_3d: Path,
    provider: str,
) -> Tuple[ort.InferenceSession, ort.InferenceSession, str]:
    providers = board.resolve_provider(provider)
    session_2d = ort.InferenceSession(str(model_2d), providers=providers)
    session_3d = ort.InferenceSession(str(model_3d), providers=providers)
    actual = board.validate_session_providers(session_2d, session_3d, provider)
    return session_2d, session_3d, actual


def load_stage_inputs(root: Path) -> Dict[str, np.ndarray]:
    """读取 legacy golden 或板端 dump，文件不存在时保留阶段缺失状态。"""
    root = Path(root)
    result: Dict[str, np.ndarray] = {}
    aliases = {
        "input_tensor": ("input_tensor.npy", "input.npy"),
        "feature_2d": ("feature_2d.npy", "features_2d.npy"),
        "bev_input": ("bev_input.npy", "neck_3d_input.npy"),
        "raw_image": ("raw_image.npy",),
        "projection_dynamic": ("projection_dynamic.npy", "projection.npy"),
        "projection_gather": ("projection_gather.npy", "gather.npy"),
        "projection_scatter": ("projection_scatter.npy", "scatter.npy"),
        "anchors": ("anchors.npy",),
        "decoded_anchors": ("decoded_anchors.npy",),
    }
    for name, candidates in aliases.items():
        for candidate in candidates:
            path = root / candidate
            if path.is_file():
                result[name] = np.load(path)
                if name == "input_tensor" and result[name].ndim == 5:
                    if result[name].shape[1] != 1:
                        raise ValueError(
                            f"legacy input view count={result[name].shape[1]}，"
                            "analyzer 只支持 S0")
                    result[name] = result[name][:, 0]
                break
        raw_path = root / f"{name}_raw.npy"
        dequant_path = root / f"{name}_dequant.npy"
        if raw_path.is_file():
            result[f"{name}_raw"] = np.load(raw_path)
        if dequant_path.is_file():
            result[name] = np.load(dequant_path)
    for archive_name in ("head_logits.npz", "raw_logits.npz"):
        path = root / archive_name
        if path.is_file():
            with np.load(path) as archive:
                for semantic in ("head_cls", "head_bbox", "head_dir"):
                    matches = [
                        name for name in archive.files
                        if name.startswith(semantic)]
                    if len(matches) == 1:
                        result[semantic] = archive[matches[0]]
                    elif matches:
                        raise ValueError(
                            f"{archive_name} 中 {semantic} 匹配不唯一: {matches}")
            break
    return result


def load_analysis_assets(
    args: argparse.Namespace,
) -> Tuple[Mapping[str, Any], Optional[Mapping[str, Any]], Optional[Path],
           Optional[Path], np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
           Optional[Mapping[str, Any]]]:
    """统一读取 manifest/anchors；fixed 模式再严格读取车辆 LUT。"""
    from tools.mono_front_board_assets import load_board_model_assets

    weights = Path(args.weights).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    spec = load_board_model_assets(weights, asset_root)
    anchors_path = (
        asset_root / str(spec["assets"]["anchors_relative"])).resolve()
    anchors = board.load_anchors(anchors_path, spec)

    contract = None
    model_2d = None
    model_3d = None
    if args.backend.startswith("onnx"):
        contract = spec["onnx_contracts"].get(args.backend)
        if not isinstance(contract, dict) or not contract.get("available"):
            reason = contract.get("reason") if isinstance(contract, dict) else "missing"
            raise RuntimeError(
                f"manifest 不包含可用 {args.backend} contract: {reason}。"
                f"{ASSET_INIT_HINT}")
        paths = []
        for part in ("2d", "3d"):
            record = contract["models"][part]
            path = board._resolve_recorded_file(
                record["file"], weights, f"{args.backend} {part} ONNX")
            if board.file_sha256(path) != record["sha256"]:
                raise RuntimeError(f"{args.backend} {part} ONNX SHA256 不匹配: {path}")
            paths.append(path)
        model_2d, model_3d = paths

    gather = None
    scatter = None
    lut_metadata = None
    if args.geometry == "fixed":
        # standalone loader 还会交叉核对 manifest、模型 hash 和车辆目录结构。
        manifest_backend = args.backend if args.backend.startswith("onnx") else "onnx-fp"
        fixed_spec, _fixed_contract, _m2d, _m3d, _anchors = board.load_model_spec(
            weights, Path(args.lut_dir), manifest_backend)
        if fixed_spec["geometry_contract_hash"] != spec["geometry_contract_hash"]:
            raise RuntimeError("fixed LUT 与公共 manifest geometry hash 不一致")
        gather, scatter, lut_metadata = board.load_lut(Path(args.lut_dir), spec)
    return (
        spec, contract, model_2d, model_3d, anchors,
        gather, scatter, lut_metadata,
    )


def _first_array(value: Any) -> np.ndarray:
    """从 bbox head 的单 level 嵌套输出中取出唯一 tensor。"""
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    if isinstance(value, (list, tuple)):
        arrays = [_first_array(item) for item in value]
        if len(arrays) != 1:
            raise RuntimeError(f"预期单 level tensor，实际数量={len(arrays)}")
        return arrays[0]
    return np.asarray(value)


def prepare_canonical_context(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    model_2d: Optional[Path],
    model_3d: Optional[Path],
) -> Tuple[Any, Any, Any, np.ndarray, Any, np.ndarray]:
    """构造和生产入口相同的 S0 sample、pipeline tensor 与 img_meta。"""
    import torch
    from mmdet3d.datasets.pipelines import Compose
    from tools import infer_mono_front_image as production

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"config 不存在: {config_path}。{ASSET_INIT_HINT}")
    if board.file_sha256(config_path) != spec["config_sha256"]:
        raise RuntimeError(f"config SHA256 与 board_model_spec 不一致: {config_path}")
    checkpoint = None
    if args.backend == "pth":
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"checkpoint 不存在: {checkpoint}。{ASSET_INIT_HINT}")
        if board.file_sha256(checkpoint) != spec["checkpoint_sha256"]:
            raise RuntimeError(
                f"checkpoint SHA256 与 board_model_spec 不一致: {checkpoint}")

    build_args = SimpleNamespace(
        config=str(config_path),
        # analyzer 会直接调用 extract_feat/onnx_export_*，绕过模型标准 forward
        # 上的 auto_fp16 边界。这里统一关闭训练配置继承的 MMCV FP16 包装，
        # 让 PTH 权重、输入和固定 LUT 回灌 tensor 都保持 FP32；ONNX I/O dtype
        # 仍严格由 board_model_spec contract 决定。
        cfg_options=("fp16=None",),
        force_resize=True,
        onnx_custom_op_path=args.onnx_custom_op_path,
        fuse_conv_bn=False,
    )
    cfg, model, n_times, _classes, box_type_3d, box_mode_3d = (
        production.build_cfg_and_model(
            build_args, args.backend, checkpoint, model_2d, model_3d,
            args.device))
    if n_times != 1:
        raise RuntimeError(f"analyzer 直接执行只支持原生 S0 n_times=1，实际 {n_times}")
    if int(spec["geometry"]["n_images"]) != 1:
        raise RuntimeError("manifest 不是 mono-front n_images=1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求 {device}，但 torch.cuda.is_available()=False")
    model = model.to(device)
    model.eval()

    image_path = Path(args.image).expanduser().resolve()
    raw_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_image is None:
        raise ValueError(f"OpenCV 无法读取图片: {image_path}")
    calibration = production.read_json(Path(args.info_json))
    sensor = production.find_sensor(calibration, args.sensor_name)
    camera_args = SimpleNamespace(
        intrinsic_size=args.intrinsic_size,
        intrinsic_size_source=args.intrinsic_size_source,
        info_extrinsic_coordinate=args.info_extrinsic_coordinate,
        aspect_tolerance=args.aspect_tolerance,
        allow_aspect_mismatch=args.allow_aspect_mismatch,
    )
    camera_template = production.build_camera_template(
        sensor, production.read_image_size(image_path), camera_args)
    sample, _info = production.build_sample_results(
        [image_path], camera_template, args.camera_id, image_path.stem, 0,
        box_type_3d, box_mode_3d, camera_args)
    pipeline = Compose(copy.deepcopy(cfg.data.test.pipeline))
    data = pipeline(sample)
    image_tensor = production.unwrap_data_container(data["img"])
    img_meta = production.unwrap_data_container(data["img_metas"])
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)
    if image_tensor.shape[1] != 1:
        raise RuntimeError(f"S0 输入应为 1 view，实际 {tuple(image_tensor.shape)}")

    pc_input = image_tensor[:, 0].detach().cpu().numpy()
    if args.preprocess == "board":
        board_input, _board_raw, _source_size = board.preprocess_image(
            image_path, spec["preprocess"])
        image_tensor = torch.from_numpy(board_input[:, None]).to(device)
        selected_input = board_input
    elif args.pc_input_tensor:
        selected_input = np.asarray(
            np.load(Path(args.pc_input_tensor)), dtype=np.float32)
        expected = tuple(pc_input.shape)
        if selected_input.shape != expected:
            raise ValueError(
                f"--pc-input-tensor shape={selected_input.shape}，预期 {expected}")
        image_tensor = torch.from_numpy(selected_input[:, None]).to(device)
    else:
        selected_input = pc_input
    return model, device, image_tensor, selected_input, img_meta, raw_image


def dynamic_projection_array(
    model: Any,
    img_meta: Mapping[str, Any],
    input_tensor: np.ndarray,
    feature_nchw: np.ndarray,
) -> np.ndarray:
    """保存 Fast-BEV 本次动态反投影实际使用的 stride/projection。"""
    import math

    stride = math.ceil(int(input_tensor.shape[-1]) / int(feature_nchw.shape[-1]))
    projection = model._compute_projection(
        copy.deepcopy(img_meta), stride, noise=model.extrinsic_noise)
    return projection.detach().float().cpu().numpy()


def validate_fixed_image_contract(
    raw_image: np.ndarray,
    metadata: Mapping[str, Any],
    image_path: Path,
) -> None:
    actual = [int(raw_image.shape[1]), int(raw_image.shape[0])]
    expected = metadata.get("source_image_size")
    if actual != expected:
        raise RuntimeError(
            f"图片 {image_path} 尺寸={actual}，fixed LUT={expected}；"
            "禁止跨图片坐标系复用 LUT")


def run_fixed_onnx(args: argparse.Namespace, output_dir: Path) -> Tuple[
    Dict[str, np.ndarray], Dict[str, Any], np.ndarray, Mapping[str, Any]
]:
    spec, contract, model_2d, model_3d, anchors_path = board.load_model_spec(
        Path(args.weights), Path(args.lut_dir), args.backend)
    gather, scatter, metadata = board.load_lut(Path(args.lut_dir), spec)
    anchors = board.load_anchors(anchors_path, spec)
    session_2d, session_3d, actual_provider = create_sessions(
        model_2d, model_3d, args.provider)

    if args.preprocess == "board":
        if not args.image:
            raise ValueError("--preprocess board 直接执行时必须传 --image")
        input_tensor, raw_image, _ = board.preprocess_image(
            Path(args.image), spec["preprocess"])
    else:
        if args.pc_input_tensor:
            input_tensor = np.load(Path(args.pc_input_tensor))
            raw_image = cv2.imread(str(Path(args.image)), cv2.IMREAD_COLOR)
            if raw_image is None:
                raise ValueError(f"OpenCV 无法读取图片: {args.image}")
        else:
            _model, _device, _img, input_tensor, _meta, raw_image = (
                prepare_canonical_context(
                    args, spec, model_2d, model_3d))

    stages: Dict[str, np.ndarray] = {
        "raw_image": raw_image,
        "input_tensor": np.asarray(input_tensor, dtype=np.float32),
        "projection_gather": gather,
        "projection_scatter": scatter,
        "anchors": anchors,
    }
    validate_fixed_image_contract(
        raw_image, metadata, Path(args.image).expanduser().resolve())
    input_spec = contract["2d"]["input"]
    input_raw = board.quantize_external(input_tensor, input_spec)
    feature_raw = board.run_2d_onnx(
        session_2d, input_tensor, contract["2d"])
    feature_spec = contract["2d"]["output"]
    feature_dequant = board.dequantize_external(feature_raw, feature_spec)
    stages["input_tensor_raw"] = input_raw
    stages["feature_2d_raw"] = feature_raw
    stages["feature_2d"] = feature_dequant

    if contract["feature_to_bev_bridge"] == "raw-quantized-direct":
        feature_lut = board.feature_to_nchw_once(
            feature_raw, feature_spec["layout"])
        fill = feature_spec["quantization"]["zero_point"]
    else:
        feature_lut = board.feature_to_nchw_once(
            feature_dequant, feature_spec["layout"])
        fill = 0.0
    bev_raw = board.apply_lut(
        feature_lut, gather, scatter, spec["geometry"], fill)
    bev_spec = contract["3d"]["input"]
    if str(bev_raw.dtype) != bev_spec["dtype"]:
        bev_raw = board.quantize_external(bev_raw, bev_spec)
    bev_dequant = board.dequantize_external(bev_raw, bev_spec)
    stages["bev_input_raw"] = bev_raw
    stages["bev_input"] = bev_dequant

    names = contract["3d"]["semantic_output_names"]
    logits_raw = board.run_3d_onnx(
        session_3d, bev_raw, names, contract["3d"])
    logits = {
        name: board.dequantize_external(
            value, contract["3d"]["outputs"][name])
        for name, value in logits_raw.items()
    }
    for name in ("head_cls", "head_bbox", "head_dir"):
        stages[f"{name}_raw"] = logits_raw[name]
        stages[name] = logits[name]
    runtime = {
        "provider": actual_provider,
        "backend": args.backend,
        "external_io_mode": contract["external_io_mode"],
        "feature_to_bev_bridge": contract["feature_to_bev_bridge"],
        "lut_metadata": metadata,
    }
    return stages, runtime, anchors, spec


def run_dynamic_pth(args: argparse.Namespace) -> Tuple[
    Dict[str, np.ndarray], Dict[str, Any], np.ndarray, Mapping[str, Any]
]:
    """直接执行 canonical PTH 前向，并在 decode 前捕获全部关键 tensor。"""
    import torch
    from tools.compare_mono_front_dataset_production import ForwardCapture

    spec, _contract, _m2d, _m3d, anchors, _g, _s, _meta = (
        load_analysis_assets(args))
    model, device, image_tensor, selected_input, img_meta, raw_image = (
        prepare_canonical_context(args, spec, None, None))
    with ForwardCapture(model) as capture, torch.inference_mode():
        feature_bev, _valids, _features_2d = model.extract_feat(
            image_tensor, [copy.deepcopy(img_meta)], "test_pth")
        head_output = model.bbox_head(feature_bev)
    feature = _first_array(capture.values["feature_2d"])
    bev = _first_array(capture.values["bev_input"])
    stages = {
        "raw_image": raw_image,
        "input_tensor": np.asarray(selected_input, dtype=np.float32),
        "feature_2d": np.asarray(feature, dtype=np.float32),
        "projection_dynamic": dynamic_projection_array(
            model, img_meta, selected_input, feature),
        "bev_input": np.asarray(bev, dtype=np.float32),
        "anchors": anchors,
    }
    for name, value in zip(
        ("head_cls", "head_bbox", "head_dir"), head_output
    ):
        stages[name] = np.asarray(_first_array(value), dtype=np.float32)
    runtime = {
        "provider": str(device),
        "backend": "pth",
        "execution": "direct-canonical-dynamic",
    }
    return stages, runtime, anchors, spec


def run_fixed_pth(args: argparse.Namespace) -> Tuple[
    Dict[str, np.ndarray], Dict[str, Any], np.ndarray, Mapping[str, Any]
]:
    """PTH 权重配合固定 LUT，供 analyzer 隔离几何差异。"""
    import torch

    spec, _contract, _m2d, _m3d, anchors, gather, scatter, metadata = (
        load_analysis_assets(args))
    assert gather is not None and scatter is not None and metadata is not None
    model, device, image_tensor, selected_input, img_meta, raw_image = (
        prepare_canonical_context(args, spec, None, None))
    validate_fixed_image_contract(
        raw_image, metadata, Path(args.image).expanduser().resolve())
    with torch.inference_mode():
        feature_tensor = model.onnx_export_2d(
            image_tensor[:, 0], [copy.deepcopy(img_meta)])
        feature = _first_array(feature_tensor)
        bev = board.apply_lut(
            np.asarray(feature, dtype=np.float32), gather, scatter,
            spec["geometry"], 0.0)
        head_output = model.onnx_export_3d(
            torch.from_numpy(bev).to(device), None)
    stages = {
        "raw_image": raw_image,
        "input_tensor": np.asarray(selected_input, dtype=np.float32),
        "feature_2d": np.asarray(feature, dtype=np.float32),
        "projection_gather": gather,
        "projection_scatter": scatter,
        "bev_input": np.asarray(bev, dtype=np.float32),
        "anchors": anchors,
    }
    for name, value in zip(
        ("head_cls", "head_bbox", "head_dir"), head_output
    ):
        stages[name] = np.asarray(_first_array(value), dtype=np.float32)
    runtime = {
        "provider": str(device),
        "backend": "pth",
        "execution": "direct-pth-fixed-lut",
        "lut_metadata": metadata,
    }
    return stages, runtime, anchors, spec


def run_dynamic_onnx(args: argparse.Namespace) -> Tuple[
    Dict[str, np.ndarray], Dict[str, Any], np.ndarray, Mapping[str, Any]
]:
    """ONNX 负责 2D/3D，项目模型只负责 canonical dynamic backprojection。"""
    import torch

    spec, contract, model_2d, model_3d, anchors, _g, _s, _metadata = (
        load_analysis_assets(args))
    assert contract is not None and model_2d is not None and model_3d is not None
    model, device, image_tensor, selected_input, img_meta, raw_image = (
        prepare_canonical_context(args, spec, model_2d, model_3d))
    session_2d, session_3d, actual_provider = create_sessions(
        model_2d, model_3d, args.provider)

    input_spec = contract["2d"]["input"]
    input_raw = board.quantize_external(selected_input, input_spec)
    feature_raw = board.run_2d_onnx(session_2d, selected_input, contract["2d"])
    feature_spec = contract["2d"]["output"]
    feature = board.feature_to_nchw_once(
        board.dequantize_external(feature_raw, feature_spec),
        feature_spec["layout"])
    captured: Dict[str, Any] = {}
    original_backbone = model._extract_onnx_backbone
    original_head = model._onnx_head_forward

    def analyzer_backbone(_img):
        return [torch.from_numpy(feature).to(device)]

    def analyzer_head(deploy_head_inputs, output_device):
        if len(deploy_head_inputs) != 1:
            raise RuntimeError(
                f"S0 3D ONNX 应只有一个 BEV input，实际 {len(deploy_head_inputs)}")
        bev_float = deploy_head_inputs[0].detach().float().cpu().numpy()
        bev_raw = board.quantize_external(bev_float, contract["3d"]["input"])
        names = contract["3d"]["semantic_output_names"]
        logits_raw = board.run_3d_onnx(
            session_3d, bev_raw, names, contract["3d"])
        logits = {
            name: board.dequantize_external(
                value, contract["3d"]["outputs"][name])
            for name, value in logits_raw.items()
        }
        captured.update({
            "bev_raw": bev_raw,
            "bev": board.dequantize_external(
                bev_raw, contract["3d"]["input"]),
            "logits_raw": logits_raw,
            "logits": logits,
        })
        return tuple(
            [torch.from_numpy(logits[name]).to(output_device)]
            for name in ("head_cls", "head_bbox", "head_dir"))

    model._extract_onnx_backbone = analyzer_backbone
    model._onnx_head_forward = analyzer_head
    try:
        with torch.inference_mode():
            # 只执行到 raw head logits；decode/NMS 统一在 decode_outputs 中执行一次。
            model.extract_feat(
                image_tensor, [copy.deepcopy(img_meta)], "test_onnx")
    finally:
        model._extract_onnx_backbone = original_backbone
        model._onnx_head_forward = original_head
    if not captured:
        raise RuntimeError("dynamic ONNX 未捕获到 3D input/output")

    stages = {
        "raw_image": raw_image,
        "input_tensor_raw": input_raw,
        "input_tensor": board.dequantize_external(input_raw, input_spec),
        "feature_2d_raw": feature_raw,
        "feature_2d": feature,
        "projection_dynamic": dynamic_projection_array(
            model, img_meta, selected_input, feature),
        "bev_input_raw": captured["bev_raw"],
        "bev_input": captured["bev"],
        "anchors": anchors,
    }
    for name in ("head_cls", "head_bbox", "head_dir"):
        stages[f"{name}_raw"] = captured["logits_raw"][name]
        stages[name] = captured["logits"][name]
    runtime = {
        "provider": actual_provider,
        "backend": args.backend,
        "execution": "direct-onnx-dynamic-canonical-geometry",
        "external_io_mode": contract["external_io_mode"],
        "feature_to_bev_bridge": "dynamic-float-backprojection",
    }
    return stages, runtime, anchors, spec


def decode_all_anchors_center(
    logits: Mapping[str, np.ndarray],
    anchors: np.ndarray,
) -> np.ndarray:
    """保存 score/top-k/NMS 之前的全部 anchor delta decode（center-origin）。"""
    bbox = np.asarray(logits["head_bbox"], dtype=np.float32)
    if bbox.ndim != 4 or bbox.shape[0] != 1:
        raise ValueError(f"head_bbox 必须是 batch=1 NCHW，实际 {bbox.shape}")
    locations = int(bbox.shape[-2] * bbox.shape[-1])
    if anchors.shape[0] % locations:
        raise ValueError("anchors 数量不能按 head 空间位置展开")
    anchors_per_location = anchors.shape[0] // locations
    if bbox.shape[1] % anchors_per_location:
        raise ValueError("bbox channels 不能按 anchors_per_location 展开")
    code_size = bbox.shape[1] // anchors_per_location
    deltas = bbox[0].transpose(1, 2, 0).reshape(-1, code_size)
    decoded = board.decode_anchor_deltas(anchors, deltas)
    decoded = np.asarray(decoded, dtype=np.float32).copy()
    decoded[:, 2] += decoded[:, 5] * 0.5
    return decoded


def normalize_canonical_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """把 canonical LiDAR box 对象统一为 center-origin NumPy。"""
    from tools.n7_box_origin import boxes_to_numpy, tensor_like_to_numpy

    boxes = boxes_to_numpy(
        result.get("boxes_3d", np.zeros((0, 7), dtype=np.float32)),
        target_origin="center", source_origin=result.get("box_origin"),
        dtype=np.float32, box_dim=None)
    scores = tensor_like_to_numpy(
        result.get("scores_3d"), dtype=np.float32).reshape(-1)
    labels = tensor_like_to_numpy(
        result.get("labels_3d"), dtype=np.int64).reshape(-1)
    count = min(len(boxes), len(scores), len(labels))
    return {
        "boxes": boxes[:count],
        "scores": scores[:count],
        "labels": labels[:count],
        "box_origin": "center",
    }


def build_canonical_postprocessor(config_path: Path, device_text: str):
    """按 config 构建项目 bbox_head；只在选择 PC 后处理时延迟导入。"""
    mmdet3d_root = os.environ.get("MMDET3D")
    if mmdet3d_root and Path(mmdet3d_root).exists():
        sys.path.insert(0, mmdet3d_root)

    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.core.bbox import get_box_type
    from mmdet3d.models import build_model

    cfg = Config.fromfile(str(config_path))
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求 {device} 做 canonical NMS，但 CUDA 不可用")
    bbox_head = model.bbox_head.to(device).eval()
    box_type_3d, _ = get_box_type("LiDAR")
    return bbox_head, box_type_3d, device


def canonical_decode(
    bbox_head: Any,
    box_type_3d: Any,
    device: Any,
    logits: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    """只调用 canonical bbox_head.get_bboxes，不复制正式解码实现。"""
    import torch

    parts = []
    for semantic in ("head_cls", "head_bbox", "head_dir"):
        tensor = torch.from_numpy(
            np.ascontiguousarray(logits[semantic])).float().to(device)
        parts.append([tensor])
    with torch.no_grad():
        decoded = bbox_head.get_bboxes(
            parts[0], parts[1], parts[2],
            [{"box_type_3d": box_type_3d}], valid=None)
    if len(decoded) != 1:
        raise RuntimeError(
            f"canonical get_bboxes 应返回一个样本，实际 {len(decoded)}")
    boxes, scores, labels = decoded[0]
    return normalize_canonical_result({
        "boxes_3d": boxes,
        "scores_3d": scores,
        "labels_3d": labels,
    })


def trace_canonical_postprocess(
    bbox_head: Any,
    box_type_3d: Any,
    device: Any,
    logits: Mapping[str, np.ndarray],
    decoded: Mapping[str, Any],
) -> Dict[str, Any]:
    """旁路追踪 canonical top-k、score threshold 与 class NMS 选择。"""
    import torch

    tensors = {
        name: torch.from_numpy(
            np.ascontiguousarray(logits[name])).float().to(device)
        for name in ("head_cls", "head_bbox", "head_dir")
    }
    cls_tensor = tensors["head_cls"]
    bbox_tensor = tensors["head_bbox"]
    dir_tensor = tensors["head_dir"]
    if cls_tensor.ndim != 4 or cls_tensor.shape[0] != 1:
        raise ValueError(
            f"后处理追踪只支持 batch=1 4D logits，实际 {tuple(cls_tensor.shape)}")
    featmap_sizes = [cls_tensor.shape[-2:]]
    anchors = bbox_head.anchor_generator.grid_anchors(
        featmap_sizes, device=device)
    anchors = [
        value.reshape(-1, bbox_head.box_code_size) for value in anchors]
    cls_single = cls_tensor[0].detach()
    bbox_single = bbox_tensor[0].detach()
    dir_single = dir_tensor[0].detach()
    raw_cls = cls_single.permute(1, 2, 0).reshape(
        -1, bbox_head.num_classes)
    if bbox_head.use_sigmoid_cls:
        raw_scores = raw_cls.sigmoid()
        max_scores = raw_scores.max(dim=1)[0]
    else:
        raw_scores = raw_cls.softmax(-1)
        max_scores = raw_scores[:, :-1].max(dim=1)[0]
    total_candidates = int(max_scores.numel())
    nms_pre = int(bbox_head.test_cfg.get("nms_pre", -1))
    if nms_pre > 0 and total_candidates > nms_pre:
        _values, topk_ids = max_scores.topk(nms_pre)
    else:
        topk_ids = torch.arange(
            total_candidates, device=device, dtype=torch.long)

    with torch.no_grad():
        pre_boxes, pre_scores, _pre_dir = (
            bbox_head.get_tta_mlvl_output_single(
                [cls_single], [bbox_single], [dir_single], anchors,
                {"box_type_3d": box_type_3d}))
    foreground_scores = pre_scores[:, :bbox_head.num_classes]
    expected_scores = raw_scores[topk_ids]
    if foreground_scores.shape != expected_scores.shape or not torch.equal(
        foreground_scores, expected_scores
    ):
        max_diff = float(
            (foreground_scores - expected_scores).abs().max().item())
        raise RuntimeError(
            "后处理追踪与 head 的 top-k score 未精确复现，"
            f"max_abs={max_diff}")

    pre_box_object = box_type_3d(
        pre_boxes.detach().clone(), box_dim=bbox_head.box_code_size)
    pre_count = int(pre_boxes.shape[0])
    pre_center = normalize_canonical_result({
        "boxes_3d": pre_box_object,
        "scores_3d": pre_boxes.new_zeros((pre_count,)),
        "labels_3d": pre_boxes.new_zeros((pre_count,), dtype=torch.long),
    })["boxes"]
    pre_scores_np = foreground_scores.detach().cpu().numpy().astype(
        np.float32, copy=False)
    topk_ids_np = topk_ids.detach().cpu().numpy().astype(
        np.int64, copy=False)
    raw_max_np = max_scores.detach().cpu().numpy().astype(
        np.float32, copy=False)
    decoded_boxes = np.asarray(decoded["boxes"], dtype=np.float32)
    decoded_scores = np.asarray(decoded["scores"], dtype=np.float32)
    decoded_labels = np.asarray(decoded["labels"], dtype=np.int64)

    used_pairs = set()
    selected_records = []
    mapping_max_abs = 0.0
    mapping_score_max_abs = 0.0
    for output_index, (box_value, score, label_value) in enumerate(zip(
        decoded_boxes, decoded_scores, decoded_labels
    )):
        label = int(label_value)
        available = np.asarray([
            index for index in range(len(pre_center))
            if (index, label) not in used_pairs
        ], dtype=np.int64)
        if label < 0 or label >= pre_scores_np.shape[1] or not available.size:
            raise RuntimeError(
                f"canonical 输出无法映射到 NMS 前候选: label={label}")
        geometry_diff = np.max(
            np.abs(pre_center[available, :6] - box_value[None, :6]), axis=1)
        score_diff = np.abs(pre_scores_np[available, label] - float(score))
        order = np.lexsort((score_diff, geometry_diff))
        pre_index = int(available[int(order[0])])
        geometry_abs = float(geometry_diff[int(order[0])])
        score_abs = float(score_diff[int(order[0])])
        if geometry_abs > 1e-4 or score_abs > 1e-6:
            raise RuntimeError(
                "无法把 canonical 输出可靠映射回 NMS 前候选: "
                f"output={output_index}, geometry_abs={geometry_abs}, "
                f"score_abs={score_abs}")
        used_pairs.add((pre_index, label))
        mapping_max_abs = max(mapping_max_abs, geometry_abs)
        mapping_score_max_abs = max(mapping_score_max_abs, score_abs)
        source_id = int(topk_ids_np[pre_index])
        selected_records.append({
            "output_index": int(output_index),
            "post_topk_index": pre_index,
            "source_anchor_index": source_id,
            "source_rank_by_max_score": int(
                1 + np.count_nonzero(raw_max_np > raw_max_np[source_id])),
            "label": label,
            "score": float(score),
            "source_max_score": float(raw_max_np[source_id]),
            "box": [float(value) for value in box_value.tolist()],
        })

    score_thr = float(bbox_head.test_cfg.get("score_thr", 0.0))
    topk_mask = np.zeros((total_candidates,), dtype=np.bool_)
    topk_mask[topk_ids_np] = True
    rejected_scores = raw_max_np[~topk_mask]
    topk_cutoff = (
        float(raw_max_np[topk_ids_np].min()) if len(topk_ids_np) else None)
    first_rejected = (
        float(rejected_scores.max()) if rejected_scores.size else None)
    return {
        "status": "PASS",
        "sigmoid_count": 1,
        "box_origin": "center",
        "total_candidates_before_topk": total_candidates,
        "nms_pre": nms_pre,
        "post_topk_candidates": int(len(topk_ids_np)),
        "topk_cutoff_max_score": topk_cutoff,
        "first_rejected_max_score": first_rejected,
        "topk_boundary_margin": (
            float(topk_cutoff - first_rejected)
            if topk_cutoff is not None and first_rejected is not None else None),
        "score_thr": score_thr,
        "per_class": [{
            "label": label,
            "above_score_thr_after_topk": int(np.count_nonzero(
                pre_scores_np[:, label] > score_thr)),
            "selected_after_nms": int(np.count_nonzero(
                decoded_labels == label)),
        } for label in range(int(bbox_head.num_classes))],
        "selected_count": int(len(selected_records)),
        "selected": selected_records,
        "output_to_pre_nms_mapping_max_abs": mapping_max_abs,
        "output_to_pre_nms_score_max_abs": mapping_score_max_abs,
    }


def attach_canonical_source_anchor_indices(
    decoded: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    """把已校验的 canonical 输出映射回填为统一业务输出契约。"""
    selected = trace.get("selected")
    if not isinstance(selected, list):
        raise RuntimeError("canonical 后处理 trace 缺少 selected 映射")
    expected_count = len(np.asarray(decoded["boxes"]))
    output_indices = [int(record["output_index"]) for record in selected]
    expected_indices = list(range(expected_count))
    if output_indices != expected_indices:
        raise RuntimeError(
            "canonical 后处理 trace 的输出顺序不完整: "
            f"actual={output_indices}, expected={expected_indices}")
    source_indices = np.asarray([
        int(record["source_anchor_index"]) for record in selected
    ], dtype=np.int64)
    if len(source_indices) != expected_count:
        raise RuntimeError(
            "canonical source anchor 数量与输出框不一致: "
            f"anchors={len(source_indices)}, boxes={expected_count}")
    result = dict(decoded)
    result["source_anchor_indices"] = source_indices
    return result


def decode_outputs(
    args: argparse.Namespace,
    stages: Mapping[str, np.ndarray],
    anchors: np.ndarray,
    spec: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    logits = {name: stages[name] for name in ("head_cls", "head_bbox", "head_dir")}
    if args.postprocess == "board":
        return board.postprocess(logits, anchors, spec["head"])

    bbox_head, box_type_3d, device = build_canonical_postprocessor(
        Path(args.config), args.device)
    decoded = canonical_decode(bbox_head, box_type_3d, device, logits)
    decoded["class_names"] = list(spec["head"]["class_names"])
    trace = trace_canonical_postprocess(
        bbox_head, box_type_3d, device, logits, decoded)
    decoded = attach_canonical_source_anchor_indices(decoded, trace)
    return decoded, trace


def business_output(
    decoded: Mapping[str, Any],
    classes: Sequence[str],
    score_thr: float,
    max_preds: int,
) -> Dict[str, Any]:
    selected, rows = board.select_business_output(
        decoded, classes, score_thr, max_preds)
    return {
        "box_origin": "center",
        "count": len(rows),
        "predictions": rows,
        "source_anchor_indices": selected["source_anchor_indices"],
    }


def compare_business_output_values(
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    """业务输出的离散字段精确比较，浮点 score/box 使用显式容差。"""
    discrete_differences = []
    if set(actual) != set(reference):
        discrete_differences.append("top_level_keys")
    for key in ("box_origin", "count", "source_anchor_indices"):
        if actual.get(key) != reference.get(key):
            discrete_differences.append(key)

    actual_rows = actual.get("predictions")
    reference_rows = reference.get("predictions")
    if not isinstance(actual_rows, list) or not isinstance(reference_rows, list):
        discrete_differences.append("predictions_type")
        actual_rows = [] if not isinstance(actual_rows, list) else actual_rows
        reference_rows = [] if not isinstance(reference_rows, list) else reference_rows
    if len(actual_rows) != len(reference_rows):
        discrete_differences.append("predictions_length")

    score_pairs = []
    box_actual = []
    box_reference = []
    for index, (actual_row, reference_row) in enumerate(zip(
        actual_rows, reference_rows
    )):
        if not isinstance(actual_row, dict) or not isinstance(reference_row, dict):
            discrete_differences.append(f"predictions[{index}]_type")
            continue
        if set(actual_row) != set(reference_row):
            discrete_differences.append(f"predictions[{index}]_keys")
        for key in set(actual_row) | set(reference_row):
            if key in ("score", "box"):
                continue
            if actual_row.get(key) != reference_row.get(key):
                discrete_differences.append(f"predictions[{index}].{key}")
        try:
            score_pairs.append((
                float(actual_row["score"]), float(reference_row["score"])))
            actual_box = np.asarray(actual_row["box"], dtype=np.float64)
            reference_box = np.asarray(reference_row["box"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            discrete_differences.append(f"predictions[{index}]_numeric_schema")
            continue
        if actual_box.shape != reference_box.shape:
            discrete_differences.append(f"predictions[{index}].box_shape")
            continue
        box_actual.append(actual_box.reshape(-1))
        box_reference.append(reference_box.reshape(-1))

    score_actual = np.asarray(
        [pair[0] for pair in score_pairs], dtype=np.float64)
    score_reference = np.asarray(
        [pair[1] for pair in score_pairs], dtype=np.float64)
    if box_actual:
        box_actual_array = np.concatenate(box_actual)
        box_reference_array = np.concatenate(box_reference)
    else:
        box_actual_array = np.zeros((0,), dtype=np.float64)
        box_reference_array = np.zeros((0,), dtype=np.float64)
    score_diff = np.abs(score_actual - score_reference)
    box_diff = np.abs(box_actual_array - box_reference_array)
    numeric_within_tolerance = bool(
        np.allclose(score_actual, score_reference, atol=atol, rtol=rtol) and
        np.allclose(
            box_actual_array, box_reference_array, atol=atol, rtol=rtol))
    discrete_exact = not discrete_differences
    return {
        "status": (
            "PASS" if discrete_exact and numeric_within_tolerance else "FAIL"),
        "exact": actual == reference,
        "discrete_exact": discrete_exact,
        "numeric_within_tolerance": numeric_within_tolerance,
        "max_abs_score_diff": float(score_diff.max()) if score_diff.size else 0.0,
        "max_abs_box_diff": float(box_diff.max()) if box_diff.size else 0.0,
        "discrete_differences": discrete_differences,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def compare_stage_dirs(
    actual_dir: Path,
    reference_dir: Optional[Path],
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    if reference_dir is None:
        return {"status": "SKIP", "reason": "--compare-dir not provided"}
    comparisons = {}
    first_divergence = None
    ordered_names = (
        "raw_image.npy",
        "input_tensor_raw.npy", "input_tensor_dequant.npy",
        "feature_2d_raw.npy", "feature_2d_dequant.npy",
        "projection_dynamic.npy", "projection_gather.npy",
        "projection_scatter.npy", "bev_input_raw.npy",
        "bev_input_dequant.npy", "head_cls_raw.npy", "head_cls_dequant.npy",
        "head_bbox_raw.npy", "head_bbox_dequant.npy", "head_dir_raw.npy",
        "head_dir_dequant.npy", "anchors.npy", "decoded_anchors.npy",
        "center_boxes.npy",
    )
    for filename in ordered_names:
        path = actual_dir / filename
        if not path.is_file():
            continue
        reference = Path(reference_dir) / path.name
        if not reference.is_file():
            comparisons[path.stem] = {"status": "MISSING_REFERENCE"}
            continue
        actual = np.load(path)
        expected = np.load(reference)
        if actual.shape != expected.shape:
            item = {"status": "FAIL", "reason": "shape", "actual": list(actual.shape),
                    "reference": list(expected.shape)}
        else:
            diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            item = {
                "status": (
                    "PASS" if np.allclose(
                        actual, expected, atol=atol, rtol=rtol,
                        equal_nan=True) else "FAIL"),
                "exact": bool(np.array_equal(actual, expected)),
                "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
            }
        comparisons[path.stem] = item
        if item["status"] == "FAIL" and first_divergence is None:
            first_divergence = path.stem
    actual_business = actual_dir / "business_output.json"
    reference_business = Path(reference_dir) / "business_output.json"
    if actual_business.is_file():
        if not reference_business.is_file():
            comparisons["business_output"] = {"status": "MISSING_REFERENCE"}
        else:
            actual_value = json.loads(actual_business.read_text(encoding="utf-8"))
            reference_value = json.loads(
                reference_business.read_text(encoding="utf-8"))
            item = compare_business_output_values(
                actual_value, reference_value, atol, rtol)
            comparisons["business_output"] = item
            if item["status"] == "FAIL" and first_divergence is None:
                first_divergence = "business_output"
    category = None
    if first_divergence:
        if first_divergence.endswith("_raw") and first_divergence.startswith(
            ("input_tensor", "feature_2d", "head_", "bev_input")
        ):
            category = "quantization error"
        elif first_divergence.startswith("input_tensor"):
            category = "preprocess or quantization error"
        elif first_divergence.startswith("feature_2d"):
            category = "ONNX graph/runtime error"
        elif first_divergence.startswith("projection_") or first_divergence.startswith("bev_input"):
            category = "LUT geometry error"
        elif first_divergence.startswith("head_"):
            category = "ONNX graph/runtime error"
        elif first_divergence in ("decoded_anchors", "center_boxes"):
            item = comparisons[first_divergence]
            category = (
                "postprocess discrete boundary difference"
                if item.get("reason") == "shape" else
                "postprocess numeric difference")
        elif first_divergence == "business_output":
            item = comparisons[first_divergence]
            category = (
                "postprocess numeric difference"
                if item.get("discrete_exact") else
                "postprocess discrete boundary difference")
    missing_references = [
        name for name, item in comparisons.items()
        if item["status"] == "MISSING_REFERENCE"]
    status = (
        "FAIL" if first_divergence is not None else
        "INCOMPLETE" if missing_references else "PASS")
    return {
        "status": status,
        "first_divergence_stage": first_divergence,
        "classification": category,
        "missing_reference_stages": missing_references,
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    output_dir = Path(args.output_dir).expanduser().resolve()
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    if args.stage_input_dir:
        stages = load_stage_inputs(Path(args.stage_input_dir))
        required = {"head_cls", "head_bbox", "head_dir"}
        if not required <= set(stages):
            raise FileNotFoundError(
                f"stage input 缺少 raw logits: {sorted(required - set(stages))}")
        spec, _contract, _m2d, _m3d, anchors, _g, _s, _metadata = (
            load_analysis_assets(args))
        stages.setdefault("anchors", anchors)
        runtime = {"provider": "stage-input", "backend": args.backend}
    elif args.backend.startswith("onnx") and args.geometry == "fixed":
        stages, runtime, anchors, spec = run_fixed_onnx(args, output_dir)
    elif args.backend.startswith("onnx"):
        stages, runtime, anchors, spec = run_dynamic_onnx(args)
    elif args.geometry == "fixed":
        stages, runtime, anchors, spec = run_fixed_pth(args)
    else:
        stages, runtime, anchors, spec = run_dynamic_pth(args)

    stages["decoded_anchors"] = decode_all_anchors_center(stages, anchors)
    if args.postprocess == "pc":
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(
                f"config 不存在: {config_path}。{ASSET_INIT_HINT}")
        if board.file_sha256(config_path) != spec["config_sha256"]:
            raise RuntimeError(
                f"PC postprocess config SHA256 与 manifest 不一致: {config_path}")

    tensor_manifest: Dict[str, Any] = {}
    if "raw_image" in stages:
        np.save(tensor_dir / "raw_image.npy", stages["raw_image"])
    for base in ("input_tensor", "feature_2d", "bev_input", "head_cls", "head_bbox", "head_dir"):
        if base not in stages:
            continue
        raw = stages.get(f"{base}_raw", stages[base])
        tensor_manifest[base] = save_tensor(tensor_dir, base, raw, stages[base])
    for name in (
        "projection_dynamic", "projection_gather", "projection_scatter",
        "anchors", "decoded_anchors",
    ):
        if name in stages:
            np.save(tensor_dir / f"{name}.npy", stages[name])

    decoded, trace = decode_outputs(args, stages, anchors, spec)
    if trace.get("sigmoid_count") != 1:
        raise RuntimeError(f"后处理 sigmoid_count={trace.get('sigmoid_count')}，预期 1")
    center_boxes = np.asarray(decoded["boxes"], dtype=np.float32)
    np.save(tensor_dir / "center_boxes.npy", center_boxes)
    business = business_output(
        decoded, args.output_classes, args.score_thr, args.max_preds)
    write_json(output_dir / "business_output.json", business)
    write_json(tensor_dir / "business_output.json", business)
    write_json(output_dir / "postprocess_trace.json", trace)
    comparison = compare_stage_dirs(
        tensor_dir, Path(args.compare_dir) if args.compare_dir else None,
        args.atol, args.rtol)
    report = {
        "status": (
            comparison["status"] if args.compare_dir else "PASS"),
        "selection": {
            "backend": args.backend,
            "geometry": args.geometry,
            "preprocess": args.preprocess,
            "postprocess": args.postprocess,
        },
        "runtime": runtime,
        "tensor_manifest": tensor_manifest,
        "postprocess": {
            "sigmoid_count": trace.get("sigmoid_count"),
            "box_origin": decoded.get("box_origin", "center"),
            "decoded_count": int(len(center_boxes)),
        },
        "comparison": comparison,
        "int8_validation_statement": (
            "未做真实 INT8 模型数值验证"
            if args.backend == "onnx-int8" and
            spec["onnx_contracts"]["onnx-int8"].get("numerical_validation") ==
            "not_performed" else None),
    }
    write_json(output_dir / "analysis_report.json", report)
    status = comparison["status"] if args.compare_dir else "PASS"
    print(
        f"MONO_FRONT_ANALYSIS: {status}; "
        f"backend={args.backend}; geometry={args.geometry}; output={output_dir}")
    if args.strict and args.compare_dir and status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
