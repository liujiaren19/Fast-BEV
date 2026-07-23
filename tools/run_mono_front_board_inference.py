#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单文件 PC 板端参考推理。

这个脚本只做一条直线链路：

``图片 -> 前处理 -> 2D ONNX -> 固定 LUT -> 3D ONNX -> 后处理 -> 可视化``

它不导入项目内的推理、资产或可视化模块，不读取 config/PTH/pkl/info.json，
也不生成 LUT。板端工程师只需阅读本文件即可逐段核对前后处理。

首次运行前先执行 tools/build_mono_front_board_assets.py 初始化模型和车辆资产。

默认输出 ``business_predictions.jsonl``（一行一帧）和可视化；只有显式传入
``--save-frame-details`` 时才额外保存每帧完整 decoded/trace JSON。

示例：

  python tools/run_mono_front_board_inference.py \
  --images data/gt/20260514_9797/20260514103014_1.dat_img \
  --weights work_dirs/n7_mono_single_frame/fp_onnx \
  --lut-dir data/board_lut/9797_UKEF \
  --output-dir work_dirs/n7_board_runtime/9797_UKEF \
  --provider cuda \
  --output-classes car
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# =============================================================================
# 0. 文件内公共常量；模型数值契约全部来自 board_model_spec.json
# =============================================================================

ASSET_INIT_HINT = (
    "首次运行前先执行 tools/build_mono_front_board_assets.py 初始化模型和车辆资产。")

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
FRONT_EDGES = ((0, 1), (4, 5), (0, 4), (1, 5))
CLASS_COLORS = (
    (0, 220, 255),
    (255, 150, 40),
    (80, 220, 80),
    (220, 80, 220),
)

LOGGER = logging.getLogger("mono_front_board_inference")


# =============================================================================
# 1. 文件读取
# =============================================================================

def natural_key(value: Any) -> List[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def list_images(path: Path) -> List[Path]:
    """读取单图或递归读取目录中的图片。"""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"图片路径不存在: {path}")
    images = set()
    for pattern in IMAGE_PATTERNS:
        images.update(item.resolve() for item in path.rglob(pattern))
    return sorted(images, key=natural_key)


def extract_image_timestamp(path: Path) -> str:
    """从 ``1709058510936_1709058510936_1.jpg`` 的首段提取时间戳。"""
    first_field = Path(path).stem.split("_", 1)[0]
    if not first_field.isdigit():
        raise ValueError(
            f"图片名无法提取时间戳，要求首个下划线字段为纯数字: {path.name}")
    return first_field


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Mapping[str, Any]) -> str:
    """按资产构建器的稳定 JSON 编码计算契约 SHA256。"""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return value


def _require_sha(value: Any, label: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise RuntimeError(f"board_model_spec 缺少有效 {label} SHA256")
    return text


def _resolve_recorded_file(value: Any, weights_dir: Path, label: str) -> Path:
    recorded = Path(str(value or "")).expanduser()
    candidates = [recorded] if recorded.is_absolute() else [weights_dir / recorded]
    if not recorded.is_absolute():
        candidates.append(weights_dir / recorded.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{label} 不存在: recorded={value}; tried={candidates}。{ASSET_INIT_HINT}")


def _validate_tensor_contract(tensor: Mapping[str, Any], label: str) -> None:
    for key in ("name", "shape", "dtype", "layout"):
        if key not in tensor:
            raise RuntimeError(f"{label} 缺少 {key}: {tensor}")
    dtype = str(tensor["dtype"])
    if dtype in ("int8", "uint8"):
        quant = tensor.get("quantization")
        required = {
            "mode", "scale", "zero_point", "channel_axis", "round_policy",
            "saturation_policy", "clamp_min", "clamp_max",
        }
        if not isinstance(quant, dict) or not required <= set(quant):
            raise RuntimeError(f"{label} 缺少 raw {dtype} 量化参数")


def quantization_domain(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """提取跨 split 模型必须一致的量化数值域，忽略 Q/DQ 来源算子。"""
    keys = (
        "mode", "scale", "zero_point", "channel_axis", "round_policy",
        "saturation_policy", "clamp_min", "clamp_max",
    )
    return {key: value.get(key) for key in keys} if value else {}


def load_model_spec(
    weights_dir: Path,
    lut_dir: Path,
    backend: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path, Path]:
    """严格读取唯一 manifest，并解析模型与 anchors；不做候选文件猜测。"""
    weights_dir = Path(weights_dir).expanduser().resolve()
    spec_path = weights_dir / "board_model_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"缺少 {spec_path}。{ASSET_INIT_HINT}")
    spec = read_json(spec_path)
    if spec.get("schema_version") != 2:
        raise RuntimeError(
            f"不支持 board_model_spec schema_version={spec.get('schema_version')}。"
            f"{ASSET_INIT_HINT}")
    for key in (
        "config_sha256", "checkpoint_sha256", "model_contract_sha256",
        "geometry_contract_hash",
    ):
        _require_sha(spec.get(key), key)
    if spec.get("geometry_contract_hash") != spec.get("asset_contract_sha256"):
        raise RuntimeError("manifest geometry_contract_hash 字段互相不一致")
    geometry = spec.get("geometry", {})
    required_geometry = {
        "n_images", "n_times", "camera_types", "feature_shape_nchw",
        "n_voxels", "voxel_size", "origin", "bev_shape_nchw", "stride",
        "point_cloud_range", "channel_layout", "use_distortion", "data_config",
    }
    missing_geometry = sorted(required_geometry - set(geometry))
    if missing_geometry:
        raise RuntimeError(
            f"manifest geometry contract 不完整: {missing_geometry}")
    if geometry["channel_layout"] != "ZC":
        raise RuntimeError(
            f"standalone 只支持 BEV channel_layout=ZC，实际 "
            f"{geometry['channel_layout']}")
    if (geometry.get("n_images"), geometry.get("n_times")) != (1, 1):
        raise RuntimeError(
            "standalone runtime 只支持原生 S0 n_images=1,n_times=1")
    head = spec.get("head", {})
    if head.get("sigmoid_count") != 1 or head.get("public_box_origin") != "center":
        raise RuntimeError("manifest 必须声明 sigmoid_count=1 且 box_origin=center")
    required_head = {
        "class_names", "num_classes", "box_code_size", "num_dir_bins",
        "feature_map_hw", "num_anchors_per_location", "anchor_generator",
        "output_shapes",
        "score_thr", "nms_pre", "max_num", "nms_type_list",
        "nms_thr_list", "nms_radius_thr_list", "nms_rescale_factor",
        "circle_nms_post_max_size", "dir_offset", "dir_limit_offset",
    }
    missing_head = sorted(required_head - set(head))
    if missing_head:
        raise RuntimeError(f"manifest head contract 不完整: {missing_head}")
    num_classes = int(head["num_classes"])
    for key in (
        "class_names", "nms_type_list", "nms_thr_list",
        "nms_radius_thr_list", "nms_rescale_factor",
    ):
        if len(head[key]) != num_classes:
            raise RuntimeError(
                f"manifest head.{key} 数量={len(head[key])}，"
                f"num_classes={num_classes}")
    preprocess = spec.get("preprocess", {})
    required_preprocess = {
        "input_size_hw", "resize_backend", "to_rgb", "mean", "std",
    }
    missing_preprocess = sorted(required_preprocess - set(preprocess))
    if missing_preprocess:
        raise RuntimeError(
            f"manifest preprocess contract 不完整: {missing_preprocess}")

    model_contract = {
        "config_sha256": spec["config_sha256"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "preprocess": preprocess,
        "geometry": geometry,
        "head": head,
    }
    if json_sha256(model_contract) != spec["model_contract_sha256"]:
        raise RuntimeError("manifest model_contract_sha256 与模型契约内容不一致")
    asset_contract = {
        "geometry": {
            key: geometry[key]
            for key in (
                "n_images", "n_times", "camera_types", "n_voxels",
                "voxel_size", "origin", "feature_shape_nchw",
                "bev_shape_nchw", "stride", "channel_layout",
                "use_distortion", "data_config")
        },
        "head": {
            key: head[key]
            for key in (
                "feature_map_hw", "num_anchors_per_location", "box_code_size",
                "anchor_generator")
        },
    }
    if json_sha256(asset_contract) != spec["geometry_contract_hash"]:
        raise RuntimeError(
            "manifest geometry_contract_hash 与几何/anchor 契约内容不一致")

    contracts = spec.get("onnx_contracts")
    contract = contracts.get(backend) if isinstance(contracts, dict) else None
    if not isinstance(contract, dict) or not contract.get("available"):
        reason = contract.get("reason") if isinstance(contract, dict) else "missing"
        raise RuntimeError(
            f"manifest 不包含可用 {backend} contract: {reason}。{ASSET_INIT_HINT}")
    tensors = [
        contract.get("2d", {}).get("input", {}),
        contract.get("2d", {}).get("output", {}),
        contract.get("3d", {}).get("input", {}),
        *contract.get("3d", {}).get("outputs", {}).values(),
    ]
    for index, tensor in enumerate(tensors):
        _validate_tensor_contract(tensor, f"{backend}.tensor[{index}]")
    if contract["2d"]["input"]["layout"] != "nchw":
        raise RuntimeError("2D ONNX input layout 必须为 nchw")
    if contract["2d"]["output"]["layout"] not in ("nchw", "nhwc"):
        raise RuntimeError("2D ONNX output layout 必须为 nchw 或 nhwc")
    if contract["3d"]["input"]["layout"] != "nchw-zc":
        raise RuntimeError("3D ONNX input layout 必须为 nchw-zc")
    if any(
        output["layout"] != "nchw"
        for output in contract["3d"]["outputs"].values()
    ):
        raise RuntimeError("3D ONNX outputs layout 必须为 nchw")
    semantic_names = contract.get("3d", {}).get("semantic_output_names", {})
    if set(semantic_names) != {"head_cls", "head_bbox", "head_dir"}:
        raise RuntimeError(f"3D semantic output contract 不完整: {semantic_names}")
    if set(contract["3d"]["outputs"]) != set(semantic_names):
        raise RuntimeError(
            f"3D outputs contract 包含错误语义: "
            f"{sorted(contract['3d']['outputs'])}")
    for semantic, output_name in semantic_names.items():
        recorded_name = contract["3d"]["outputs"].get(semantic, {}).get("name")
        if output_name != recorded_name:
            raise RuntimeError(
                f"3D {semantic} 输出名契约不一致: semantic="
                f"{output_name}, tensor={recorded_name}")
    if contract["3d"]["outputs"]["head_cls"].get("has_upstream_sigmoid"):
        raise RuntimeError(
            f"{backend} head_cls 已包含 Sigmoid，不能满足 sigmoid_count=1")
    raw_io = any(
        tensor["dtype"] in ("int8", "uint8") for tensor in tensors)
    if bool(contract.get("manual_external_quantization")) != raw_io:
        raise RuntimeError(
            f"{backend} manual_external_quantization 与真实外部 dtype 不一致")
    feature_spec = contract["2d"]["output"]
    bev_spec = contract["3d"]["input"]
    raw_bridge = feature_spec["dtype"] in ("int8", "uint8")
    if raw_bridge:
        if (
            bev_spec["dtype"] != feature_spec["dtype"] or
            quantization_domain(bev_spec.get("quantization")) !=
            quantization_domain(feature_spec.get("quantization")) or
            contract.get("feature_to_bev_bridge") != "raw-quantized-direct"
        ):
            raise RuntimeError(
                f"{backend} 2D feature 与 3D input 量化域不一致")
    elif (
        bev_spec["dtype"] in ("int8", "uint8") or
        contract.get("feature_to_bev_bridge") != "float-direct"
    ):
        raise RuntimeError(f"{backend} float feature-to-BEV 契约不一致")
    for semantic, output in contract["3d"]["outputs"].items():
        if output["shape"] != head["output_shapes"].get(semantic):
            raise RuntimeError(
                f"{backend} {semantic} shape={output['shape']} 与 head contract="
                f"{head['output_shapes'].get(semantic)} 不一致")

    paths = []
    for part in ("2d", "3d"):
        record = contract.get("models", {}).get(part, {})
        path = _resolve_recorded_file(record.get("file"), weights_dir, f"{backend} {part} ONNX")
        if file_sha256(path) != _require_sha(record.get("sha256"), f"{backend} {part} ONNX"):
            raise RuntimeError(f"{backend} {part} ONNX SHA256 不匹配: {path}")
        paths.append(path)

    root = Path(lut_dir).expanduser().resolve()
    if root.name == "LUT":
        root = root.parent
    asset_root = root.parent
    anchors_record = spec.get("assets", {})
    anchors_path = asset_root / str(anchors_record.get("anchors_relative", ""))
    if not anchors_path.is_file():
        raise FileNotFoundError(f"anchors.npy 不存在: {anchors_path}。{ASSET_INIT_HINT}")
    if file_sha256(anchors_path) != _require_sha(
        anchors_record.get("anchors_sha256"), "anchors"
    ):
        raise RuntimeError(f"anchors.npy SHA256 不匹配: {anchors_path}")
    points_path = asset_root / str(anchors_record.get("points_relative", ""))
    if not points_path.is_file():
        raise FileNotFoundError(
            f"points.npy 不存在: {points_path}。{ASSET_INIT_HINT}")
    if file_sha256(points_path) != _require_sha(
        anchors_record.get("points_sha256"), "points"
    ):
        raise RuntimeError(f"points.npy SHA256 不匹配: {points_path}")
    points = np.load(points_path, mmap_mode="r")
    expected_points = tuple(
        int(value) for value in anchors_record.get("points_shape", ()))
    if tuple(points.shape) != expected_points:
        raise RuntimeError(
            f"points.npy shape={points.shape}，manifest={expected_points}")
    return spec, contract, paths[0], paths[1], anchors_path


def load_lut(
    lut_dir: Path,
    spec: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """直接读取板端三个 bin；缺少任意文件立即报错。"""
    root = Path(lut_dir).expanduser().resolve()
    if root.name == "LUT":
        root = root.parent
    board_dir = root / "LUT"
    gather_path = board_dir / "gather_new_0.bin"
    scatter_path = board_dir / "scatter_nd_new_0.bin"
    length_path = board_dir / "featurePointLength.bin"
    missing = [
        str(path) for path in (gather_path, scatter_path, length_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"固定 LUT 不完整，缺少: {missing}。{ASSET_INIT_HINT}")
    arr_paths = {
        "LUT_arr/gather_0.npy": root / "LUT_arr" / "gather_0.npy",
        "LUT_arr/scatter_nd_0.npy": root / "LUT_arr" / "scatter_nd_0.npy",
    }
    missing_arr = [str(path) for path in arr_paths.values() if not path.is_file()]
    if missing_arr:
        raise FileNotFoundError(
            f"固定 LUT_arr 不完整，缺少: {missing_arr}。{ASSET_INIT_HINT}")

    lengths = np.fromfile(length_path, dtype=np.int32)
    if lengths.shape != (1,):
        raise ValueError(
            "原生单帧 featurePointLength.bin 必须只有一个 int32，"
            f"实际 shape={lengths.shape}")
    gather = np.fromfile(gather_path, dtype=np.int32).astype(np.int64)
    scatter = np.fromfile(scatter_path, dtype=np.int32).astype(np.int64)
    expected_length = int(lengths[0])
    if gather.size != expected_length or scatter.size != expected_length:
        raise ValueError(
            f"featurePointLength={expected_length}，但 gather/scatter="
            f"{gather.size}/{scatter.size}")

    geometry = spec["geometry"]
    feature_shape = [int(value) for value in geometry["feature_shape_nchw"]]
    n_voxels = [int(value) for value in geometry["n_voxels"]]
    feature_points = feature_shape[2] * feature_shape[3]
    volume_points = int(np.prod(n_voxels))
    if gather.size and (gather.min() < 0 or gather.max() >= feature_points):
        raise ValueError(f"gather index 越界，feature_points={feature_points}")
    if scatter.size and (scatter.min() < 0 or scatter.max() >= volume_points):
        raise ValueError(f"scatter index 越界，volume_points={volume_points}")

    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"LUT 缺少 metadata.json，无法核对图片尺寸和可视化标定: {root}。"
            f"{ASSET_INIT_HINT}")
    metadata = read_json(metadata_path)
    if metadata.get("n_images") != 1 or metadata.get("n_times") != 1:
        raise ValueError(
            "本脚本只支持 n_images=1、n_times=1，LUT metadata 为 "
            f"{metadata.get('n_images')}/{metadata.get('n_times')}")
    if metadata.get("n_voxels") != n_voxels:
        raise ValueError(
            f"LUT n_voxels={metadata.get('n_voxels')}，预期 {n_voxels}")
    if metadata.get("feature_shape") != feature_shape:
        raise ValueError(
            f"LUT feature_shape={metadata.get('feature_shape')} 与模型不一致")
    if metadata.get("model_asset_profile") != spec.get("asset_profile"):
        raise RuntimeError("LUT asset profile 与模型不一致")
    if metadata.get("model_asset_contract_sha256") != spec.get(
        "asset_contract_sha256"
    ):
        raise RuntimeError("LUT asset contract hash 与模型不一致")
    if metadata.get("geometry_contract_hash") != spec.get("geometry_contract_hash"):
        raise RuntimeError("LUT geometry_contract_hash 与模型不一致")
    expected_hashes = metadata.get("lut_file_sha256")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("LUT metadata 缺少 lut_file_sha256")
    files = {
        "LUT/gather_new_0.bin": gather_path,
        "LUT/scatter_nd_new_0.bin": scatter_path,
        "LUT/featurePointLength.bin": length_path,
        **arr_paths,
    }
    for name, path in files.items():
        if file_sha256(path) != _require_sha(expected_hashes.get(name), name):
            raise RuntimeError(f"LUT SHA256 不匹配: {path}")
    gather_array = np.asarray(
        np.load(arr_paths["LUT_arr/gather_0.npy"]), dtype=np.int64).reshape(-1)
    scatter_array = np.asarray(
        np.load(arr_paths["LUT_arr/scatter_nd_0.npy"]), dtype=np.int64).reshape(-1)
    if not np.array_equal(gather, gather_array):
        raise RuntimeError("LUT gather bin 与 LUT_arr 不一致")
    if not np.array_equal(scatter, scatter_array):
        raise RuntimeError("LUT scatter bin 与 LUT_arr 不一致")
    return gather, scatter, metadata


def load_anchors(path: Path, spec: Mapping[str, Any]) -> np.ndarray:
    """读取固定 anchor；其行数和维度稍后由真实 3D ONNX 输出校验。"""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"anchors.npy 不存在: {path}。{ASSET_INIT_HINT}")
    anchors = np.asarray(np.load(path), dtype=np.float32)
    if anchors.ndim != 2 or anchors.shape[1] < 7:
        raise ValueError(f"anchors shape 无效: {anchors.shape}")
    expected = tuple(int(value) for value in spec["assets"]["anchors_shape"])
    if anchors.shape != expected:
        raise ValueError(f"anchors shape={anchors.shape}，manifest={expected}")
    return anchors


# =============================================================================
# 2. 图片前处理
# =============================================================================

def preprocess_image(
    path: Path,
    contract: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """BGR 读取 -> PIL bicubic resize -> RGB -> mmcv 兼容 normalize。"""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV 无法读取图片: {path}")
    image_h, image_w = image_bgr.shape[:2]

    # dataset 把 BGR 数组直接交给 RGB mode 的 Pillow；Pillow 只做 resize，
    # 真正的 BGR->RGB 发生在下面的 cv2.cvtColor。
    input_h, input_w = [int(value) for value in contract["input_size_hw"]]
    if contract.get("resize_backend") != "pipeline_pil_bicubic":
        raise RuntimeError(
            f"不支持的 resize_backend={contract.get('resize_backend')}")
    resized = np.asarray(
        Image.fromarray(image_bgr, mode="RGB").resize(
            (input_w, input_h), resample=Image.BICUBIC))
    normalized = resized.copy().astype(np.float32)
    if bool(contract.get("to_rgb")):
        cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB, normalized)

    # NormalizeMultiviewImage 先把 mean/std 保存成 float32，再提升到 float64
    # 交给 OpenCV 运算。这个顺序会影响约 1e-7 的末位，不能写成普通 NumPy 除法。
    mean = np.float64(np.asarray(
        contract["mean"], dtype=np.float32).reshape(1, -1))
    std_inv = 1.0 / np.float64(
        np.asarray(contract["std"], dtype=np.float32).reshape(1, -1))
    cv2.subtract(normalized, mean, normalized)
    cv2.multiply(normalized, std_inv, normalized)

    tensor = np.ascontiguousarray(
        normalized.transpose(2, 0, 1)[None], dtype=np.float32)
    expected = (1, 3, input_h, input_w)
    if tensor.shape != expected:
        raise ValueError(f"input shape={tensor.shape}，预期 {expected}")
    return tensor, image_bgr, [int(image_w), int(image_h)]


# =============================================================================
# 3. ONNX Runtime
# =============================================================================

def resolve_provider(requested: str) -> List[str]:
    available = ort.get_available_providers()
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"请求 CUDAExecutionProvider，但 available={available}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def validate_session_providers(
    session_2d: ort.InferenceSession,
    session_3d: ort.InferenceSession,
    requested: str,
) -> str:
    """核对 session 实际 provider，禁止显式 CUDA 请求静默回退 CPU。"""
    providers_2d = session_2d.get_providers()
    providers_3d = session_3d.get_providers()
    if not providers_2d or not providers_3d:
        raise RuntimeError(
            f"ONNX Runtime 未返回实际 provider: 2d={providers_2d}, "
            f"3d={providers_3d}")
    primary_2d = providers_2d[0]
    primary_3d = providers_3d[0]
    if primary_2d != primary_3d:
        raise RuntimeError(
            f"2D/3D ONNX 实际 provider 不一致: "
            f"2d={providers_2d}, 3d={providers_3d}")
    if requested == "cuda" and primary_2d != "CUDAExecutionProvider":
        raise RuntimeError(
            "请求 --provider cuda，但 CUDAExecutionProvider 创建失败并回退到了 "
            f"{primary_2d}: 2d={providers_2d}, 3d={providers_3d}")
    if requested == "cpu" and primary_2d != "CPUExecutionProvider":
        raise RuntimeError(
            f"请求 --provider cpu，但实际 primary provider={primary_2d}")
    return primary_2d


def semantic_3d_output_names(session: ort.InferenceSession) -> Dict[str, str]:
    """按真实输出名称识别 cls/bbox/dir，禁止按位置猜测。"""
    names = [item.name for item in session.get_outputs()]

    def unique_match(label: str, predicate) -> str:
        matches = [name for name in names if predicate(name.lower())]
        if len(matches) != 1:
            raise RuntimeError(
                f"3D ONNX 无法唯一识别 {label} 输出: outputs={names}, matches={matches}")
        return matches[0]

    return {
        "head_cls": unique_match(
            "cls", lambda name: "cls" in name and "dir" not in name),
        "head_bbox": unique_match("bbox", lambda name: "bbox" in name),
        "head_dir": unique_match("dir", lambda name: "dir" in name),
    }


def _quant_vector(
    quantization: Mapping[str, Any],
    ndim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(quantization["scale"], dtype=np.float32)
    zero = np.asarray(quantization["zero_point"], dtype=np.float32)
    if scale.size == 1:
        return scale.reshape(()), zero.reshape(())
    axis = quantization.get("channel_axis")
    if axis is None:
        raise RuntimeError("per-channel 量化缺少 channel_axis")
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise RuntimeError(f"per-channel channel_axis={axis} 超出 ndim={ndim}")
    shape = [1] * ndim
    shape[axis] = scale.size
    if zero.size == 1:
        zero = np.full(scale.shape, zero.item(), dtype=np.float32)
    return scale.reshape(shape), zero.reshape(shape)


def quantize_external(value: np.ndarray, tensor: Mapping[str, Any]) -> np.ndarray:
    """仅在真实 graph 外部 I/O 为 INT8/UINT8 时手工量化。"""
    dtype = str(tensor["dtype"])
    if dtype not in ("int8", "uint8"):
        return np.ascontiguousarray(value, dtype=np.dtype(dtype))
    quant = tensor["quantization"]
    if quant.get("round_policy") != "round-to-nearest-ties-to-even":
        raise RuntimeError(f"不支持的 round_policy={quant.get('round_policy')}")
    scale, zero = _quant_vector(quant, value.ndim)
    raw = np.rint(np.asarray(value, dtype=np.float32) / scale + zero)
    raw = np.clip(raw, int(quant["clamp_min"]), int(quant["clamp_max"]))
    return np.ascontiguousarray(raw.astype(np.dtype(dtype)))


def dequantize_external(value: np.ndarray, tensor: Mapping[str, Any]) -> np.ndarray:
    """raw cls 等 INT8/UINT8 输出必须在 sigmoid 前执行此处反量化。"""
    dtype = str(tensor["dtype"])
    if dtype not in ("int8", "uint8"):
        return np.ascontiguousarray(value, dtype=np.float32)
    scale, zero = _quant_vector(tensor["quantization"], value.ndim)
    return np.ascontiguousarray(
        (np.asarray(value, dtype=np.float32) - zero) * scale,
        dtype=np.float32)


def feature_to_nchw_once(feature: np.ndarray, layout: str) -> np.ndarray:
    layout = str(layout).lower()
    if layout == "nchw":
        return np.ascontiguousarray(feature)
    if layout == "nhwc":
        return np.ascontiguousarray(feature.transpose(0, 3, 1, 2))
    raise RuntimeError(f"不支持的 feature layout={layout}")


def run_2d_onnx(
    session: ort.InferenceSession,
    input_tensor: np.ndarray,
    contract: Mapping[str, Any],
) -> np.ndarray:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("2D ONNX 必须是单输入、单输出")
    input_spec = contract["input"]
    output_spec = contract["output"]
    if inputs[0].name != input_spec["name"] or outputs[0].name != output_spec["name"]:
        raise RuntimeError("2D ONNX session I/O 名称与 manifest 不一致")
    encoded = quantize_external(input_tensor, input_spec)
    expected_input = tuple(int(value) for value in input_spec["shape"])
    if encoded.shape != expected_input or str(encoded.dtype) != input_spec["dtype"]:
        raise RuntimeError(
            f"2D input shape/dtype={encoded.shape}/{encoded.dtype}，"
            f"manifest={expected_input}/{input_spec['dtype']}")
    feature = session.run(
        [output_spec["name"]],
        {input_spec["name"]: encoded},
    )[0]
    feature = np.asarray(feature)
    expected = tuple(int(value) for value in output_spec["shape"])
    if feature.shape != expected or str(feature.dtype) != output_spec["dtype"]:
        raise RuntimeError(
            f"2D feature shape/dtype={feature.shape}/{feature.dtype}，"
            f"manifest={expected}/{output_spec['dtype']}")
    return np.ascontiguousarray(feature)


def run_3d_onnx(
    session: ort.InferenceSession,
    bev_input: np.ndarray,
    output_names: Mapping[str, str],
    contract: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError("3D ONNX 必须是单输入")
    input_spec = contract["input"]
    if inputs[0].name != input_spec["name"]:
        raise RuntimeError("3D ONNX input name 与 manifest 不一致")
    expected_input = tuple(int(value) for value in input_spec["shape"])
    if bev_input.shape != expected_input or str(bev_input.dtype) != input_spec["dtype"]:
        raise RuntimeError(
            f"3D input shape/dtype={bev_input.shape}/{bev_input.dtype}，"
            f"manifest={expected_input}/{input_spec['dtype']}")
    requested = [
        output_names["head_cls"],
        output_names["head_bbox"],
        output_names["head_dir"],
    ]
    values = session.run(
        requested,
        {input_spec["name"]: np.ascontiguousarray(
            bev_input, dtype=np.dtype(input_spec["dtype"]))},
    )
    by_name = dict(zip(requested, values))
    result = {}
    for semantic, name in output_names.items():
        raw = np.asarray(by_name[name])
        output_spec = contract["outputs"][semantic]
        expected = tuple(int(value) for value in output_spec["shape"])
        if raw.shape != expected or str(raw.dtype) != output_spec["dtype"]:
            raise RuntimeError(
                f"{semantic} shape/dtype={raw.shape}/{raw.dtype}，"
                f"manifest={expected}/{output_spec['dtype']}")
        result[semantic] = np.ascontiguousarray(raw)
    return result


# =============================================================================
# 4. 固定 LUT：2D feature -> BEV
# =============================================================================

def apply_lut(
    feature: np.ndarray,
    gather: np.ndarray,
    scatter: np.ndarray,
    geometry: Mapping[str, Any],
    fill_value: Any = 0,
) -> np.ndarray:
    """按板端 gather/scatter 生成 [1,Z*C,X,Y]。"""
    expected = tuple(int(value) for value in geometry["feature_shape_nchw"])
    if feature.shape != expected:
        raise ValueError(f"feature shape={feature.shape}，预期 {expected}")

    _, channels, feature_h, feature_w = expected
    n_voxels = tuple(int(value) for value in geometry["n_voxels"])
    flat_feature = feature.reshape(1, channels, feature_h * feature_w)
    fill = np.asarray(fill_value, dtype=feature.dtype)
    if fill.size == 1:
        volume = np.full(
            (channels, int(np.prod(n_voxels))), fill.item(), dtype=feature.dtype)
    else:
        if fill.size != channels:
            raise ValueError(
                f"per-channel zero_point={fill.size}，feature channels={channels}")
        volume = np.broadcast_to(
            fill.reshape(channels, 1),
            (channels, int(np.prod(n_voxels)))).copy()
    volume[:, scatter] = flat_feature[0][:, gather]

    # 先恢复 [C,X,Y,Z]，再转为 [Z,C,X,Y] 后折叠 Z*C。直接 reshape 会
    # 改变已经训练好的 3D neck 通道语义。
    bev = (
        volume.reshape(channels, *n_voxels)
        .transpose(3, 0, 1, 2)
        .reshape(tuple(int(value) for value in geometry["bev_shape_nchw"]))
    )
    return np.ascontiguousarray(bev)


# =============================================================================
# 5. Anchor decode 与 CPU rotated NMS
# =============================================================================

def sigmoid_once(logits: np.ndarray) -> np.ndarray:
    """分类 raw logits 在全链只经过这一次 sigmoid。"""
    values = np.asarray(logits, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def decode_anchor_deltas(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """复现 DeltaXYZWLHRBBoxCoder.decode，输出仍为 bottom-origin。"""
    if anchors.shape != deltas.shape:
        raise ValueError(
            f"anchors/deltas shape 不一致: {anchors.shape}/{deltas.shape}")
    xa, ya, za, wa, la, ha, ra = [anchors[:, index] for index in range(7)]
    xt, yt, zt, wt, lt, ht, rt = [deltas[:, index] for index in range(7)]
    za_center = za + ha / 2.0
    diagonal = np.sqrt(la * la + wa * wa)
    decoded = np.empty_like(deltas, dtype=np.float32)
    decoded[:, 0] = xt * diagonal + xa
    decoded[:, 1] = yt * diagonal + ya
    decoded[:, 3] = np.exp(wt) * wa
    decoded[:, 4] = np.exp(lt) * la
    decoded[:, 5] = np.exp(ht) * ha
    decoded[:, 2] = zt * ha + za_center - decoded[:, 5] / 2.0
    decoded[:, 6] = rt + ra
    if anchors.shape[1] > 7:
        decoded[:, 7:] = deltas[:, 7:] + anchors[:, 7:]
    return decoded


def signed_polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def bev_box_corners(box: np.ndarray) -> np.ndarray:
    x, y, width, length, yaw = [float(value) for value in box]
    corners = np.asarray([
        [width * 0.5, length * 0.5],
        [width * 0.5, -length * 0.5],
        [-width * 0.5, -length * 0.5],
        [-width * 0.5, length * 0.5],
    ], dtype=np.float32)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.asarray(
        [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]], dtype=np.float32)
    return corners @ rotation.T + np.asarray([x, y], dtype=np.float32)


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def convex_intersection(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman 凸多边形裁剪，用于 CPU rotated IoU。"""
    if len(subject) == 0 or len(clipper) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    output = [np.asarray(point, dtype=np.float32) for point in subject]
    orientation = 1.0 if signed_polygon_area(clipper) >= 0 else -1.0

    def inside(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
        return orientation * cross_2d(end - start, point - start) >= -1e-6

    def intersection(
        line_start: np.ndarray,
        line_end: np.ndarray,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
    ) -> np.ndarray:
        line = line_end - line_start
        edge = edge_end - edge_start
        denominator = cross_2d(line, edge)
        if abs(denominator) < 1e-8:
            return line_end
        factor = cross_2d(edge_start - line_start, edge) / denominator
        return line_start + factor * line

    for index in range(len(clipper)):
        edge_start = clipper[index]
        edge_end = clipper[(index + 1) % len(clipper)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = inside(current, edge_start, edge_end)
            previous_inside = inside(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(
                        previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(intersection(
                    previous, current, edge_start, edge_end))
            previous = current
    if not output:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(output).astype(np.float32)


def rotated_nms_cpu(
    boxes_xywlr: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """不依赖 CUDA/mmcv 的 class-wise rotated NMS。"""
    boxes = np.asarray(boxes_xywlr, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape != (scores.size, 5):
        raise ValueError(
            f"NMS boxes/scores shape 错误: {boxes.shape}/{scores.shape}")
    if scores.size == 0:
        return np.zeros((0,), dtype=np.int64)

    polygons = [bev_box_corners(box) for box in boxes]
    areas = np.asarray([
        abs(signed_polygon_area(polygon)) for polygon in polygons
    ], dtype=np.float32)
    mins = np.asarray([polygon.min(axis=0) for polygon in polygons])
    maxs = np.asarray([polygon.max(axis=0) for polygon in polygons])
    order = np.argsort(-scores, kind="stable")
    keep: List[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if remaining.size == 0:
            break
        overlap_aabb = (
            (maxs[remaining, 0] >= mins[current, 0]) &
            (mins[remaining, 0] <= maxs[current, 0]) &
            (maxs[remaining, 1] >= mins[current, 1]) &
            (mins[remaining, 1] <= maxs[current, 1])
        )
        suppressed = np.zeros(remaining.size, dtype=bool)
        for local_index in np.nonzero(overlap_aabb)[0]:
            candidate = int(remaining[local_index])
            intersection = convex_intersection(
                polygons[current], polygons[candidate])
            intersection_area = abs(signed_polygon_area(intersection))
            union = float(areas[current] + areas[candidate] - intersection_area)
            iou = intersection_area / union if union > 0 else 0.0
            if iou > float(iou_threshold):
                suppressed[local_index] = True
        order = remaining[~suppressed]
    return np.asarray(keep, dtype=np.int64)


def circle_nms_cpu(
    centers_xy: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    post_max_size: int = 83,
) -> np.ndarray:
    """复现 canonical circle_nms 的距离平方阈值与默认上限。"""
    centers = np.asarray(centers_xy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(scores.size, dtype=bool)
    keep: List[int] = []
    for order_index, current in enumerate(order):
        current = int(current)
        if suppressed[current]:
            continue
        keep.append(current)
        rest = order[order_index + 1:]
        if rest.size:
            delta = centers[rest] - centers[current]
            suppressed[rest[np.sum(delta * delta, axis=1) <= threshold]] = True
    return np.asarray(keep[:post_max_size], dtype=np.int64)


def cutoff_tie(values: np.ndarray, keep_count: int) -> Dict[str, Any]:
    """记录 top-k/max_num 边界是否存在同分 tie，不替代板端排序规则。"""
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    if keep_count <= 0 or scores.size <= keep_count:
        return {"present": False, "cutoff": None, "equal_count": 0}
    ordered = np.sort(scores)[::-1]
    cutoff = float(ordered[keep_count - 1])
    equal_count = int(np.count_nonzero(scores == cutoff))
    return {
        "present": equal_count > 1,
        "cutoff": cutoff,
        "equal_count": equal_count,
    }


def postprocess(
    logits: Mapping[str, np.ndarray],
    anchors: np.ndarray,
    head: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """canonical 顺序：sigmoid/top-k/decode/score/NMS/max/yaw/center。"""
    cls = np.asarray(logits["head_cls"], dtype=np.float32)
    bbox = np.asarray(logits["head_bbox"], dtype=np.float32)
    direction = np.asarray(logits["head_dir"], dtype=np.float32)
    if cls.ndim != 4 or bbox.ndim != 4 or direction.ndim != 4:
        raise ValueError("三个 3D head 输出必须是 NCHW 4D tensor")
    if cls.shape[0] != 1 or bbox.shape[0] != 1 or direction.shape[0] != 1:
        raise ValueError("当前板端参考只支持 batch=1")
    if cls.shape[-2:] != bbox.shape[-2:] or cls.shape[-2:] != direction.shape[-2:]:
        raise ValueError("cls/bbox/dir 空间 shape 不一致")

    feature_h, feature_w = [int(value) for value in cls.shape[-2:]]
    locations = feature_h * feature_w
    if anchors.shape[0] % locations != 0:
        raise ValueError(
            f"anchor rows={anchors.shape[0]} 不能被 head locations={locations} 整除")
    anchors_per_location = anchors.shape[0] // locations
    if (
        cls.shape[1] % anchors_per_location != 0 or
        bbox.shape[1] % anchors_per_location != 0 or
        direction.shape[1] % anchors_per_location != 0
    ):
        raise ValueError("cls/bbox/dir channels 不能按 anchors_per_location 拆分")
    num_classes = cls.shape[1] // anchors_per_location
    box_code_size = bbox.shape[1] // anchors_per_location
    num_dir_bins = direction.shape[1] // anchors_per_location
    if min(num_classes, box_code_size, num_dir_bins) <= 0:
        raise ValueError("由 ONNX 输出推导的 head 维度必须大于 0")
    if anchors.shape[1] != box_code_size:
        raise ValueError(
            f"anchor dims={anchors.shape[1]}，ONNX box_code_size={box_code_size}")
    class_names = [str(value) for value in head["class_names"]]
    if num_classes != int(head["num_classes"]) or len(class_names) != num_classes:
        raise ValueError(
            f"ONNX num_classes={num_classes} 与 manifest={head.get('num_classes')} 不一致")
    if box_code_size != int(head["box_code_size"]):
        raise ValueError("ONNX box_code_size 与 manifest 不一致")
    if num_dir_bins != int(head["num_dir_bins"]):
        raise ValueError("ONNX num_dir_bins 与 manifest 不一致")

    cls_flat = cls[0].transpose(1, 2, 0).reshape(-1, num_classes)
    bbox_flat = bbox[0].transpose(1, 2, 0).reshape(-1, box_code_size)
    dir_flat = direction[0].transpose(1, 2, 0).reshape(-1, num_dir_bins)
    if cls_flat.shape[0] != anchors.shape[0]:
        raise ValueError(
            f"head candidates={cls_flat.shape[0]}，anchors={anchors.shape[0]}")

    scores = sigmoid_once(cls_flat)
    dir_labels = np.argmax(dir_flat, axis=1).astype(np.int64)
    selected_anchors = anchors
    source_indices = np.arange(anchors.shape[0], dtype=np.int64)

    nms_pre = int(head["nms_pre"])
    topk_boundary_tie = cutoff_tie(scores.max(axis=1), nms_pre)
    if 0 < nms_pre < scores.shape[0]:
        max_scores = scores.max(axis=1)
        # 与此前 PC 板端参考一致：score 降序，相同 score 保留 anchor 原序。
        topk = np.argsort(-max_scores, kind="stable")[:nms_pre]
        selected_anchors = selected_anchors[topk]
        bbox_flat = bbox_flat[topk]
        scores = scores[topk]
        dir_labels = dir_labels[topk]
        source_indices = source_indices[topk]

    decoded_bottom = decode_anchor_deltas(selected_anchors, bbox_flat)
    selected_boxes: List[np.ndarray] = []
    selected_scores: List[np.ndarray] = []
    selected_labels: List[np.ndarray] = []
    selected_dirs: List[np.ndarray] = []
    selected_source_indices: List[np.ndarray] = []
    per_class = []
    for class_index, class_name in enumerate(class_names):
        candidate_indices = np.nonzero(
            scores[:, class_index] > float(head["score_thr"]))[0]
        class_scores = scores[candidate_indices, class_index]
        if candidate_indices.size == 0:
            per_class.append({
                "class_name": class_name,
                "above_score_thr": 0,
                "selected_after_nms": 0,
                "candidate_source_anchor_indices": [],
                "selected_source_anchor_indices": [],
            })
            continue

        nms_type = str(head["nms_type_list"][class_index])
        if nms_type == "circle":
            keep_local = circle_nms_cpu(
                decoded_bottom[candidate_indices, :2], class_scores,
                float(head["nms_radius_thr_list"][class_index]),
                int(head["circle_nms_post_max_size"]))
        elif nms_type == "rotate":
            nms_boxes = decoded_bottom[candidate_indices][
                :, [0, 1, 3, 4, 6]
            ].copy()
            nms_boxes[:, 2:4] *= float(
                head["nms_rescale_factor"][class_index])
            keep_local = rotated_nms_cpu(
                nms_boxes, class_scores,
                float(head["nms_thr_list"][class_index]))
        else:
            raise RuntimeError(f"不支持的 nms_type={nms_type}")
        selected = candidate_indices[keep_local]
        selected_boxes.append(decoded_bottom[selected])
        selected_scores.append(scores[selected, class_index])
        selected_labels.append(np.full(
            selected.size, class_index, dtype=np.int64))
        selected_dirs.append(dir_labels[selected])
        selected_source_indices.append(source_indices[selected])
        per_class.append({
            "class_name": class_name,
            "above_score_thr": int(candidate_indices.size),
            "selected_after_nms": int(selected.size),
            "candidate_source_anchor_indices": source_indices[
                candidate_indices].astype(int).tolist(),
            "selected_source_anchor_indices": source_indices[
                selected].astype(int).tolist(),
        })

    if selected_boxes:
        boxes = np.concatenate(selected_boxes).astype(np.float32)
        output_scores = np.concatenate(selected_scores).astype(np.float32)
        labels = np.concatenate(selected_labels).astype(np.int64)
        directions = np.concatenate(selected_dirs).astype(np.int64)
        source_anchor_indices = np.concatenate(
            selected_source_indices).astype(np.int64)
    else:
        boxes = np.zeros((0, box_code_size), dtype=np.float32)
        output_scores = np.zeros((0,), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)
        directions = np.zeros((0,), dtype=np.int64)
        source_anchor_indices = np.zeros((0,), dtype=np.int64)

    max_num = int(head["max_num"])
    max_num_boundary_tie = cutoff_tie(output_scores, max_num)
    if max_num > 0 and len(boxes) > max_num:
        order = np.argsort(-output_scores, kind="stable")[:max_num]
        boxes = boxes[order]
        output_scores = output_scores[order]
        labels = labels[order]
        directions = directions[order]
        source_anchor_indices = source_anchor_indices[order]

    if len(boxes):
        dir_offset = float(head["dir_offset"])
        dir_limit_offset = float(head["dir_limit_offset"])
        raw_yaw = boxes[:, 6] - dir_offset
        limited_yaw = raw_yaw - np.floor(
            raw_yaw / np.pi + dir_limit_offset) * np.pi
        boxes[:, 6] = (
            limited_yaw + dir_offset + np.pi * directions.astype(np.float32))
        # bbox coder 的 z 是底中心；公共输出只在这里转换一次到 center-origin。
        boxes[:, 2] += boxes[:, 5] * 0.5

    decoded = {
        "boxes": boxes,
        "scores": output_scores,
        "labels": labels,
        "source_anchor_indices": source_anchor_indices,
        "box_origin": "center",
        "class_names": class_names,
    }
    trace = {
        "sigmoid_count": 1,
        "num_classes": num_classes,
        "anchors_per_location": anchors_per_location,
        "box_code_size": box_code_size,
        "num_dir_bins": num_dir_bins,
        "nms_pre": nms_pre,
        "post_topk_source_anchor_indices": source_indices.astype(int).tolist(),
        "topk_boundary_tie": topk_boundary_tie,
        "model_score_thr": float(head["score_thr"]),
        "nms_type_list": list(head["nms_type_list"]),
        "nms_thr_list": list(head["nms_thr_list"]),
        "max_num": max_num,
        "max_num_boundary_tie": max_num_boundary_tie,
        "per_class": per_class,
        "selected_count": int(len(boxes)),
        "selected_source_anchor_indices": source_anchor_indices.astype(
            int).tolist(),
        "box_origin": "center",
    }
    return decoded, trace


# =============================================================================
# 6. 业务过滤、JSON 与可视化
# =============================================================================

def decoded_rows(decoded: Mapping[str, Any]) -> List[Dict[str, Any]]:
    class_names = decoded["class_names"]
    rows = []
    for box, score, label, anchor in zip(
        decoded["boxes"], decoded["scores"], decoded["labels"],
        decoded["source_anchor_indices"]
    ):
        label_id = int(label)
        rows.append({
            "label": label_id,
            "class_name": class_names[label_id],
            "score": float(score),
            "box": np.asarray(box, dtype=np.float32).tolist(),
            "source_anchor_index": int(anchor),
            "box_origin": "center",
        })
    return rows


def select_business_output(
    decoded: Mapping[str, Any],
    requested_classes: Sequence[str],
    score_threshold: float,
    max_predictions: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """完整模型 decode/NMS 后才执行 car 等业务过滤。"""
    class_names = decoded["class_names"]
    requested = set(
        class_names if "all" in requested_classes else requested_classes)
    unknown = requested - set(class_names)
    if unknown:
        raise ValueError(f"模型不存在这些业务类别: {sorted(unknown)}")
    labels = np.asarray(decoded["labels"], dtype=np.int64)
    scores = np.asarray(decoded["scores"], dtype=np.float32)
    allowed_ids = {
        index for index, name in enumerate(class_names) if name in requested}
    mask = np.asarray([
        int(label) in allowed_ids and float(score) >= score_threshold
        for label, score in zip(labels, scores)
    ], dtype=bool)
    indices = np.nonzero(mask)[0]
    indices = np.asarray(sorted(
        indices.tolist(),
        key=lambda index: (
            -float(scores[index]),
            int(decoded["source_anchor_indices"][index]),
            int(labels[index])),
    ), dtype=np.int64)
    if max_predictions > 0:
        indices = indices[:max_predictions]
    selected = {
        "boxes": np.asarray(decoded["boxes"])[indices],
        "scores": scores[indices],
        "labels": labels[indices],
        "source_anchor_indices": np.asarray(
            decoded["source_anchor_indices"], dtype=np.int64)[indices],
        "box_origin": "center",
        "class_names": class_names,
    }
    rows = decoded_rows(selected)
    return selected, rows


def box_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """按 ``+X`` 前、``+Y`` 左、正 yaw 朝 ``+Y`` 生成 8 个 LiDAR 角点。"""
    corners = np.zeros((len(boxes), 8, 3), dtype=np.float32)
    for index, box in enumerate(np.asarray(boxes, dtype=np.float32)):
        x, y, z, length, width, height, yaw = [float(v) for v in box[:7]]
        local = np.asarray([
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ], dtype=np.float32)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        local_xy = local[:, :2].copy()
        local[:, 0] = local_xy[:, 0] * cos_yaw - local_xy[:, 1] * sin_yaw
        local[:, 1] = local_xy[:, 0] * sin_yaw + local_xy[:, 1] * cos_yaw
        corners[index] = local + np.asarray([x, y, z], dtype=np.float32)
    return corners


def scaled_intrinsic(calibration: Mapping[str, Any], image_size: Sequence[int]) -> np.ndarray:
    """把 calibration K 从 intrinsic size 缩放到当前原图尺寸。"""
    image_w, image_h = [int(value) for value in image_size]
    intrinsic_w = int(calibration["intrinsic_width"])
    intrinsic_h = int(calibration["intrinsic_height"])
    if min(image_w, image_h, intrinsic_w, intrinsic_h) <= 0:
        raise ValueError("相机或图片尺寸无效")
    intrinsic = np.asarray(calibration["cam_intrinsic"], dtype=np.float64).copy()
    intrinsic[0, :] *= image_w / float(intrinsic_w)
    intrinsic[1, :] *= image_h / float(intrinsic_h)
    return intrinsic


def project_lidar_points(
    points: np.ndarray,
    calibration: Mapping[str, Any],
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """将 LiDAR/Fast-BEV 坐标点投影到 original-K 去畸变图。"""
    sensor_to_lidar_r = np.asarray(
        calibration["sensor2lidar_rotation"], dtype=np.float64)
    sensor_to_lidar_t = np.asarray(
        calibration["sensor2lidar_translation"], dtype=np.float64).reshape(3)
    lidar_to_camera_r = np.linalg.inv(sensor_to_lidar_r)
    camera_points = (
        np.asarray(points, dtype=np.float64) - sensor_to_lidar_t
    ) @ lidar_to_camera_r.T
    depth = camera_points[:, 2]
    projected = camera_points @ intrinsic.T
    uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
    return uv, depth


def draw_camera_boxes(
    image: np.ndarray,
    decoded: Mapping[str, Any],
    calibration: Mapping[str, Any],
    timestamp: str,
) -> np.ndarray:
    """original-K 去畸变后，投影 3D 框并标注置信度和帧时间戳。"""
    h, w = image.shape[:2]
    intrinsic = scaled_intrinsic(calibration, (w, h))
    distortion = np.asarray(
        calibration.get("distortion", []), dtype=np.float64).reshape(-1)
    if distortion.size not in (0, 4, 5, 8, 12, 14):
        padded = np.zeros(8, dtype=np.float64)
        padded[:min(8, distortion.size)] = distortion[:8]
        distortion = padded
    undistorted = (
        image.copy() if distortion.size == 0 else
        cv2.undistort(image, intrinsic, distortion, None, intrinsic)
    )

    # 先在原始分辨率上绘制，再随整张前视图一起缩放。动态字号可以避免
    # 3840x2160 原图缩到 1600 宽后，时间戳和框置信度小到无法辨认。
    font_scale = max(0.55, min(h, w) / 1000.0)
    text_thickness = max(2, int(round(font_scale * 1.5)))

    corners = box_corners_3d(decoded["boxes"])
    for box_index, points in enumerate(corners):
        uv, depth = project_lidar_points(points, calibration, intrinsic)
        label_id = int(decoded["labels"][box_index])
        score = float(decoded["scores"][box_index])
        color = CLASS_COLORS[label_id % len(CLASS_COLORS)]
        thickness = max(1, int(round(min(h, w) / 420.0)))
        for start, end in BOX_EDGES:
            if depth[start] <= 0.1 or depth[end] <= 0.1:
                continue
            if not np.all(np.isfinite(uv[[start, end]])):
                continue
            pt1 = tuple(np.round(np.clip(uv[start], -1e6, 1e6)).astype(int))
            pt2 = tuple(np.round(np.clip(uv[end], -1e6, 1e6)).astype(int))
            ok, clipped1, clipped2 = cv2.clipLine((0, 0, w, h), pt1, pt2)
            if ok:
                edge_thickness = thickness + 1 if (start, end) in FRONT_EDGES else thickness
                cv2.line(
                    undistorted, clipped1, clipped2, color,
                    edge_thickness, cv2.LINE_AA)
        valid = (depth > 0.1) & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        inside = valid & (
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h))
        if np.any(inside):
            anchor = uv[inside][np.argmin(uv[inside][:, 1])]
            name = decoded["class_names"][label_id]
            cv2.putText(
                undistorted, f"{name} {score:.2f}",
                (int(anchor[0]), max(18, int(anchor[1]) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                text_thickness, cv2.LINE_AA)

    timestamp_text = f"timestamp: {timestamp}"
    (text_w, text_h), baseline = cv2.getTextSize(
        timestamp_text, cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, text_thickness)
    margin = max(8, int(round(font_scale * 8)))
    cv2.rectangle(
        undistorted,
        (margin // 2, margin // 2),
        (margin + text_w, margin + text_h + baseline),
        (0, 0, 0), -1)
    cv2.putText(
        undistorted, timestamp_text,
        (margin, margin + text_h),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
        text_thickness, cv2.LINE_AA)
    return undistorted


def bev_to_pixel(
    points_xy: np.ndarray,
    size: int,
    point_cloud_range: Sequence[float],
) -> np.ndarray:
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    u = (y_max - y) / (y_max - y_min) * (size - 1)
    v = (x_max - x) / (x_max - x_min) * (size - 1)
    return np.stack([u, v], axis=1)


def draw_bev(
    decoded: Mapping[str, Any],
    size: int,
    point_cloud_range: Sequence[float],
) -> np.ndarray:
    """绘制前向 +x、左向 +y 的 BEV 框和 10 m 网格。"""
    image = np.full((size, size, 3), 20, dtype=np.uint8)
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    grid_color = (50, 50, 50)
    for x in range(int(x_min), int(x_max) + 1, 10):
        points = bev_to_pixel(
            np.asarray([[x, y_min], [x, y_max]], dtype=np.float32), size,
            point_cloud_range)
        cv2.line(image, tuple(points[0].astype(int)),
                 tuple(points[1].astype(int)), grid_color, 1)
    for y in range(int(y_min), int(y_max) + 1, 10):
        points = bev_to_pixel(
            np.asarray([[x_min, y], [x_max, y]], dtype=np.float32), size,
            point_cloud_range)
        cv2.line(image, tuple(points[0].astype(int)),
                 tuple(points[1].astype(int)), grid_color, 1)

    ego = bev_to_pixel(
        np.asarray([[0.0, 0.0]], dtype=np.float32), size,
        point_cloud_range)[0]
    cv2.circle(image, tuple(ego.astype(int)), max(3, size // 150),
               (245, 245, 245), -1, cv2.LINE_AA)
    corners = box_corners_3d(decoded["boxes"])
    for index, box_corners in enumerate(corners):
        label_id = int(decoded["labels"][index])
        color = CLASS_COLORS[label_id % len(CLASS_COLORS)]
        points = bev_to_pixel(
            box_corners[:4, :2], size, point_cloud_range).astype(np.int32)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        cv2.line(image, tuple(points[0]), tuple(points[1]),
                 (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(image, tuple(points[0]), tuple(points[1]),
                 color, 1, cv2.LINE_AA)

    # 当前 BEV 像素坐标中，车辆坐标 +X（向前）对应画面向上，+Y（向左）
    # 对应画面向左。图例放在右下角，避免把图像坐标的右/下方向误认为
    # 车辆坐标的正方向。
    axis_length = max(36, size // 9)
    margin = max(18, size // 30)
    origin = (size - margin, size - margin)
    x_end = (origin[0], origin[1] - axis_length)
    y_end = (origin[0] - axis_length, origin[1])
    axis_thickness = max(2, size // 360)
    font_scale = max(0.45, size / 1200.0)
    cv2.arrowedLine(
        image, origin, x_end, (0, 0, 255), axis_thickness,
        cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(
        image, origin, y_end, (0, 255, 0), axis_thickness,
        cv2.LINE_AA, tipLength=0.18)
    cv2.circle(image, origin, axis_thickness + 1, (255, 255, 255), -1)
    x_label = "+X"
    (x_text_w, x_text_h), _ = cv2.getTextSize(
        x_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, axis_thickness)
    # +X 文字以竖直箭头为中心，并显式限制在画布内；不能从靠近右边界的
    # 箭头右侧开始绘制，否则高分辨率 BEV 缩放后字母 X 会被裁掉。
    x_text_x = min(
        max(2, x_end[0] - x_text_w // 2), size - x_text_w - 2)
    x_text_y = max(x_text_h + 2, x_end[1] - 6)
    cv2.putText(
        image, x_label, (x_text_x, x_text_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
        axis_thickness, cv2.LINE_AA)
    cv2.putText(
        image, "+Y", (max(2, y_end[0] - 34), y_end[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
        axis_thickness, cv2.LINE_AA)
    return image


def render_visualization(
    image_bgr: np.ndarray,
    decoded: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_path: Path,
    visual_width: int,
    timestamp: str,
    point_cloud_range: Sequence[float],
) -> None:
    """输出带时间戳/框置信度的 original-K 去畸变前视图和 BEV。"""
    calibration = metadata.get("calibration")
    if not isinstance(calibration, dict):
        raise RuntimeError(
            "LUT metadata 缺少 calibration；不能可靠绘制前视 3D 框")
    front = draw_camera_boxes(
        image_bgr, decoded, calibration, timestamp)
    if visual_width > 0 and front.shape[1] != visual_width:
        scale = visual_width / float(front.shape[1])
        front = cv2.resize(
            front,
            (visual_width, max(1, int(round(front.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA)
    bev = draw_bev(decoded, front.shape[0], point_cloud_range)
    visualization = np.concatenate([front, bev], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), visualization):
        raise IOError(f"可视化写入失败: {output_path}")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8")


def write_json_lines(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """写 JSON Lines；每一行独立表示一帧，便于逐帧流式读取。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(jsonable(row), ensure_ascii=False) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8")


def business_frame_record(
    timestamp: str,
    business_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """生成默认业务结果行，只保留板端业务消费所需字段。"""
    targets = []
    for row in business_rows:
        bbox = list(row["box"])
        if len(bbox) < 7:
            raise RuntimeError(f"业务 bbox 至少需要 7 个值，实际 {len(bbox)}")
        targets.append({
            "label": row["class_name"],
            "label_id": int(row["label"]),
            # JSON 数值保持 numeric 类型，只限制到三位小数；例如 1.200 在
            # JSON 文本中可能显示为 1.2，但数值精度已经按三位小数处理。
            "bbox": [round(float(value), 3) for value in bbox],
            "score": round(float(row["score"]), 3),
        })
    return {
        "timestamp": timestamp,
        "box_origin": "center",
        "count": len(targets),
        "targets": targets,
    }


def relative_output_path(
    image_root: Path,
    image_path: Path,
    output_root: Path,
    folder: str,
    suffix: str,
) -> Path:
    base = image_root.parent if image_root.is_file() else image_root
    try:
        relative = image_path.relative_to(base)
    except ValueError:
        relative = Path(image_path.name)
    return output_root / folder / relative.with_suffix(suffix)


# =============================================================================
# 7. 入口：逐图直接执行完整板端链路
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="独立单文件：图片 + 2D ONNX + LUT + 3D ONNX + 后处理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=ASSET_INIT_HINT,
    )
    parser.add_argument("--images", required=True, help="单张图片或图片目录")
    parser.add_argument(
        "--weights", required=True,
        help="board_model_spec.json 所在目录；模型文件由 manifest 精确绑定")
    parser.add_argument(
        "--backend", choices=("onnx-fp", "onnx-int8"), default="onnx-fp",
        help="模型后端；onnx-int8 必须是真实量化 ONNX contract")
    parser.add_argument(
        "--geometry", choices=("fixed",), default="fixed",
        help="standalone 只支持预构建固定 LUT，不支持 dynamic")
    parser.add_argument("--lut-dir", required=True,
                        help="车辆 LUT 目录，目录内包含 LUT/ 和 metadata.json")
    parser.add_argument("--output-dir", default="work_dirs/n7_board_runtime")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--output-classes", nargs="+", default=("car",),
                        help="完整 decode/NMS 后保留的业务类别；all 表示全部")
    parser.add_argument("--score-thr", type=float, default=0.2,
                        help="业务输出和可视化阈值")
    parser.add_argument("--max-preds", type=int, default=100)
    parser.add_argument(
        "--stride", type=int, default=1,
        help="按自然排序后的图片序列每隔 N 帧推理一帧")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--visual-width", type=int, default=1600,
                        help="前视图显示宽度；0 保持原图尺寸")
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument(
        "--save-frame-details", action="store_true",
        help="保存每帧完整 decoded/trace JSON；默认只写汇总业务结果")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    image_root = Path(args.images).expanduser().resolve()
    images = list_images(image_root)
    if args.stride <= 0:
        raise ValueError(f"--stride 必须为正整数，实际 {args.stride}")
    images = images[::args.stride]
    if args.max_frames > 0:
        images = images[:args.max_frames]
    if not images:
        raise FileNotFoundError(f"没有找到可推理图片: {image_root}")

    weights_dir = Path(args.weights).expanduser().resolve()
    if not weights_dir.is_dir():
        raise FileNotFoundError(
            f"权重目录不存在: {weights_dir}。{ASSET_INIT_HINT}")
    spec, backend_contract, model_2d_path, model_3d_path, anchors_path = (
        load_model_spec(weights_dir, Path(args.lut_dir), args.backend))
    providers = resolve_provider(args.provider)
    session_2d = ort.InferenceSession(str(model_2d_path), providers=providers)
    session_3d = ort.InferenceSession(str(model_3d_path), providers=providers)
    actual_provider = validate_session_providers(
        session_2d, session_3d, args.provider)
    output_names_3d = dict(
        backend_contract["3d"]["semantic_output_names"])
    actual_output_names = {item.name for item in session_3d.get_outputs()}
    if set(output_names_3d.values()) != actual_output_names:
        raise RuntimeError(
            f"3D session outputs={sorted(actual_output_names)} 与 manifest="
            f"{sorted(output_names_3d.values())} 不一致")

    gather, scatter, lut_metadata = load_lut(Path(args.lut_dir), spec)
    anchors = load_anchors(anchors_path, spec)
    output_root = Path(args.output_dir).expanduser().resolve()
    frame_summaries = []
    business_frame_records = []

    LOGGER.info("2D ONNX: %s", model_2d_path)
    LOGGER.info("3D ONNX: %s", model_3d_path)
    LOGGER.info("actual provider: %s", actual_provider)
    LOGGER.info("LUT: %s, valid_points=%d", args.lut_dir, gather.size)
    LOGGER.info("anchors: %s, shape=%s", anchors_path, anchors.shape)

    for image_index, image_path in enumerate(images):
        # ------------------------------------------------------------------
        # 板端主链。这里故意直接展开，不再通过项目内 wrapper 隐藏步骤。
        # ------------------------------------------------------------------
        # 1. 读取图片并执行固定前处理。
        timestamp = extract_image_timestamp(image_path)
        input_tensor, image_bgr, image_size = preprocess_image(
            image_path, spec["preprocess"])
        expected_image_size = lut_metadata.get("source_image_size")
        if image_size != expected_image_size:
            raise RuntimeError(
                f"图片 {image_path} 尺寸={image_size}，固定 LUT 对应尺寸="
                f"{expected_image_size}；禁止跨图片坐标系复用 LUT")

        # 2. 2D ONNX 得到 NCHW feature。
        feature_raw = run_2d_onnx(
            session_2d, input_tensor, backend_contract["2d"])
        feature_spec = backend_contract["2d"]["output"]
        if backend_contract["feature_to_bev_bridge"] == "raw-quantized-direct":
            feature_2d = feature_to_nchw_once(
                feature_raw, feature_spec["layout"])
            fill_value = feature_spec["quantization"]["zero_point"]
        else:
            feature_2d = feature_to_nchw_once(
                dequantize_external(feature_raw, feature_spec),
                feature_spec["layout"])
            fill_value = 0.0

        # 3. 固定 bin LUT gather/scatter 得到 [1,256,160,140]。
        bev_input = apply_lut(
            feature_2d, gather, scatter, spec["geometry"], fill_value)
        bev_spec = backend_contract["3d"]["input"]
        if str(bev_input.dtype) != bev_spec["dtype"]:
            bev_input = quantize_external(bev_input, bev_spec)

        # 4. 3D ONNX 按输出名称取得 raw cls/bbox/dir logits。
        head_logits_raw = run_3d_onnx(
            session_3d, bev_input, output_names_3d,
            backend_contract["3d"])
        head_logits = {
            name: dequantize_external(
                value, backend_contract["3d"]["outputs"][name])
            for name, value in head_logits_raw.items()
        }

        # 5. sigmoid 一次、anchor decode、方向修正、CPU NMS、center-origin。
        decoded, trace = postprocess(head_logits, anchors, spec["head"])

        # 6. 完整模型后处理结束后，再应用 car 等业务过滤。
        selected, business_rows = select_business_output(
            decoded, args.output_classes, args.score_thr, args.max_preds)

        business_frame_records.append(business_frame_record(
            timestamp, business_rows))

        # 完整 decoded/trace 是排查资料，批量生产推理默认不落盘；需要定位某帧
        # 时显式传 --save-frame-details，避免 predictions/ 产生大量冗余 JSON。
        prediction_path = None
        if args.save_frame_details:
            prediction_path = relative_output_path(
                image_root, image_path, output_root, "predictions", ".json")
            result = {
                "version": "mono-front-board-standalone-v3",
                "backend": args.backend,
                "timestamp": timestamp,
                "image_index": image_index,
                "models": {
                    "2d": str(model_2d_path),
                    "3d": str(model_3d_path),
                },
                "lut_dir": str(Path(args.lut_dir).expanduser().resolve()),
                "anchors": str(anchors_path),
                "tensor_shapes": {
                    "input": list(input_tensor.shape),
                    "feature_2d_raw": list(feature_raw.shape),
                    "feature_2d_nchw": list(feature_2d.shape),
                    "bev_input": list(bev_input.shape),
                    "head_logits": {
                        name: list(value.shape)
                        for name, value in head_logits.items()
                    },
                },
                "postprocess_trace": trace,
                "decoded_all": {
                    "count": int(len(decoded["boxes"])),
                    "box_origin": "center",
                    "predictions": decoded_rows(decoded),
                },
                "business_output": {
                    "classes": list(args.output_classes),
                    "score_thr": float(args.score_thr),
                    "count": len(business_rows),
                    "box_origin": "center",
                    "predictions": business_rows,
                },
            }
            write_json(prediction_path, result)

        visualization_path = None
        if not args.no_visualization:
            visualization_path = relative_output_path(
                image_root, image_path, output_root,
                "visualizations", ".jpg")
            render_visualization(
                image_bgr, selected, lut_metadata,
                visualization_path, args.visual_width, timestamp,
                spec["geometry"]["point_cloud_range"])

        frame_summaries.append({
            "timestamp": timestamp,
            "prediction_detail": (
                str(prediction_path) if prediction_path else None),
            "visualization": (
                str(visualization_path) if visualization_path else None),
            "decoded_count": int(len(decoded["boxes"])),
            "business_count": len(business_rows),
        })
        LOGGER.info(
            "%d/%d %s: decoded=%d, output=%d",
            image_index + 1, len(images), image_path.name,
            len(decoded["boxes"]), len(business_rows))

    business_output_path = output_root / "business_predictions.jsonl"
    write_json_lines(business_output_path, business_frame_records)
    summary = {
        "version": "mono-front-board-standalone-summary-v3",
        "status": "PASS",
        "image_count": len(images),
        "stride": int(args.stride),
        "provider": actual_provider,
        "backend": args.backend,
        "geometry": "fixed",
        "business_predictions": str(business_output_path),
        "frame_details_enabled": bool(args.save_frame_details),
        "frames": frame_summaries,
    }
    write_json(output_root / "prediction_summary.json", summary)
    print(
        "BOARD_RUNTIME: PASS; "
        f"images={len(images)}; provider={actual_provider}; output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
