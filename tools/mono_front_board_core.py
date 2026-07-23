#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N7 S0 单目前视板端资产构建实现。

本文件只服务一次性模型/LUT 资产生成。正式板端推理入口是自包含单文件，
不会从这里读取前处理、LUT gather、ONNX 执行或后处理私有函数。用户通常
不应直接执行本文件，而应使用：

- ``tools/build_mono_front_board_assets.py``：一次性生成公共资料和车辆 LUT；
- ``tools/run_mono_front_board_inference.py``：只读取现成资料执行推理。

推理直线链路为：

``图片 -> 板端前处理 -> 2D ONNX -> 车辆固定 LUT -> 3D ONNX -> 板端后处理``

完整的黄金 tensor、canonical 旁路和逐级 comparison 统一由
``tools/analyze_mono_front_pipeline.py`` 负责。运行时只读加载位于
``tools/mono_front_board_assets.py``，本文件不会反向导入数值 runtime。

公开函数 :func:`build_board_model_assets` 生成：

- ``<weights>/board_model_spec.json``：类别、张量布局和后处理契约；
- ``data/board_lut/common/<profile>/points.npy``：固定 BEV voxel 中心；
- ``data/board_lut/common/<profile>/anchors.npy``：固定 3D head anchors；
- ``data/board_lut/common/<profile>/asset_manifest.json``：公共资产 hash。

车辆 LUT 由 :func:`build_board_vehicle_lut` 写入
``data/board_lut/<vehicle_id>/{LUT,LUT_arr}``。完整性检查由
``tools/mono_front_board_assets.py`` 完成；只读加载不会在推理阶段补写资产。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
N7_TOOL_ROOT = REPO_ROOT / "tools" / "data_converter" / "n7"
if str(N7_TOOL_ROOT) not in sys.path:
    # 直接导入脚本文件，避免 tools.data_converter 包初始化时要求 legacy mmcv。
    sys.path.insert(0, str(N7_TOOL_ROOT))

from build_fastbev_lut import (  # noqa: E402
    build_img_meta_for_sample,
    export_one_sequence,
    find_pipeline_step,
    get_nested,
    get_points,
    load_py_config,
    normalize_first_item,
)
from tools.infer_mono_front_image import (  # noqa: E402
    build_camera_info,
    build_camera_template,
    check_aspect,
    find_sensor,
    read_image_size,
    read_json,
    safe_output_name,
)
from tools.mono_front_board_assets import (  # noqa: E402
    DEFAULT_LUT_ROOT,
    board_bin_state as _board_bin_state,
    load_board_vehicle_lut,
    validate_model_assets as _validate_existing_model_assets,
    validate_semantic_output_names,
)


LOGGER = logging.getLogger("mono_front_board_assets")
DEFAULT_CONFIG = (
    "configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18.py")
DEFAULT_ASSET_PROFILE = "n7_s0_v1"
SEMANTIC_LOGITS = ("head_cls", "head_bbox", "head_dir")


@dataclass(frozen=True)
class VehicleJob:
    """一辆车对应的标定、图片、LUT 和输出目录。"""

    vehicle_id: str
    info_json: Optional[Path]
    images: Tuple[Path, ...]
    image_base: Path
    lut_dir: Path
    output_dir: Path


class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter):
    """让 ``--help`` 同时保留示例换行并显示默认值。"""


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """流式计算文件 SHA256，避免一次性读取较大的 ONNX。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Mapping[str, Any]) -> str:
    """计算结构化契约 hash；排序后不受 dict 插入顺序影响。"""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_jsonable(value: Any) -> Any:
    """把 Path/NumPy 等值转换成 JSON 可写对象。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """以 UTF-8 写 JSON；父目录由调用方无需预先创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8")


def natural_key(value: Any) -> List[Any]:
    """对含时间戳/编号的路径做自然排序，保证批量样本可复现。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def _resolve_recorded_path(value: str, base_dir: Path) -> Path:
    """解析 metadata 中可能来自另一工作目录的模型路径。"""
    raw = Path(value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend((Path.cwd() / raw, base_dir / raw, base_dir / raw.name))
    else:
        # 内网仓库迁移后历史绝对路径可能失效，但文件名通常仍在权重目录中。
        candidates.append(base_dir / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"metadata 记录的文件不存在: {value}; tried={candidates}")


def _resolve_onnx_assets(weights_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    """从 export metadata 或约定文件名定位 2D/3D ONNX。"""
    weights_dir = weights_dir.expanduser().resolve()
    if not weights_dir.is_dir():
        raise NotADirectoryError(f"--weights 必须是 split ONNX 目录: {weights_dir}")
    metadata_path = weights_dir / "export_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    exports = metadata.get("exports", {})

    def resolve_part(part: str, names: Sequence[str]) -> Path:
        item = exports.get(part, {})
        recorded = item.get("simplified") or item.get("raw")
        if recorded:
            try:
                return _resolve_recorded_path(str(recorded), weights_dir)
            except FileNotFoundError:
                LOGGER.warning("export_metadata 中的 %s 路径失效，按文件名回退", part)
        for name in names:
            candidate = weights_dir / name
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"{weights_dir} 中找不到 {part} ONNX")

    model_2d = resolve_part(
        "2d", ("simplified_export_2d_model.onnx", "export_2d_model.onnx"))
    model_3d = resolve_part(
        "3d", ("simplified_export_3d_model.onnx", "export_3d_model.onnx"))
    return model_2d, model_3d, metadata


def _onnx_dtype_name(elem_type: int) -> str:
    """把 ONNX TensorProto dtype 规范成 NumPy/ORT 常用名称。"""
    import onnx

    mapping = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.DOUBLE: "float64",
        onnx.TensorProto.INT8: "int8",
        onnx.TensorProto.UINT8: "uint8",
        onnx.TensorProto.INT16: "int16",
        onnx.TensorProto.UINT16: "uint16",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.UINT32: "uint32",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.UINT64: "uint64",
        onnx.TensorProto.BOOL: "bool",
    }
    if elem_type not in mapping:
        raise ValueError(
            f"当前板端契约不支持 ONNX dtype="
            f"{onnx.TensorProto.DataType.Name(elem_type)}")
    return mapping[elem_type]


def _ort_dtype_name(value: str) -> str:
    """把 ORT NodeArg.type 规范成与 graph dtype 相同的名称。"""
    mapping = {
        "tensor(float)": "float32",
        "tensor(float16)": "float16",
        "tensor(double)": "float64",
        "tensor(int8)": "int8",
        "tensor(uint8)": "uint8",
        "tensor(int16)": "int16",
        "tensor(uint16)": "uint16",
        "tensor(int32)": "int32",
        "tensor(uint32)": "uint32",
        "tensor(int64)": "int64",
        "tensor(uint64)": "uint64",
        "tensor(bool)": "bool",
    }
    if value not in mapping:
        raise ValueError(f"当前板端契约不支持 ORT dtype={value}")
    return mapping[value]


def _initializer_arrays(model: Any) -> Dict[str, np.ndarray]:
    """读取量化节点引用的常量 initializer。"""
    import onnx

    return {
        item.name: np.asarray(onnx.numpy_helper.to_array(item))
        for item in model.graph.initializer
    }


def _has_upstream_op(model: Any, tensor_name: str, op_type: str) -> bool:
    """沿输出 producer 反向检查算子，避免 raw cls 契约误接已 sigmoid tensor。"""
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    pending = [tensor_name]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        node = producers.get(current)
        if node is None:
            continue
        if node.op_type == op_type:
            return True
        pending.extend(name for name in node.input if name)
    return False


def _quantization_for_external_tensor(
    model: Any,
    tensor_name: str,
    dtype: str,
    direction: str,
) -> Optional[Dict[str, Any]]:
    """从外部 I/O 邻接的 QuantizeLinear/DequantizeLinear 读取量化域。

    外部 raw INT8/UINT8 输入通常直接进入 ``DequantizeLinear``；raw 输出通常
    由 ``QuantizeLinear`` 产生。若图没有暴露这层关系，就不能可靠恢复比例，
    因而必须拒绝生成可运行契约，不能从文件名或经验值猜测。
    """
    if dtype not in ("int8", "uint8"):
        return None
    nodes = []
    for node in model.graph.node:
        if direction == "input":
            matched = node.op_type == "DequantizeLinear" and node.input[0] == tensor_name
        else:
            matched = node.op_type == "QuantizeLinear" and tensor_name in node.output
        if matched:
            nodes.append(node)
    if len(nodes) != 1:
        raise ValueError(
            f"外部 {dtype} {direction} tensor {tensor_name!r} 无法唯一定位量化参数；"
            f"匹配到 {len(nodes)} 个 Q/DQ 节点")
    node = nodes[0]
    if len(node.input) < 2:
        raise ValueError(f"{node.op_type} 缺少 scale 输入: {tensor_name}")
    arrays = _initializer_arrays(model)
    if node.input[1] not in arrays:
        raise ValueError(
            f"{tensor_name}: scale={node.input[1]!r} 不是静态 initializer")
    scale = np.asarray(arrays[node.input[1]], dtype=np.float64)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"{tensor_name}: scale 必须为有限正数，实际 {scale}")
    if len(node.input) >= 3 and node.input[2]:
        if node.input[2] not in arrays:
            raise ValueError(
                f"{tensor_name}: zero_point={node.input[2]!r} 不是静态 initializer")
        zero_point = np.asarray(arrays[node.input[2]])
    else:
        zero_point = np.zeros_like(scale, dtype=np.dtype(dtype))
    if zero_point.shape != scale.shape and zero_point.size != 1:
        raise ValueError(
            f"{tensor_name}: scale/zero_point shape 不兼容: "
            f"{scale.shape}/{zero_point.shape}")
    axis = None
    for attr in node.attribute:
        if attr.name == "axis":
            axis = int(attr.i)
    per_channel = scale.size > 1
    if per_channel and axis is None:
        # ONNX QuantizeLinear/DequantizeLinear 的默认 axis 为 1。
        axis = 1
    limits = np.iinfo(np.dtype(dtype))
    return {
        "mode": "per-channel" if per_channel else "per-tensor",
        "scale": scale.reshape(-1).astype(float).tolist(),
        "zero_point": zero_point.reshape(-1).astype(int).tolist(),
        "channel_axis": axis if per_channel else None,
        "round_policy": "round-to-nearest-ties-to-even",
        "saturation_policy": "clamp",
        "clamp_min": int(limits.min),
        "clamp_max": int(limits.max),
        "source": f"onnx:{node.op_type}",
    }


def _quantization_domain(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """提取跨 split 模型必须一致的量化域；不比较 Q/DQ 来源算子。"""
    keys = (
        "mode", "scale", "zero_point", "channel_axis", "round_policy",
        "saturation_policy", "clamp_min", "clamp_max",
    )
    return {key: value.get(key) for key in keys} if value else {}


def _onnx_static_io(path: Path) -> Dict[str, Any]:
    """读取并交叉核对真实 ONNX graph/session 静态 I/O。"""
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)

    def describe(items: Iterable[Any], direction: str) -> List[Dict[str, Any]]:
        result = []
        for item in items:
            tensor_type = item.type.tensor_type
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(int(dim.dim_value))
                else:
                    shape.append(None)
            dtype = _onnx_dtype_name(int(tensor_type.elem_type))
            record = {
                "name": item.name,
                "shape": shape,
                "dtype": dtype,
                "quantization": _quantization_for_external_tensor(
                    model, item.name, dtype, direction),
            }
            if direction == "output":
                record["has_upstream_sigmoid"] = _has_upstream_op(
                    model, item.name, "Sigmoid")
            result.append(record)
        return result

    graph_inputs = describe(model.graph.input, "input")
    graph_outputs = describe(model.graph.output, "output")
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"])

    def describe_session(items: Iterable[Any]) -> List[Dict[str, Any]]:
        return [{
            "name": item.name,
            "shape": [int(v) if isinstance(v, int) else None for v in item.shape],
            "dtype": _ort_dtype_name(item.type),
        } for item in items]

    session_inputs = describe_session(session.get_inputs())
    session_outputs = describe_session(session.get_outputs())
    graph_projection = [
        {key: item[key] for key in ("name", "shape", "dtype")}
        for item in graph_inputs
    ]
    if session_inputs != graph_projection:
        raise RuntimeError(
            f"ONNX graph/session input contract 不一致: "
            f"graph={graph_projection}, session={session_inputs}")
    graph_projection = [
        {key: item[key] for key in ("name", "shape", "dtype")}
        for item in graph_outputs
    ]
    if session_outputs != graph_projection:
        raise RuntimeError(
            f"ONNX graph/session output contract 不一致: "
            f"graph={graph_projection}, session={session_outputs}")
    node_types = {node.op_type for node in model.graph.node}
    has_q = "QuantizeLinear" in node_types
    has_dq = "DequantizeLinear" in node_types
    has_integer_op = any(
        name.startswith("QLinear") or name.endswith("Integer")
        for name in node_types)
    return {
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "session_inputs": session_inputs,
        "session_outputs": session_outputs,
        "graph_format": (
            "qdq" if has_q and has_dq
            else "quantized-io" if has_q or has_dq
            else "quantized-ops" if has_integer_op
            else "float"),
    }


def _require_static_shape(shape: Sequence[Optional[int]], label: str) -> Tuple[int, ...]:
    if any(value is None or int(value) <= 0 for value in shape):
        raise ValueError(f"{label} 必须是正整数静态 shape，实际 {list(shape)}")
    return tuple(int(value) for value in shape)


def _semantic_output_map(outputs: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    """严格按输出名称识别 cls/bbox/dir，禁止按位置猜测语义。"""
    result: Dict[str, Mapping[str, Any]] = {}
    for output in outputs:
        name = str(output["name"]).lower()
        if "bbox" in name:
            if "head_bbox" in result:
                raise ValueError(f"3D ONNX 存在多个 bbox 输出: {outputs}")
            result["head_bbox"] = output
        elif "dir" in name:
            if "head_dir" in result:
                raise ValueError(f"3D ONNX 存在多个 dir 输出: {outputs}")
            result["head_dir"] = output
        elif "cls" in name:
            if "head_cls" in result:
                raise ValueError(f"3D ONNX 存在多个 cls 输出: {outputs}")
            result["head_cls"] = output
    if set(result) != set(SEMANTIC_LOGITS):
        raise ValueError(
            "无法仅凭名称识别 3D ONNX 三个语义输出；禁止按 output index "
            f"回退: {outputs}")
    return result


def _repeat_or_slice(value: Any, count: int, default: Any) -> List[Any]:
    """把 test_cfg 的标量/列表规范成每类别一项。"""
    if value is None:
        values = [default]
    elif isinstance(value, (tuple, list)):
        values = list(value)
    else:
        values = [value]
    if len(values) == 1:
        values = values * count
    if len(values) < count:
        raise ValueError(f"配置只有 {len(values)} 项，少于 num_classes={count}")
    return values[:count]


def _extract_model_contract(
    config_path: Path,
    model_2d: Path,
    model_3d: Path,
    export_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """从 resolved config 和真实 ONNX 图提取板端模型契约。"""
    cfg = load_py_config(config_path)
    dataset_cfg = get_nested(cfg, ("data", "test"), {})
    pipeline = dataset_cfg.get("pipeline") or cfg.get("test_pipeline") or []
    aug_step = find_pipeline_step(pipeline, "RandomAugImageMultiViewImage")
    norm_step = find_pipeline_step(pipeline, "NormalizeMultiviewImage")
    origin_step = find_pipeline_step(pipeline, "KittiSetOrigin")
    if aug_step is None or norm_step is None or origin_step is None:
        raise KeyError("test pipeline 必须包含 RandomAug/Normalize/KittiSetOrigin")

    model_cfg = cfg.get("model", {})
    head_cfg = model_cfg.get("bbox_head", {})
    test_cfg = model_cfg.get("test_cfg", {})
    anchor_cfg = head_cfg.get("anchor_generator", {})
    if anchor_cfg.get("type") != "AlignedAnchor3DRangeGenerator":
        raise ValueError(
            "当前纯 NumPy 后处理只审计过 AlignedAnchor3DRangeGenerator，"
            f"实际 {anchor_cfg.get('type')}")

    n_images = int(model_cfg.get("n_images", dataset_cfg.get("n_images", -1)))
    n_times = int(dataset_cfg.get("n_times", -1))
    sequential = bool(dataset_cfg.get("sequential", True))
    if (n_images, n_times, sequential) != (1, 1, False):
        raise ValueError(
            "正式运行工具只支持原生单目单帧 S0；"
            f"observed={(n_images, n_times, sequential)}")
    camera_types = list(dataset_cfg.get("camera_types") or ["cam0"])
    if len(camera_types) != 1:
        raise ValueError(f"当前运行工具只支持一个前视相机，实际 {camera_types}")

    io_2d = _onnx_static_io(model_2d)
    io_3d = _onnx_static_io(model_3d)
    if len(io_2d["inputs"]) != 1 or len(io_2d["outputs"]) != 1:
        raise ValueError(f"2D ONNX 必须单输入单输出: {io_2d}")
    if len(io_3d["inputs"]) != 1 or len(io_3d["outputs"]) != 3:
        raise ValueError(f"S0 3D ONNX 必须单输入三输出: {io_3d}")

    input_2d_shape = _require_static_shape(io_2d["inputs"][0]["shape"], "2D input")
    output_2d_shape = _require_static_shape(io_2d["outputs"][0]["shape"], "2D output")
    input_3d_shape = _require_static_shape(io_3d["inputs"][0]["shape"], "3D input")
    semantic_outputs = _semantic_output_map(io_3d["outputs"])
    semantic_shapes = {
        name: _require_static_shape(item["shape"], name)
        for name, item in semantic_outputs.items()
    }


    metadata_layout = get_nested(
        dict(export_metadata), ("exports", "2d", "output_layout"), None)
    if metadata_layout is None:
        raise ValueError(
            "export_metadata 缺少 exports.2d.output_layout；"
            "NCHW/NHWC 不能仅凭 shape 猜测")
    feature_layout = str(metadata_layout).lower()
    if feature_layout not in ("nchw", "nhwc"):
        raise ValueError(f"不支持的 2D output layout: {feature_layout}")
    if feature_layout == "nchw":
        _, feature_channels, feature_h, feature_w = output_2d_shape
    else:
        _, feature_h, feature_w, feature_channels = output_2d_shape

    input_h, input_w = [
        int(value) for value in aug_step["data_config"]["test_input_size"]]
    if not bool(aug_step.get("force_resize", False)):
        raise ValueError(
            "当前板端预处理只实现已完成数值验证的 force_resize=True 路径")
    if input_2d_shape != (1, 3, input_h, input_w):
        raise ValueError(
            f"2D ONNX input={input_2d_shape} 与 config 输入不一致: "
            f"{(1, 3, input_h, input_w)}")
    stride_h = input_h / float(feature_h)
    stride_w = input_w / float(feature_w)
    if stride_h != stride_w or not stride_h.is_integer():
        raise ValueError(
            f"2D feature stride 不一致: h={stride_h}, w={stride_w}")

    n_voxels = tuple(
        int(value) for value in normalize_first_item(model_cfg.get("n_voxels")))
    voxel_size = tuple(
        float(value) for value in normalize_first_item(model_cfg.get("voxel_size")))
    point_cloud_range = tuple(float(value) for value in origin_step["point_cloud_range"])
    origin = tuple(
        float(value) for value in (
            (np.asarray(point_cloud_range[:3], dtype=np.float32) +
             np.asarray(point_cloud_range[3:], dtype=np.float32)) / 2.0))
    expected_bev = (1, feature_channels * n_voxels[2], n_voxels[0], n_voxels[1])
    if input_3d_shape != expected_bev:
        raise ValueError(
            f"3D ONNX input={input_3d_shape} 与 ZC BEV={expected_bev} 不一致")

    class_names = list(cfg.get("class_names") or [])
    num_classes = int(head_cfg.get("num_classes", len(class_names)))
    if not class_names:
        class_names = [f"class_{index}" for index in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names={class_names} 与 num_classes={num_classes} 不一致")

    sizes = [list(map(float, item)) for item in anchor_cfg.get("sizes", [])]
    rotations = [float(value) for value in anchor_cfg.get("rotations", [0, 1.5707963])]
    ranges = [list(map(float, item)) for item in anchor_cfg.get("ranges", [])]
    scales = [float(value) for value in anchor_cfg.get("scales", [1])]
    if not sizes or not ranges or len(scales) != 1:
        raise ValueError(
            f"当前单层 head 要求非空 sizes/ranges 且 scales 只有一项: {anchor_cfg}")
    anchors_per_location = len(sizes) * len(rotations)
    box_code_size = int(head_cfg.get("bbox_coder", {}).get("code_size", 7))
    num_dir_bins = 2
    cls_shape = semantic_shapes["head_cls"]
    bbox_shape = semantic_shapes["head_bbox"]
    dir_shape = semantic_shapes["head_dir"]
    if cls_shape[1] != anchors_per_location * num_classes:
        raise ValueError(
            f"cls channels={cls_shape[1]}，但 A*C="
            f"{anchors_per_location}*{num_classes}")
    if bbox_shape[1] != anchors_per_location * box_code_size:
        raise ValueError(
            f"bbox channels={bbox_shape[1]}，但 A*code_size="
            f"{anchors_per_location}*{box_code_size}")
    if dir_shape[1] != anchors_per_location * num_dir_bins:
        raise ValueError(
            f"dir channels={dir_shape[1]}，但 A*dir_bins="
            f"{anchors_per_location}*{num_dir_bins}")
    head_hw = tuple(int(value) for value in cls_shape[-2:])
    if any(tuple(shape[-2:]) != head_hw for shape in semantic_shapes.values()):
        raise ValueError(f"三个 head 输出空间 shape 不一致: {semantic_shapes}")

    custom_values = [float(value) for value in anchor_cfg.get("custom_values", [])]
    if 7 + len(custom_values) != box_code_size:
        raise ValueError(
            f"anchor dims={7 + len(custom_values)} 与 box_code_size={box_code_size} 不一致")
    use_sigmoid = bool(head_cfg.get("loss_cls", {}).get("use_sigmoid", True))
    if not use_sigmoid:
        raise ValueError("当前板端后处理只支持 sigmoid 分类头")

    nms_types = [str(value) for value in _repeat_or_slice(
        test_cfg.get("nms_type_list"), num_classes, "rotate")]
    nms_thresholds = [float(value) for value in _repeat_or_slice(
        test_cfg.get("nms_thr_list"), num_classes, test_cfg.get("nms_thr", 0.2))]
    nms_radius = [float(value) for value in _repeat_or_slice(
        test_cfg.get("nms_radius_thr_list"), num_classes, 1.0)]
    nms_rescale = [float(value) for value in _repeat_or_slice(
        test_cfg.get("nms_rescale_factor"), num_classes, 1.0)]
    if any(value not in ("rotate", "circle") for value in nms_types):
        raise ValueError(f"不支持的 nms_type_list: {nms_types}")

    checkpoint_value = export_metadata.get("checkpoint")
    checkpoint_sha = str(export_metadata.get("checkpoint_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
        if not checkpoint_value:
            raise ValueError(
                "export_metadata 缺少 checkpoint/checkpoint_sha256，"
                "无法建立模型权重契约")
        checkpoint_path = _resolve_recorded_path(
            str(checkpoint_value), model_2d.parent)
        checkpoint_sha = file_sha256(checkpoint_path)

    return {
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "checkpoint": checkpoint_value,
        "checkpoint_sha256": checkpoint_sha,
        "preprocess": {
            "input_size_hw": [input_h, input_w],
            "resize_backend": "pipeline_pil_bicubic",
            "read_color": "BGR",
            "model_color": "RGB" if bool(norm_step.get("to_rgb", True)) else "BGR",
            "to_rgb": bool(norm_step.get("to_rgb", True)),
            # NormalizeMultiviewImage 会先把 mean/std 保存为 float32；在模型契约
            # 中也固化同一量化结果，避免 JSON 原始小数绕过这一步。
            "mean": np.asarray(
                norm_step["mean"], dtype=np.float32).astype(float).tolist(),
            "std": np.asarray(
                norm_step["std"], dtype=np.float32).astype(float).tolist(),
            "force_resize": bool(aug_step.get("force_resize", False)),
        },
        "geometry": {
            "n_images": n_images,
            "n_times": n_times,
            "sequential": sequential,
            "camera_types": camera_types,
            "n_voxels": list(n_voxels),
            "voxel_size": list(voxel_size),
            "point_cloud_range": list(point_cloud_range),
            "origin": list(origin),
            "stride": int(stride_h),
            "use_distortion": bool(model_cfg.get("use_distortion", False)),
            "data_config": aug_step["data_config"],
            "feature_shape_nchw": [1, feature_channels, feature_h, feature_w],
            "bev_shape_nchw": list(expected_bev),
            "channel_layout": "ZC",
        },
        "head": {
            "class_names": class_names,
            "num_classes": num_classes,
            "num_anchors_per_location": anchors_per_location,
            "box_code_size": box_code_size,
            "num_dir_bins": num_dir_bins,
            "use_sigmoid_cls": True,
            "feature_map_hw": list(head_hw),
            "output_shapes": {
                name: list(shape) for name, shape in semantic_shapes.items()
            },
            "anchor_generator": {
                "type": anchor_cfg["type"],
                "ranges": ranges,
                "sizes": sizes,
                "scales": scales,
                "rotations": rotations,
                "custom_values": custom_values,
                "reshape_out": bool(anchor_cfg.get("reshape_out", True)),
                "size_per_range": bool(anchor_cfg.get("size_per_range", True)),
                "align_corner": bool(anchor_cfg.get("align_corner", False)),
            },
            "score_thr": float(test_cfg.get("score_thr", 0.0)),
            "nms_pre": int(test_cfg.get("nms_pre", -1)),
            "max_num": int(test_cfg.get("max_num", 500)),
            "nms_type_list": nms_types,
            "nms_thr_list": nms_thresholds,
            "nms_radius_thr_list": nms_radius,
            "nms_rescale_factor": nms_rescale,
            # 当前 anchor3d_multiclass_nms 调用 circle_nms 时使用其默认上限；
            # 固化进 manifest，standalone 不再内嵌这个数值。
            "circle_nms_post_max_size": int(
                test_cfg.get("circle_nms_post_max_size", 83)),
            "dir_offset": float(head_cfg.get("dir_offset", 0.0)),
            "dir_limit_offset": float(head_cfg.get("dir_limit_offset", 0.0)),
            "decoded_native_box_origin": "bottom",
            "public_box_origin": "center",
            "sigmoid_count": 1,
        },
        "onnx_runtime": {
            "2d": {
                "input_name": io_2d["inputs"][0]["name"],
                "input_shape": list(input_2d_shape),
                "input_dtype": io_2d["inputs"][0]["dtype"],
                "input_layout": "nchw",
                "input_quantization": io_2d["inputs"][0]["quantization"],
                "output_name": io_2d["outputs"][0]["name"],
                "output_shape": list(output_2d_shape),
                "output_dtype": io_2d["outputs"][0]["dtype"],
                "output_layout": feature_layout,
                "output_quantization": io_2d["outputs"][0]["quantization"],
                "graph_format": io_2d["graph_format"],
                "graph_inputs": io_2d["inputs"],
                "graph_outputs": io_2d["outputs"],
                "session_inputs": io_2d["session_inputs"],
                "session_outputs": io_2d["session_outputs"],
            },
            "3d": {
                "input_name": io_3d["inputs"][0]["name"],
                "input_shape": list(input_3d_shape),
                "input_dtype": io_3d["inputs"][0]["dtype"],
                "input_layout": "nchw-zc",
                "input_quantization": io_3d["inputs"][0]["quantization"],
                "semantic_output_names": {
                    name: item["name"] for name, item in semantic_outputs.items()
                },
                "semantic_outputs": {
                    name: {
                        "name": item["name"],
                        "shape": list(semantic_shapes[name]),
                        "dtype": item["dtype"],
                        "layout": "nchw",
                        "quantization": item["quantization"],
                    }
                    for name, item in semantic_outputs.items()
                },
                "graph_format": io_3d["graph_format"],
                "graph_inputs": io_3d["inputs"],
                "graph_outputs": io_3d["outputs"],
                "session_inputs": io_3d["session_inputs"],
                "session_outputs": io_3d["session_outputs"],
            },
        },
    }


def _build_onnx_backend_contract(
    model_2d: Path,
    model_3d: Path,
    feature_layout: str,
    expected_runtime: Mapping[str, Any],
    backend: str,
) -> Dict[str, Any]:
    """为 FP/INT8 split ONNX 生成统一的真实 I/O 契约。"""
    io_2d = _onnx_static_io(model_2d)
    io_3d = _onnx_static_io(model_3d)
    if len(io_2d["inputs"]) != 1 or len(io_2d["outputs"]) != 1:
        raise ValueError(f"{backend} 2D ONNX 必须单输入单输出: {io_2d}")
    if len(io_3d["inputs"]) != 1 or len(io_3d["outputs"]) != 3:
        raise ValueError(f"{backend} 3D ONNX 必须单输入三输出: {io_3d}")
    if feature_layout not in ("nchw", "nhwc"):
        raise ValueError(f"{backend} 2D output layout 无效: {feature_layout}")

    semantic = _semantic_output_map(io_3d["outputs"])
    if semantic["head_cls"].get("has_upstream_sigmoid"):
        raise ValueError(
            f"{backend} head_cls 输出已经包含 Sigmoid；契约要求 raw logits，"
            "后处理只能执行一次 sigmoid")
    expected_2d = expected_runtime["2d"]
    expected_3d = expected_runtime["3d"]
    if io_2d["inputs"][0]["shape"] != expected_2d["input_shape"]:
        raise ValueError(
            f"{backend} 2D input shape={io_2d['inputs'][0]['shape']}，"
            f"FP/config 要求 {expected_2d['input_shape']}")
    expected_feature_nchw = list(expected_2d["output_shape"])
    if expected_2d["output_layout"] == "nhwc":
        expected_feature_nchw = [
            expected_feature_nchw[0], expected_feature_nchw[3],
            expected_feature_nchw[1], expected_feature_nchw[2],
        ]
    actual_feature = list(io_2d["outputs"][0]["shape"])
    actual_feature_nchw = (
        actual_feature if feature_layout == "nchw" else
        [actual_feature[0], actual_feature[3], actual_feature[1], actual_feature[2]])
    if actual_feature_nchw != expected_feature_nchw:
        raise ValueError(
            f"{backend} 2D feature shape/layout={actual_feature}/{feature_layout}，"
            f"FP/config NCHW 要求 {expected_feature_nchw}")
    if io_3d["inputs"][0]["shape"] != expected_3d["input_shape"]:
        raise ValueError(
            f"{backend} 3D input shape={io_3d['inputs'][0]['shape']}，"
            f"FP/config 要求 {expected_3d['input_shape']}")
    expected_outputs = expected_3d["semantic_outputs"]
    for name, item in semantic.items():
        if item["shape"] != expected_outputs[name]["shape"]:
            raise ValueError(
                f"{backend} {name} shape={item['shape']}，"
                f"FP/config 要求 {expected_outputs[name]['shape']}")

    feature_output = io_2d["outputs"][0]
    bev_input = io_3d["inputs"][0]
    feature_raw_quantized = feature_output["dtype"] in ("int8", "uint8")
    bev_raw_quantized = bev_input["dtype"] in ("int8", "uint8")
    if feature_raw_quantized != bev_raw_quantized:
        raise ValueError(
            f"{backend} 2D feature/3D input 量化域类型不一致: "
            f"{feature_output['dtype']} -> {bev_input['dtype']}")
    if feature_raw_quantized:
        same_domain = (
            feature_output["dtype"] == bev_input["dtype"] and
            _quantization_domain(feature_output["quantization"]) ==
            _quantization_domain(bev_input["quantization"]))
        if not same_domain:
            raise ValueError(
                f"{backend} 2D feature 与 3D input 量化参数不一致；"
                "固定 LUT 之间没有隐式 requant 步骤")
        bridge = "raw-quantized-direct"
    else:
        bridge = "float-direct"

    external_dtypes = [
        item["dtype"]
        for item in (
            io_2d["inputs"] + io_2d["outputs"] +
            io_3d["inputs"] + io_3d["outputs"])
    ]
    if backend == "onnx-int8" and not (
        any(dtype in ("int8", "uint8") for dtype in external_dtypes) or
        io_2d["graph_format"] != "float" or
        io_3d["graph_format"] != "float"
    ):
        raise ValueError(
            "onnx-int8 必须是真实量化图；当前两个 ONNX 均为纯浮点图，"
            "禁止把 FP ONNX 作为 INT8 别名")
    return {
        "backend": backend,
        "available": True,
        "numerical_validation": "not_performed",
        "models": {
            "2d": {"file": str(model_2d.resolve()), "sha256": file_sha256(model_2d)},
            "3d": {"file": str(model_3d.resolve()), "sha256": file_sha256(model_3d)},
        },
        "external_io_mode": (
            "raw-int8" if any(
                dtype in ("int8", "uint8") for dtype in external_dtypes)
            else "float32-qdq" if (
                io_2d["graph_format"] == "qdq" or io_3d["graph_format"] == "qdq")
            else "float"),
        "manual_external_quantization": any(
            dtype in ("int8", "uint8") for dtype in external_dtypes),
        "feature_to_bev_bridge": bridge,
        "2d": {
            "input": {**io_2d["inputs"][0], "layout": "nchw"},
            "output": {**feature_output, "layout": feature_layout},
            "graph_format": io_2d["graph_format"],
            "session_inputs": io_2d["session_inputs"],
            "session_outputs": io_2d["session_outputs"],
        },
        "3d": {
            "input": {**bev_input, "layout": "nchw-zc"},
            "outputs": {
                name: {**item, "layout": "nchw"}
                for name, item in semantic.items()
            },
            "semantic_output_names": {
                name: item["name"] for name, item in semantic.items()
            },
            "graph_format": io_3d["graph_format"],
            "session_inputs": io_3d["session_inputs"],
            "session_outputs": io_3d["session_outputs"],
        },
    }


def _aligned_anchor_single_range(
    feature_hw: Sequence[int],
    anchor_range: Sequence[float],
    scale: float,
    sizes: Sequence[Sequence[float]],
    rotations: Sequence[float],
    custom_values: Sequence[float],
    align_corner: bool,
) -> np.ndarray:
    """按 MMDetection3D 的确切轴序生成一个 range 的 aligned anchors。

    最容易混淆的是 flatten 次序。原实现先生成 ``[W,H,Z,S,R,D]``，再
    permute 成 ``[Z,H,W,S,R,D]`` 后 flatten，因此一个空间位置内是 rotation
    先变化，随后才是 size；这里按同一顺序实现，不能使用任意 meshgrid 排列。
    """
    import torch

    feature_h, feature_w = [int(value) for value in feature_hw]
    feature_size = [1, feature_h, feature_w]
    anchor_range_t = torch.tensor(anchor_range, dtype=torch.float32)
    z_centers = torch.linspace(anchor_range_t[2], anchor_range_t[5], 2)
    y_centers = torch.linspace(
        anchor_range_t[1], anchor_range_t[4], feature_size[1] + 1)
    x_centers = torch.linspace(
        anchor_range_t[0], anchor_range_t[3], feature_size[2] + 1)
    if not align_corner:
        z_centers += (z_centers[1] - z_centers[0]) / 2
        y_centers += (y_centers[1] - y_centers[0]) / 2
        x_centers += (x_centers[1] - x_centers[0]) / 2

    sizes_t = torch.tensor(sizes, dtype=torch.float32).reshape(-1, 3) * scale
    rotations_t = torch.tensor(rotations, dtype=torch.float32)
    grids = list(torch.meshgrid(
        x_centers[:feature_w],
        y_centers[:feature_h],
        z_centers[:1],
        rotations_t,
        indexing="ij"))
    tile_shape = [1, 1, 1, int(sizes_t.shape[0]), 1]
    for index in range(len(grids)):
        grids[index] = grids[index].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)
    sizes_t = sizes_t.reshape(1, 1, 1, -1, 1, 3)
    sizes_t = sizes_t.repeat(
        grids[0].shape[0], grids[0].shape[1], grids[0].shape[2], 1,
        grids[0].shape[4], 1)
    grids.insert(3, sizes_t)
    anchors = torch.cat(grids, dim=-1).permute(2, 1, 0, 3, 4, 5)
    if custom_values:
        # 上游实现对 custom_values 只追加零通道，并不会把配置值本身填入。
        # 当前两个通道是 vx/vy anchor 初值，必须保持这个历史语义。
        custom = torch.zeros(
            (*anchors.shape[:-1], len(custom_values)), dtype=anchors.dtype)
        anchors = torch.cat([anchors, custom], dim=-1)
    return anchors.cpu().numpy().astype(np.float32, copy=False)


def _generate_aligned_anchors(head: Mapping[str, Any]) -> np.ndarray:
    """生成并 flatten 当前单层 3D head 的 anchors。"""
    cfg = head["anchor_generator"]
    ranges = list(cfg["ranges"])
    sizes = list(cfg["sizes"])
    size_per_range = bool(cfg["size_per_range"])
    if size_per_range:
        if len(ranges) == 1 and len(sizes) > 1:
            ranges = ranges * len(sizes)
        if len(ranges) != len(sizes):
            raise ValueError("size_per_range=True 时 ranges/sizes 数量必须一致")
        parts = [
            _aligned_anchor_single_range(
                head["feature_map_hw"], anchor_range, cfg["scales"][0],
                [anchor_size], cfg["rotations"], cfg["custom_values"],
                cfg["align_corner"])
            for anchor_range, anchor_size in zip(ranges, sizes)
        ]
        anchors = np.concatenate(parts, axis=-3)
    else:
        if len(ranges) != 1:
            raise ValueError("size_per_range=False 时只允许一个 anchor range")
        anchors = _aligned_anchor_single_range(
            head["feature_map_hw"], ranges[0], cfg["scales"][0], sizes,
            cfg["rotations"], cfg["custom_values"], cfg["align_corner"])
    flattened = np.ascontiguousarray(
        anchors.reshape(-1, int(head["box_code_size"])), dtype=np.float32)
    expected = (
        int(np.prod(head["feature_map_hw"])) *
        int(head["num_anchors_per_location"]))
    if flattened.shape != (expected, int(head["box_code_size"])):
        raise ValueError(f"anchors shape 错误: {flattened.shape}, expected rows={expected}")
    return flattened


def _resolve_config_path(
    config_path: Optional[Path],
    export_metadata: Mapping[str, Any],
    weights_dir: Path,
) -> Path:
    """优先使用显式 config；否则尝试 export_metadata 中记录的路径。"""
    if config_path is not None:
        candidate = config_path.expanduser()
        if candidate.is_file():
            return candidate.resolve()
        repo_candidate = REPO_ROOT / candidate
        if repo_candidate.is_file():
            return repo_candidate.resolve()
        raise FileNotFoundError(f"config 不存在: {config_path}")
    recorded = export_metadata.get("config")
    if recorded:
        return _resolve_recorded_path(str(recorded), weights_dir)
    raise FileNotFoundError(
        "公共模型资产尚未生成，且无法解析 config；首次运行请传 --config")


def build_board_model_assets(
    weights_dir: Path,
    config_path: Optional[Path] = None,
    asset_root: Path = Path(DEFAULT_LUT_ROOT),
    profile_name: str = DEFAULT_ASSET_PROFILE,
    int8_model_2d: Optional[Path] = None,
    int8_model_3d: Optional[Path] = None,
    int8_output_layout: str = "nchw",
    force: bool = False,
) -> Dict[str, Any]:
    """一次性构建板端公共模型资产，并在后续运行中只做完整性检查。

    Args:
        weights_dir: 包含 split 2D/3D ONNX 与 ``export_metadata.json`` 的目录。
        config_path: 首次构建时使用的训练/导出 config。资产已完整时不会读取。
        asset_root: 车辆 LUT 与公共模型资产的共同根目录。
        profile_name: anchor/voxel 几何版本名；改变模板时必须使用新版本名。
        force: 显式重建。默认绝不覆盖一个 hash 不匹配的已有 profile。

    这个函数把 anchors 放在公共 profile 而不是车辆目录，因为 anchors 只由
    模型 head 决定；把 points 放在同一公共目录，因为 voxel 网格也与车辆标定
    无关。车辆内外参只影响 gather/scatter 和 ``featurePointLength.bin``。
    """
    weights_dir = Path(weights_dir).expanduser().resolve()
    asset_root = Path(asset_root).expanduser().resolve()
    if bool(int8_model_2d) != bool(int8_model_3d):
        raise ValueError("INT8 契约要求同时提供 2D 和 3D ONNX")
    if not force and int8_model_2d is None:
        existing = _validate_existing_model_assets(
            weights_dir, asset_root, profile_name)
        if existing is not None:
            if config_path is not None:
                requested_config = _resolve_config_path(
                    Path(config_path), {}, weights_dir)
                if file_sha256(requested_config) != existing["config_sha256"]:
                    raise RuntimeError(
                        f"--config 与现有 board_model_spec SHA256 不一致: "
                        f"{requested_config}")
            return {"status": "existing", "spec": existing}

    model_2d, model_3d, export_metadata = _resolve_onnx_assets(weights_dir)
    config_path = _resolve_config_path(
        Path(config_path) if config_path is not None else None,
        export_metadata,
        weights_dir)
    contract = _extract_model_contract(
        config_path, model_2d, model_3d, export_metadata)
    fp_contract = _build_onnx_backend_contract(
        model_2d,
        model_3d,
        contract["onnx_runtime"]["2d"]["output_layout"],
        contract["onnx_runtime"],
        "onnx-fp",
    )
    export_diffs = []
    for part in ("2d", "3d"):
        value = get_nested(
            dict(export_metadata), ("exports", part, "verify_max_abs_diff"), None)
        if value is not None:
            export_diffs.extend(float(item) for item in value)
    if export_diffs:
        fp_contract["numerical_validation"] = {
            "scope": "export-time-random-input",
            "max_abs_diff": max(export_diffs),
        }

    int8_contract: Dict[str, Any]
    if int8_model_2d is None:
        int8_contract = {
            "backend": "onnx-int8",
            "available": False,
            "reason": "real_int8_onnx_not_provided",
            "numerical_validation": "not_performed",
            "statement": "未做真实 INT8 模型数值验证",
        }
    else:
        int8_2d = Path(int8_model_2d).expanduser().resolve()
        int8_3d = Path(int8_model_3d).expanduser().resolve()
        if not int8_2d.is_file() or not int8_3d.is_file():
            raise FileNotFoundError(
                f"INT8 ONNX 不完整: 2d={int8_2d}, 3d={int8_3d}")
        int8_contract = _build_onnx_backend_contract(
            int8_2d,
            int8_3d,
            str(int8_output_layout).lower(),
            contract["onnx_runtime"],
            "onnx-int8",
        )
        int8_contract["statement"] = "未做真实 INT8 模型数值验证"

    geometry = contract["geometry"]
    head = contract["head"]
    points = get_points(
        geometry["n_voxels"], geometry["voxel_size"], geometry["origin"])
    points = np.ascontiguousarray(points, dtype=np.float32)
    anchors = _generate_aligned_anchors(head)

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
    asset_contract_sha = json_sha256(asset_contract)
    common_dir = asset_root / "common" / profile_name
    common_dir.mkdir(parents=True, exist_ok=True)
    anchors_path = common_dir / "anchors.npy"
    points_path = common_dir / "points.npy"

    # profile 已存在时先逐值核验。默认不能因为用户换了一份 config 就悄悄
    # 覆盖公共 anchor，否则其他权重会继续引用同名 profile 并产生错误框。
    for path, expected in ((anchors_path, anchors), (points_path, points)):
        if path.is_file() and not force:
            actual = np.load(path)
            if not np.array_equal(actual, expected):
                raise RuntimeError(
                    f"公共 profile {profile_name} 的 {path.name} 与当前模型不一致；"
                    "请为新几何使用新的 --asset-profile")
        else:
            np.save(path, expected)

    manifest = {
        "schema_version": 2,
        "version": "mono-front-board-common-assets-v1",
        "profile": profile_name,
        "asset_contract": asset_contract,
        "asset_contract_sha256": asset_contract_sha,
        "geometry_contract_hash": asset_contract_sha,
        "anchors": {
            "path": str(anchors_path.resolve()),
            "shape": list(anchors.shape),
            "dtype": str(anchors.dtype),
            "sha256": file_sha256(anchors_path),
            "box_origin": "bottom",
        },
        "points": {
            "path": str(points_path.resolve()),
            "shape": list(points.shape),
            "dtype": str(points.dtype),
            "sha256": file_sha256(points_path),
        },
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json(common_dir / "asset_manifest.json", manifest)

    model_contract_sha = json_sha256({
        "config_sha256": contract["config_sha256"],
        "checkpoint_sha256": contract.get("checkpoint_sha256"),
        "preprocess": contract["preprocess"],
        "geometry": contract["geometry"],
        "head": contract["head"],
    })
    spec = {
        "schema_version": 2,
        "version": "mono-front-board-model-spec-v2",
        "asset_profile": profile_name,
        "asset_contract_sha256": asset_contract_sha,
        "geometry_contract_hash": asset_contract_sha,
        "model_contract_sha256": model_contract_sha,
        "config": contract["config"],
        "config_sha256": contract["config_sha256"],
        "checkpoint": contract.get("checkpoint"),
        "checkpoint_sha256": contract.get("checkpoint_sha256"),
        "onnx": {
            "2d": {
                "file": model_2d.name,
                "sha256": file_sha256(model_2d),
            },
            "3d": {
                "file": model_3d.name,
                "sha256": file_sha256(model_3d),
            },
        },
        "preprocess": contract["preprocess"],
        "geometry": contract["geometry"],
        "head": contract["head"],
        "onnx_runtime": contract["onnx_runtime"],
        "onnx_contracts": {
            "onnx-fp": fp_contract,
            "onnx-int8": int8_contract,
        },
        "assets": {
            "anchors_relative": f"common/{profile_name}/anchors.npy",
            "anchors_shape": list(anchors.shape),
            "anchors_sha256": file_sha256(anchors_path),
            "points_relative": f"common/{profile_name}/points.npy",
            "points_shape": list(points.shape),
            "points_sha256": file_sha256(points_path),
        },
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json(weights_dir / "board_model_spec.json", spec)
    LOGGER.info("已生成公共模型资产: %s", common_dir)
    return {"status": "generated", "spec": spec}


def _generate_vehicle_lut(
    job: VehicleJob,
    spec: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    """由生产车 info.json 直接生成 compact board bin + LUT_arr。"""
    if job.info_json is None or not job.info_json.is_file():
        raise FileNotFoundError(
            f"{job.vehicle_id}: LUT 不存在，且没有可读取的 info.json")
    calibration = read_json(job.info_json)
    sensor = find_sensor(calibration, args.sensor_name)
    first_image = job.images[0]
    first_image_size = read_image_size(first_image)
    camera_args = SimpleNamespace(
        intrinsic_size=args.intrinsic_size,
        intrinsic_size_source=args.intrinsic_size_source,
        info_extrinsic_coordinate=(
            "raw_n7" if args.info_extrinsic_coordinate == "raw" else
            args.info_extrinsic_coordinate),
        aspect_tolerance=args.aspect_tolerance,
        allow_aspect_mismatch=args.allow_aspect_mismatch,
    )
    camera_template = build_camera_template(sensor, first_image_size, camera_args)
    check_aspect(
        first_image_size,
        (camera_template["intrinsic_width"], camera_template["intrinsic_height"]),
        camera_args,
        first_image)
    cam_info = build_camera_info(camera_template, first_image, first_image_size)
    info = {
        "token": f"board_lut_{safe_output_name(job.vehicle_id)}",
        "timestamp": 0,
        "cams": {args.camera_id: cam_info},
    }
    geometry = spec["geometry"]
    img_meta, camera_ids, frame_records = build_img_meta_for_sample(
        info=info,
        camera_types=(args.camera_id,),
        n_times=1,
        sequential=False,
        test_adj_ids=None,
        min_interval=0,
        temporal_compensate=False,
        data_config=geometry["data_config"],
        force_resize=bool(spec["preprocess"]["force_resize"]),
    )
    metadata = {
        "version": "mono-front-board-vehicle-lut-v1",
        "vehicle_id": job.vehicle_id,
        "info_json": str(job.info_json.resolve()),
        "info_json_sha256": file_sha256(job.info_json),
        "source_image": str(first_image.resolve()),
        "source_image_sha256": file_sha256(first_image),
        "source_image_size": list(first_image_size),
        "sample_token": info["token"],
        "frame_records": frame_records,
        "camera_types": [args.camera_id],
        "camera_id_sequence": camera_ids,
        "n_images": 1,
        "n_times": 1,
        "n_voxels": geometry["n_voxels"],
        "voxel_size": geometry["voxel_size"],
        "origin": geometry["origin"],
        "stride": geometry["stride"],
        "use_distortion": geometry["use_distortion"],
        "force_resize": spec["preprocess"]["force_resize"],
        "data_config": geometry["data_config"],
        "camera_overwrite_order": [0],
        "projection_backend": "torch",
        "torch_device": args.lut_device,
        "model_asset_profile": spec["asset_profile"],
        "model_asset_contract_sha256": spec["asset_contract_sha256"],
        "geometry_contract_hash": spec["geometry_contract_hash"],
        "calibration": camera_template,
    }
    export_one_sequence(
        out_dir=job.lut_dir,
        img_meta=img_meta,
        seq_id=0,
        n_images=1,
        n_voxels=geometry["n_voxels"],
        voxel_size=geometry["voxel_size"],
        origin=geometry["origin"],
        stride=int(geometry["stride"]),
        use_distortion=bool(geometry["use_distortion"]),
        camera_overwrite_order=(0,),
        feature_channels=int(geometry["feature_shape_nchw"][1]),
        dump_debug_volume=False,
        projection_backend="torch",
        torch_device=args.lut_device,
        common_metadata=metadata,
        compact_output=True,
    )
    metadata_path = job.lut_dir / "metadata.json"
    written_metadata = read_json(metadata_path)
    lut_paths = {
        "LUT/gather_new_0.bin": job.lut_dir / "LUT" / "gather_new_0.bin",
        "LUT/scatter_nd_new_0.bin": job.lut_dir / "LUT" / "scatter_nd_new_0.bin",
        "LUT/featurePointLength.bin": job.lut_dir / "LUT" / "featurePointLength.bin",
        "LUT_arr/gather_0.npy": job.lut_dir / "LUT_arr" / "gather_0.npy",
        "LUT_arr/scatter_nd_0.npy": job.lut_dir / "LUT_arr" / "scatter_nd_0.npy",
    }
    missing = [str(path) for path in lut_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LUT 导出不完整: {missing}")
    written_metadata["lut_file_sha256"] = {
        name: file_sha256(path) for name, path in lut_paths.items()
    }
    write_json(metadata_path, written_metadata)


def build_board_vehicle_lut(
    vehicle_id: str,
    info_json: Path,
    image_path: Path,
    spec: Mapping[str, Any],
    asset_root: Path = Path(DEFAULT_LUT_ROOT),
    sensor_name: str = "front_wide",
    camera_id: str = "cam0",
    intrinsic_size: Optional[Sequence[int]] = None,
    intrinsic_size_source: str = "auto",
    info_extrinsic_coordinate: str = "raw",
    aspect_tolerance: float = 0.02,
    allow_aspect_mismatch: bool = False,
    torch_device: str = "cuda:0",
    force: bool = False,
) -> Dict[str, Any]:
    """一次性生成或校验车辆 LUT，供资产准备 CLI 和自动化脚本复用。

    车辆 LUT 只依赖模型几何、相机标定以及相机图像坐标系。``image_path`` 只
    用于确定生产图片的宽高和记录资产来源，不会运行模型。已有完整 LUT 默认
    直接校验并复用；只有 ``force=True`` 才允许显式覆盖。
    """
    info_json = Path(info_json).expanduser().resolve()
    image_path = Path(image_path).expanduser().resolve()
    if not info_json.is_file():
        raise FileNotFoundError(f"info.json 不存在: {info_json}")
    if not image_path.is_file():
        raise FileNotFoundError(f"LUT 参考图片不存在: {image_path}")

    vehicle_id = safe_output_name(str(vehicle_id))
    lut_dir = Path(asset_root).expanduser().resolve() / vehicle_id
    state, exists = _board_bin_state(lut_dir)
    if state == "partial" and not force:
        raise RuntimeError(
            f"{vehicle_id}: LUT bin 不完整，拒绝自动补齐: {exists}；"
            "确认目录后使用 --force-vehicle-lut 重建")
    if state == "complete" and not force:
        _, _, record = load_board_vehicle_lut(
            vehicle_id, spec, asset_root, require_calibration=True)
        recorded_sha = record["metadata"].get("info_json_sha256")
        actual_sha = file_sha256(info_json)
        if recorded_sha != actual_sha:
            raise RuntimeError(
                f"{vehicle_id}: 已有 LUT 对应 info.json SHA={recorded_sha}，"
                f"当前为 {actual_sha}；确认标定版本后用 --force-vehicle-lut")
        record["status"] = "existing"
        return record

    job = VehicleJob(
        vehicle_id=vehicle_id,
        info_json=info_json,
        images=(image_path,),
        image_base=image_path.parent,
        lut_dir=lut_dir,
        output_dir=Path("."),
    )
    args = SimpleNamespace(
        sensor_name=sensor_name,
        camera_id=camera_id,
        intrinsic_size=(
            list(map(int, intrinsic_size)) if intrinsic_size is not None else None),
        intrinsic_size_source=intrinsic_size_source,
        info_extrinsic_coordinate=info_extrinsic_coordinate,
        aspect_tolerance=float(aspect_tolerance),
        allow_aspect_mismatch=bool(allow_aspect_mismatch),
        lut_device=torch_device,
    )
    _generate_vehicle_lut(job, spec, args)
    _, _, record = load_board_vehicle_lut(
        vehicle_id, spec, asset_root, require_calibration=True)
    record["status"] = "generated"
    return record


# 本文件仅负责离线资产构建，不反向导入 standalone、analyzer 或 legacy runtime。
def _candidate_images(path: Path, patterns: Sequence[str]) -> List[Path]:
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError(f"图片路径不存在: {path}")
    candidates = set()
    for pattern in patterns:
        candidates.update(item.resolve() for item in path.rglob(pattern) if item.is_file())
    return sorted(candidates, key=natural_key)


def _resolve_info_json(root: Path, vehicle_id: str, template: str) -> Optional[Path]:
    if root.is_file():
        return root.resolve()
    if not root.exists():
        return None
    name = template.format(vehicle_id=vehicle_id, vehicle=vehicle_id)
    candidate = root / name
    if candidate.is_file():
        return candidate.resolve()
    matches = sorted(root.rglob(f"*{vehicle_id}*.json"), key=natural_key)
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(
            f"车辆 {vehicle_id} 匹配到多个 info.json，请调整 --info-json-template: {matches}")
    return None


def _infer_vehicle_id_from_info(path: Path) -> str:
    match = re.search(r"_info_(.+)$", path.stem)
    if match:
        return safe_output_name(match.group(1))
    return safe_output_name(path.stem)
