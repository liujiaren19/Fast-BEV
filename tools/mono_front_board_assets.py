#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端参考推理所需的只读资产加载。

本模块只读取并校验已经生成的 ``board_model_spec.json``、公共 anchors 和
车辆固定 LUT。它不会读取 config/info.json，也不会生成、补齐或覆盖任何资产。
资产生成统一由 ``tools/build_mono_front_board_assets.py`` 完成。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LUT_ROOT = "data/board_lut"
BOARD_BIN_NAMES = (
    "gather_new_0.bin",
    "scatter_nd_new_0.bin",
    "featurePointLength.bin",
)
BOARD_ARRAY_NAMES = (
    "gather_0.npy",
    "scatter_nd_0.npy",
)
SEMANTIC_LOGITS = ("head_cls", "head_bbox", "head_dir")
LOGGER = logging.getLogger("mono_front_board_assets")
ASSET_INIT_HINT = (
    "首次运行前先执行 tools/build_mono_front_board_assets.py 初始化模型和车辆资产。")


def safe_vehicle_id(value: str) -> str:
    """把车辆编号限制为可安全用作目录名的字符。"""
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("._")
    if not normalized:
        raise ValueError(f"车辆编号无法转换成有效目录名: {value!r}")
    return normalized


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """流式计算文件 SHA256，避免一次性读取 ONNX。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
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


def quantization_domain(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """提取跨 split 模型必须一致的量化数值域，忽略 Q/DQ 来源算子。"""
    keys = (
        "mode", "scale", "zero_point", "channel_axis", "round_policy",
        "saturation_policy", "clamp_min", "clamp_max",
    )
    return {key: value.get(key) for key in keys} if value else {}


def validate_semantic_output_names(mapping: Mapping[str, Any]) -> Dict[str, str]:
    """要求 3D ONNX 输出名明确表达 cls/bbox/dir 语义。"""
    if set(mapping) != set(SEMANTIC_LOGITS):
        raise RuntimeError(
            f"3D 输出语义不完整: {mapping}; expected={SEMANTIC_LOGITS}")
    names = {semantic: str(mapping[semantic]) for semantic in SEMANTIC_LOGITS}
    if len(set(names.values())) != len(names):
        raise RuntimeError(f"3D 输出语义映射存在重复名称: {names}")
    lowered = {semantic: name.lower() for semantic, name in names.items()}
    valid = (
        "cls" in lowered["head_cls"] and "dir" not in lowered["head_cls"] and
        "bbox" in lowered["head_bbox"] and
        "dir" in lowered["head_dir"]
    )
    if not valid:
        raise RuntimeError(
            "3D 输出名没有明确表达 cls/bbox/dir 语义；拒绝使用可能由位置回退"
            f"生成的 mapping: {names}")
    return names


def validate_model_assets(
    weights_dir: Path,
    asset_root: Path,
    profile_name: str,
) -> Optional[Dict[str, Any]]:
    """校验已存在的模型公共资产；缺文件返回 ``None``，不生成文件。"""
    weights_dir = Path(weights_dir).expanduser().resolve()
    asset_root = Path(asset_root).expanduser().resolve()
    spec_path = weights_dir / "board_model_spec.json"
    if not spec_path.is_file():
        return None
    spec = read_json(spec_path)
    if spec.get("schema_version") != 2:
        LOGGER.info("旧版 board_model_spec 需要由 asset builder 升级: %s", spec_path)
        return None
    if spec.get("asset_profile") != profile_name:
        raise ValueError(
            f"现有 board_model_spec profile={spec.get('asset_profile')}，"
            f"本次请求 profile={profile_name}")

    runtime_contract = spec.get("onnx_runtime", {})
    contract_2d = runtime_contract.get("2d", {})
    contract_3d = runtime_contract.get("3d", {})
    layout = str(contract_2d.get("output_layout", "")).lower()
    if layout not in ("nchw", "nhwc"):
        raise RuntimeError(
            "现有 board_model_spec 缺少明确的 2D output_layout，拒绝复用")
    semantic_names = contract_3d.get("semantic_output_names")
    if not isinstance(semantic_names, dict):
        raise RuntimeError(
            "现有 board_model_spec 缺少 3D semantic_output_names，拒绝复用")
    validate_semantic_output_names(semantic_names)

    def require_sha(value: Any, label: str) -> str:
        text = str(value or "")
        if not re.fullmatch(r"[0-9a-f]{64}", text):
            raise RuntimeError(f"board_model_spec 缺少有效 {label} SHA256")
        return text

    require_sha(spec.get("config_sha256"), "config")
    require_sha(spec.get("checkpoint_sha256"), "checkpoint")
    require_sha(spec.get("model_contract_sha256"), "model contract")
    require_sha(spec.get("geometry_contract_hash"), "geometry contract")
    if spec.get("geometry_contract_hash") != spec.get("asset_contract_sha256"):
        raise RuntimeError("geometry_contract_hash 与 asset_contract_sha256 不一致")

    geometry = spec.get("geometry", {})
    head = spec.get("head", {})
    preprocess = spec.get("preprocess", {})
    geometry_keys = (
        "n_images", "n_times", "camera_types", "n_voxels", "voxel_size",
        "origin", "feature_shape_nchw", "bev_shape_nchw", "stride",
        "channel_layout", "use_distortion", "data_config",
    )
    head_keys = (
        "feature_map_hw", "num_anchors_per_location", "box_code_size",
        "anchor_generator",
    )
    missing_geometry = [key for key in geometry_keys if key not in geometry]
    missing_head = [key for key in head_keys if key not in head]
    if missing_geometry or missing_head or not preprocess:
        raise RuntimeError(
            "board_model_spec 契约不完整: "
            f"geometry={missing_geometry}, head={missing_head}, "
            f"preprocess={bool(preprocess)}")
    model_contract = {
        "config_sha256": spec["config_sha256"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "preprocess": preprocess,
        "geometry": geometry,
        "head": head,
    }
    if json_sha256(model_contract) != spec["model_contract_sha256"]:
        raise RuntimeError("model_contract_sha256 与模型契约内容不一致")
    asset_contract = {
        "geometry": {key: geometry[key] for key in geometry_keys},
        "head": {key: head[key] for key in head_keys},
    }
    if json_sha256(asset_contract) != spec["geometry_contract_hash"]:
        raise RuntimeError("geometry_contract_hash 与几何/anchor 契约内容不一致")

    contracts = spec.get("onnx_contracts")
    if not isinstance(contracts, dict):
        raise RuntimeError("board_model_spec 缺少 FP/INT8 ONNX contracts")
    fp_contract = contracts.get("onnx-fp")
    if not isinstance(fp_contract, dict) or not fp_contract.get("available"):
        raise RuntimeError("board_model_spec 缺少可用 FP ONNX contract")
    int8_contract = contracts.get("onnx-int8")
    if not isinstance(int8_contract, dict):
        raise RuntimeError("board_model_spec 缺少 INT8 ONNX contract 状态")

    for backend, contract in contracts.items():
        if not contract.get("available"):
            continue
        for part in ("2d", "3d"):
            model_record = contract.get("models", {}).get(part, {})
            recorded_path = Path(
                str(model_record.get("file", ""))).expanduser()
            candidates = (
                [recorded_path, weights_dir / recorded_path.name]
                if recorded_path.is_absolute() else
                [weights_dir / recorded_path, weights_dir / recorded_path.name]
            )
            path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                None)
            if path is None:
                raise FileNotFoundError(
                    f"{backend} {part} ONNX 不存在: tried={candidates}；"
                    f"{ASSET_INIT_HINT}")
            expected_sha = require_sha(
                model_record.get("sha256"), f"{backend} {part} ONNX")
            if file_sha256(path) != expected_sha:
                raise RuntimeError(f"{backend} {part} ONNX SHA256 不匹配: {path}")
        tensors = [
            contract.get("2d", {}).get("input", {}),
            contract.get("2d", {}).get("output", {}),
            contract.get("3d", {}).get("input", {}),
            *contract.get("3d", {}).get("outputs", {}).values(),
        ]
        for tensor in tensors:
            for key in ("name", "shape", "dtype", "layout"):
                if key not in tensor:
                    raise RuntimeError(
                        f"{backend} tensor contract 缺少 {key}: {tensor}")
            if tensor["dtype"] in ("int8", "uint8"):
                quant = tensor.get("quantization")
                required = {
                    "mode", "scale", "zero_point", "channel_axis",
                    "round_policy", "saturation_policy", "clamp_min", "clamp_max",
                }
                if not isinstance(quant, dict) or not required <= set(quant):
                    raise RuntimeError(
                        f"{backend} raw INT8/UINT8 tensor 缺少量化参数: {tensor}")
        semantic_names = contract.get("3d", {}).get(
            "semantic_output_names", {})
        validate_semantic_output_names(semantic_names)
        if set(contract["3d"]["outputs"]) != set(semantic_names):
            raise RuntimeError(
                f"{backend} 3D outputs contract 包含错误语义: "
                f"{sorted(contract['3d']['outputs'])}")
        for semantic, output_name in semantic_names.items():
            recorded_name = contract["3d"]["outputs"].get(
                semantic, {}).get("name")
            if output_name != recorded_name:
                raise RuntimeError(
                    f"{backend} 3D {semantic} 输出名契约不一致: "
                    f"semantic={output_name}, tensor={recorded_name}")
        if contract["3d"]["outputs"]["head_cls"].get(
            "has_upstream_sigmoid"
        ):
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

    common_dir = asset_root / "common" / profile_name
    anchors_path = common_dir / "anchors.npy"
    points_path = common_dir / "points.npy"
    manifest_path = common_dir / "asset_manifest.json"
    if not all(path.is_file() for path in (
        anchors_path, points_path, manifest_path
    )):
        return None

    for part in ("2d", "3d"):
        record = spec["onnx"][part]
        path = weights_dir / record["file"]
        if not path.is_file():
            return None
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(
                f"{part} ONNX 已变化，拒绝复用旧 board_model_spec: {path}")
    if file_sha256(anchors_path) != spec["assets"]["anchors_sha256"]:
        raise RuntimeError(
            f"anchors.npy SHA256 与 board_model_spec 不一致: {anchors_path}")
    if file_sha256(points_path) != spec["assets"]["points_sha256"]:
        raise RuntimeError(
            f"points.npy SHA256 与 board_model_spec 不一致: {points_path}")

    anchors = np.load(anchors_path, mmap_mode="r")
    points = np.load(points_path, mmap_mode="r")
    if list(anchors.shape) != spec["assets"]["anchors_shape"]:
        raise RuntimeError(f"anchors.npy shape 被修改: {anchors.shape}")
    if list(points.shape) != spec["assets"]["points_shape"]:
        raise RuntimeError(f"points.npy shape 被修改: {points.shape}")
    manifest = read_json(manifest_path)
    if manifest.get("asset_contract_sha256") != spec.get(
        "asset_contract_sha256"
    ):
        raise RuntimeError(
            "公共 asset_manifest 与 board_model_spec 的几何契约 hash 不一致")
    return spec


def load_board_model_assets(
    weights_dir: Path,
    asset_root: Path = Path(DEFAULT_LUT_ROOT),
) -> Dict[str, Any]:
    """严格读取现成模型公共资产；缺失或变化时直接报错。"""
    weights_dir = Path(weights_dir).expanduser().resolve()
    asset_root = Path(asset_root).expanduser().resolve()
    spec_path = weights_dir / "board_model_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"缺少模型公共资料 {spec_path}；{ASSET_INIT_HINT}")
    recorded = read_json(spec_path)
    profile_name = recorded.get("asset_profile")
    if not profile_name:
        raise RuntimeError(f"board_model_spec 缺少 asset_profile: {spec_path}")
    spec = validate_model_assets(weights_dir, asset_root, str(profile_name))
    if spec is None:
        raise FileNotFoundError(
            f"模型公共资料不完整；{ASSET_INIT_HINT}"
            "推理时禁止自动生成资产。")
    LOGGER.info("已加载模型公共资产: %s", profile_name)
    return spec


def board_bin_state(lut_dir: Path) -> Tuple[str, Dict[str, bool]]:
    """返回车辆 LUT bin 是 complete、missing 还是 partial。"""
    board_dir = Path(lut_dir).expanduser().resolve() / "LUT"
    exists = {name: (board_dir / name).is_file() for name in BOARD_BIN_NAMES}
    count = sum(exists.values())
    if count == len(BOARD_BIN_NAMES):
        return "complete", exists
    if count == 0:
        return "missing", exists
    return "partial", exists


def _read_single_camera_lut(lut_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """按板端 int32 bin 格式读取 cam0 gather/scatter 和有效点数。"""
    board_dir = Path(lut_dir) / "LUT"
    length_values = np.fromfile(
        board_dir / "featurePointLength.bin", dtype=np.int32)
    if length_values.shape != (1,):
        raise ValueError(
            "原生单帧 LUT 的 featurePointLength.bin 必须只包含 1 个 int32，"
            f"实际 shape={length_values.shape}")
    gather = np.fromfile(board_dir / "gather_new_0.bin", dtype=np.int32)
    scatter = np.fromfile(board_dir / "scatter_nd_new_0.bin", dtype=np.int32)
    expected_length = int(length_values[0])
    if gather.size != expected_length or scatter.size != expected_length:
        raise ValueError(
            f"featurePointLength={expected_length}，但 gather/scatter="
            f"{gather.size}/{scatter.size}")
    return gather.astype(np.int64), scatter.astype(np.int64)


def _validate_lut_indices(
    gather: np.ndarray,
    scatter: np.ndarray,
    geometry: Mapping[str, Any],
) -> None:
    feature_shape = [int(value) for value in geometry["feature_shape_nchw"]]
    n_voxels = [int(value) for value in geometry["n_voxels"]]
    feature_points = feature_shape[2] * feature_shape[3]
    volume_points = int(np.prod(n_voxels))
    if gather.size and (gather.min() < 0 or gather.max() >= feature_points):
        raise ValueError(f"gather index 越界，feature_points={feature_points}")
    if scatter.size and (scatter.min() < 0 or scatter.max() >= volume_points):
        raise ValueError(f"scatter index 越界，volume_points={volume_points}")


def load_board_vehicle_lut(
    vehicle_id: str,
    spec: Mapping[str, Any],
    asset_root: Path = Path(DEFAULT_LUT_ROOT),
    require_calibration: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """严格读取一辆车的固定 LUT；运行时绝不自动生成或补齐。"""
    normalized_id = safe_vehicle_id(vehicle_id)
    lut_dir = Path(asset_root).expanduser().resolve() / normalized_id
    state, exists = board_bin_state(lut_dir)
    if state != "complete":
        raise FileNotFoundError(
            f"{normalized_id}: 固定 LUT 不完整: {exists}。{ASSET_INIT_HINT}"
            "推理入口不会自动生成 LUT。")

    gather, scatter = _read_single_camera_lut(lut_dir)
    array_paths = [lut_dir / "LUT_arr" / name for name in BOARD_ARRAY_NAMES]
    missing_arrays = [str(path) for path in array_paths if not path.is_file()]
    if missing_arrays:
        raise FileNotFoundError(
            f"{normalized_id}: 固定 LUT_arr 不完整: {missing_arrays}。"
            f"{ASSET_INIT_HINT}")
    geometry = spec["geometry"]
    _validate_lut_indices(gather, scatter, geometry)

    metadata_path = lut_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{normalized_id}: 缺少 LUT metadata.json: {lut_dir}。"
            f"{ASSET_INIT_HINT}")
    metadata = read_json(metadata_path)
    checks = {
        "n_images": 1,
        "n_times": 1,
        "n_voxels": geometry["n_voxels"],
        "feature_shape": geometry["feature_shape_nchw"],
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"{normalized_id}: LUT metadata {key}={metadata.get(key)}，"
                f"模型要求 {expected}")
    if metadata.get("model_asset_profile") != spec["asset_profile"]:
        raise RuntimeError(
            f"{normalized_id}: LUT profile={metadata.get('model_asset_profile')}，"
            f"模型 profile={spec['asset_profile']}")
    if metadata.get("model_asset_contract_sha256") != spec[
        "asset_contract_sha256"
    ]:
        raise RuntimeError(f"{normalized_id}: LUT 几何契约 hash 与当前模型不一致")
    if metadata.get("geometry_contract_hash") != spec["geometry_contract_hash"]:
        raise RuntimeError(
            f"{normalized_id}: LUT geometry_contract_hash 与当前模型不一致")

    expected_hashes = metadata.get("lut_file_sha256")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError(f"{normalized_id}: LUT metadata 缺少文件 SHA256")
    actual_paths = {
        **{f"LUT/{name}": lut_dir / "LUT" / name for name in BOARD_BIN_NAMES},
        **{f"LUT_arr/{name}": lut_dir / "LUT_arr" / name for name in BOARD_ARRAY_NAMES},
    }
    for name, path in actual_paths.items():
        expected_sha = expected_hashes.get(name)
        if not isinstance(expected_sha, str) or file_sha256(path) != expected_sha:
            raise RuntimeError(f"{normalized_id}: LUT 文件 SHA256 不匹配: {path}")
    gather_array = np.asarray(
        np.load(lut_dir / "LUT_arr" / "gather_0.npy"),
        dtype=np.int64).reshape(-1)
    scatter_array = np.asarray(
        np.load(lut_dir / "LUT_arr" / "scatter_nd_0.npy"),
        dtype=np.int64).reshape(-1)
    if not np.array_equal(gather, gather_array):
        raise RuntimeError(f"{normalized_id}: gather bin 与 LUT_arr 不一致")
    if not np.array_equal(scatter, scatter_array):
        raise RuntimeError(f"{normalized_id}: scatter bin 与 LUT_arr 不一致")

    source_image_size = metadata.get("source_image_size")
    if (
        not isinstance(source_image_size, list) or
        len(source_image_size) != 2 or
        min(int(value) for value in source_image_size) <= 0
    ):
        raise RuntimeError(
            f"{normalized_id}: LUT metadata 缺少有效 source_image_size")
    if require_calibration and not isinstance(metadata.get("calibration"), dict):
        raise RuntimeError(
            f"{normalized_id}: LUT metadata 缺少 calibration，无法可靠可视化")

    return gather, scatter, {
        "source": "existing_board_bin",
        "root": str(lut_dir),
        "gather_count": int(gather.size),
        "scatter_count": int(scatter.size),
        "metadata": metadata,
        "files": {
            name: file_sha256(path) for name, path in actual_paths.items()
        },
    }


__all__ = [
    "BOARD_BIN_NAMES",
    "BOARD_ARRAY_NAMES",
    "DEFAULT_LUT_ROOT",
    "SEMANTIC_LOGITS",
    "ASSET_INIT_HINT",
    "board_bin_state",
    "file_sha256",
    "load_board_model_assets",
    "load_board_vehicle_lut",
    "read_json",
    "safe_vehicle_id",
    "validate_model_assets",
    "validate_semantic_output_names",
]
