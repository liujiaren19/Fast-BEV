#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐级比较 N7 mono-front 标准 dataset 与生产 PTH 推理链路。

真实模式只在 legacy MMDetection3D 环境中导入 mmcv/mmdet/mmdet3d。本工具只
保留生成/复核 PTH 黄金参考的 legacy 职责；synthetic 覆盖位于 ``tools/tests``。

对齐链路固定为：

1. ``CustomMultiViewDataset`` 的标准 test pipeline；
2. ``tools/infer_mono_front_image.py`` 的生产 sample 构造与 test pipeline；
3. 同一个 FP32 ``model.eval()`` 与同一个 ``bbox_head.get_bboxes`` 后处理。

工具侧通过临时 forward hooks 捕获 2D feature、3D neck 输入和 bbox head 原始
logits，不修改 FastBEV 正常 train/test 行为，也不会持续保留 GPU 大张量。
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import infer_mono_front_image as production_infer  # noqa: E402
from tools.n7_box_origin import boxes_to_numpy as n7_boxes_to_numpy  # noqa: E402


REPORT_SCHEMA_VERSION = 1
CRITICAL_STAGE_ORDER = (
    "raw_image",
    "calibration",
    "input",
    "feature_2d",
    "bev_input",
    "head_logits",
    "decoded_boxes",
)
CALIBRATION_FIELDS = (
    "cam_intrinsic",
    "intrinsic_width",
    "intrinsic_height",
    "image_width",
    "image_height",
    "distortion",
    "sensor2lidar_rotation",
    "sensor2lidar_translation",
    "post_rot",
    "post_tran",
    "projection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="N7 mono-front 标准 dataset 与生产 PTH 同图逐级数值对齐",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="S0 Fast-BEV 配置")
    parser.add_argument("--checkpoint", help="S0 PTH checkpoint")
    parser.add_argument("--ann-file", help="覆盖 config.data.test.ann_file 的 val pkl")
    parser.add_argument("--data-root", help="覆盖 config.data.test.data_root")
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("--sample-index", type=int, help="排序后的 val dataset 索引")
    sample_group.add_argument("--token", help="精确选择 val token")
    parser.add_argument("--info-json", help="可选：再运行真实车型 info.json 标定模式")
    parser.add_argument(
        "--intrinsic-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
        help="info.json 中 K 对应的尺寸",
    )
    parser.add_argument(
        "--info-extrinsic-coordinate", choices=("raw", "raw_n7", "fastbev"),
        default="raw",
        help="info.json 外参坐标；raw 与生产脚本 raw_n7 同义",
    )
    parser.add_argument("--sensor-name", default="front_wide", help="info.json 相机名")
    parser.add_argument("--camera-id", default="cam0", help="val pkl 前视相机 id")
    parser.add_argument("--output-dir", required=True, help="诊断输出目录")
    parser.add_argument("--device", default="cuda:0", help="PTH 推理设备")
    parser.add_argument("--atol", type=float, default=1e-6, help="数值比较绝对容差")
    parser.add_argument("--rtol", type=float, default=1e-5, help="数值比较相对容差")
    parser.add_argument("--dump-tensors", action="store_true", help="保存完整核心 tensor")
    parser.add_argument("--strict", action="store_true", help="关键层不一致时返回非零")
    return parser.parse_args()


def sha256_file(path: Optional[Path], chunk_size: int = 8 * 1024 * 1024) -> Optional[str]:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(arr.shape)).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_text_command(command: Sequence[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(command), cwd=str(REPO_ROOT), check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except OSError:
        return None
    return completed.stdout.strip()


def module_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "installed"))


def array_summary(value: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(value)
    summary: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": sha256_array(arr),
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        summary.update({
            "finite": bool(finite.all()),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr.astype(np.float64))),
        })
    return summary


def compare_arrays(
    reference: Any,
    candidate: Any,
    atol: float,
    rtol: float,
    *,
    exact_required: bool = False,
) -> Dict[str, Any]:
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    report: Dict[str, Any] = {
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(cand.dtype),
        "atol": float(atol),
        "rtol": float(rtol),
        "exact_required": bool(exact_required),
    }
    if ref.shape != cand.shape:
        report.update({
            "exact_equal": False,
            "allclose": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "max_rel_diff": None,
            "status": "FAIL",
            "reason": "shape_mismatch",
        })
        return report

    exact_equal = bool(np.array_equal(ref, cand, equal_nan=True))
    numeric = np.issubdtype(ref.dtype, np.number) and np.issubdtype(cand.dtype, np.number)
    if numeric:
        ref64 = ref.astype(np.float64, copy=False)
        cand64 = cand.astype(np.float64, copy=False)
        diff = np.abs(ref64 - cand64)
        denom = np.maximum(np.abs(ref64), np.finfo(np.float64).eps)
        allclose = bool(np.allclose(ref64, cand64, atol=atol, rtol=rtol, equal_nan=True))
        max_abs = float(np.nanmax(diff)) if diff.size else 0.0
        mean_abs = float(np.nanmean(diff)) if diff.size else 0.0
        max_rel = float(np.nanmax(diff / denom)) if diff.size else 0.0
    else:
        allclose = exact_equal
        max_abs = mean_abs = max_rel = 0.0 if exact_equal else None

    passed = exact_equal if exact_required else allclose
    report.update({
        "exact_equal": exact_equal,
        "allclose": allclose,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_rel_diff": max_rel,
        "status": "PASS" if passed else "FAIL",
    })
    if not passed and ref.size:
        unequal = ~(np.isclose(ref, cand, atol=atol, rtol=rtol, equal_nan=True) if numeric else ref == cand)
        indices = np.argwhere(unequal)
        if indices.size:
            index = tuple(int(x) for x in indices[0])
            report["first_diff_index"] = list(index)
            report["first_reference_value"] = json_ready(ref[index])
            report["first_candidate_value"] = json_ready(cand[index])
    return report


def flatten_tensor_tree(value: Any, prefix: str = "tensor") -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    if value is None:
        return output
    if hasattr(value, "detach"):
        output[prefix] = value.detach().float().cpu().numpy()
    elif isinstance(value, np.ndarray):
        output[prefix] = value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.update(flatten_tensor_tree(item, f"{prefix}_{index}"))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            output.update(flatten_tensor_tree(item, f"{prefix}_{key}"))
    return output


def compare_named_arrays(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    ref_keys = list(reference.keys())
    cand_keys = list(candidate.keys())
    fields: Dict[str, Any] = {}
    for key in sorted(set(ref_keys) | set(cand_keys)):
        if key not in reference or key not in candidate:
            fields[key] = {
                "status": "FAIL",
                "reason": "missing_key",
                "in_reference": key in reference,
                "in_candidate": key in candidate,
            }
        else:
            fields[key] = compare_arrays(reference[key], candidate[key], atol, rtol)
    passed = ref_keys == cand_keys and all(item["status"] == "PASS" for item in fields.values())
    return {
        "reference_keys": ref_keys,
        "candidate_keys": cand_keys,
        "fields": fields,
        "status": "PASS" if passed else "FAIL",
    }


def calibration_from_cam_and_meta(cam_info: Mapping[str, Any], img_meta: Mapping[str, Any]) -> Dict[str, Any]:
    lidar2img = img_meta.get("lidar2img", {})
    aug_list = lidar2img.get("lidar2img_aug", [])
    aug = aug_list[0] if aug_list else {}
    projections = lidar2img.get("extrinsic", [])
    projection = projections[0] if projections else np.zeros((0,), dtype=np.float32)
    return {
        "cam_intrinsic": np.asarray(cam_info.get("cam_intrinsic", []), dtype=np.float32),
        "intrinsic_width": int(cam_info.get("intrinsic_width", -1)),
        "intrinsic_height": int(cam_info.get("intrinsic_height", -1)),
        "image_width": int(cam_info.get("image_width", -1)),
        "image_height": int(cam_info.get("image_height", -1)),
        "distortion": np.asarray(cam_info.get("distortion", []), dtype=np.float32).reshape(-1),
        "sensor2lidar_rotation": np.asarray(cam_info.get("sensor2lidar_rotation", []), dtype=np.float32),
        "sensor2lidar_translation": np.asarray(cam_info.get("sensor2lidar_translation", []), dtype=np.float32),
        "post_rot": np.asarray(aug.get("post_rot", []), dtype=np.float32),
        "post_tran": np.asarray(aug.get("post_tran", []), dtype=np.float32),
        "projection": np.asarray(projection, dtype=np.float32),
    }


def compare_calibration(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], atol: float, rtol: float
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for key in CALIBRATION_FIELDS:
        fields[key] = compare_arrays(reference.get(key), candidate.get(key), atol, rtol)
    first = next((key for key in CALIBRATION_FIELDS if fields[key]["status"] == "FAIL"), None)
    return {
        "fields": fields,
        "first_mismatch": first,
        "status": "PASS" if first is None else "FAIL",
    }


def normalize_decoded_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    if "pts_bbox" in result:
        result = result["pts_bbox"]
    boxes = n7_boxes_to_numpy(
        result.get("boxes_3d", np.zeros((0, 7), dtype=np.float32)),
        target_origin="center",
        source_origin=result.get("box_origin"),
        dtype=np.float32,
        box_dim=None,
    )
    scores = production_infer.tensor_like_to_numpy(result.get("scores_3d"), dtype=np.float32).reshape(-1)
    labels = production_infer.tensor_like_to_numpy(result.get("labels_3d"), dtype=np.int64).reshape(-1)
    count = min(len(boxes), len(scores), len(labels))
    return {
        "boxes": boxes[:count],
        "scores": scores[:count],
        "labels": labels[:count],
        "box_origin": "center",
    }


def compare_decoded(reference: Mapping[str, Any], candidate: Mapping[str, Any], atol: float, rtol: float) -> Dict[str, Any]:
    ref_boxes = np.asarray(reference["boxes"])
    cand_boxes = np.asarray(candidate["boxes"])
    fields: Dict[str, Any] = {
        "count": {
            "reference": int(len(ref_boxes)),
            "candidate": int(len(cand_boxes)),
            "status": "PASS" if len(ref_boxes) == len(cand_boxes) else "FAIL",
        },
        "box_origin": {
            "reference": reference.get("box_origin"),
            "candidate": candidate.get("box_origin"),
            "status": "PASS" if reference.get("box_origin") == candidate.get("box_origin") == "center" else "FAIL",
        },
        "labels": compare_arrays(reference["labels"], candidate["labels"], 0.0, 0.0, exact_required=True),
        "scores": compare_arrays(reference["scores"], candidate["scores"], atol, rtol),
        "boxes_full": compare_arrays(ref_boxes, cand_boxes, atol, rtol),
    }
    names = ("xyz", "lwh", "yaw")
    slices = (slice(0, 3), slice(3, 6), slice(6, 7))
    if ref_boxes.ndim == 2 and cand_boxes.ndim == 2:
        for name, box_slice in zip(names, slices):
            fields[name] = compare_arrays(ref_boxes[:, box_slice], cand_boxes[:, box_slice], atol, rtol)
    first = next((name for name, item in fields.items() if item["status"] == "FAIL"), None)
    return {
        "preserves_raw_order": True,
        "diagnostic_matching_applied": False,
        "fields": fields,
        "first_mismatch": first,
        "status": "PASS" if first is None else "FAIL",
    }


def compare_captures(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], atol: float, rtol: float
) -> Dict[str, Any]:
    stages: Dict[str, Any] = {
        "raw_image": compare_arrays(reference["raw_image"], candidate["raw_image"], 0.0, 0.0, exact_required=True),
        "calibration": compare_calibration(reference["calibration"], candidate["calibration"], atol, rtol),
        "input": compare_arrays(reference["input"], candidate["input"], atol, rtol),
        "feature_2d": compare_arrays(reference["feature_2d"], candidate["feature_2d"], atol, rtol),
        "bev_input": compare_arrays(reference["bev_input"], candidate["bev_input"], atol, rtol),
        "head_logits": compare_named_arrays(reference["head_logits"], candidate["head_logits"], atol, rtol),
        "decoded_boxes": compare_decoded(reference["decoded"], candidate["decoded"], atol, rtol),
    }
    first = next((name for name in CRITICAL_STAGE_ORDER if stages[name]["status"] == "FAIL"), None)
    input_shape = list(np.asarray(reference["input"]).shape)
    bev_shape = list(np.asarray(reference["bev_input"]).shape)
    contracts = {
        "single_frame_input": {
            "observed": input_shape,
            "expected": [1, 1, 3, 256, 704],
            "status": "PASS" if input_shape == [1, 1, 3, 256, 704] else "FAIL",
        },
        "single_frame_no_repeat": {
            "observed_views": input_shape[1] if len(input_shape) == 5 else None,
            "expected_views": 1,
            "status": "PASS" if len(input_shape) == 5 and input_shape[1] == 1 else "FAIL",
        },
        "bev_neck_input": {
            "observed": bev_shape,
            "expected": [1, 256, 160, 140],
            "status": "PASS" if bev_shape == [1, 256, 160, 140] else "FAIL",
        },
        "box_origin": {
            "observed": reference["decoded"].get("box_origin"),
            "expected": "center",
            "status": "PASS" if reference["decoded"].get("box_origin") == "center" else "FAIL",
        },
    }
    contract_first = next((name for name, item in contracts.items() if item["status"] == "FAIL"), None)
    if first is None and contract_first is not None:
        first = f"contract:{contract_first}"
    return {
        "stages": stages,
        "contracts": contracts,
        "first_mismatch": first,
        "status": "PASS" if first is None else "FAIL",
        "sigmoid": {
            "raw_logits_capture": "bbox_head.forward output, before sigmoid/top-k/NMS",
            "execution": "Anchor3DHead.get_bboxes_single applies cls_score.sigmoid() once",
            "expected_count": 1,
            "source": "mmdet3d/models/dense_heads/anchor3d_head.py",
        },
    }


def capture_metadata(capture: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "input": array_summary(capture["input"]),
        "feature_2d": array_summary(capture["feature_2d"]),
        "bev_input": array_summary(capture["bev_input"]),
        "head_logits": {key: array_summary(value) for key, value in capture["head_logits"].items()},
        "raw_image": array_summary(capture["raw_image"]),
        "calibration": json_ready(capture["calibration"]),
        "box_origin": capture["decoded"]["box_origin"],
        "decoded_count": int(len(capture["decoded"]["boxes"])),
        "img_meta": json_ready(capture.get("img_meta_summary", {})),
    }


def decoded_json(decoded: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for box, score, label in zip(decoded["boxes"], decoded["scores"], decoded["labels"]):
        rows.append({
            "label": int(label),
            "score": float(score),
            "xyz": [float(x) for x in box[:3]],
            "lwh": [float(x) for x in box[3:6]],
            "yaw": float(box[6]),
            "box": [float(x) for x in box],
            "box_origin": "center",
        })
    return {"box_origin": "center", "count": len(rows), "boxes": rows}


def save_capture(output_dir: Path, capture: Mapping[str, Any], dump_tensors: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", capture_metadata(capture))
    write_json(output_dir / "boxes.json", decoded_json(capture["decoded"]))
    if dump_tensors:
        np.save(output_dir / "input.npy", capture["input"])
        np.save(output_dir / "feature_2d.npy", capture["feature_2d"])
        np.save(output_dir / "bev_input.npy", capture["bev_input"])
        np.savez(output_dir / "head_logits.npz", **capture["head_logits"])


def comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# N7 单帧 dataset / production 数值对齐",
        "",
        f"- Overall: **{report['status']}**",
        f"- First mismatch: `{report.get('first_mismatch')}`",
        f"- atol / rtol: `{report['atol']}` / `{report['rtol']}`",
        "- Raw output order preserved: `true`; diagnostic matching: `false`",
        "",
        "## PKL-calibration stages",
        "",
        "| Stage | Status | Exact | Allclose | Max abs diff | Mean abs diff | Max rel diff |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    stages = report["pkl_calibration"]["stages"]
    for name in CRITICAL_STAGE_ORDER:
        item = stages[name]
        if name in ("calibration", "head_logits", "decoded_boxes"):
            exact = allclose = max_abs = mean_abs = max_rel = "-"
        else:
            exact = str(item.get("exact_equal", "-")).lower()
            allclose = str(item.get("allclose", "-")).lower()
            max_abs = item.get("max_abs_diff", "-")
            mean_abs = item.get("mean_abs_diff", "-")
            max_rel = item.get("max_rel_diff", "-")
        lines.append(f"| {name} | {item['status']} | {exact} | {allclose} | {max_abs} | {mean_abs} | {max_rel} |")
    lines.extend(["", "## Shape / semantic contracts", ""])
    for name, item in report["pkl_calibration"]["contracts"].items():
        lines.append(f"- `{name}`: **{item['status']}**, observed=`{item.get('observed', item.get('observed_views'))}`")
    info_report = report.get("info_json_calibration")
    if info_report is not None:
        lines.extend([
            "",
            "## info.json mode",
            "",
            f"- Asset relation: **{info_report['asset_relation']}**",
            f"- Calibration version identity: **{info_report['version_identity']}**",
            f"- First calibration mismatch: `{info_report.get('first_calibration_mismatch')}`",
            f"- Downstream comparison status: `{info_report.get('downstream_status')}`",
        ])
    lines.extend([
        "",
        "## Sigmoid contract",
        "",
        "Raw cls logits are captured before sigmoid/top-k/NMS. Canonical `Anchor3DHead.get_bboxes_single()` applies `sigmoid()` once.",
        "",
    ])
    return "\n".join(lines)


def base_manifest(args: argparse.Namespace, started_at: str) -> Dict[str, Any]:
    torch_version = module_version("torch")
    cuda_version = None
    if torch_version is not None:
        try:
            import torch
            cuda_version = torch.version.cuda
        except Exception:
            pass
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task": "Fast-BEV N7 原生单帧——标准 dataset 与生产推理同图逐级数值对齐",
        "started_at_utc": started_at,
        "git_commit": run_text_command(("git", "rev-parse", "HEAD")),
        "git_status": run_text_command(("git", "status", "--short", "--branch")),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch_version,
            "cuda": cuda_version,
            "mmcv": module_version("mmcv"),
            "mmdet": module_version("mmdet"),
            "mmdet3d": module_version("mmdet3d"),
        },
        "device": args.device,
        "inference_dtype": "float32",
        "n_images": 1,
        "n_times": 1,
        "box_origin": "center",
        "paths": {},
    }


def finalise_outputs(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    dataset_capture: Mapping[str, Any],
    production_capture: Mapping[str, Any],
    comparison: Dict[str, Any],
    info_capture: Optional[Mapping[str, Any]] = None,
) -> None:
    output_dir = Path(args.output_dir)
    save_capture(output_dir / "dataset", dataset_capture, args.dump_tensors)
    save_capture(output_dir / "production", production_capture, args.dump_tensors)
    if info_capture is not None:
        save_capture(output_dir / "production_info_json", info_capture, args.dump_tensors)
    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["duration_seconds"] = float(time.monotonic() - manifest.pop("_monotonic_start"))
    write_json(output_dir / "sample_manifest.json", manifest)
    write_json(output_dir / "comparison.json", comparison)
    (output_dir / "comparison.md").write_text(comparison_markdown(comparison), encoding="utf-8")


def require_real_args(args: argparse.Namespace) -> None:
    missing = [name for name in ("config", "checkpoint", "ann_file", "data_root") if getattr(args, name.replace("-", "_"), None) is None]
    if missing:
        raise ValueError("真实对齐缺少参数: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    if args.sample_index is None and args.token is None:
        raise ValueError("真实对齐必须二选一传入 --sample-index 或 --token")
    for name in ("config", "checkpoint", "ann_file"):
        path = Path(getattr(args, name.replace("-", "_")))
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} 不存在: {path}")
    if not Path(args.data_root).is_dir():
        raise FileNotFoundError(f"--data-root 不存在: {args.data_root}")


def unwrap_data_container(value: Any) -> Any:
    return value._data if hasattr(value, "_data") else value


def prepare_test_dataset_cfg(dataset_cfg: Any, replace_image_to_tensor: Any) -> int:
    """复刻 ``tools/test.py`` 在 ``build_dataset`` 前的 test config 清理。

    ``samples_per_gpu`` 属于 dataloader，不是 dataset 构造参数。如果直接把含有
    该字段的 ``cfg.data.test`` 交给 registry，就会被透传到
    ``CustomMultiViewDataset.__init__`` 并触发 unexpected keyword argument。
    """
    dataset_cfg["test_mode"] = True
    samples_per_gpu = int(dataset_cfg.pop("samples_per_gpu", 1))
    if samples_per_gpu > 1:
        dataset_cfg["pipeline"] = replace_image_to_tensor(dataset_cfg["pipeline"])
    return samples_per_gpu


def prepare_model_input(data: Mapping[str, Any], device: Any) -> Tuple[Any, Dict[str, Any]]:
    img = unwrap_data_container(data["img"])
    img_meta = unwrap_data_container(data["img_metas"])
    if isinstance(img, (list, tuple)) and len(img) == 1:
        img = img[0]
    if isinstance(img_meta, (list, tuple)) and len(img_meta) == 1 and isinstance(img_meta[0], dict):
        img_meta = img_meta[0]
    if img.ndim == 4:
        img = img.unsqueeze(0)
    return img.to(device=device, dtype=__import__("torch").float32), img_meta


class ForwardCapture:
    """一次 forward 的临时 hooks；退出后立即移除。"""

    def __init__(self, model: Any):
        self.model = model
        self.handles: List[Any] = []
        self.values: Dict[str, Any] = {}

    @staticmethod
    def _cpu(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach().float().cpu()
        if isinstance(value, (list, tuple)):
            return type(value)(ForwardCapture._cpu(item) for item in value)
        if isinstance(value, dict):
            return {key: ForwardCapture._cpu(item) for key, item in value.items()}
        return value

    def __enter__(self) -> "ForwardCapture":
        fuse = getattr(self.model, "neck_fuse_0", None) or getattr(self.model, "neck_fuse", None)
        if fuse is None:
            raise RuntimeError("找不到 neck_fuse_0/neck_fuse，无法捕获 2D feature")

        def fuse_hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.values["feature_2d"] = self._cpu(output)

        def bev_pre_hook(_module: Any, inputs: Any) -> None:
            self.values["bev_input"] = self._cpu(inputs[0])

        def head_hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.values["head_logits"] = self._cpu(output)

        self.handles.append(fuse.register_forward_hook(fuse_hook))
        self.handles.append(self.model.neck_3d.register_forward_pre_hook(bev_pre_hook))
        self.handles.append(self.model.bbox_head.register_forward_hook(head_hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run_model_capture(model: Any, data: Mapping[str, Any], device: Any, raw_image: np.ndarray, cam_info: Mapping[str, Any]) -> Dict[str, Any]:
    import torch

    img, img_meta = prepare_model_input(data, device)
    with ForwardCapture(model) as hooks, torch.inference_mode():
        result = model(img=img, img_metas=[copy.deepcopy(img_meta)], return_loss=False)
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError(f"模型输出应为单样本 list，实际 {type(result)}")
    required = ("feature_2d", "bev_input", "head_logits")
    missing = [key for key in required if key not in hooks.values]
    if missing:
        raise RuntimeError("hooks 未捕获: " + ", ".join(missing))
    head_parts = hooks.values["head_logits"]
    head_names = ("head_cls", "head_bbox", "head_dir")
    logits: Dict[str, np.ndarray] = {}
    for index, value in enumerate(head_parts):
        prefix = head_names[index] if index < len(head_names) else f"head_{index}"
        logits.update(flatten_tensor_tree(value, prefix))
    feature_tree = flatten_tensor_tree(hooks.values["feature_2d"], "feature_2d")
    if len(feature_tree) != 1:
        raise RuntimeError(f"2D feature 应只有一个 tensor，实际 keys={list(feature_tree)}")
    bev_tree = flatten_tensor_tree(hooks.values["bev_input"], "bev_input")
    if len(bev_tree) != 1:
        raise RuntimeError(f"3D neck 输入应只有一个 tensor，实际 keys={list(bev_tree)}")
    return {
        "raw_image": raw_image,
        "input": img.detach().float().cpu().numpy(),
        "feature_2d": next(iter(feature_tree.values())),
        "bev_input": next(iter(bev_tree.values())),
        "head_logits": logits,
        "decoded": normalize_decoded_result(result[0]),
        "calibration": calibration_from_cam_and_meta(cam_info, img_meta),
        "img_meta_summary": {
            "sample_idx": img_meta.get("sample_idx"),
            "filename": img_meta.get("filename"),
            "img_shape": img_meta.get("img_shape"),
            "ori_shape": img_meta.get("ori_shape"),
            "pad_shape": img_meta.get("pad_shape"),
            "view_count": int(img.shape[1]),
        },
    }


def pkl_camera_template(cam_info: Mapping[str, Any]) -> Dict[str, Any]:
    """直接复用同一 val info 的精确字段，不做坐标或标定数值转换。"""
    template = copy.deepcopy(dict(cam_info))
    required = (
        "cam_intrinsic", "sensor2lidar_rotation", "sensor2lidar_translation",
        "distortion", "intrinsic_width", "intrinsic_height",
    )
    missing = [key for key in required if key not in template]
    if missing:
        raise KeyError("val pkl cam info 缺少: " + ", ".join(missing))
    template.setdefault("width", int(template["intrinsic_width"]))
    template.setdefault("height", int(template["intrinsic_height"]))
    template["intrinsic_size_source"] = "val_pkl"
    template["info_extrinsic_coordinate"] = "fastbev"
    return template


def production_namespace(args: argparse.Namespace, *, allow_aspect_mismatch: bool) -> argparse.Namespace:
    coordinate = "raw_n7" if args.info_extrinsic_coordinate in ("raw", "raw_n7") else "fastbev"
    return argparse.Namespace(
        intrinsic_size=args.intrinsic_size,
        intrinsic_size_source="auto",
        info_extrinsic_coordinate=coordinate,
        aspect_tolerance=0.02,
        allow_aspect_mismatch=allow_aspect_mismatch,
    )


def read_raw_image(path: Path) -> np.ndarray:
    import mmcv
    image = mmcv.imread(str(path), flag="color")
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return image


def build_production_data(
    cfg: Any,
    image_path: Path,
    template: Mapping[str, Any],
    info: Mapping[str, Any],
    camera_id: str,
    box_type_3d: Any,
    box_mode_3d: Any,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from mmdet3d.datasets.pipelines import Compose

    prod_args = production_namespace(args, allow_aspect_mismatch=True)
    sample_results, portable_info = production_infer.build_sample_results(
        [image_path], template, camera_id, str(info["token"]), int(info["timestamp"]),
        box_type_3d, box_mode_3d, prod_args,
    )
    pipeline_cfg = copy.deepcopy(cfg.data.test.pipeline)
    shadow_cfg = copy.deepcopy(cfg)
    shadow_cfg.data.test.pipeline = pipeline_cfg
    production_infer.patch_test_pipeline(shadow_cfg, n_images=1, n_times=1, force_resize=True)
    pipeline = Compose(copy.deepcopy(shadow_cfg.data.test.pipeline))
    return pipeline(sample_results), portable_info


def select_dataset_index(dataset: Any, sample_index: Optional[int], token: Optional[str]) -> int:
    if token is not None:
        matches = [index for index, info in enumerate(dataset.data_infos) if str(info.get("token")) == str(token)]
        if len(matches) != 1:
            raise ValueError(f"token={token!r} 匹配数量应为 1，实际 {len(matches)}")
        return matches[0]
    assert sample_index is not None
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample-index={sample_index} 超出 [0, {len(dataset)})")
    return int(sample_index)


def configure_real_runtime(args: argparse.Namespace) -> Tuple[Any, Any, Any, Any, Any]:
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet.datasets import replace_ImageToTensor
    from mmdet3d.core.bbox import get_box_type
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.data.test.ann_file = str(Path(args.ann_file).resolve())
    cfg.data.test.data_root = str(Path(args.data_root).resolve())
    # 与标准 tools/test.py 保持一致：在 build_dataset 前剥离 dataloader 字段。
    prepare_test_dataset_cfg(cfg.data.test, replace_ImageToTensor)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.test_cfg.test_mode = "test_pth"
    if int(cfg.model.n_images) != 1:
        raise ValueError(f"要求 model.n_images=1，实际 {cfg.model.n_images}")
    if int(production_infer.infer_pipeline_value(cfg, "n_times", 1)) != 1:
        raise ValueError("要求原生单帧 n_times=1")
    camera_types = list(cfg.data.test.get("camera_types", []))
    if camera_types != ["cam0"]:
        raise ValueError(f"要求 data.test.camera_types=['cam0']，实际 {camera_types}")
    if not bool(cfg.model.get("use_distortion", False)):
        raise ValueError("本任务要求 model.use_distortion=True")
    if bool(cfg.model.test_cfg.get("use_tta", False)):
        raise ValueError("黄金参考要求关闭 TTA")
    dataset_sequential = bool(cfg.data.test.get("sequential", False))
    pipeline_step = production_infer.find_pipeline_step(
        cfg.data.test.pipeline, "MultiViewPipeline")
    pipeline_sequential = bool(pipeline_step.get("sequential", False)) if pipeline_step else None
    pipeline_n_images = int(pipeline_step.get("n_images", -1)) if pipeline_step else -1
    pipeline_n_times = int(pipeline_step.get("n_times", -1)) if pipeline_step else -1
    if (dataset_sequential or pipeline_sequential is not False or
            pipeline_n_images != 1 or pipeline_n_times != 1):
        raise ValueError(
            "要求原生单帧 dataset/MultiViewPipeline 契约，实际 "
            f"dataset_sequential={dataset_sequential}, "
            f"pipeline_sequential={pipeline_sequential}, "
            f"n_images={pipeline_n_images}, n_times={pipeline_n_times}")
    image_aug_step = production_infer.find_pipeline_step(
        cfg.data.test.pipeline, "RandomAugImageMultiViewImage")
    if (image_aug_step is None or bool(image_aug_step.get("is_train", True)) or
            not bool(image_aug_step.get("force_resize", False))):
        raise ValueError(
            "标准 test pipeline 必须使用 is_train=False、force_resize=True 的确定性图像变换")
    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    checkpoint_meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    model.CLASSES = checkpoint_meta.get("CLASSES", cfg.get("class_names", ["car", "truck"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求 {device}，但 torch.cuda.is_available()=False")
    # 黄金参考显式保持 FP32；不调用 wrap_fp16_model。
    model.to(device=device, dtype=torch.float32)
    model.eval()
    box_type_3d, box_mode_3d = get_box_type("LiDAR")
    return cfg, dataset, model, device, (box_type_3d, box_mode_3d)


def info_identity(info: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("dataset", "sequence", "clip", "clip_id", "vehicle", "vehicle_id", "car_id", "date")
    # 显式保留空字段，避免报告阅读者误以为工具忘记采集车辆信息。
    return {key: json_ready(info.get(key)) for key in keys}


def run_real(args: argparse.Namespace) -> int:
    require_real_args(args)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = base_manifest(args, started_at)
    manifest["_monotonic_start"] = time.monotonic()
    cfg, dataset, model, device, box_types = configure_real_runtime(args)
    box_type_3d, box_mode_3d = box_types
    index = select_dataset_index(dataset, args.sample_index, args.token)
    info = dataset.data_infos[index]
    if args.camera_id not in info.get("cams", {}):
        raise KeyError(f"token={info.get('token')} 不含 camera {args.camera_id}")
    cam_info = info["cams"][args.camera_id]
    image_path = Path(dataset._resolve_path(cam_info["data_path"])).resolve()
    raw_image = read_raw_image(image_path)

    dataset_data = dataset[index]
    dataset_capture = run_model_capture(model, dataset_data, device, raw_image.copy(), cam_info)

    template = pkl_camera_template(cam_info)
    production_data, production_info = build_production_data(
        cfg, image_path, template, info, args.camera_id, box_type_3d, box_mode_3d, args,
    )
    production_cam = production_info["cams"][args.camera_id]
    production_capture = run_model_capture(model, production_data, device, raw_image.copy(), production_cam)
    pkl_report = compare_captures(dataset_capture, production_capture, args.atol, args.rtol)

    info_capture = None
    info_report = None
    if args.info_json:
        info_json_path = Path(args.info_json).resolve()
        calibration_json = production_infer.read_json(info_json_path)
        sensor = production_infer.find_sensor(calibration_json, args.sensor_name)
        json_args = production_namespace(args, allow_aspect_mismatch=True)
        json_template = production_infer.build_camera_template(
            sensor, (int(cam_info["image_width"]), int(cam_info["image_height"])), json_args,
        )
        info_data, info_portable = build_production_data(
            cfg, image_path, json_template, info, args.camera_id, box_type_3d, box_mode_3d, args,
        )
        info_cam = info_portable["cams"][args.camera_id]
        info_capture = run_model_capture(model, info_data, device, raw_image.copy(), info_cam)
        calibration_report = compare_calibration(
            dataset_capture["calibration"], info_capture["calibration"], args.atol, args.rtol,
        )
        if calibration_report["status"] == "PASS":
            downstream = compare_captures(dataset_capture, info_capture, args.atol, args.rtol)
            info_report = {
                "asset_relation": "NUMERIC_CALIBRATION_MATCH",
                "version_identity": "UNCONFIRMED_NO_SHARED_VERSION_ID",
                "first_calibration_mismatch": None,
                "calibration": calibration_report,
                "downstream_status": downstream["status"],
                "downstream": downstream,
            }
        else:
            info_report = {
                "asset_relation": "ASSET_MISMATCH_OR_UNCONFIRMED",
                "version_identity": "UNCONFIRMED_NO_SHARED_VERSION_ID",
                "first_calibration_mismatch": calibration_report["first_mismatch"],
                "calibration": calibration_report,
                "downstream_status": "NOT_GRADED",
                "downstream": compare_captures(dataset_capture, info_capture, args.atol, args.rtol),
            }

    overall_status = pkl_report["status"]
    first_mismatch = pkl_report["first_mismatch"]
    if (info_report and info_report["asset_relation"] == "NUMERIC_CALIBRATION_MATCH" and
            info_report["downstream_status"] != "PASS"):
        overall_status = "FAIL"
        first_mismatch = first_mismatch or "info_json:" + str(info_report["downstream"].get("first_mismatch"))
    comparison = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": overall_status,
        "first_mismatch": first_mismatch,
        "atol": args.atol,
        "rtol": args.rtol,
        "real_pth_alignment_performed": True,
        "pkl_calibration": pkl_report,
        "info_json_calibration": info_report,
    }

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    ann_path = Path(args.ann_file).resolve()
    info_json_path = Path(args.info_json).resolve() if args.info_json else None
    manifest.update({
        "mode": "real-pth",
        "sample_index": index,
        "token": str(info["token"]),
        "selection": "exact token" if args.token else "explicit dataset index; no prediction-based selection",
        "image_path": str(image_path),
        "image_paths": {
            "dataset": str(image_path),
            "production_pkl": str(image_path),
            "production_info_json": str(image_path) if info_json_path else None,
        },
        "image_sha256": sha256_file(image_path),
        "raw_pixel_sha256": sha256_array(raw_image),
        "sample_identity": info_identity(info),
        "vehicle_info": {
            "from_val_pkl": {
                "vehicle": info.get("vehicle"),
                "vehicle_id": info.get("vehicle_id"),
                "car_id": info.get("car_id"),
            },
            "info_json": str(info_json_path) if info_json_path else None,
        },
        "pipeline_contract": {
            "dataset_sequential": bool(cfg.data.test.get("sequential", False)),
            "dataset_pipeline_sequential": bool(production_infer.find_pipeline_step(
                cfg.data.test.pipeline, "MultiViewPipeline").get("sequential", False)),
            "production_sequential": False,
            "note": "原生 S0 生产脚本保持 sequential=False，只选择当前 cam0 一张图。",
        },
        "paths": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "ann_file": {"path": str(ann_path), "sha256": sha256_file(ann_path)},
            "data_root": str(Path(args.data_root).resolve()),
            "info_json": {"path": str(info_json_path), "sha256": sha256_file(info_json_path)} if info_json_path else None,
        },
        "production_pkl_info": production_info,
    })
    finalise_outputs(args, manifest, dataset_capture, production_capture, comparison, info_capture)
    print(f"real PTH alignment: {overall_status}; token={info['token']}; first_mismatch={first_mismatch}")
    return 2 if args.strict and overall_status != "PASS" else 0


def main() -> int:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        return run_real(args)
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "真实对齐需要 legacy mmcv/mmdet/mmdet3d runtime；当前导入失败: "
            f"{type(exc).__name__}: {exc}", file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
