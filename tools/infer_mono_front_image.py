#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 mono-front Fast-BEV 权重对生产前视单目图片做推理。

输入是 N7 格式 ``info.json`` 和前视图片路径。脚本只读取 ``front_wide``
相机，构造与 ``CustomMultiViewDataset`` 测试 pipeline 等价的临时 sample，
然后支持两种推理方式：

- PTH：加载 Fast-BEV checkpoint 后走 PyTorch backbone/3D head；
- ONNX：使用当前项目的 split ONNX 约定，即 2D backbone ONNX + 3D head
  ONNX，后处理仍走项目内 ``bbox_head.get_bboxes``。

当前 mono-front baseline 是 temporal mono：``n_images=1, n_times=4``。
生产侧只有一张前视图时，默认把同一张图重复为 4 个时序输入。

完整使用示例：

1. 单张图片 PTH 推理。``--intrinsic-size`` 表示 ``cam_intrinsic/K`` 对应的
   图像坐标尺寸，不是输入图片文件尺寸。若 K 是原始 4K 标定，填
   ``3840 2160``；若 K 已适配到 1600x900，填 ``1600 900``。

   .. code-block:: bash

      python tools/infer_mono_front_image.py \
        --config configs/fastbev/custom/custom_fastbev_mono_front_r18.py \
        --checkpoint work_dirs/mono_front/epoch_5.pth \
        --info-json data/info_json/2025_04_18_2k_byd_info_9797_UKEF.json \
        --intrinsic-size 1600 900 \
        --image /path/to/front_3840x2160.jpg \
        --output-dir outputs/mono_front_single

2. 指定目录批量推理。默认可视化文件使用原图文件名。``--display-aspect
   native`` 会保持输入图显示比例，适合检查车辆是否被拉伸；``16:9`` 只适合
   输入图本身就是 16:9 或希望统一审片比例的场景。

   .. code-block:: bash

      python tools/infer_mono_front_image.py \
        --checkpoint work_dirs/mono_front/epoch_5.pth \
        --info-json data/info_json/2025_04_18_2k_byd_info_9797_UKEF.json \
        --intrinsic-size 1600 900 \
        --image-dir /path/to/front_images \
        --image-glob '*.jpg' \
        --display-aspect native \
        --output-dir outputs/mono_front_dir

3. 目录抽帧和快速可视化。``--stride`` / ``--max-frames`` 会减少实际推理帧；
   ``--visualization-stride`` / ``--max-visualizations`` 只减少保存可视化的帧；
   ``--raw-distorted`` 跳过去畸变，``--no-bev`` 不拼 BEV，二者都能减少渲染开销。

   .. code-block:: bash

      python tools/infer_mono_front_image.py \
        --checkpoint work_dirs/mono_front/epoch_5.pth \
        --info-json data/info_json/2025_04_18_2k_byd_info_9797_UKEF.json \
        --intrinsic-size 1600 900 \
        --image-dir /path/to/front_images \
        --image-glob '*.jpg' \
        --stride 5 \
        --max-frames 200 \
        --visualization-stride 5 \
        --max-visualizations 40 \
        --camera-width 960 \
        --raw-distorted \
        --no-bev \
        --display-aspect native \
        --output-dir outputs/mono_front_fast_sample

4. 多车多片段批量推理。默认读取
   ``data/info_json/2k/2025_04_18_2k_byd_info_{vehicle_id}.json``，并在
   ``data/qingping/2k/{vehicle_id}/`` 下递归查找 ``jpg/jpeg/png``。每个车辆
   单独输出到 ``--output-dir/{vehicle_id}/``，可视化保留相对片段路径，避免
   不同片段同名图片互相覆盖。

   .. code-block:: bash

      python tools/infer_mono_front_image.py \
        --checkpoint work_dirs/mono_front/epoch_5.pth \
        --vehicle-id 9797_UKEF 1234_ABC \
        --info-json-dir data/info_json/2k \
        --info-json-template '2025_04_18_2k_byd_info_{vehicle_id}.json' \
        --image-root data/qingping/2k \
        --intrinsic-size 1600 900 \
        --stride 5 \
        --max-frames 200 \
        --visualization-stride 5 \
        --camera-width 960 \
        --display-aspect native \
        --output-dir outputs/mono_front_batch

5. Split ONNX 推理。``export_metadata.json`` 中的 2D output layout 必须是
   NCHW；若是 NHWC，本脚本会报错，因为当前后处理按项目 ``test_onnx`` 的
   NCHW 约定连接 2D backbone 和 3D head。

   .. code-block:: bash

      python tools/infer_mono_front_image.py \
        --onnx-dir output/onnx_mono_front \
        --info-json data/info_json/2025_04_18_2k_byd_info.json \
        --intrinsic-size 1600 900 \
        --image /path/to/front_wide.jpg \
        --output-dir outputs/mono_front_onnx

注意：
- 没有 pose/车辆位姿时，不建议传真实历史帧做时序；默认重复当前图更稳。
- ``--raw-distorted`` 会直接在原始畸变图上画框；如果边缘框明显飘，去掉该
  参数，改用默认去畸变可视化。
- 输出包括 ``infos.pkl``、portable ``pred_results.pkl``、
  ``prediction_summary.json`` 和可选 ``frames/`` 可视化。
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import logging
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.n7_box_origin import boxes_to_numpy as n7_boxes_to_numpy  # noqa: E402


LOGGER = logging.getLogger("infer_mono_front_image")

# 原始 N7/自采标定坐标系：x 向左，y 向后，z 向上。
# Fast-BEV/MMDet3D LiDAR 坐标系：x 向前，y 向左，z 向上。
RAW_TO_FASTBEV = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

MONO_FRONT_ROI = (0.0, -35.0, 80.0, 35.0)
DEFAULT_CONFIG = "configs/fastbev/custom/custom_fastbev_mono_front_r18.py"
DEFAULT_CLASSES = ["car", "truck"]


class RawDefaultsHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter):
    """保留 epilog 示例换行，同时展示默认值。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对生产前视单目图片运行 Fast-BEV mono-front 推理并可视化",
        formatter_class=RawDefaultsHelpFormatter,
        epilog="""
示例：
  # PTH checkpoint 推理，使用 info.json 中 front_wide，输出预测 pkl/json 和可视化。
  python tools/infer_mono_front_image.py \\
    --config configs/fastbev/custom/custom_fastbev_mono_front_r18.py \\
    --checkpoint work_dirs/mono_front/epoch_5.pth \\
    --info-json data/info_json/2025_04_18_2k_byd_info.json \\
    --image /path/to/front_wide.jpg \\
    --output-dir outputs/mono_front_prod_pth

  # 9797_UKEF 这类 K 已经适配到 1600x900、图片是 2560x1440 的情况，
  # 建议显式声明 K 对应尺寸，避免 info.json 的 width/height 旧字段误导。
  python tools/infer_mono_front_image.py \\
    --checkpoint work_dirs/mono_front/epoch_5.pth \\
    --info-json data/info_json/2025_04_18_2k_byd_info_9797_UKEF.json \\
    --intrinsic-size 1600 900 \\
    --image /path/to/front_2560x1440.jpg \\
    --output-dir outputs/mono_front_9797

  # Split ONNX 推理。也可以把 --onnx-dir 指到 tools/export_onnx.py 的输出目录。
  python tools/infer_mono_front_image.py \\
    --onnx-dir output/onnx_mono_front \\
    --info-json data/info_json/2025_04_18_2k_byd_info.json \\
    --image /path/to/front_wide.jpg \\
    --output-dir outputs/mono_front_onnx
""")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="mono-front Fast-BEV config")
    parser.add_argument("--checkpoint", default=None, help="PTH checkpoint；PTH 模式必填")
    parser.add_argument("--weights", default=None,
                        help="自动识别权重：.pth 作为 checkpoint；目录作为 ONNX export 目录")
    parser.add_argument("--onnx-dir", default=None,
                        help="tools/export_onnx.py 输出目录，优先读取 export_metadata.json")
    parser.add_argument("--backbone-onnx", default=None, help="2D backbone ONNX 路径")
    parser.add_argument("--head-onnx", default=None, help="3D head ONNX 路径")
    parser.add_argument("--onnx-custom-op-path", default=None,
                        help="ONNXRuntime custom op so；普通 ONNX 不需要")
    parser.add_argument("--mode", choices=["auto", "pth", "onnx"], default="auto",
                        help="推理模式；auto 按权重参数判断")
    parser.add_argument("--info-json", default=None,
                        help="N7 格式标定 info.json；未使用 --vehicle-id 时必填")
    parser.add_argument("--vehicle-id", nargs="+", default=None,
                        help="按车辆编号批量推理；可传一个或多个编号")
    parser.add_argument("--info-json-dir", default="data/info_json/2k",
                        help="车辆批量模式下 info.json 所在目录")
    parser.add_argument("--info-json-template", default="2025_04_18_2k_byd_info_{vehicle_id}.json",
                        help="车辆批量模式下 info.json 文件名模板，支持 {vehicle_id} 或 {vehicle}")
    parser.add_argument("--image-root", default="data/qingping/2k",
                        help="车辆批量模式下图片根目录；脚本读取 image-root/vehicle-id 下的图片")
    parser.add_argument("--vehicle-image-glob", nargs="+",
                        default=["**/*.jpg", "**/*.jpeg", "**/*.png"],
                        help="车辆批量模式下相对 image-root/vehicle-id 的递归图片 glob，可传多个")
    parser.add_argument("--sensor-name", default="front_wide", help="info.json 中前视相机 sensor 名")
    parser.add_argument("--camera-id", default="cam0", help="输出临时 Fast-BEV info 中的 camera id")
    parser.add_argument("--image", action="append", default=[],
                        help="前视图片路径；可重复传多张，逐张推理")
    parser.add_argument("--image-dir", default=None, help="前视图片目录；配合 --image-glob 使用")
    parser.add_argument("--image-glob", default="*.jpg", help="--image-dir 下的图片 glob")
    parser.add_argument("--image-list", default=None, help="逐行保存图片路径的文本文件")
    parser.add_argument("--start-index", type=int, default=0, help="从候选图片列表的第几个索引开始推理")
    parser.add_argument("--stride", type=int, default=1, help="候选图片抽帧步长；1 表示不抽帧")
    parser.add_argument("--max-frames", type=int, default=-1, help="最多推理多少张；<0 表示不限制")
    parser.add_argument("--temporal-images", nargs="+", default=None,
                        help=("单样本完整时序图片，数量必须等于 n_times，顺序为 current prev1 prev2 ...；"
                              "真实历史帧会直接使用，不做 ego-motion compensation（没有 pose 输入），"
                              "不同于训练时 prev frames；不传则重复 --image，匹配训练中 clip-start 样本的 fallback 分布"))
    parser.add_argument("--temporal-policy", choices=["repeat", "error"], default="repeat",
                        help="只有单张图但模型需要多时序时的处理方式")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--device", default=None, help="推理设备，默认优先 cuda:0，否则 cpu")
    parser.add_argument("--cfg-options", nargs="+", default=None,
                        help="覆盖 config，格式 key=value；支持 model.xxx=... 形式")
    parser.add_argument("--fuse-conv-bn", action="store_true", help="PTH 推理前 fuse conv/bn")
    parser.add_argument("--score-thr", type=float, default=0.2, help="可视化和 summary 使用的分数阈值")
    parser.add_argument("--max-preds", type=int, default=100, help="每帧最多写入 summary/可视化的预测框数；<=0 不限制")
    parser.add_argument("--no-visualization", action="store_true", help="只保存预测结果，不输出可视化")
    parser.add_argument("--visualization-stride", type=int, default=1,
                        help="每隔多少个已推理样本保存一张可视化；1 表示每张都保存")
    parser.add_argument("--max-visualizations", type=int, default=-1,
                        help="最多保存多少张可视化；<0 表示不限制")
    parser.add_argument("--visualization-name", choices=["image", "token"], default="image",
                        help="可视化文件命名方式；image 保留原图文件名，token 使用样本 token")
    parser.add_argument("--display-aspect", choices=["16:9", "native"], default="16:9",
                        help="相机可视化面板显示比例；16:9 只影响显示层，不改变模型输入几何")
    parser.add_argument("--no-bev", action="store_true", help="可视化只保存相机面板，不拼接 BEV 面板")
    parser.add_argument("--camera-width", type=int, default=1280, help="可视化前视图宽度")
    parser.add_argument("--bev-size", type=int, default=700, help="可视化 BEV 面板基础尺寸")
    parser.add_argument("--bev-range", nargs=4, type=float, default=list(MONO_FRONT_ROI),
                        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
                        help="BEV 可视化范围")
    parser.add_argument("--raw-distorted", dest="undistort", action="store_false",
                        default=True, help="可视化不去畸变，直接在原始畸变图上画框")
    parser.add_argument("--undistort-alpha", type=float, default=0.0, help="OpenCV 去畸变 alpha")
    parser.add_argument("--min-depth", type=float, default=0.1, help="画框最小相机深度")
    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg",
                        help="visualization-name=token 时的可视化图片格式；image 模式保留原图文件名")
    parser.add_argument("--intrinsic-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        default=None, help="cam_intrinsic/K 对应的图像尺寸；9797_UKEF 建议填 1600 900")
    parser.add_argument("--intrinsic-size-source", choices=["auto", "info", "image"], default="auto",
                        help="未显式传 --intrinsic-size 时的尺寸来源")
    parser.add_argument("--aspect-tolerance", type=float, default=0.02,
                        help="实际图片宽高比和 K 对应宽高比允许差值；<=0 表示不检查")
    parser.add_argument("--allow-aspect-mismatch", action="store_true",
                        help="实际图片比例和 K 对应尺寸不一致时只告警不报错")
    parser.add_argument("--info-extrinsic-coordinate", choices=["raw_n7", "fastbev"], default="raw_n7",
                        help="info.json 的 to_lidar_main 外参坐标系；N7 converter 兼容格式默认 raw_n7")
    parser.add_argument("--force-resize", dest="force_resize", action="store_true", default=True,
                        help="强制把测试 pipeline 的 RandomAugImageMultiViewImage 设置为 force_resize=True")
    parser.add_argument("--use-config-resize", dest="force_resize", action="store_false",
                        help="不覆盖 config 中的 force_resize 设置")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()
    if args.vehicle_id and args.temporal_images:
        parser.error("--vehicle-id 批量模式暂不支持 --temporal-images；当前会按 repeat 使用单图时序")
    if not args.vehicle_id and not args.info_json:
        parser.error("未使用 --vehicle-id 时必须传 --info-json")
    if args.temporal_images and (args.image_dir or args.image_list or len(args.image) != 1):
        parser.error("--temporal-images 只支持和单个 --image 一起使用")
    return args


def natural_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.as_posix())
    return [int(x) if x.isdigit() else x for x in parts]


def parse_cfg_value(text: str) -> Any:
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_cfg_options(options: Optional[Sequence[str]]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in options or []:
        if "=" not in item:
            raise ValueError(f"--cfg-options 项缺少 '=': {item}")
        key, value = item.split("=", 1)
        parsed[key] = parse_cfg_value(value)
    return parsed


def read_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, exc.lineno - 3)
        end = min(len(lines), exc.lineno + 2)
        context = "\n".join(
            f"{idx + 1}: {lines[idx]}" for idx in range(start, end))
        raise ValueError(
            f"标定 JSON 解析失败: {path}\n"
            f"line={exc.lineno}, column={exc.colno}, msg={exc.msg}\n"
            f"附近内容:\n{context}") from exc


def find_sensor(calib: Dict, sensor_name: str) -> Dict:
    for sensor in calib.get("sensors", []):
        if sensor.get("name") == sensor_name:
            return sensor
    available = [str(sensor.get("name")) for sensor in calib.get("sensors", [])]
    raise KeyError(f"Cannot find sensor {sensor_name!r}; available={available}")


def normalize_rotation_matrix(mat: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(mat, dtype=np.float64))
    rot = u @ vh
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vh
    return rot.astype(np.float32)


def quat_xyzw_to_matrix(q_xyzw: Sequence[float]) -> np.ndarray:
    qx, qy, qz, qw = [float(x) for x in q_xyzw]
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0:
        raise ValueError("zero-norm quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array([
        [1 - 2 * (qy * qy + qz * qz),
         2 * (qx * qy - qz * qw),
         2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),
         1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),
         2 * (qy * qz + qx * qw),
         1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def matrix_to_quat_wxyz(rot: np.ndarray) -> List[float]:
    rot = normalize_rotation_matrix(rot).astype(np.float64)
    trace = float(np.trace(rot))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    return [float(x) for x in quat.tolist()]


def parse_sensor_to_lidar(raw_ext: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw_ext, dtype=np.float64)
    if arr.shape == (4, 4):
        rot = normalize_rotation_matrix(arr[:3, :3])
        tran = arr[:3, 3].astype(np.float32)
        return rot, tran
    flat = arr.reshape(-1)
    if flat.size != 7:
        raise ValueError(f"expected 7-vector or 4x4 extrinsic, got shape {arr.shape}")
    tran = flat[:3].astype(np.float32)
    rot = quat_xyzw_to_matrix(flat[3:7])
    return rot, tran


def read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        width, height = img.size
    return int(width), int(height)


def dedupe_and_validate_images(paths: Sequence[Path], empty_message: str) -> List[Path]:
    deduped: List[Path] = []
    seen = set()
    for path in paths:
        resolved_key = str(path.resolve()) if path.exists() else str(path)
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        deduped.append(path)
    if not deduped:
        raise ValueError(empty_message)
    missing = [str(path) for path in deduped if not path.exists()]
    if missing:
        raise FileNotFoundError("图片不存在:\n- " + "\n- ".join(missing[:20]))
    return deduped


def candidate_images(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = [Path(x) for x in args.image]
    if args.image_dir:
        image_dir = Path(args.image_dir)
        paths.extend(sorted(image_dir.glob(args.image_glob), key=natural_key))
    if args.image_list:
        list_path = Path(args.image_list)
        base = list_path.parent
        for line in list_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            path = Path(text)
            paths.append(path if path.is_absolute() else base / path)
    return dedupe_and_validate_images(
        paths, "请通过 --image、--image-dir、--image-list 或 --vehicle-id 指定前视图片")


def resolve_vehicle_info_json(vehicle_id: str, args: argparse.Namespace) -> Path:
    filename = args.info_json_template.format(vehicle_id=vehicle_id, vehicle=vehicle_id)
    return Path(args.info_json_dir) / filename


def vehicle_image_base_dir(vehicle_id: str, args: argparse.Namespace) -> Path:
    return Path(args.image_root) / vehicle_id


def candidate_vehicle_images(vehicle_id: str, args: argparse.Namespace) -> List[Path]:
    base_dir = vehicle_image_base_dir(vehicle_id, args)
    if not base_dir.exists():
        raise FileNotFoundError(f"车辆图片目录不存在: {base_dir}")
    paths: List[Path] = []
    for pattern in args.vehicle_image_glob:
        paths.extend(sorted(base_dir.glob(pattern), key=natural_key))
    return dedupe_and_validate_images(
        paths,
        f"车辆 {vehicle_id} 在 {base_dir} 下未找到图片，glob={args.vehicle_image_glob}")


def select_image_subset(paths: Sequence[Path], args: argparse.Namespace) -> List[Path]:
    stride = max(int(args.stride), 1)
    max_frames = int(args.max_frames)
    selected: List[Path] = []
    for path in list(paths)[max(int(args.start_index), 0)::stride]:
        if max_frames >= 0 and len(selected) >= max_frames:
            break
        selected.append(path)
    if not selected:
        raise ValueError(
            f"抽帧后没有图片可推理：total={len(paths)}, start_index={args.start_index}, "
            f"stride={args.stride}, max_frames={args.max_frames}")
    return selected


def round_principal_size(value: float) -> int:
    raw = max(1.0, 2.0 * float(value))
    nearest_10 = round(raw / 10.0) * 10
    if abs(nearest_10 - raw) <= 3.0:
        return int(nearest_10)
    return int(round(raw))


def sensor_size_is_plausible(k: np.ndarray, width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    return 0.30 <= cx / float(width) <= 0.70 and 0.30 <= cy / float(height) <= 0.70


def principal_point_matches_size(k: np.ndarray, width: int, height: int,
                                 tolerance: float = 0.15) -> bool:
    if width <= 0 or height <= 0:
        return False
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    return (
        abs(2.0 * cx - float(width)) / float(width) <= tolerance and
        abs(2.0 * cy - float(height)) / float(height) <= tolerance
    )


def infer_intrinsic_size(
    sensor: Dict,
    k: np.ndarray,
    image_size: Tuple[int, int],
    args: argparse.Namespace,
) -> Tuple[int, int, str]:
    if args.intrinsic_size:
        width, height = [int(x) for x in args.intrinsic_size]
        return width, height, "cli"

    image_width, image_height = image_size
    sensor_width = int(sensor.get("width", 0) or 0)
    sensor_height = int(sensor.get("height", 0) or 0)

    if args.intrinsic_size_source == "image":
        return image_width, image_height, "image"
    if args.intrinsic_size_source == "info":
        if sensor_width <= 0 or sensor_height <= 0:
            raise ValueError("info.json sensor width/height 无效，无法作为 intrinsic size")
        return sensor_width, sensor_height, "info"

    if (
        sensor_size_is_plausible(k, sensor_width, sensor_height) and
        principal_point_matches_size(k, sensor_width, sensor_height)
    ):
        return sensor_width, sensor_height, "auto_info"

    inferred_width = round_principal_size(float(k[0, 2]))
    inferred_height = round_principal_size(float(k[1, 2]))
    LOGGER.warning(
        "info.json 中 %s 的 width/height=%sx%s 与 K 主点 cx/cy=(%.3f, %.3f) "
        "不匹配，自动推断 K 对应尺寸为 %sx%s。若这是 9797_UKEF，请优先显式传 "
        "--intrinsic-size 1600 900。",
        sensor.get("name", "sensor"),
        sensor_width,
        sensor_height,
        float(k[0, 2]),
        float(k[1, 2]),
        inferred_width,
        inferred_height)
    return inferred_width, inferred_height, "auto_principal_point"


def check_aspect(
    image_size: Tuple[int, int],
    intrinsic_size: Tuple[int, int],
    args: argparse.Namespace,
    image_path: Path,
) -> None:
    if args.aspect_tolerance <= 0:
        return
    img_w, img_h = image_size
    intr_w, intr_h = intrinsic_size
    if min(img_w, img_h, intr_w, intr_h) <= 0:
        return
    image_aspect = img_w / float(img_h)
    intrinsic_aspect = intr_w / float(intr_h)
    delta = abs(image_aspect - intrinsic_aspect) / max(intrinsic_aspect, 1e-6)
    if delta <= args.aspect_tolerance:
        return
    message = (
        f"图片 {image_path} 的尺寸 {img_w}x{img_h} 与 K 对应尺寸 "
        f"{intr_w}x{intr_h} 宽高比不一致，delta={delta:.4f}。这通常表示 "
        "原图到 K 尺寸之间存在裁剪/非等比缩放，需要先把图片处理到和 K 同一"
        "坐标系，或把 K 重新适配到当前图片。")
    if args.allow_aspect_mismatch:
        LOGGER.warning(message)
    else:
        raise ValueError(message)


def build_camera_template(sensor: Dict, first_image_size: Tuple[int, int],
                          args: argparse.Namespace) -> Dict:
    raw_ext = sensor.get("extrinsic", {}).get("to_lidar_main")
    if raw_ext is None:
        raise ValueError(f"sensor {sensor.get('name')} lacks extrinsic.to_lidar_main")
    sensor_to_lidar_rot, sensor_to_lidar_tran = parse_sensor_to_lidar(raw_ext)
    if args.info_extrinsic_coordinate == "raw_n7":
        sensor_to_lidar_rot = RAW_TO_FASTBEV @ sensor_to_lidar_rot
        sensor_to_lidar_tran = RAW_TO_FASTBEV @ sensor_to_lidar_tran

    intrinsic = np.asarray(sensor.get("intrinsic", {}).get("K", []), dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"sensor {sensor.get('name')} has invalid K shape {intrinsic.shape}")
    distortion = np.asarray(sensor.get("intrinsic", {}).get("D", []), dtype=np.float32).reshape(-1)

    intrinsic_width, intrinsic_height, size_source = infer_intrinsic_size(
        sensor, intrinsic, first_image_size, args)
    return {
        "sensor_name": sensor.get("name", ""),
        "cam_intrinsic": intrinsic,
        "sensor2lidar_rotation": sensor_to_lidar_rot.astype(np.float32),
        "sensor2lidar_translation": sensor_to_lidar_tran.astype(np.float32),
        "sensor2ego_rotation": matrix_to_quat_wxyz(sensor_to_lidar_rot),
        "sensor2ego_translation": sensor_to_lidar_tran.astype(np.float32).tolist(),
        "distortion": distortion,
        "intrinsic_width": int(intrinsic_width),
        "intrinsic_height": int(intrinsic_height),
        "width": int(intrinsic_width),
        "height": int(intrinsic_height),
        "intrinsic_size_source": size_source,
        "info_extrinsic_coordinate": args.info_extrinsic_coordinate,
    }


def build_camera_info(camera_template: Dict, image_path: Path,
                      image_size: Tuple[int, int]) -> Dict:
    image_width, image_height = image_size
    cam_info = copy.deepcopy(camera_template)
    cam_info.update({
        "data_path": str(image_path.resolve()),
        "image_width": int(image_width),
        "image_height": int(image_height),
        # width/height 是兼容字段，语义保持为 cam_intrinsic 对应尺寸。
        "width": int(camera_template["intrinsic_width"]),
        "height": int(camera_template["intrinsic_height"]),
    })
    return cam_info


def dataset_lidar2img_from_cam_info(cam_info: Dict) -> Tuple[np.ndarray, Dict, Dict]:
    intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float32)
    sensor2lidar_r = np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float32)
    sensor2lidar_t = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float32).reshape(3)

    lidar2cam_r = np.linalg.inv(sensor2lidar_r)
    lidar2cam_t = sensor2lidar_t @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    distortion = np.asarray(cam_info.get("distortion", []), dtype=np.float32).reshape(-1)
    dimension_fields = dict(
        intrinsic_width=int(cam_info["intrinsic_width"]),
        intrinsic_height=int(cam_info["intrinsic_height"]),
        image_width=int(cam_info["image_width"]),
        image_height=int(cam_info["image_height"]),
    )
    lidar2img_aug = dict(
        intrin=intrinsic.astype(np.float32),
        rot=sensor2lidar_r.astype(np.float32),
        tran=sensor2lidar_t.astype(np.float32),
        post_rot=np.eye(3, dtype=np.float32),
        post_tran=np.zeros(3, dtype=np.float32),
        temporal_compensated=False,
        distortion=distortion,
        **dimension_fields,
    )
    lidar2img_extra = dict(
        distortion=distortion,
        temporal_compensated=False,
        **dimension_fields,
    )
    return lidar2img, lidar2img_aug, lidar2img_extra


def cfg_get(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def iter_dataset_cfgs(data_cfg: Any) -> Iterable[Any]:
    if data_cfg is None:
        return []
    items: List[Any] = []
    for split in ("test", "val", "train"):
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
        if isinstance(step, dict) and step.get("type") == step_type:
            return step
    return None


def infer_pipeline_value(cfg: Any, key: str, default: Any = None) -> Any:
    for dataset_cfg in iter_dataset_cfgs(cfg_get(cfg, "data")):
        if cfg_get(dataset_cfg, key) is not None:
            return cfg_get(dataset_cfg, key)
        step = find_pipeline_step(cfg_get(dataset_cfg, "pipeline"), "MultiViewPipeline")
        if step is not None and step.get(key) is not None:
            return step.get(key)
    return default


def patch_test_pipeline(cfg: Any, n_images: int, n_times: int,
                        force_resize: bool) -> None:
    pipeline = cfg.data.test.pipeline
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "MultiViewPipeline":
            step["sequential"] = True
            step["n_images"] = int(n_images)
            step["n_times"] = int(n_times)
        elif step.get("type") == "RandomAugImageMultiViewImage":
            step["is_train"] = False
            if force_resize:
                step["force_resize"] = True
        elif step.get("type") == "DefaultFormatBundle3D":
            step["with_label"] = False
    if cfg_get(cfg.data.test, "n_times") is not None:
        cfg.data.test.n_times = int(n_times)


def resolve_weight_args(args: argparse.Namespace) -> Tuple[str, Optional[Path], Optional[Path], Optional[Path]]:
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    backbone_onnx = Path(args.backbone_onnx) if args.backbone_onnx else None
    head_onnx = Path(args.head_onnx) if args.head_onnx else None
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else None

    if args.weights:
        weights = Path(args.weights)
        if weights.suffix.lower() in (".pth", ".pt"):
            checkpoint = weights
        elif weights.is_dir():
            onnx_dir = weights
        elif weights.suffix.lower() == ".onnx":
            raise ValueError(
                "当前 Fast-BEV ONNX 推理使用 split ONNX：2D backbone + 3D head。"
                "请传 --backbone-onnx 和 --head-onnx，或传 --onnx-dir 指向 export_metadata.json。")
        else:
            raise ValueError(f"无法识别 --weights: {weights}")

    if onnx_dir is not None:
        metadata_path = onnx_dir / "export_metadata.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            exports = metadata.get("exports", {})
            two_d = exports.get("2d", {})
            three_d = exports.get("3d", {})
            output_layout = two_d.get("output_layout")
            if output_layout is None:
                LOGGER.warning(
                    "%s 缺少 exports['2d']['output_layout']，当前脚本按 NCHW test_onnx 约定继续。",
                    metadata_path)
            else:
                layout = str(output_layout).lower()
                if layout == "nhwc":
                    raise ValueError(
                        f"{metadata_path} 记录的 exports['2d']['output_layout']='nhwc'，"
                        "但当前 infer_mono_front_image.py 的 ONNX 推理按 NCHW test_onnx 约定读取 2D 输出。"
                        "请重新导出 NCHW 2D ONNX，或修改脚本的 2D->3D 输入适配。")
                if layout != "nchw":
                    LOGGER.warning(
                        "%s 记录的 2D output_layout=%r 不是已知的 nchw/nhwc，当前脚本按 NCHW test_onnx 约定继续。",
                        metadata_path,
                        output_layout)
            backbone_value = two_d.get("simplified") or two_d.get("raw")
            head_value = three_d.get("simplified") or three_d.get("raw")
            if backbone_value:
                backbone_onnx = Path(backbone_value)
            if head_value:
                head_onnx = Path(head_value)
        else:
            LOGGER.warning(
                "未找到 %s，改用 glob 推断 split ONNX；无法确认 2D output_layout，"
                "当前脚本按 NCHW test_onnx 约定继续。",
                metadata_path)
            candidates_2d = sorted(onnx_dir.glob("*2d*.onnx"), key=natural_key)
            candidates_3d = sorted(onnx_dir.glob("*3d*.onnx"), key=natural_key)
            if candidates_2d:
                backbone_onnx = candidates_2d[-1]
            if candidates_3d:
                head_onnx = candidates_3d[-1]

    mode = args.mode
    if mode == "auto":
        mode = "onnx" if (backbone_onnx or head_onnx or onnx_dir) else "pth"

    if mode == "pth":
        if checkpoint is None:
            raise ValueError("PTH 模式需要 --checkpoint 或 --weights *.pth")
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    elif mode == "onnx":
        if backbone_onnx is None or head_onnx is None:
            raise ValueError("ONNX 模式需要 --backbone-onnx 和 --head-onnx，或 --onnx-dir")
        if not backbone_onnx.exists():
            raise FileNotFoundError(f"2D backbone ONNX 不存在: {backbone_onnx}")
        if not head_onnx.exists():
            raise FileNotFoundError(f"3D head ONNX 不存在: {head_onnx}")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return mode, checkpoint, backbone_onnx, head_onnx


def add_mmdet3d_root_to_path() -> None:
    mmdet3d_root = os.environ.get("MMDET3D")
    if mmdet3d_root and Path(mmdet3d_root).exists():
        sys.path.insert(0, mmdet3d_root)
        LOGGER.info("using mmdet3d: %s", mmdet3d_root)


def build_cfg_and_model(args: argparse.Namespace, mode: str,
                        checkpoint: Optional[Path],
                        backbone_onnx: Optional[Path],
                        head_onnx: Optional[Path],
                        device_str: str):
    add_mmdet3d_root_to_path()
    import torch
    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.core.bbox import get_box_type
    from mmdet3d.models import build_model

    cfg = Config.fromfile(args.config)
    cfg_options = parse_cfg_options(args.cfg_options)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg["custom_imports"])
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    model_n_images = int(cfg_get(cfg.model, "n_images", 1))
    if model_n_images != 1:
        raise ValueError(
            f"该脚本只服务 mono-front，config model.n_images={model_n_images}，"
            "请使用 mono-front config/权重。")
    n_times = int(infer_pipeline_value(cfg, "n_times", 4))
    patch_test_pipeline(cfg, model_n_images, n_times, args.force_resize)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if mode == "onnx":
        cfg.model.test_cfg.test_mode = "test_onnx"
        cfg.model.test_cfg.backbone_onnx = str(backbone_onnx)
        cfg.model.test_cfg.head_onnx = str(head_onnx)
        if args.onnx_custom_op_path:
            cfg.model.test_cfg.onnx_custom_op_path = args.onnx_custom_op_path
    else:
        cfg.model.test_cfg.test_mode = "test_pth"

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if mode == "pth" and str(device_str).lower().startswith("cpu"):
            LOGGER.warning(
                "PTH CPU 推理跳过 wrap_fp16_model；mmcv auto_fp16 会把 img cast 到 fp16，"
                "CPU half conv 不支持。")
        else:
            wrap_fp16_model(model)
    checkpoint_meta = {}
    if checkpoint is not None:
        loaded = load_checkpoint(model, str(checkpoint), map_location="cpu")
        checkpoint_meta = loaded.get("meta", {}) if isinstance(loaded, dict) else {}
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    class_names = list(cfg.get("class_names", checkpoint_meta.get("CLASSES", DEFAULT_CLASSES)))
    model.CLASSES = checkpoint_meta.get("CLASSES", class_names)
    model.cfg = cfg
    box_type_3d, box_mode_3d = get_box_type("LiDAR")
    return cfg, model, n_times, class_names, box_type_3d, box_mode_3d


def unwrap_data_container(value: Any) -> Any:
    return value._data if hasattr(value, "_data") else value


def build_sample_results(
    image_paths: Sequence[Path],
    camera_template: Dict,
    camera_id: str,
    token: str,
    timestamp: int,
    box_type_3d: Any,
    box_mode_3d: Any,
    args: argparse.Namespace,
) -> Tuple[Dict, Dict]:
    img_infos: List[Dict] = []
    img_prefix: List[None] = []
    lidar2img_rts: List[np.ndarray] = []
    lidar2img_augs: List[Dict] = []
    lidar2img_extras: List[Dict] = []
    first_cam_info: Optional[Dict] = None

    for image_path in image_paths:
        image_size = read_image_size(image_path)
        check_aspect(
            image_size=image_size,
            intrinsic_size=(camera_template["intrinsic_width"], camera_template["intrinsic_height"]),
            args=args,
            image_path=image_path)
        cam_info = build_camera_info(camera_template, image_path, image_size)
        if first_cam_info is None:
            first_cam_info = cam_info
        lidar2img_rt, lidar2img_aug, lidar2img_extra = dataset_lidar2img_from_cam_info(cam_info)
        img_infos.append({"filename": str(image_path.resolve())})
        img_prefix.append(None)
        lidar2img_rts.append(lidar2img_rt)
        lidar2img_augs.append(lidar2img_aug)
        lidar2img_extras.append(lidar2img_extra)

    if first_cam_info is None:
        raise ValueError("empty image_paths")

    results = dict(
        sample_idx=token,
        timestamp=float(timestamp) / 1e6,
        img_prefix=img_prefix,
        img_info=img_infos,
        lidar2img=dict(
            extrinsic=[x.astype(np.float32) for x in lidar2img_rts],
            intrinsic=np.eye(4, dtype=np.float32),
            lidar2img_aug=lidar2img_augs,
            lidar2img_extra=lidar2img_extras,
        ),
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[],
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
    )
    info = dict(
        token=token,
        timestamp=int(timestamp),
        cams={camera_id: first_cam_info},
        gt_boxes=np.zeros((0, 7), dtype=np.float32),
        gt_names=[],
        gt_velocity=np.zeros((0, 2), dtype=np.float32),
    )
    return results, info


def make_temporal_paths(image_path: Path, n_times: int,
                        args: argparse.Namespace) -> List[Path]:
    if args.temporal_images:
        temporal = [Path(x) for x in args.temporal_images]
        if len(temporal) != n_times:
            raise ValueError(
                f"--temporal-images 数量需要等于 n_times={n_times}，当前 {len(temporal)}")
        missing = [str(path) for path in temporal if not path.exists()]
        if missing:
            raise FileNotFoundError("时序图片不存在:\n- " + "\n- ".join(missing))
        return temporal
    if n_times == 1:
        return [image_path]
    if args.temporal_policy == "error":
        raise ValueError(
            f"当前 config 需要 n_times={n_times}，但只提供了一张图片；"
            "请传 --temporal-images 或使用 --temporal-policy repeat")
    return [image_path for _ in range(n_times)]


def run_one_sample(model, pipeline, sample_results: Dict, device) -> Tuple[Dict, Dict]:
    import torch

    data = pipeline(sample_results)
    img = unwrap_data_container(data["img"])
    img_metas = unwrap_data_container(data["img_metas"])
    if img.ndim == 4:
        img = img.unsqueeze(0)
    img = img.to(device)
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_context():
        result = model(img=img, img_metas=[img_metas], return_loss=False)
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError(f"unexpected model result type/len: {type(result)} / {len(result) if isinstance(result, list) else 'n/a'}")
    return result[0], img_metas


def tensor_like_to_numpy(value, dtype=None) -> np.ndarray:
    if value is None:
        arr = np.asarray([])
    else:
        if hasattr(value, "tensor"):
            value = value.tensor
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            arr = value.numpy()
        else:
            arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def prediction_boxes_to_center_numpy(
        boxes: Any, box_origin: Optional[str] = None) -> np.ndarray:
    """通过 N7 公共 origin 契约输出重心 z numpy box。"""
    return n7_boxes_to_numpy(
        boxes,
        target_origin="center",
        source_origin=box_origin,
        box_dim=None,
        dtype=np.float32)


def result_to_numpy_native(result: Dict) -> Dict:
    if isinstance(result, dict) and "pts_bbox" in result:
        return {"pts_bbox": result_to_numpy_native(result["pts_bbox"])}
    if not isinstance(result, dict):
        return {}
    boxes = result.get("boxes_3d", result.get("bboxes_3d"))
    scores = result.get("scores_3d", result.get("scores"))
    labels = result.get("labels_3d", result.get("labels"))
    out: Dict[str, Any] = {"box_origin": "center"}
    if boxes is not None:
        out["boxes_3d"] = prediction_boxes_to_center_numpy(
            boxes, result.get("box_origin"))
    if scores is not None:
        out["scores_3d"] = tensor_like_to_numpy(scores, dtype=np.float32).reshape(-1)
    if labels is not None:
        out["labels_3d"] = tensor_like_to_numpy(labels, dtype=np.int64).reshape(-1)
    return out


def result_to_summary(result: Dict, class_names: Sequence[str],
                      score_thr: float, max_preds: int) -> List[Dict]:
    if isinstance(result, dict) and "pts_bbox" in result:
        result = result["pts_bbox"]
    boxes = result.get("boxes_3d", result.get("bboxes_3d")) if isinstance(result, dict) else None
    scores = result.get("scores_3d", result.get("scores")) if isinstance(result, dict) else None
    labels = result.get("labels_3d", result.get("labels")) if isinstance(result, dict) else None
    if boxes is None or scores is None or labels is None:
        return []
    boxes_np = prediction_boxes_to_center_numpy(
        boxes, result.get("box_origin"))
    scores_np = tensor_like_to_numpy(scores, dtype=np.float32).reshape(-1)
    labels_np = tensor_like_to_numpy(labels, dtype=np.int64).reshape(-1)
    if boxes_np.ndim == 1:
        boxes_np = boxes_np.reshape(1, -1)
    elif boxes_np.ndim > 2:
        boxes_np = boxes_np.reshape(-1, boxes_np.shape[-1])
    count = min(len(boxes_np), len(scores_np), len(labels_np))
    rows = []
    for idx in np.argsort(-scores_np[:count]):
        score = float(scores_np[idx])
        if score < score_thr:
            continue
        label = int(labels_np[idx])
        name = class_names[label] if 0 <= label < len(class_names) else f"class_{label}"
        rows.append({
            "score": score,
            "label": label,
            "class_name": name,
            "box_origin": "center",
            "box_lidar": [float(x) for x in boxes_np[idx, :7].tolist()],
        })
        if max_preds > 0 and len(rows) >= max_preds:
            break
    return rows


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(payload, f)


def render_visualization(
    info: Dict,
    result: Dict,
    class_names: Sequence[str],
    frame_output: Path,
    args: argparse.Namespace,
) -> None:
    import cv2
    from tools.data_converter.n7.visualize_n7_fastbev_pkl import render_info

    image = render_info(
        info=info,
        data_root=Path("."),
        class_names=class_names,
        pred_result=result,
        pred_score_thr=args.score_thr,
        max_preds=args.max_preds,
        draw_gt=False,
        draw_pred=True,
        camera_ids=[args.camera_id],
        camera_width=args.camera_width,
        bev_range=tuple(float(x) for x in args.bev_range),
        bev_size=args.bev_size,
        bev_heading_style="front-edge",
        undistort=args.undistort,
        undistort_alpha=args.undistort_alpha,
        min_depth=args.min_depth,
        max_edge_px=0.0,
        draw_fullres=False,
        no_bev=args.no_bev,
        box_label_mode="compact",
        display_aspect=args.display_aspect,
        gt_view_mode="raw",
        gt_filter_visible_camera=None,
        gt_filter_range=None,
    )
    frame_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(frame_output), image):
        raise IOError(f"可视化写入失败: {frame_output}")


def visualization_output_path(output_dir: Path, image_path: Path, token: str,
                              args: argparse.Namespace,
                              image_base_dir: Optional[Path] = None) -> Path:
    if args.visualization_name == "image":
        if image_base_dir is None:
            return output_dir / "frames" / image_path.name
        return output_dir / "frames" / relative_path_from_base(image_path, image_base_dir)
    else:
        filename = f"{token}.{args.image_ext}"
    return output_dir / "frames" / filename


def relative_path_from_base(path: Path, base_dir: Path) -> Path:
    try:
        return path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return Path(path.name)


def should_render_visualization(processed_index: int, visualized_count: int,
                                args: argparse.Namespace) -> bool:
    if args.no_visualization:
        return False
    if processed_index % max(int(args.visualization_stride), 1) != 0:
        return False
    return int(args.max_visualizations) < 0 or visualized_count < int(args.max_visualizations)


def safe_output_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("._") or "vehicle"


def run_inference_job(
    *,
    job_name: str,
    info_json_path: Path,
    all_candidate_paths: Sequence[Path],
    output_dir: Path,
    args: argparse.Namespace,
    mode: str,
    checkpoint: Optional[Path],
    backbone_onnx: Optional[Path],
    head_onnx: Optional[Path],
    n_times: int,
    class_names: Sequence[str],
    box_type_3d: Any,
    box_mode_3d: Any,
    model: Any,
    pipeline: Any,
    device: Any,
    image_base_dir: Optional[Path] = None,
    vehicle_id: Optional[str] = None,
) -> Dict:
    image_paths = select_image_subset(all_candidate_paths, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "[%s] selected %d/%d images for inference (start_index=%d, stride=%d, max_frames=%d)",
        job_name,
        len(image_paths),
        len(all_candidate_paths),
        args.start_index,
        args.stride,
        args.max_frames)

    first_image_size = read_image_size(image_paths[0])
    calib = read_json(info_json_path)
    sensor = find_sensor(calib, args.sensor_name)
    camera_template = build_camera_template(sensor, first_image_size, args)
    LOGGER.info(
        "[%s] effective intrinsic size for %s: %sx%s (%s); first image=%sx%s",
        job_name,
        args.sensor_name,
        camera_template["intrinsic_width"],
        camera_template["intrinsic_height"],
        camera_template["intrinsic_size_source"],
        first_image_size[0],
        first_image_size[1])

    all_infos: List[Dict] = []
    all_results: List[Dict] = []
    all_summaries: List[Dict] = []
    visualized_count = 0
    visualization_names = set()

    for index, image_path in enumerate(image_paths):
        temporal_paths = make_temporal_paths(image_path, n_times, args)
        token = image_path.stem if len(image_paths) == 1 else f"{index:06d}_{image_path.stem}"
        # 没有真实同步时间戳时用 index 占位，保持 pkl 字段类型稳定。
        timestamp = index
        sample_results, info = build_sample_results(
            temporal_paths, camera_template, args.camera_id, token, timestamp,
            box_type_3d, box_mode_3d, args)
        result, img_meta = run_one_sample(model, pipeline, sample_results, device)
        all_infos.append(info)
        all_results.append(result)
        predictions = result_to_summary(result, class_names, args.score_thr, args.max_preds)
        all_summaries.append({
            "token": token,
            "image": str(image_path.resolve()),
            "relative_image": (
                str(relative_path_from_base(image_path, image_base_dir))
                if image_base_dir is not None else image_path.name),
            "temporal_images": [str(path.resolve()) for path in temporal_paths],
            "num_predictions": len(predictions),
            "predictions": predictions,
        })

        if should_render_visualization(index, visualized_count, args):
            frame_path = visualization_output_path(output_dir, image_path, token, args, image_base_dir)
            try:
                visualization_key = str(frame_path.relative_to(output_dir / "frames"))
            except ValueError:
                visualization_key = frame_path.name
            if visualization_key in visualization_names:
                LOGGER.warning("可视化文件名重复，将覆盖本轮前序输出: %s", visualization_key)
            visualization_names.add(visualization_key)
            render_visualization(info, result, class_names, frame_path, args)
            visualized_count += 1
            LOGGER.info("wrote visualization: %s", frame_path)

    metadata = dict(
        version="mono-front-production-image-infer",
        job_name=job_name,
        vehicle_id=vehicle_id,
        info_json=str(info_json_path),
        image_base_dir=str(image_base_dir) if image_base_dir is not None else None,
        mode=mode,
        config=str(Path(args.config)),
        checkpoint=str(checkpoint) if checkpoint else None,
        backbone_onnx=str(backbone_onnx) if backbone_onnx else None,
        head_onnx=str(head_onnx) if head_onnx else None,
        classes=list(class_names),
        camera_ids=[args.camera_id],
        sensor_name=args.sensor_name,
        coordinate="mmdet3d_lidar:x_front_y_left_z_up; origin follows converted info.json to_lidar_main",
        prediction_box_origin="center",
        info_extrinsic_coordinate=args.info_extrinsic_coordinate,
        raw_to_fastbev=RAW_TO_FASTBEV.tolist(),
        n_times=n_times,
        temporal_policy=args.temporal_policy,
        input_image_count=len(all_candidate_paths),
        selected_image_count=len(image_paths),
        start_index=int(args.start_index),
        stride=max(int(args.stride), 1),
        max_frames=int(args.max_frames),
        visualization_stride=max(int(args.visualization_stride), 1),
        max_visualizations=int(args.max_visualizations),
        display_aspect=args.display_aspect,
        no_bev=bool(args.no_bev),
        visualization_name=args.visualization_name,
        visualization_count=visualized_count,
        temporal_note=(
            "Real --temporal-images history frames are used without ego-motion compensation "
            "because no pose input is provided; this differs from training-time prev frames. "
            "The repeat fallback matches the training distribution for clip-start samples."
        ),
        temporal_images_without_ego_motion_compensation=bool(args.temporal_images),
        calibration=json_ready(camera_template),
    )
    write_pickle(output_dir / "infos.pkl", {"infos": all_infos, "metadata": metadata})
    write_pickle(
        output_dir / "pred_results.pkl",
        [result_to_numpy_native(result) for result in all_results])
    (output_dir / "prediction_summary.json").write_text(
        json.dumps({"metadata": metadata, "frames": all_summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    LOGGER.info("wrote infos: %s", output_dir / "infos.pkl")
    LOGGER.info("wrote predictions: %s", output_dir / "pred_results.pkl")
    LOGGER.info("wrote summary: %s", output_dir / "prediction_summary.json")
    return dict(
        job_name=job_name,
        vehicle_id=vehicle_id,
        info_json=str(info_json_path),
        image_base_dir=str(image_base_dir) if image_base_dir is not None else None,
        output_dir=str(output_dir),
        input_image_count=len(all_candidate_paths),
        selected_image_count=len(image_paths),
        visualization_count=visualized_count,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s")

    mode, checkpoint, backbone_onnx, head_onnx = resolve_weight_args(args)
    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device
    if device_str is None:
        import torch
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    import torch
    device = torch.device(device_str)
    cfg, model, n_times, class_names, box_type_3d, box_mode_3d = build_cfg_and_model(
        args, mode, checkpoint, backbone_onnx, head_onnx, device_str)
    model = model.to(device)
    model.eval()

    from mmdet3d.datasets.pipelines import Compose
    pipeline = Compose(copy.deepcopy(cfg.data.test.pipeline))

    if args.vehicle_id:
        if args.image or args.image_dir or args.image_list:
            LOGGER.warning(
                "--vehicle-id 批量模式会忽略 --image/--image-dir/--image-list，"
                "图片来自 --image-root/<vehicle-id>。")
        batch_jobs = []
        for vehicle_id in args.vehicle_id:
            info_json_path = resolve_vehicle_info_json(vehicle_id, args)
            if not info_json_path.exists():
                raise FileNotFoundError(f"车辆 {vehicle_id} 的 info.json 不存在: {info_json_path}")
            image_base_dir = vehicle_image_base_dir(vehicle_id, args)
            all_candidate_paths = candidate_vehicle_images(vehicle_id, args)
            job_output_dir = root_output_dir / safe_output_name(vehicle_id)
            batch_jobs.append(run_inference_job(
                job_name=vehicle_id,
                vehicle_id=vehicle_id,
                info_json_path=info_json_path,
                all_candidate_paths=all_candidate_paths,
                output_dir=job_output_dir,
                args=args,
                mode=mode,
                checkpoint=checkpoint,
                backbone_onnx=backbone_onnx,
                head_onnx=head_onnx,
                n_times=n_times,
                class_names=class_names,
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d,
                model=model,
                pipeline=pipeline,
                device=device,
                image_base_dir=image_base_dir))
        batch_summary = dict(
            version="mono-front-production-image-infer-batch",
            info_json_dir=str(Path(args.info_json_dir)),
            info_json_template=args.info_json_template,
            image_root=str(Path(args.image_root)),
            vehicle_image_glob=list(args.vehicle_image_glob),
            jobs=batch_jobs,
        )
        (root_output_dir / "batch_summary.json").write_text(
            json.dumps(batch_summary, ensure_ascii=False, indent=2),
            encoding="utf-8")
        LOGGER.info("wrote batch summary: %s", root_output_dir / "batch_summary.json")
        return

    all_candidate_paths = candidate_images(args)
    run_inference_job(
        job_name="single",
        vehicle_id=None,
        info_json_path=Path(args.info_json),
        all_candidate_paths=all_candidate_paths,
        output_dir=root_output_dir,
        args=args,
        mode=mode,
        checkpoint=checkpoint,
        backbone_onnx=backbone_onnx,
        head_onnx=head_onnx,
        n_times=n_times,
        class_names=class_names,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        model=model,
        pipeline=pipeline,
        device=device,
        image_base_dir=None)


if __name__ == "__main__":
    main()
