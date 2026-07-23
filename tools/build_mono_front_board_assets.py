#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性准备 N7 S0 板端模型公共资料和车辆固定 LUT。

这个工具只负责“生成资料”，不运行图片推理。生成结果分为两层：

- 模型公共资料：``board_model_spec.json``、``anchors.npy``、``points.npy``；
- 每车资料：``data/board_lut/<vehicle_id>/{LUT,LUT_arr,metadata.json}``。

``board_model_spec.json`` 同时记录 FP ONNX 契约和可选的真实 INT8 ONNX
契约。未提供真实 INT8 图时会明确记录“未做真实 INT8 模型数值验证”；不会把
FP ONNX 当作 INT8 别名。

公共资料已经完整时不会重复生成；车辆 LUT 已存在且标定 SHA、模型几何契约
一致时也会直接复用。标定或模型几何变化必须显式重建，避免静默覆盖资产。

只准备模型公共资料：

  python tools/build_mono_front_board_assets.py \
  --weights work_dirs/n7_mono_single_frame/fp_onnx \
  --config configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18.py

准备 9797 生产车 LUT：

  python tools/build_mono_front_board_assets.py \
  --weights work_dirs/n7_mono_single_frame/fp_onnx \
  --config configs/fastbev/custom/custom_fastbev_mono_front_single_frame_r18.py \
  --vehicle-id 9797_UKEF \
  --info-json data/info_json/2025_04_18_byd_info_9797_UKEF.json \
  --images data/gt/20260514_9797/20260514103014_1.dat_img \
  --intrinsic-size 3840 2160 \
  --device cuda:0

多车模式下，``--info-json`` 和 ``--images`` 都传根目录，图片目录约定为
``<images>/<vehicle_id>``；info.json 可用模板定位，也会按车辆编号唯一搜索。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mono_front_board_core import (
    DEFAULT_ASSET_PROFILE,
    DEFAULT_CONFIG,
    DEFAULT_LUT_ROOT,
    HelpFormatter,
    _candidate_images,
    _infer_vehicle_id_from_info,
    _resolve_info_json,
    build_board_model_assets,
    build_board_vehicle_lut,
    natural_key,
)


LOGGER = logging.getLogger("mono_front_board_asset_builder")


def _first_image(images_root: Path, vehicle_id: str, multi_vehicle: bool) -> Path:
    """选择目录中自然排序第一张图，仅用其尺寸建立 LUT 坐标系。"""
    if images_root.is_file():
        return images_root.resolve()
    if multi_vehicle or (images_root / vehicle_id).is_dir():
        vehicle_root = images_root / vehicle_id
    else:
        vehicle_root = images_root
    candidates = _candidate_images(
        vehicle_root, ("*.jpg", "*.jpeg", "*.png", "*.bmp"))
    if not candidates:
        raise FileNotFoundError(f"车辆 {vehicle_id} 没有可用于 LUT 的图片: {vehicle_root}")
    return sorted(candidates, key=natural_key)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成板端公共 anchors/points/spec 和每车固定 LUT",
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="split 2D/3D ONNX 权重目录")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="模型训练/导出配置")
    parser.add_argument("--lut-root", default=DEFAULT_LUT_ROOT,
                        help="公共资料和每车 LUT 根目录")
    parser.add_argument("--asset-profile", default=DEFAULT_ASSET_PROFILE,
                        help="anchor/voxel 几何版本名")
    parser.add_argument("--int8-2d-onnx",
                        help="真实量化 2D ONNX；必须与 --int8-3d-onnx 同时提供")
    parser.add_argument("--int8-3d-onnx",
                        help="真实量化 3D ONNX；必须与 --int8-2d-onnx 同时提供")
    parser.add_argument(
        "--int8-output-layout", choices=("nchw", "nhwc"), default="nchw",
        help="真实 INT8 2D feature 的 graph output layout")
    parser.add_argument("--force-model-assets", action="store_true",
                        help="显式重建模型公共资料")

    parser.add_argument("--vehicle-id", nargs="+", help="要准备 LUT 的一个或多个车辆编号")
    parser.add_argument("--info-json", help="单车 info.json 或多车 info.json 根目录")
    parser.add_argument("--images", help="单图、单车图片目录或多车图片根目录")
    parser.add_argument("--info-json-template", default="{vehicle_id}.json",
                        help="多车 info.json 文件名模板")
    parser.add_argument("--sensor-name", default="front_wide",
                        help="info.json 中前视相机 sensor 名")
    parser.add_argument("--intrinsic-size", nargs=2, type=int,
                        metavar=("WIDTH", "HEIGHT"), help="K 对应的图像尺寸")
    parser.add_argument("--intrinsic-size-source", choices=("auto", "info", "image"),
                        default="auto", help="未显式给尺寸时的推断来源")
    parser.add_argument("--info-extrinsic-coordinate", choices=("raw", "fastbev"),
                        default="raw", help="info.json 外参坐标系语义")
    parser.add_argument("--allow-aspect-mismatch", action="store_true",
                        help="图片与 K 宽高比不一致时只告警")
    parser.add_argument("--device", default="cuda:0",
                        help="生成 LUT 的 Torch 投影设备；黄金资产建议 cuda:0")
    parser.add_argument("--force-vehicle-lut", action="store_true",
                        help="显式重建车辆 LUT")
    args = parser.parse_args()
    if bool(args.info_json) != bool(args.images):
        parser.error("准备车辆 LUT 时 --info-json 和 --images 必须同时提供")
    if bool(args.int8_2d_onnx) != bool(args.int8_3d_onnx):
        parser.error("--int8-2d-onnx 和 --int8-3d-onnx 必须同时提供")
    if args.vehicle_id and not args.info_json:
        parser.error("传入 --vehicle-id 时必须同时传 --info-json 和 --images")
    if args.intrinsic_size and min(args.intrinsic_size) <= 0:
        parser.error("--intrinsic-size 必须是正整数")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    weights = Path(args.weights).expanduser().resolve()
    lut_root = Path(args.lut_root).expanduser().resolve()
    model_result = build_board_model_assets(
        weights_dir=weights,
        config_path=Path(args.config),
        asset_root=lut_root,
        profile_name=args.asset_profile,
        int8_model_2d=(
            Path(args.int8_2d_onnx) if args.int8_2d_onnx else None),
        int8_model_3d=(
            Path(args.int8_3d_onnx) if args.int8_3d_onnx else None),
        int8_output_layout=args.int8_output_layout,
        force=args.force_model_assets,
    )
    spec = model_result["spec"]

    vehicle_records: List[dict] = []
    if args.info_json:
        info_root = Path(args.info_json).expanduser()
        images_root = Path(args.images).expanduser()
        vehicle_ids = list(args.vehicle_id or [])
        if not vehicle_ids:
            if not info_root.is_file():
                raise ValueError("多车 info.json 目录模式必须显式传 --vehicle-id")
            vehicle_ids = [_infer_vehicle_id_from_info(info_root)]
        multi_vehicle = len(vehicle_ids) > 1
        if multi_vehicle and info_root.is_file():
            raise ValueError("多车模式下 --info-json 必须是包含各车标定的根目录")
        if multi_vehicle and images_root.is_file():
            raise ValueError("多车模式下 --images 必须是包含车辆子目录的根目录")
        for vehicle_id in vehicle_ids:
            info_path = _resolve_info_json(
                info_root, vehicle_id, args.info_json_template)
            if info_path is None:
                raise FileNotFoundError(
                    f"车辆 {vehicle_id} 找不到 info.json: {info_root}")
            image_path = _first_image(images_root, vehicle_id, multi_vehicle)
            record = build_board_vehicle_lut(
                vehicle_id=vehicle_id,
                info_json=info_path,
                image_path=image_path,
                spec=spec,
                asset_root=lut_root,
                sensor_name=args.sensor_name,
                intrinsic_size=args.intrinsic_size,
                intrinsic_size_source=args.intrinsic_size_source,
                info_extrinsic_coordinate=args.info_extrinsic_coordinate,
                allow_aspect_mismatch=args.allow_aspect_mismatch,
                torch_device=args.device,
                force=args.force_vehicle_lut,
            )
            vehicle_records.append(record)
            LOGGER.info("车辆 %s LUT: %s", vehicle_id, record["status"])

    print(
        "BOARD_ASSET_BUILD: PASS; "
        f"model={model_result['status']}; vehicles={len(vehicle_records)}; "
        f"lut_root={lut_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
