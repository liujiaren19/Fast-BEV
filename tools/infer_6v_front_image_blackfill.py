#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 6V Fast-BEV checkpoint 对单目前视图做黑图补齐推理。

这个脚本用于诊断性 A/B：加载原 6V 模型和 6V 标定，真实图片只放到
``cam0/front_wide``，其余相机用全黑图片占位，使 6V 模型的输入 shape
和相机顺序保持不变。由于生产侧没有 pose，时序维默认重复同一组 6V 输入，
不做 ego-motion compensation。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import infer_mono_front_image as mono_utils  # noqa: E402


LOGGER = logging.getLogger("infer_6v_front_image_blackfill")

DEFAULT_CONFIG = "configs/fastbev/custom/custom_fastbev_6v_r18.py"
DEFAULT_CAMERA_IDS = ["cam0", "cam11", "cam9", "cam3", "cam8", "cam10"]
DEFAULT_CAMERA_SENSOR_MAP = {
    "cam0": "front_wide",
    "cam11": "right_front",
    "cam9": "left_front",
    "cam3": "back",
    "cam8": "left_back",
    "cam10": "right_back",
}
DEFAULT_BEV_RANGE = (-50.0, -50.0, 50.0, 50.0)


class RawDefaultsHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter):
    """保留示例换行，同时展示默认值。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 6V Fast-BEV checkpoint，前视用真实图，其余相机用黑图补齐推理",
        formatter_class=RawDefaultsHelpFormatter,
        epilog="""
示例：
  python tools/infer_6v_front_image_blackfill.py \\
    --config configs/fastbev/custom/custom_fastbev_6v_r18.py \\
    --checkpoint work_dirs/6v/epoch_20.pth \\
    --info-json data/info_json/2025_04_18_2k_byd_info.json \\
    --image-dir /path/to/front_3840x2160 \\
    --image-glob '*.jpg' \\
    --output-dir outputs/6v_blackfill_front

  # 如果 info.json 中 K 是 4K 原图坐标，建议显式声明：
  python tools/infer_6v_front_image_blackfill.py \\
    --checkpoint work_dirs/6v/epoch_20.pth \\
    --info-json /path/to/original_4k_info.json \\
    --intrinsic-size 3840 2160 \\
    --image /path/to/front.jpg \\
    --output-dir outputs/6v_blackfill_4k
""")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="6V Fast-BEV config")
    parser.add_argument("--checkpoint", required=True, help="6V PTH checkpoint")
    parser.add_argument("--info-json", required=True, help="包含 6V pinhole sensors 的 N7 格式 info.json")
    parser.add_argument("--image", action="append", default=[], help="前视图片路径；可重复传多张")
    parser.add_argument("--image-dir", default=None, help="前视图片目录；配合 --image-glob 使用")
    parser.add_argument("--image-glob", default="*.jpg", help="--image-dir 下的图片 glob")
    parser.add_argument("--image-list", default=None, help="逐行保存前视图片路径的文本文件")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--device", default=None, help="推理设备，默认优先 cuda:0，否则 cpu")
    parser.add_argument("--camera-ids", nargs="+", default=None,
                        help="6V 相机顺序；默认优先从 config.data.test.camera_types 读取")
    parser.add_argument("--front-camera-id", default="cam0", help="真实前视图对应的 Fast-BEV camera id")
    parser.add_argument("--camera-sensor-map", nargs="+", default=None,
                        help="覆盖 camera id 到 info.json sensor name 映射，格式 cam0=front_wide")
    parser.add_argument("--black-value", type=int, default=0, help="黑图像素值，范围 0-255")
    parser.add_argument("--black-dir", default=None,
                        help="黑图缓存目录；默认写到 output-dir/_black_fill")
    parser.add_argument("--intrinsic-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        default=None, help="所有相机 cam_intrinsic/K 对应尺寸；不传则每个 sensor 自动判断")
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
    parser.add_argument("--cfg-options", nargs="+", default=None,
                        help="覆盖 config，格式 key=value；支持 model.xxx=... 形式")
    parser.add_argument("--fuse-conv-bn", action="store_true", help="PTH 推理前 fuse conv/bn")
    parser.add_argument("--score-thr", type=float, default=0.2, help="可视化和 summary 使用的分数阈值")
    parser.add_argument("--max-preds", type=int, default=100, help="每帧最多写入 summary/可视化的预测框数；<=0 不限制")
    parser.add_argument("--no-visualization", action="store_true", help="只保存预测结果，不输出可视化")
    parser.add_argument("--render-camera-ids", nargs="+", default=None,
                        help="可视化相机列表；默认画全部 6V 相机")
    parser.add_argument("--camera-width", type=int, default=640, help="可视化每个相机视图宽度")
    parser.add_argument("--bev-size", type=int, default=700, help="可视化 BEV 面板基础尺寸")
    parser.add_argument("--bev-range", nargs=4, type=float, default=list(DEFAULT_BEV_RANGE),
                        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"), help="BEV 可视化范围")
    parser.add_argument("--raw-distorted", dest="undistort", action="store_false",
                        default=True, help="可视化不去畸变，直接在原始畸变图上画框")
    parser.add_argument("--undistort-alpha", type=float, default=0.0, help="OpenCV 去畸变 alpha")
    parser.add_argument("--min-depth", type=float, default=0.1, help="画框最小相机深度")
    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg", help="可视化图片格式")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def parse_camera_sensor_map(items: Optional[Sequence[str]]) -> Dict[str, str]:
    mapping = dict(DEFAULT_CAMERA_SENSOR_MAP)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--camera-sensor-map 项缺少 '=': {item}")
        camera_id, sensor_name = item.split("=", 1)
        mapping[camera_id.strip()] = sensor_name.strip()
    return mapping


def cfg_get(mapping: Any, key: str, default: Any = None) -> Any:
    return mono_utils.cfg_get(mapping, key, default)


def infer_pipeline_value(cfg: Any, key: str, default: Any) -> Any:
    return mono_utils.infer_pipeline_value(cfg, key, default)


def camera_ids_from_cfg(cfg: Any, args: argparse.Namespace) -> List[str]:
    if args.camera_ids:
        return [str(x) for x in args.camera_ids]
    test_cfg = cfg_get(cfg.data, "test")
    camera_ids = cfg_get(test_cfg, "camera_types")
    if camera_ids is None:
        camera_ids = cfg_get(cfg, "camera_types")
    if camera_ids is None:
        camera_ids = DEFAULT_CAMERA_IDS
    return [str(x) for x in camera_ids]


def select_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_cfg_and_model(args: argparse.Namespace, device_str: str):
    mono_utils.add_mmdet3d_root_to_path()
    import torch
    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.core.bbox import get_box_type
    from mmdet3d.models import build_model

    cfg = Config.fromfile(args.config)
    cfg_options = mono_utils.parse_cfg_options(args.cfg_options)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg["custom_imports"])
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    camera_ids = camera_ids_from_cfg(cfg, args)
    n_images = int(cfg_get(cfg.model, "n_images", len(camera_ids)))
    if n_images != len(camera_ids):
        raise ValueError(
            f"config model.n_images={n_images}，但 camera_ids 数量={len(camera_ids)}: {camera_ids}")
    n_times = int(infer_pipeline_value(cfg, "n_times", cfg_get(cfg.data.test, "n_times", 4)))
    mono_utils.patch_test_pipeline(cfg, n_images, n_times, args.force_resize)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.test_cfg.test_mode = "test_pth"

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if str(device_str).lower().startswith("cpu"):
            LOGGER.warning(
                "PTH CPU 推理跳过 wrap_fp16_model；mmcv auto_fp16 会把 img cast 到 fp16，"
                "CPU half conv 不支持。")
        else:
            wrap_fp16_model(model)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    loaded = load_checkpoint(model, str(checkpoint), map_location="cpu")
    checkpoint_meta = loaded.get("meta", {}) if isinstance(loaded, dict) else {}
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    class_names = list(cfg.get("class_names", checkpoint_meta.get("CLASSES", mono_utils.DEFAULT_CLASSES)))
    model.CLASSES = checkpoint_meta.get("CLASSES", class_names)
    model.cfg = cfg
    box_type_3d, box_mode_3d = get_box_type("LiDAR")
    return cfg, model, camera_ids, n_times, class_names, box_type_3d, box_mode_3d


def make_black_image(path: Path, image_size: Tuple[int, int], black_value: int) -> Path:
    width, height = image_size
    path.parent.mkdir(parents=True, exist_ok=True)
    black_value = int(np.clip(black_value, 0, 255))
    if path.exists() and mono_utils.read_image_size(path) == (width, height):
        return path
    image = Image.new("RGB", (width, height), (black_value, black_value, black_value))
    image.save(path)
    return path


def build_camera_templates(
    calib: Dict,
    camera_ids: Sequence[str],
    sensor_map: Dict[str, str],
    first_image_size: Tuple[int, int],
    args: argparse.Namespace,
) -> Dict[str, Dict]:
    templates: Dict[str, Dict] = {}
    for camera_id in camera_ids:
        sensor_name = sensor_map.get(camera_id)
        if sensor_name is None:
            raise KeyError(
                f"缺少 camera id {camera_id!r} 到 info.json sensor name 的映射，"
                "请通过 --camera-sensor-map camX=sensor_name 指定。")
        sensor = mono_utils.find_sensor(calib, sensor_name)
        templates[camera_id] = mono_utils.build_camera_template(sensor, first_image_size, args)
    return templates


def build_blackfill_sample_results(
    front_image_path: Path,
    black_image_path: Path,
    camera_templates: Dict[str, Dict],
    camera_ids: Sequence[str],
    front_camera_id: str,
    n_times: int,
    token: str,
    timestamp: int,
    box_type_3d: Any,
    box_mode_3d: Any,
    args: argparse.Namespace,
) -> Tuple[Dict, Dict, List[Path]]:
    img_infos: List[Dict] = []
    img_prefix: List[None] = []
    lidar2img_rts: List[np.ndarray] = []
    lidar2img_augs: List[Dict] = []
    lidar2img_extras: List[Dict] = []
    current_cams: Dict[str, Dict] = {}
    view_paths: List[Path] = []

    for time_id in range(n_times):
        for camera_id in camera_ids:
            image_path = front_image_path if camera_id == front_camera_id else black_image_path
            image_size = mono_utils.read_image_size(image_path)
            camera_template = camera_templates[camera_id]
            mono_utils.check_aspect(
                image_size=image_size,
                intrinsic_size=(camera_template["intrinsic_width"], camera_template["intrinsic_height"]),
                args=args,
                image_path=image_path)
            cam_info = mono_utils.build_camera_info(camera_template, image_path, image_size)
            if time_id == 0:
                current_cams[camera_id] = cam_info
            lidar2img_rt, lidar2img_aug, lidar2img_extra = mono_utils.dataset_lidar2img_from_cam_info(cam_info)
            img_infos.append({"filename": str(image_path.resolve())})
            img_prefix.append(None)
            lidar2img_rts.append(lidar2img_rt)
            lidar2img_augs.append(lidar2img_aug)
            lidar2img_extras.append(lidar2img_extra)
            view_paths.append(image_path)

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
        cams=current_cams,
        gt_boxes=np.zeros((0, 7), dtype=np.float32),
        gt_names=[],
        gt_velocity=np.zeros((0, 2), dtype=np.float32),
    )
    return results, info, view_paths


def render_visualization(
    info: Dict,
    result: Dict,
    class_names: Sequence[str],
    camera_ids: Sequence[str],
    frame_output: Path,
    args: argparse.Namespace,
) -> None:
    import cv2
    from tools.data_converter.n7.visualize_n7_fastbev_pkl import render_info

    render_camera_ids = args.render_camera_ids or list(camera_ids)
    image = render_info(
        info=info,
        data_root=Path("."),
        class_names=class_names,
        pred_result=result,
        pred_score_thr=args.score_thr,
        max_preds=args.max_preds,
        draw_gt=False,
        draw_pred=True,
        camera_ids=render_camera_ids,
        camera_width=args.camera_width,
        bev_range=tuple(float(x) for x in args.bev_range),
        bev_size=args.bev_size,
        bev_heading_style="front-edge",
        undistort=args.undistort,
        undistort_alpha=args.undistort_alpha,
        min_depth=args.min_depth,
        max_edge_px=0.0,
        draw_fullres=False,
        no_bev=False,
        box_label_mode="compact",
        display_aspect="native",
        gt_view_mode="raw",
        gt_filter_visible_camera=None,
        gt_filter_range=None,
    )
    frame_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(frame_output), image):
        raise IOError(f"可视化写入失败: {frame_output}")


def write_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(payload, f)


def json_ready(value: Any) -> Any:
    return mono_utils.json_ready(value)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s")

    image_paths = mono_utils.candidate_images(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    black_dir = Path(args.black_dir) if args.black_dir else output_dir / "_black_fill"

    first_image_size = mono_utils.read_image_size(image_paths[0])
    calib = mono_utils.read_json(Path(args.info_json))
    sensor_map = parse_camera_sensor_map(args.camera_sensor_map)
    device_str = select_device(args.device)

    import torch
    device = torch.device(device_str)
    cfg, model, camera_ids, n_times, class_names, box_type_3d, box_mode_3d = build_cfg_and_model(
        args, device_str)
    if args.front_camera_id not in camera_ids:
        raise ValueError(f"--front-camera-id {args.front_camera_id!r} 不在 camera_ids={camera_ids} 中")
    camera_templates = build_camera_templates(
        calib, camera_ids, sensor_map, first_image_size, args)

    model = model.to(device)
    model.eval()

    from mmdet3d.datasets.pipelines import Compose
    pipeline = Compose(copy.deepcopy(cfg.data.test.pipeline))

    all_infos: List[Dict] = []
    all_results: List[Dict] = []
    all_summaries: List[Dict] = []

    for index, image_path in enumerate(image_paths):
        image_size = mono_utils.read_image_size(image_path)
        black_path = make_black_image(
            black_dir / f"black_{image_size[0]}x{image_size[1]}_{args.black_value}.png",
            image_size,
            args.black_value)
        token = image_path.stem if len(image_paths) == 1 else f"{index:06d}_{image_path.stem}"
        timestamp = index
        sample_results, info, view_paths = build_blackfill_sample_results(
            front_image_path=image_path,
            black_image_path=black_path,
            camera_templates=camera_templates,
            camera_ids=camera_ids,
            front_camera_id=args.front_camera_id,
            n_times=n_times,
            token=token,
            timestamp=timestamp,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            args=args)
        result, img_meta = mono_utils.run_one_sample(model, pipeline, sample_results, device)
        all_infos.append(info)
        all_results.append(result)
        predictions = mono_utils.result_to_summary(
            result, class_names, args.score_thr, args.max_preds)
        all_summaries.append({
            "token": token,
            "front_image": str(image_path.resolve()),
            "black_image": str(black_path.resolve()),
            "view_images": [str(path.resolve()) for path in view_paths],
            "num_predictions": len(predictions),
            "predictions": predictions,
        })

        if not args.no_visualization:
            frame_path = output_dir / "frames" / f"{token}.{args.image_ext}"
            render_visualization(info, result, class_names, camera_ids, frame_path, args)
            LOGGER.info("wrote visualization: %s", frame_path)

    metadata = dict(
        version="6v-front-image-blackfill-infer",
        mode="pth",
        config=str(Path(args.config)),
        checkpoint=str(Path(args.checkpoint)),
        classes=list(class_names),
        camera_ids=list(camera_ids),
        front_camera_id=args.front_camera_id,
        camera_sensor_map={camera_id: sensor_map.get(camera_id) for camera_id in camera_ids},
        black_fill_camera_ids=[camera_id for camera_id in camera_ids if camera_id != args.front_camera_id],
        black_value=int(np.clip(args.black_value, 0, 255)),
        coordinate="mmdet3d_lidar:x_front_y_left_z_up; origin follows converted info.json to_lidar_main",
        info_extrinsic_coordinate=args.info_extrinsic_coordinate,
        raw_to_fastbev=mono_utils.RAW_TO_FASTBEV.tolist(),
        n_images=len(camera_ids),
        n_times=n_times,
        temporal_policy="repeat_same_6v_blackfill",
        temporal_note=(
            "The same front image plus black-fill 6V set is repeated for all temporal steps; "
            "no ego-motion compensation or pose input is used. This is a diagnostic hack, "
            "not a geometry-correct 6V production path."
        ),
        calibration={camera_id: json_ready(camera_templates[camera_id]) for camera_id in camera_ids},
    )
    write_pickle(output_dir / "infos.pkl", {"infos": all_infos, "metadata": metadata})
    write_pickle(
        output_dir / "pred_results.pkl",
        [mono_utils.result_to_numpy_native(result) for result in all_results])
    (output_dir / "prediction_summary.json").write_text(
        json.dumps({"metadata": metadata, "frames": all_summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    LOGGER.info("wrote infos: %s", output_dir / "infos.pkl")
    LOGGER.info("wrote predictions: %s", output_dir / "pred_results.pkl")
    LOGGER.info("wrote summary: %s", output_dir / "prediction_summary.json")


if __name__ == "__main__":
    main()
