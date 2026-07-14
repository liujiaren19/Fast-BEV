#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 Windows 10 或 Linux 上按 eval pkl 精确拉取 Ozone 图片。

这个脚本面向“服务器完成推理，Windows 本地只做预测可视化”的场景。它会：

1. 读取 ``--gt-pkl`` 中本次 eval 的 infos；
2. 按 ``--start-index/--stride/--max-frames`` 选择和可视化脚本相同的帧；
3. 通过一次 ``rclone copy --files-from`` 只下载这些帧引用的相机图片；
4. 默认把 Ozone 原图 resize 到 pkl ``image_width/image_height`` 记录的尺寸；
5. 保留 pkl 中的相对目录，因此下载目录可直接作为可视化的 ``--data-root``。

注意：脚本只拉取离线可视化需要的当前帧，不拉 temporal ``prev`` 图片，也不
在 Windows 上执行模型推理。预测结果 pkl 和 GT/info pkl 仍需从 eval 服务器
复制到 Windows。

Windows 10 示例（从仓库根目录运行）：

    py -3 tools/data_converter/n7/pull_n7_eval_images_from_ozone.py ^
      --gt-pkl D:\\FastBEV\\eval\\custom_fastbev_xxx_infos_val_20260703.pkl ^
      --remote-root ozone:/gcy-truevalplat/liujiaren/data/nuscenes_20251017_20251030_20251031_20251203/nuscenes ^
      --output-root D:\\FastBEV\\data\\N7_704_256 ^
      --camera-ids cam0 --max-frames -1 --progress

下载完成后可直接运行：

    py -3 tools/data_converter/n7/visualize_n7_fastbev_pkl.py ^
      --gt-pkl D:\\FastBEV\\eval\\custom_fastbev_xxx_infos_val_20260703.pkl ^
      --pred-pkl D:\\FastBEV\\eval\\epoch_5_val_results.pkl ^
      --data-root D:\\FastBEV\\data\\N7_704_256 ^
      --output-dir D:\\FastBEV\\eval_vis\\epoch_5 ^
      --camera-ids cam0 --max-frames -1 --video-group sequence --workers 4

前置条件：

* Windows 已安装 Python、Pillow、numpy、opencv-python、tqdm；
* Windows 已安装 rclone，且 ``rclone listremotes`` 能看到 ``ozone:``；
* 不要把包含 Ozone 密钥的 ``rclone.conf`` 提交到仓库。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import datetime as dt
import json
import os
import pickle
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - 只在缺少 Windows 依赖时触发
    raise SystemExit("缺少 Pillow，请先运行：py -3 -m pip install pillow") from exc


@dataclass(frozen=True)
class ImageSpec:
    """一张待下载图片及 pkl 要求的最终尺寸。"""

    relative_path: str
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 Fast-BEV eval GT/info pkl 从 Ozone 精确拉取可视化图片。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gt-pkl", type=Path, required=True,
                        help="eval 使用的 GT/info pkl；不是 tools/test.py 输出的预测 pkl")
    parser.add_argument("--remote-root", required=True,
                        help="Ozone 上与 pkl data_path 相对根对应的 rclone 路径")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Windows 本地图片根目录；后续直接作为可视化 --data-root")
    parser.add_argument("--camera-ids", nargs="+", default=None,
                        help="只拉指定相机；单目前视 eval 建议填写 cam0，默认拉 pkl 当前帧全部相机")
    parser.add_argument("--start-index", type=int, default=0,
                        help="从第几个 info 开始，语义与可视化脚本一致")
    parser.add_argument("--stride", type=int, default=1,
                        help="每隔多少个 info 拉一帧，语义与可视化脚本一致")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="最多选择多少个 info；负数表示全部")
    parser.add_argument("--path-prefix", default=None,
                        help="pkl data_path 为绝对路径时，先剥离此前缀再映射到 Ozone；新版 N7 pkl 通常不需要")
    parser.add_argument("--rclone", default="rclone",
                        help="rclone 或 rclone.exe 路径")
    parser.add_argument("--rclone-config", type=Path, default=None,
                        help="可选 rclone.conf 路径；不传时使用 rclone 的 Windows 默认配置目录")
    parser.add_argument("--transfers", type=int, default=32,
                        help="rclone 并发传输数；Windows 本地不建议直接使用服务器侧 128")
    parser.add_argument("--checkers", type=int, default=64,
                        help="rclone 并发检查数")
    parser.add_argument("--retries", type=int, default=3,
                        help="rclone 高层重试次数")
    parser.add_argument("--low-level-retries", type=int, default=10,
                        help="rclone 底层重试次数")
    parser.add_argument("--progress", action="store_true",
                        help="显示 rclone 实时进度")
    parser.add_argument("--overwrite", action="store_true",
                        help="重新下载所选图片；默认跳过尺寸已经符合 pkl 的本地图片")
    parser.add_argument("--no-resize", action="store_true",
                        help="保留 Ozone 源尺寸；源图尺寸和 pkl image_* 不同时会导致现有可视化脚本拒绝读取")
    parser.add_argument("--resize-workers", type=int, default=4,
                        help="下载后并行 resize 线程数")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="resize 后 JPEG 保存质量")
    parser.add_argument("--dry-run", action="store_true",
                        help="生成 manifest 并打印 rclone 命令，不下载、不 resize")
    args = parser.parse_args()
    if args.stride <= 0:
        parser.error("--stride 必须大于 0")
    if args.transfers <= 0 or args.checkers <= 0:
        parser.error("--transfers 和 --checkers 必须大于 0")
    if args.resize_workers <= 0:
        parser.error("--resize-workers 必须大于 0")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality 必须在 1 到 100 之间")
    return args


def load_infos(pkl_path: Path) -> Tuple[List[Dict], Dict]:
    """读取 converter 新格式 pkl，也兼容只保存 info list 的旧格式。"""
    with pkl_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)
    if isinstance(payload, dict):
        infos = payload.get("infos", [])
        metadata = payload.get("metadata", {})
    elif isinstance(payload, list):
        infos = payload
        metadata = {}
    else:
        raise TypeError("不支持的 pkl payload 类型：{}".format(type(payload).__name__))
    if not isinstance(infos, list):
        raise TypeError("pkl infos 必须是 list，实际为：{}".format(type(infos).__name__))
    return infos, metadata if isinstance(metadata, dict) else {}


def selected_infos(
    infos: Sequence[Dict], start_index: int, stride: int, max_frames: int
) -> Iterable[Tuple[int, Dict]]:
    """使用和 visualize_n7_fastbev_pkl.py 相同的下标选择语义。"""
    count = 0
    for index in range(max(start_index, 0), len(infos), stride):
        if max_frames >= 0 and count >= max_frames:
            break
        yield index, infos[index]
        count += 1


def normalize_prefix(prefix: Optional[str]) -> Optional[str]:
    if prefix is None:
        return None
    normalized = str(prefix).replace("\\", "/").rstrip("/")
    return normalized or None


def relative_data_path(data_path: object, path_prefix: Optional[str]) -> str:
    """把 pkl 路径安全地转换成 rclone ``--files-from`` 相对路径。"""
    text = str(data_path or "").strip().replace("\\", "/")
    if not text:
        raise ValueError("camera info 缺少 data_path")

    prefix = normalize_prefix(path_prefix)
    if prefix is not None:
        lower_text = text.lower()
        lower_prefix = prefix.lower()
        if lower_text == lower_prefix:
            text = ""
        elif lower_text.startswith(lower_prefix + "/"):
            text = text[len(prefix):].lstrip("/")
        elif text.startswith("/") or re.match(r"^[A-Za-z]:/", text):
            raise ValueError(
                "绝对 data_path 不在 --path-prefix 下：data_path={!r}, prefix={!r}".format(
                    data_path, path_prefix))

    if text.startswith("/") or re.match(r"^[A-Za-z]:/", text):
        raise ValueError(
            "pkl data_path 是绝对路径，请用 --path-prefix 指定要剥离的数据根：{}".format(
                data_path))

    parts = [part for part in PurePosixPath(text).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("不安全或为空的 data_path：{}".format(data_path))
    return PurePosixPath(*parts).as_posix()


def camera_target_size(cam_info: Dict) -> Tuple[int, int]:
    """读取 pkl 中 data_path 图片应有的尺寸；旧 pkl 缺字段时返回 0。"""
    width = int(cam_info.get("image_width", 0) or 0)
    height = int(cam_info.get("image_height", 0) or 0)
    if (width <= 0) != (height <= 0):
        raise ValueError("image_width/image_height 必须同时为正数或同时缺失")
    return max(width, 0), max(height, 0)


def collect_image_specs(
    selected: Sequence[Tuple[int, Dict]],
    camera_ids: Optional[Sequence[str]],
    path_prefix: Optional[str],
) -> Tuple[List[ImageSpec], Dict[str, int]]:
    """从所选 key frame 收集图片并去重。"""
    specs_by_path: Dict[str, ImageSpec] = {}
    missing_camera_counts: Dict[str, int] = {}
    requested = [str(item) for item in camera_ids] if camera_ids else None

    for pkl_index, info in selected:
        cams = info.get("cams") or {}
        if not isinstance(cams, dict):
            raise TypeError("infos[{}].cams 不是 dict".format(pkl_index))
        current_ids = requested if requested is not None else list(cams.keys())
        for camera_id in current_ids:
            if camera_id not in cams:
                missing_camera_counts[camera_id] = missing_camera_counts.get(camera_id, 0) + 1
                continue
            cam_info = cams[camera_id]
            relative_path = relative_data_path(cam_info.get("data_path"), path_prefix)
            width, height = camera_target_size(cam_info)
            spec = ImageSpec(relative_path=relative_path, width=width, height=height)
            old = specs_by_path.get(relative_path)
            if old is not None and (old.width, old.height) != (width, height):
                raise ValueError(
                    "同一图片在 pkl 中出现不同 image size：{} -> {}x{} / {}x{}".format(
                        relative_path, old.width, old.height, width, height))
            specs_by_path[relative_path] = spec

    specs = [specs_by_path[key] for key in sorted(specs_by_path)]
    return specs, missing_camera_counts


def local_path(output_root: Path, relative_path: str) -> Path:
    """把 POSIX manifest 路径映射成当前平台本地路径。"""
    return output_root.joinpath(*PurePosixPath(relative_path).parts)


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    """只读取图片 header；无法读取时返回 None。"""
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (OSError, ValueError):
        return None


def image_is_ready(path: Path, spec: ImageSpec, no_resize: bool) -> bool:
    if not path.is_file():
        return False
    if no_resize or spec.width <= 0 or spec.height <= 0:
        return image_size(path) is not None
    return image_size(path) == (spec.width, spec.height)


def manifest_path(output_root: Path, gt_pkl: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", gt_pkl.stem)[:100]
    directory = output_root / "_ozone_eval_manifests"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "{}_{}.txt".format(safe_stem, timestamp)


def write_manifest(path: Path, specs: Sequence[ImageSpec]) -> None:
    path.write_text("".join(spec.relative_path + "\n" for spec in specs), encoding="utf-8")


def rclone_command(args: argparse.Namespace, manifest: Path) -> List[str]:
    command = [
        args.rclone,
        "copy",
        args.remote_root.rstrip("/"),
        str(args.output_root),
        "--files-from",
        str(manifest),
        "--no-traverse",
        "--transfers",
        str(args.transfers),
        "--checkers",
        str(args.checkers),
        "--retries",
        str(args.retries),
        "--low-level-retries",
        str(args.low_level_retries),
    ]
    if args.rclone_config is not None:
        command.extend(["--config", str(args.rclone_config.expanduser().resolve())])
    if args.overwrite:
        # rclone copy 默认会跳过它认为相同的文件；overwrite 语义要求强制重传。
        command.append("--ignore-times")
    if args.progress:
        command.append("--progress")
    return command


def printable_command(command: Sequence[str]) -> str:
    """只用于日志，兼容 Windows 路径中可能出现的空格。"""
    return subprocess.list2cmdline([str(item) for item in command])


def ensure_rclone(executable: str) -> None:
    if Path(executable).is_file() or shutil.which(executable):
        return
    raise FileNotFoundError(
        "找不到 rclone：{}。请安装 rclone 并加入 PATH，或传 --rclone C:\\\\path\\\\rclone.exe".format(
            executable))


def pil_bilinear_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def save_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    if suffix == ".bmp":
        return "BMP"
    return "JPEG"


def resize_one(path: Path, spec: ImageSpec, jpeg_quality: int) -> str:
    """必要时原位缩放，并用 os.replace 避免留下半张目标图片。"""
    current_size = image_size(path)
    if current_size is None:
        raise OSError("图片不存在或无法读取：{}".format(path))
    target_size = (spec.width, spec.height)
    if spec.width <= 0 or spec.height <= 0 or current_size == target_size:
        return "unchanged"

    file_format = save_format(path)
    temp_path = path.with_name(path.stem + ".resize_tmp" + path.suffix)
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize(target_size, pil_bilinear_resample())
            if file_format == "JPEG":
                image.save(temp_path, format=file_format, quality=jpeg_quality, optimize=False)
            else:
                image.save(temp_path, format=file_format)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return "resized"


def resize_downloaded_images(
    output_root: Path,
    specs: Sequence[ImageSpec],
    workers: int,
    jpeg_quality: int,
) -> Dict[str, int]:
    counters = {"resized": 0, "unchanged": 0, "failed": 0}
    errors: List[str] = []

    def work(spec: ImageSpec) -> Tuple[str, Optional[str]]:
        path = local_path(output_root, spec.relative_path)
        try:
            return resize_one(path, spec, jpeg_quality), None
        except Exception as exc:  # 保留所有帧的批处理能力，在末尾统一失败
            return "failed", "{}: {}".format(path, exc)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(work, spec): spec for spec in specs}
        for done, future in enumerate(as_completed(future_map), 1):
            status, error = future.result()
            counters[status] += 1
            if error and len(errors) < 20:
                errors.append(error)
            if done == len(specs) or done % 500 == 0:
                print("[resize] {}/{} resized={} unchanged={} failed={}".format(
                    done, len(specs), counters["resized"], counters["unchanged"], counters["failed"]))

    if counters["failed"]:
        raise RuntimeError(
            "有 {} 张图片 resize 失败，前几个错误：\n{}".format(
                counters["failed"], "\n".join(errors)))
    return counters


def verify_images(
    output_root: Path,
    specs: Sequence[ImageSpec],
    no_resize: bool,
) -> Tuple[int, List[str]]:
    failed: List[str] = []
    for spec in specs:
        path = local_path(output_root, spec.relative_path)
        if not image_is_ready(path, spec, no_resize):
            actual = image_size(path)
            expected = "readable" if no_resize or spec.width <= 0 else "{}x{}".format(
                spec.width, spec.height)
            failed.append("{} expected={} actual={}".format(path, expected, actual))
    return len(specs) - len(failed), failed


def write_summary(path: Path, summary: Dict) -> Path:
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    gt_pkl = args.gt_pkl.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not gt_pkl.is_file():
        raise FileNotFoundError("找不到 --gt-pkl：{}".format(gt_pkl))
    output_root.mkdir(parents=True, exist_ok=True)

    infos, metadata = load_infos(gt_pkl)
    selected = list(selected_infos(infos, args.start_index, args.stride, args.max_frames))
    specs, missing_camera_counts = collect_image_specs(
        selected, args.camera_ids, args.path_prefix)
    if not specs:
        raise RuntimeError("没有从所选 info 收集到图片，请检查 --camera-ids 和 pkl 内容")

    ready_specs = []
    download_specs = []
    for spec in specs:
        path = local_path(output_root, spec.relative_path)
        if not args.overwrite and image_is_ready(path, spec, args.no_resize):
            ready_specs.append(spec)
        else:
            download_specs.append(spec)

    manifest = manifest_path(output_root, gt_pkl)
    write_manifest(manifest, download_specs)
    print("[pkl] {} infos={} selected_infos={}".format(gt_pkl, len(infos), len(selected)))
    print("[camera] requested={} metadata={} missing={}".format(
        args.camera_ids or "all",
        metadata.get("camera_ids", "unknown"),
        missing_camera_counts or "none"))
    print("[images] unique={} ready={} need_download={}".format(
        len(specs), len(ready_specs), len(download_specs)))
    print("[manifest] {}".format(manifest))

    resize_counters = {"resized": 0, "unchanged": 0, "failed": 0}
    if download_specs:
        command = rclone_command(args, manifest)
        print("[rclone] {}".format(printable_command(command)))
        if not args.dry_run:
            ensure_rclone(args.rclone)
            subprocess.run(command, check=True)
            if not args.no_resize:
                resize_counters = resize_downloaded_images(
                    output_root, download_specs, args.resize_workers, args.jpeg_quality)
    else:
        print("[rclone] 所选图片均已存在且尺寸正确，无需下载")

    verified = 0
    failures: List[str] = []
    if not args.dry_run:
        verified, failures = verify_images(output_root, specs, args.no_resize)
        print("[verify] ok={}/{} failed={}".format(verified, len(specs), len(failures)))
        if failures:
            for error in failures[:20]:
                print("[verify-fail] {}".format(error), file=sys.stderr)

    summary = {
        "created": dt.datetime.now().isoformat(),
        "gt_pkl": str(gt_pkl),
        "remote_root": args.remote_root,
        "output_root": str(output_root),
        "camera_ids": args.camera_ids,
        "total_infos": len(infos),
        "selected_infos": len(selected),
        "unique_images": len(specs),
        "already_ready": len(ready_specs),
        "download_manifest_images": len(download_specs),
        "resize": resize_counters,
        "verified_images": verified,
        "verify_failures": len(failures),
        "dry_run": bool(args.dry_run),
        "manifest": str(manifest),
    }
    summary_path = write_summary(manifest, summary)
    print("[summary] {}".format(summary_path))
    if failures:
        return 2
    print("[done] 下载目录可直接作为 visualize_n7_fastbev_pkl.py 的 --data-root")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("用户中断")
