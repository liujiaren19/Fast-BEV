#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ozone 批量拉取 N7 图片和 3D_OD 标签，不做 resize。

职责边界：
    这个脚本只负责网络下载，不检查训练可用性，也不做图片尺寸变换。下载完成后：
    1. 如果希望离线缓存到 704x256 或 1600x900，再运行 tools/data_converter/n7/resize_n7_raw_tmp.py。
    2. 如果 ozone 源图已经是 1600x900 且想直接训练，可把 --output-root 设为
       data/N7_raw_tmp，让标签和原图落在同一个 data root，然后 converter 直接
       使用 data/N7_raw_tmp。

尺寸约定：
    --source-image-size 只用于日志和交接说明，不会改动文件。当前四天高快数据
    是从 N7 4K 原始图预处理到 1600x900 后上传到 ozone，因此默认写成
    HEIGHT=900, WIDTH=1600。后续如果直接拉 2560x1440 原图，需要显式传
    --source-image-size 1440 2560，并在后续强制做离线缓存。

目录约定：
    原始图片临时目录：data/N7_raw_tmp/<day>/<sequence>/parsed_data/<clip>/frames/.../images/...
    标签输出目录：    <output-root>/<day>/<sequence>/output/<clip>/3D_OD/lidar/*.json

示例：
    # 拉当前 ozone 中已经预处理到 1600x900 的四天数据，标签写入 704 缓存根目录。
    python tools/data_converter/n7/pull_n7_raw_from_ozone.py \
      --remote-root ozone:/gcy-truevalplat/liujiaren/data/nuscenes_20251017_20251030_20251031_20251203/nuscenes \
      --days 20251017 20251030 20251031 20251203 \
      --source-image-size 900 1600 \
      --rclone-transfers 128 \
      --rclone-checkers 256

    # 如果不做离线缓存，直接用 1600x900 原图训练/convert，则让标签也落到 raw_tmp。
    python tools/data_converter/n7/pull_n7_raw_from_ozone.py \
      --remote-root ozone:/.../nuscenes \
      --days 20251017 \
      --output-root data/N7_raw_tmp \
      --source-image-size 900 1600

    # 只拉某个 sequence。
    python tools/data_converter/n7/pull_n7_raw_from_ozone.py \
      --remote-root ozone:/.../nuscenes \
      --sequences 20251017/20251017_143548
"""

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LABEL_EXTS = (".json",)


@dataclass
class WorkUnit:
    """一个下载单元，通常是 sequence 或 clip。"""
    name: str
    mode: str
    image_src: str
    image_tmp: Path
    image_out: Path
    image_filter: str
    label_src: str
    label_out: Path
    label_filter: str


def format_seconds(seconds):
    """把耗时格式化成便于日志观察的字符串。"""
    seconds = float(seconds)
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return "{}m{:.0f}s".format(int(minutes), sec)
    hours, minutes = divmod(minutes, 60)
    return "{}h{}m".format(int(hours), int(minutes))


def run_capture(cmd):
    """执行轻量命令并返回 stdout。"""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("命令执行失败：{}\n{}".format(
            " ".join(str(x) for x in cmd), proc.stderr.decode("utf-8", "ignore")))
    return proc.stdout.decode("utf-8", "ignore")


def run_logged(cmd, dry_run=False):
    """执行长时间命令，直接把 rclone 输出交给终端。"""
    if dry_run:
        print("[dry-run] {}".format(" ".join(str(x) for x in cmd)))
        return
    print("[run] {}".format(" ".join(str(x) for x in cmd)))
    subprocess.run(cmd, check=True)


def remote_join(*parts):
    """拼接 rclone 远端路径，避免重复或遗漏斜杠。"""
    cleaned = []
    for idx, part in enumerate(parts):
        text = str(part)
        cleaned.append(text.rstrip("/") if idx == 0 else text.strip("/"))
    return "/".join(x for x in cleaned if x)


def list_remote_dirs(remote_path):
    """列出远端一级子目录。"""
    out = run_capture(["rclone", "lsf", remote_path, "--dirs-only"])
    return [line.strip().strip("/") for line in out.splitlines() if line.strip()]


def is_image_file(path):
    """判断本地文件是否是图片。"""
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_frame_image_path(path):
    """判断路径是否位于 frames/<lidar_ts>/images 目录下。"""
    normalized = str(path).replace("\\", "/")
    parts = normalized.split("/")
    return is_image_file(normalized) and "frames" in parts and "images" in parts


def count_local_images_until(root, limit):
    """粗略统计本地图片数量，达到 limit 后提前返回。"""
    root = Path(root)
    if not root.exists():
        return 0
    limit = max(1, int(limit))
    count = 0
    for file_path in root.rglob("*"):
        if file_path.is_file() and is_frame_image_path(file_path):
            count += 1
            if count >= limit:
                return count
    return count


def count_local_labels(root):
    """统计标签 json 数量。"""
    root = Path(root)
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in LABEL_EXTS)


def rclone_copy_filtered(src, dst, include_pattern, args):
    """用 rclone copy 批量复制匹配 include_pattern 的文件。"""
    cmd = [
        "rclone", "copy", src, str(dst),
        "--filter", "+ {}".format(include_pattern),
        "--filter", "- **",
        "--transfers", str(args.rclone_transfers),
        "--checkers", str(args.rclone_checkers),
        "--retries", str(args.rclone_retries),
        "--low-level-retries", str(args.rclone_low_level_retries),
    ]
    if args.fast_list:
        cmd.append("--fast-list")
    if args.rclone_progress:
        cmd.append("--progress")
    run_logged(cmd, dry_run=args.dry_run)


def parse_clip_scope(clip_scope):
    """解析 day/sequence/parsed_data/clip 格式。"""
    parts = clip_scope.strip("/").split("/")
    if len(parts) < 4 or parts[2] != "parsed_data":
        raise ValueError("clip 必须是 day/sequence/parsed_data/clip 格式：{}".format(clip_scope))
    return parts[0], parts[1], parts[3]


def make_sequence_unit(args, sequence_scope):
    """构造 sequence 下载单元。"""
    sequence_scope = sequence_scope.strip("/")
    return WorkUnit(
        name=sequence_scope,
        mode="sequence",
        image_src=remote_join(args.remote_root, sequence_scope, "parsed_data"),
        image_tmp=Path(args.raw_tmp_root) / sequence_scope / "parsed_data",
        image_out=Path(args.output_root) / sequence_scope / "parsed_data",
        image_filter="**/frames/**/images/**",
        label_src=remote_join(args.remote_root, sequence_scope, "output"),
        label_out=Path(args.output_root) / sequence_scope / "output",
        label_filter="*/3D_OD/lidar/*.json",
    )


def make_clip_unit(args, clip_scope):
    """构造 clip 下载单元。"""
    clip_scope = clip_scope.strip("/")
    day, sequence, clip = parse_clip_scope(clip_scope)
    label_scope = "/".join([day, sequence, "output", clip, "3D_OD", "lidar"])
    return WorkUnit(
        name=clip_scope,
        mode="clip",
        image_src=remote_join(args.remote_root, clip_scope),
        image_tmp=Path(args.raw_tmp_root) / clip_scope,
        image_out=Path(args.output_root) / clip_scope,
        image_filter="frames/**/images/**",
        label_src=remote_join(args.remote_root, label_scope),
        label_out=Path(args.output_root) / label_scope,
        label_filter="*.json",
    )


def expand_work_units(args):
    """根据 days / sequences / clips 展开下载单元。"""
    units = []
    if args.clips:
        units.extend(make_clip_unit(args, clip) for clip in args.clips)
    if args.sequences:
        units.extend(make_sequence_unit(args, sequence) for sequence in args.sequences)
    if args.days:
        for day in args.days:
            day = day.strip("/")
            seq_names = list_remote_dirs(remote_join(args.remote_root, day))
            print("[list] {}: {} sequences".format(day, len(seq_names)))
            for seq_name in seq_names:
                units.append(make_sequence_unit(args, "/".join([day, seq_name])))
    return units


def should_skip_existing_pull(unit, args):
    """粗粒度跳过已经拉过或已经 resize 完成的下载单元。"""
    if not args.skip_existing_units or args.overwrite:
        return False
    image_ready = True
    label_ready = True
    raw_image_count = 0
    resized_image_count = 0
    label_count = 0
    if not args.labels_only:
        raw_image_count = count_local_images_until(unit.image_tmp, args.min_existing_images)
        resized_image_count = count_local_images_until(unit.image_out, args.min_existing_images)
        # 分阶段处理时，raw 可能已经被删除；只要最终 704x256 图片已经存在，也不需要重新拉原图。
        image_ready = max(raw_image_count, resized_image_count) >= args.min_existing_images
    if not args.images_only:
        label_count = count_local_labels(unit.label_out)
        label_ready = label_count > 0
    if image_ready and label_ready:
        print("[skip-existing] {} raw_images_seen={} resized_images_seen={} label_json={}，跳过下载".format(
            unit.name, raw_image_count, resized_image_count, label_count))
        return True
    print("[skip-check] {} raw_images_seen={} resized_images_seen={} label_json={} -> 不跳过".format(
        unit.name, raw_image_count, resized_image_count, label_count))
    return False


def process_unit(unit, args):
    """下载一个 sequence 或 clip。"""
    start = time.time()
    print("\n[unit] {} ({})".format(unit.name, unit.mode))
    if should_skip_existing_pull(unit, args):
        print("[unit-done] {} skipped elapsed={}".format(unit.name, format_seconds(time.time() - start)))
        return

    if not args.labels_only:
        step_start = time.time()
        print("[copy-images] {} -> {}".format(unit.image_src, unit.image_tmp))
        rclone_copy_filtered(unit.image_src, unit.image_tmp, unit.image_filter, args)
        print("[copy-images-done] raw_images_seen={} elapsed={}".format(
            count_local_images_until(unit.image_tmp, args.min_existing_images),
            format_seconds(time.time() - step_start)))

    if not args.images_only:
        step_start = time.time()
        print("[copy-labels] {} -> {}".format(unit.label_src, unit.label_out))
        rclone_copy_filtered(unit.label_src, unit.label_out, unit.label_filter, args)
        print("[copy-labels-done] label_json={} elapsed={}".format(
            count_local_labels(unit.label_out), format_seconds(time.time() - step_start)))

    print("[unit-done] {} elapsed={}".format(unit.name, format_seconds(time.time() - start)))


def parse_args():
    parser = argparse.ArgumentParser(description="从 ozone 拉取 N7 原始图片和 3D_OD 标签，不做 resize。")
    parser.add_argument("--remote-root", required=True, help="ozone 上 nuscenes 根目录")
    parser.add_argument("--raw-tmp-root", default="data/N7_raw_tmp", help="远端原图临时输出根目录")
    parser.add_argument("--output-root", default="data/N7_704_256", help="3D_OD 标签输出根目录；如果不做离线缓存可设为 data/N7_raw_tmp")
    parser.add_argument("--days", nargs="*", default=None, help="按天处理，例如 20251017 20251030")
    parser.add_argument("--source-image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=(900, 1600), help="ozone 源图实际尺寸，仅用于日志说明；当前四天预处理数据默认 900 1600")
    parser.add_argument("--sequences", nargs="*", default=None, help="按 sequence 处理，例如 20251031/20251031_164821")
    parser.add_argument("--clips", nargs="*", default=None, help="按 clip 处理，格式 day/sequence/parsed_data/clip")
    parser.add_argument("--images-only", action="store_true", help="只拉图片，不同步标签")
    parser.add_argument("--labels-only", action="store_true", help="只同步标签，不拉图片")
    parser.add_argument("--skip-existing-units", action="store_true", help="如果 raw 图片和标签看起来已经存在，跳过整个 sequence/clip")
    parser.add_argument("--min-existing-images", type=int, default=100, help="skip 检查中至少看到多少张图片才认为图片已存在")
    parser.add_argument("--overwrite", action="store_true", help="传给 skip 逻辑使用；rclone 仍按自身规则判断是否覆盖")
    parser.add_argument("--rclone-transfers", type=int, default=128, help="rclone 并发传输数")
    parser.add_argument("--rclone-checkers", type=int, default=256, help="rclone 并发检查数")
    parser.add_argument("--rclone-retries", type=int, default=3, help="rclone 高层重试次数")
    parser.add_argument("--rclone-low-level-retries", type=int, default=10, help="rclone 底层重试次数")
    parser.add_argument("--no-fast-list", dest="fast_list", action="store_false", help="关闭 rclone --fast-list")
    parser.set_defaults(fast_list=True)
    parser.add_argument("--rclone-progress", action="store_true", help="显示 rclone 进度")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际执行")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.days and not args.sequences and not args.clips:
        raise SystemExit("必须指定 --days、--sequences 或 --clips 至少一种")
    if args.images_only and args.labels_only:
        raise SystemExit("--images-only 和 --labels-only 不能同时指定")
    units = expand_work_units(args)
    print("[plan] {} work units".format(len(units)))
    print("[root] raw_tmp={} output={}".format(args.raw_tmp_root, args.output_root))
    print("[source] ozone_image_size_hxw={}".format(tuple(args.source_image_size)))
    print("[rclone] transfers={} checkers={} fast_list={}".format(
        args.rclone_transfers, args.rclone_checkers, args.fast_list))
    for unit in units:
        process_unit(unit, args)
    print("[done] pull finished")


if __name__ == "__main__":
    main()
