#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 N7_704_256 中已经存在的 3D_OD 标签复制回 N7_raw_tmp。

使用场景：
    你已经把 1600x900 原图完整下载到 data/N7_raw_tmp，但这个目录里没有 output 标签；
    同时 data/N7_704_256 里保留了 labels/output 目录。此时可以运行本脚本，
    只把标签目录复制回 raw tmp，使 N7_raw_tmp 能直接作为 1600x900 训练数据根目录。

目录约定：
    源标签：data/N7_704_256/<day>/<sequence>/output/<clip>/3D_OD/lidar/*.json
    目标标签：data/N7_raw_tmp/<day>/<sequence>/output/<clip>/3D_OD/lidar/*.json

示例：
    # 先预览某一天会复制多少标签。
    python tools/data_converter/n7/copy_n7_labels_to_raw_tmp.py \
      --days 20251203 \
      --dry-run

    # 把四天标签复制回 raw tmp，默认跳过已经存在的 json。
    python tools/data_converter/n7/copy_n7_labels_to_raw_tmp.py \
      --days 20251017 20251030 20251031 20251203

    # 只复制一个 sequence。
    python tools/data_converter/n7/copy_n7_labels_to_raw_tmp.py \
      --sequences 20251203/20251203_151928

    # 只复制一个 clip；clip 可以写 parsed_data 或 output 形式。
    python tools/data_converter/n7/copy_n7_labels_to_raw_tmp.py \
      --clips 20251203/20251203_151928/parsed_data/20251203_151928_16
"""

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

LABEL_EXTS = (".json",)


@dataclass
class LabelUnit:
    """一个 clip 的标签复制任务。"""
    name: str
    src_lidar: Path
    dst_lidar: Path


def format_seconds(seconds):
    """把耗时格式化成便于阅读的字符串。"""
    seconds = float(seconds)
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return "{}m{:.0f}s".format(int(minutes), sec)
    hours, minutes = divmod(minutes, 60)
    return "{}h{}m".format(int(hours), int(minutes))


def is_label_file(path):
    """判断文件是否是标签 json。"""
    return Path(path).suffix.lower() in LABEL_EXTS


def parse_clip_scope(clip_scope):
    """解析 clip 参数，兼容 parsed_data/output/简写三种形式。"""
    parts = clip_scope.strip("/").split("/")
    if len(parts) == 3:
        day, sequence, clip = parts
        return day, sequence, clip
    if len(parts) >= 4 and parts[2] in ("parsed_data", "output"):
        day, sequence, _, clip = parts[:4]
        return day, sequence, clip
    raise ValueError(
        "clip 格式应为 day/sequence/clip、day/sequence/parsed_data/clip "
        "或 day/sequence/output/clip：{}".format(clip_scope)
    )


def make_unit(src_root, dst_root, day, sequence, clip):
    """根据 day/sequence/clip 构造标签复制任务。"""
    rel_lidar = Path(day) / sequence / "output" / clip / "3D_OD" / "lidar"
    return LabelUnit(
        name="{}/{}/{}".format(day, sequence, clip),
        src_lidar=Path(src_root) / rel_lidar,
        dst_lidar=Path(dst_root) / rel_lidar,
    )


def list_sequences(src_root, day):
    """列出源标签根目录下某一天的 sequence。"""
    day_dir = Path(src_root) / day
    if not day_dir.exists():
        print("[list-warn] day 不存在，跳过：{}".format(day_dir), flush=True)
        return []
    return sorted(p.name for p in day_dir.iterdir() if p.is_dir())


def list_clips(src_root, day, sequence):
    """列出源标签根目录下某个 sequence 的 clip。"""
    output_dir = Path(src_root) / day / sequence / "output"
    if not output_dir.exists():
        print("[list-warn] output 不存在，跳过：{}".format(output_dir), flush=True)
        return []
    clips = []
    for clip_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        lidar_dir = clip_dir / "3D_OD" / "lidar"
        if lidar_dir.exists():
            clips.append(clip_dir.name)
    return clips


def expand_units(args):
    """把 days/sequences/clips 参数展开成 clip 级复制任务。"""
    units = []
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if args.days:
        for day in args.days:
            day = day.strip("/")
            sequences = list_sequences(src_root, day)
            print("[list] {}: {} sequences".format(day, len(sequences)), flush=True)
            for sequence in sequences:
                clips = list_clips(src_root, day, sequence)
                print("[list] {}/{}: {} clips".format(day, sequence, len(clips)), flush=True)
                for clip in clips:
                    units.append(make_unit(src_root, dst_root, day, sequence, clip))

    if args.sequences:
        for sequence_scope in args.sequences:
            parts = sequence_scope.strip("/").split("/")
            if len(parts) != 2:
                raise ValueError("sequence 格式应为 day/sequence：{}".format(sequence_scope))
            day, sequence = parts
            clips = list_clips(src_root, day, sequence)
            print("[list] {}/{}: {} clips".format(day, sequence, len(clips)), flush=True)
            for clip in clips:
                units.append(make_unit(src_root, dst_root, day, sequence, clip))

    if args.clips:
        for clip_scope in args.clips:
            day, sequence, clip = parse_clip_scope(clip_scope)
            units.append(make_unit(src_root, dst_root, day, sequence, clip))

    # 去重，避免同时传 days 和 sequences 时重复复制。
    deduped = []
    seen = set()
    for unit in units:
        key = unit.name
        if key in seen:
            continue
        seen.add(key)
        deduped.append(unit)
    return deduped


def copy_label_unit(unit, args):
    """复制一个 clip 的所有 lidar json 标签。"""
    start = time.time()
    if not unit.src_lidar.exists():
        message = "[missing] 源标签目录不存在：{}".format(unit.src_lidar)
        if args.allow_missing:
            print(message, flush=True)
            return {"missing": 1, "copied": 0, "skip": 0, "overwrite": 0, "fail": 0, "total": 0}
        raise FileNotFoundError(message)

    label_files = sorted(p for p in unit.src_lidar.iterdir() if p.is_file() and is_label_file(p))
    counters = {"missing": 0, "copied": 0, "skip": 0, "overwrite": 0, "fail": 0, "total": len(label_files)}
    if not label_files:
        print("[warn] 标签目录为空：{}".format(unit.src_lidar), flush=True)
        return counters

    print("[clip] {} labels={} : {} -> {}".format(
        unit.name, len(label_files), unit.src_lidar, unit.dst_lidar), flush=True)

    if args.dry_run:
        print("[dry-run] 将复制 {} 个标签到 {}".format(len(label_files), unit.dst_lidar), flush=True)
        return counters

    unit.dst_lidar.mkdir(parents=True, exist_ok=True)
    for index, src in enumerate(label_files, 1):
        dst = unit.dst_lidar / src.name
        try:
            if dst.exists():
                if not args.overwrite:
                    counters["skip"] += 1
                    continue
                counters["overwrite"] += 1
            else:
                counters["copied"] += 1
            shutil.copy2(src, dst)
        except Exception as exc:
            counters["fail"] += 1
            print("[copy-fail] {} -> {}: {}".format(src, dst, exc), flush=True)
        if args.progress_interval > 0 and index % args.progress_interval == 0:
            print("[progress] {} {}/{} copied={} skip={} overwrite={} fail={}".format(
                unit.name, index, len(label_files), counters["copied"], counters["skip"],
                counters["overwrite"], counters["fail"]), flush=True)

    elapsed = format_seconds(time.time() - start)
    print("[clip-done] {} elapsed={} copied={} skip={} overwrite={} fail={}".format(
        unit.name, elapsed, counters["copied"], counters["skip"],
        counters["overwrite"], counters["fail"]), flush=True)
    return counters


def merge_counters(total, part):
    """汇总多个 clip 的复制统计。"""
    for key, value in part.items():
        total[key] = total.get(key, 0) + int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="把 N7_704_256 中的 3D_OD 标签复制回 N7_raw_tmp。")
    parser.add_argument("--src-root", default="data/N7_704_256", help="已有标签的源数据根目录")
    parser.add_argument("--dst-root", default="data/N7_raw_tmp", help="需要补标签的目标 raw tmp 根目录")
    parser.add_argument("--days", nargs="*", default=None, help="按 day 复制，例如 20251203")
    parser.add_argument("--sequences", nargs="*", default=None, help="按 sequence 复制，例如 20251203/20251203_151928")
    parser.add_argument("--clips", nargs="*", default=None, help="按 clip 复制，支持 day/sequence/clip 或 parsed_data/output 形式")
    parser.add_argument("--overwrite", action="store_true", help="覆盖目标目录中已经存在的 json；默认跳过")
    parser.add_argument("--allow-missing", action="store_true", help="源标签目录缺失时只警告不失败")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要复制的内容，不实际写入")
    parser.add_argument("--progress-interval", type=int, default=5000, help="每复制多少个标签打印一次 clip 内进度；0 表示关闭")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.days and not args.sequences and not args.clips:
        raise SystemExit("必须指定 --days、--sequences 或 --clips 至少一种")

    print("[root] src={} dst={}".format(args.src_root, args.dst_root), flush=True)
    units = expand_units(args)
    print("[plan] {} clip units overwrite={} dry_run={}".format(
        len(units), args.overwrite, args.dry_run), flush=True)

    totals = {}
    failed_units = 0
    for unit in units:
        try:
            counters = copy_label_unit(unit, args)
        except Exception as exc:
            failed_units += 1
            print("[unit-fail] {}: {}".format(unit.name, exc), flush=True)
            continue
        merge_counters(totals, counters)
        if counters.get("fail", 0) > 0:
            failed_units += 1

    print("[done] totals={} failed_units={}".format(totals, failed_units), flush=True)
    if failed_units:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
