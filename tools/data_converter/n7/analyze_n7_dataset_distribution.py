#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 N7 自采 3D_OD 数据集的标签分布，辅助划分 train/val/test。

这个脚本只读取本地已经拉取或离线缓存后的 N7 数据目录，不修改数据文件，
也不依赖 Fast-BEV pkl。它的目标是在生成 pkl 和正式训练之前，先回答几个
划分数据集时最关心的问题：

    - 每天、每个 sequence、每个 clip 有多少帧标签；
    - 每类目标有多少，目标密度是否均衡；
    - 目标距离、自车前后左右分布是否偏斜；
    - 哪些 sequence/clip 适合作为 val/test 的完整切分单元。

默认会统计所有可识别类别；如果只想关注当前乘用车任务，可以通过
``--classes car`` 或 ``--classes car truck`` 限制统计和画图关注的类别。
类别名默认使用 converter 中一致的 mapped 类别空间，例如 N7 的
``no_motor_bike`` 会映射为 ``bicycle``。

常用示例：

    # 分析四天数据，默认统计所有已知映射类别并生成 csv/json/png。
    python tools/data_converter/n7/analyze_n7_dataset_distribution.py \
        --data-root data/N7_704_256 \
        --datasets 20251017 20251030 20251031 20251203 \
        --output-dir work_dirs/n7_dataset_analysis

    # 只关注车辆两类，适合当前高快车辆检测训练划分。
    python tools/data_converter/n7/analyze_n7_dataset_distribution.py \
        --data-root data/N7_704_256 \
        --datasets 20251017 20251030 20251031 20251203 \
        --classes car truck \
        --output-dir work_dirs/n7_dataset_analysis_vehicle

    # 如果只想快速得到 csv/json，不需要画图。
    python tools/data_converter/n7/analyze_n7_dataset_distribution.py \
        --data-root data/N7_raw_tmp \
        --datasets 20251031 \
        --classes car \
        --no-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 和 tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py 保持一致的类别映射。这里复制一份，
# 避免分析脚本 import converter 时触发不必要的三方依赖或全局逻辑。
CLASS_MAPPING = {
    "car": "car",
    "smallMot": "car",
    "van": "truck",
    "van-less": "truck",
    "truck": "truck",
    "bigMot": "truck",
    "tanker": "truck",
    "bus": "bus",
    "otherMot": "construction_vehicle",
    "construction_vehicle": "construction_vehicle",
    "bicycle": "bicycle",
    "nonMot": "bicycle",
    "no_motor_bike": "bicycle",
    "tricycle": "bicycle",
    "rider": "motorcycle",
    "ride_person": "motorcycle",
    "motorcycle": "motorcycle",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
}

DEFAULT_DISTANCE_BINS = [0.0, 20.0, 50.0, 80.0, 120.0]


@dataclass(frozen=True)
class ClipRef:
    """一个可分析的 N7 clip 引用。"""

    dataset: str
    sequence: str
    clip: str
    label_dir: Path
    frames_dir: Optional[Path]


@dataclass
class ClipStats:
    """一个 clip 的统计结果。"""

    dataset: str
    sequence: str
    clip: str
    label_frames: int = 0
    image_frames: int = 0
    aligned_frames: int = 0
    frames_with_selected: int = 0
    frames_with_any_known: int = 0
    objects_total: int = 0
    objects_known: int = 0
    objects_unknown: int = 0
    objects_selected: int = 0
    invalid_box_size: int = 0
    raw_class_counts: Counter = field(default_factory=Counter)
    mapped_class_counts: Counter = field(default_factory=Counter)
    selected_class_counts: Counter = field(default_factory=Counter)
    distance_bin_counts: Counter = field(default_factory=Counter)
    direction_counts: Counter = field(default_factory=Counter)
    selected_objects_per_frame: List[int] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """保留说明中的换行，同时展示 argparse 默认值。"""


def natural_sort_key(value):
    """自然排序 key，避免 sequence/clip 的 10 排在 2 前面。"""

    parts = re.split(r"(\d+)", str(value))
    key = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def sorted_dir_paths(path: Path) -> List[Path]:
    """用 os.scandir 快速列出一级子目录，并按自然顺序排序。"""

    try:
        with os.scandir(path) as entries:
            return sorted(
                (Path(entry.path) for entry in entries if entry.is_dir()),
                key=lambda p: natural_sort_key(p.name),
            )
    except FileNotFoundError:
        return []


def sorted_json_files(path: Path) -> List[Path]:
    """用 os.scandir 快速列出 JSON 文件，并按自然顺序排序。"""

    try:
        with os.scandir(path) as entries:
            files = [
                Path(entry.path)
                for entry in entries
                if entry.is_file() and entry.name.lower().endswith('.json')
            ]
    except FileNotFoundError:
        return []
    return sorted(files, key=lambda p: natural_sort_key(p.name))


def has_json_files(path: Path) -> bool:
    """快速判断目录下是否至少有一个 JSON 文件。"""

    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.json'):
                    return True
    except FileNotFoundError:
        return False
    return False


def iter_recursive_label_dirs(base: Path) -> List[Path]:
    """兜底递归发现 3D_OD/lidar 标签目录；标准结构不会走到这里。"""

    label_dirs = []
    for dirpath, _, filenames in os.walk(base):
        path = Path(dirpath)
        if path.name != 'lidar' or path.parent.name != '3D_OD':
            continue
        if any(name.lower().endswith('.json') for name in filenames):
            label_dirs.append(path)
    return sorted(label_dirs, key=lambda p: natural_sort_key(p.as_posix()))


def load_json(path: Path) -> Dict:
    """读取单个 JSON 文件。"""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def annotation_payload(label: Dict) -> Dict:
    """兼容带 3d_od 包装和不带包装的 N7 标签 JSON。"""

    return label.get("3d_od", label)


def is_image_file(path: Path) -> bool:
    """判断是否是常见图片文件。"""

    return path.suffix.lower() in IMAGE_EXTS


def normalize_name_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    """清洗命令行传入的名称列表，保持顺序并去重。"""

    if not values:
        return None
    names = [x.strip() for x in values if x and x.strip()]
    return list(dict.fromkeys(names)) or None


def discover_datasets(data_root: Path, requested: Optional[Sequence[str]]) -> List[str]:
    """决定要扫描哪些 dataset/day 目录。"""

    requested = normalize_name_list(requested)
    if requested:
        return requested
    if not data_root.exists():
        raise SystemExit("data-root 不存在：{}".format(data_root))
    return [p.name for p in sorted_dir_paths(data_root)]


def resolve_frames_dir(base: Path, sequence: str, clip: str) -> Optional[Path]:
    """根据已知 N7 目录结构查找 clip 的 frames 目录。"""

    candidates = [
        base / sequence / "parsed_data" / clip / "frames",
        base / "parsed_data" / sequence / clip / "frames",
        base / "parsed_data" / clip / "frames",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_structured_refs(
    data_root: Path,
    dataset: str,
    requested_sequences: Optional[Sequence[str]],
    requested_clips: Optional[Sequence[str]],
) -> List[ClipRef]:
    """按 N7 常见层级结构发现 clip，避免无边界递归扫描。"""

    base = data_root / dataset
    if not base.exists():
        base = data_root

    requested_sequences = normalize_name_list(requested_sequences)
    requested_clips = normalize_name_list(requested_clips)
    sequence_filter = set(requested_sequences or [])
    clip_filter = set(requested_clips or [])
    refs: List[ClipRef] = []

    sequence_dirs = sorted_dir_paths(base)
    for sequence_dir in sequence_dirs:
        sequence = sequence_dir.name
        if requested_sequences and sequence not in sequence_filter:
            continue
        output_root = sequence_dir / "output"
        if not output_root.exists():
            continue
        for clip_dir in sorted_dir_paths(output_root):
            clip = clip_dir.name
            if requested_clips and clip not in clip_filter:
                continue
            label_dir = clip_dir / "3D_OD" / "lidar"
            if not label_dir.exists():
                continue
            if not has_json_files(label_dir):
                continue
            refs.append(ClipRef(
                dataset=dataset,
                sequence=sequence,
                clip=clip,
                label_dir=label_dir,
                frames_dir=resolve_frames_dir(base, sequence, clip),
            ))

    return refs


def discover_recursive_refs(
    data_root: Path,
    dataset: str,
    requested_sequences: Optional[Sequence[str]],
    requested_clips: Optional[Sequence[str]],
) -> List[ClipRef]:
    """兜底递归发现标签目录，兼容历史导出路径。"""

    base = data_root / dataset
    if not base.exists():
        base = data_root

    requested_sequences = normalize_name_list(requested_sequences)
    requested_clips = normalize_name_list(requested_clips)
    sequence_filter = set(requested_sequences or [])
    clip_filter = set(requested_clips or [])
    refs: List[ClipRef] = []

    for label_dir in iter_recursive_label_dirs(base):
        parts = label_dir.parts
        clip = label_dir.parents[1].name
        sequence = ""
        if "output" in parts:
            output_index = parts.index("output")
            # 常见结构：base/sequence/output/clip/3D_OD/lidar。
            if output_index > 0 and output_index + 1 < len(parts):
                sequence = parts[output_index - 1]
                clip = parts[output_index + 1]
            # 历史结构：base/output/sequence/clip/3D_OD/lidar。
            if output_index + 2 < len(parts) and parts[output_index + 2] == clip:
                sequence = parts[output_index + 1]
        if requested_sequences and sequence not in sequence_filter:
            continue
        if requested_clips and clip not in clip_filter:
            continue
        refs.append(ClipRef(
            dataset=dataset,
            sequence=sequence,
            clip=clip,
            label_dir=label_dir,
            frames_dir=resolve_frames_dir(base, sequence, clip),
        ))

    # 去重但保持路径排序后的稳定顺序。
    dedup: Dict[Tuple[str, str, str], ClipRef] = {}
    for ref in refs:
        dedup.setdefault((ref.dataset, ref.sequence, ref.clip), ref)
    return list(dedup.values())


def discover_refs(
    data_root: Path,
    dataset: str,
    requested_sequences: Optional[Sequence[str]],
    requested_clips: Optional[Sequence[str]],
) -> List[ClipRef]:
    """发现一个 dataset/day 下的所有 clip。"""

    refs = discover_structured_refs(data_root, dataset, requested_sequences, requested_clips)
    if refs:
        return refs
    return discover_recursive_refs(data_root, dataset, requested_sequences, requested_clips)


def list_frame_timestamps(frames_dir: Optional[Path]) -> set:
    """列出 frames 目录下的帧时间戳；不检查每帧相机图片是否齐全。"""

    if frames_dir is None:
        return set()
    try:
        with os.scandir(frames_dir) as entries:
            return {entry.name for entry in entries if entry.is_dir()}
    except FileNotFoundError:
        return set()


def label_timestamp(label_path: Path, payload: Dict) -> str:
    """从标签内容或文件名获取帧时间戳。"""

    value = payload.get("frame_timestamp")
    if value is None:
        return label_path.stem
    return str(int(value))


def raw_location_to_fastbev_xy(anno: Dict) -> Tuple[float, float]:
    """把 N7 原始 lidar 坐标中的位置转为 Fast-BEV 水平坐标。

    N7 原始坐标按 x 左、y 后、z 上理解；Fast-BEV lidar 坐标按 x 前、y 左、
    z 上理解。因此水平位置变换为 x_fast=-y_raw, y_fast=x_raw。分析脚本只
    用这个结果计算距离和前后左右方向，不改变原标签文件。
    """

    loc = anno.get("location", {}) or {}
    raw_x = float(loc.get("x", 0.0) or 0.0)
    raw_y = float(loc.get("y", 0.0) or 0.0)
    return -raw_y, raw_x


def object_direction(x: float, y: float) -> str:
    """按 Fast-BEV 水平坐标给目标粗分前后左右。"""

    if x >= abs(y):
        return "front"
    if -x >= abs(y):
        return "rear"
    if y >= 0:
        return "left"
    return "right"


def distance_bin_name(distance: float, bins: Sequence[float]) -> str:
    """把距离落入命令行配置的区间。"""

    if not bins:
        return "all"
    for left, right in zip(bins[:-1], bins[1:]):
        if left <= distance < right:
            return "{:.0f}-{:.0f}m".format(left, right)
    return ">={:.0f}m".format(bins[-1])


def valid_box_size(anno: Dict) -> bool:
    """检查 3D box 尺寸是否为正。"""

    size = anno.get("size", {}) or {}
    try:
        return float(size.get("l", 0.0)) > 0 and float(size.get("w", 0.0)) > 0 and float(size.get("h", 0.0)) > 0
    except Exception:
        return False


def selected_class_name(raw_name: str, mapped_name: Optional[str], args: argparse.Namespace) -> Optional[str]:
    """根据类别空间和 --classes 判断一个目标是否进入重点统计。"""

    if args.class_space == "raw":
        name = raw_name
    else:
        name = mapped_name
    if name is None:
        return None
    if args.classes and name not in args.classes:
        return None
    return name


def scan_clip(ref: ClipRef, args: argparse.Namespace) -> ClipStats:
    """读取一个 clip 的所有标签并统计分布。"""

    stats = ClipStats(dataset=ref.dataset, sequence=ref.sequence, clip=ref.clip)
    image_ts = list_frame_timestamps(ref.frames_dir)
    label_ts = set()

    for label_path in sorted_json_files(ref.label_dir):
        try:
            payload = annotation_payload(load_json(label_path))
        except Exception as exc:
            if len(stats.examples) < args.max_examples:
                stats.examples.append("{}:read_error={}".format(label_path.name, exc))
            continue

        ts = label_timestamp(label_path, payload)
        label_ts.add(ts)
        annotations = payload.get("annotations", []) or []
        selected_in_frame = 0
        known_in_frame = 0

        for anno in annotations:
            stats.objects_total += 1
            raw_name = str(anno.get("type", "unknown"))
            mapped_name = CLASS_MAPPING.get(raw_name)
            stats.raw_class_counts[raw_name] += 1
            if mapped_name is None:
                stats.objects_unknown += 1
                continue

            stats.objects_known += 1
            known_in_frame += 1
            stats.mapped_class_counts[mapped_name] += 1
            selected_name = selected_class_name(raw_name, mapped_name, args)
            if selected_name is None:
                continue

            stats.objects_selected += 1
            selected_in_frame += 1
            stats.selected_class_counts[selected_name] += 1

            if not valid_box_size(anno):
                stats.invalid_box_size += 1
            x, y = raw_location_to_fastbev_xy(anno)
            dist = math.hypot(x, y)
            stats.distance_bin_counts[distance_bin_name(dist, args.distance_bins)] += 1
            stats.direction_counts[object_direction(x, y)] += 1

        if known_in_frame > 0:
            stats.frames_with_any_known += 1
        if selected_in_frame > 0:
            stats.frames_with_selected += 1
        stats.selected_objects_per_frame.append(selected_in_frame)

    stats.label_frames = len(label_ts)
    stats.image_frames = len(image_ts)
    stats.aligned_frames = len(label_ts & image_ts) if image_ts else 0
    return stats


def counter_to_plain(counter: Counter) -> Dict[str, int]:
    """把 Counter 转成稳定排序的普通 dict，便于写 json。"""

    return {k: int(counter[k]) for k in sorted(counter, key=natural_sort_key)}


def merge_stats(rows: Iterable[ClipStats], key_fields: Sequence[str]) -> List[Dict]:
    """把 clip 级统计聚合到 sequence/day 等层级。"""

    grouped: Dict[Tuple[str, ...], Dict] = {}
    for row in rows:
        key = tuple(getattr(row, field_name) for field_name in key_fields)
        if key not in grouped:
            grouped[key] = {
                field_name: value for field_name, value in zip(key_fields, key)
            }
            grouped[key].update({
                "clips": 0,
                "label_frames": 0,
                "image_frames": 0,
                "aligned_frames": 0,
                "frames_with_selected": 0,
                "frames_with_any_known": 0,
                "objects_total": 0,
                "objects_known": 0,
                "objects_unknown": 0,
                "objects_selected": 0,
                "invalid_box_size": 0,
                "selected_class_counts": Counter(),
                "raw_class_counts": Counter(),
                "mapped_class_counts": Counter(),
                "distance_bin_counts": Counter(),
                "direction_counts": Counter(),
            })
        item = grouped[key]
        item["clips"] += 1
        for name in (
            "label_frames", "image_frames", "aligned_frames",
            "frames_with_selected", "frames_with_any_known",
            "objects_total", "objects_known", "objects_unknown",
            "objects_selected", "invalid_box_size",
        ):
            item[name] += int(getattr(row, name))
        item["selected_class_counts"].update(row.selected_class_counts)
        item["raw_class_counts"].update(row.raw_class_counts)
        item["mapped_class_counts"].update(row.mapped_class_counts)
        item["distance_bin_counts"].update(row.distance_bin_counts)
        item["direction_counts"].update(row.direction_counts)

    merged = []
    for item in grouped.values():
        frames = item["label_frames"]
        nonempty = item["frames_with_selected"]
        item["avg_selected_per_label_frame"] = item["objects_selected"] / frames if frames else 0.0
        item["avg_selected_per_nonempty_frame"] = item["objects_selected"] / nonempty if nonempty else 0.0
        merged.append(item)
    return sorted(merged, key=lambda x: tuple(natural_sort_key(x.get(k, "")) for k in key_fields))


def all_counter_keys(rows: Sequence[ClipStats], attr_name: str) -> List[str]:
    """收集若干 Counter 字段的所有 key。"""

    keys = set()
    for row in rows:
        keys.update(getattr(row, attr_name).keys())
    return sorted(keys, key=natural_sort_key)


def write_clip_csv(path: Path, rows: Sequence[ClipStats]) -> None:
    """写出 clip 级 CSV。"""

    selected_classes = all_counter_keys(rows, "selected_class_counts")
    distance_bins = all_counter_keys(rows, "distance_bin_counts")
    directions = ["front", "left", "rear", "right"]
    fieldnames = [
        "dataset", "sequence", "clip",
        "label_frames", "image_frames", "aligned_frames",
        "frames_with_selected", "frames_with_any_known",
        "objects_total", "objects_known", "objects_unknown", "objects_selected",
        "invalid_box_size",
        "avg_selected_per_label_frame", "avg_selected_per_nonempty_frame",
    ]
    fieldnames += ["class_{}".format(x) for x in selected_classes]
    fieldnames += ["dist_{}".format(x) for x in distance_bins]
    fieldnames += ["dir_{}".format(x) for x in directions]
    fieldnames += ["examples"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            nonempty = row.frames_with_selected
            record = {
                "dataset": row.dataset,
                "sequence": row.sequence,
                "clip": row.clip,
                "label_frames": row.label_frames,
                "image_frames": row.image_frames,
                "aligned_frames": row.aligned_frames,
                "frames_with_selected": row.frames_with_selected,
                "frames_with_any_known": row.frames_with_any_known,
                "objects_total": row.objects_total,
                "objects_known": row.objects_known,
                "objects_unknown": row.objects_unknown,
                "objects_selected": row.objects_selected,
                "invalid_box_size": row.invalid_box_size,
                "avg_selected_per_label_frame": "{:.6f}".format(row.objects_selected / row.label_frames if row.label_frames else 0.0),
                "avg_selected_per_nonempty_frame": "{:.6f}".format(row.objects_selected / nonempty if nonempty else 0.0),
                "examples": " | ".join(row.examples),
            }
            for name in selected_classes:
                record["class_{}".format(name)] = row.selected_class_counts.get(name, 0)
            for name in distance_bins:
                record["dist_{}".format(name)] = row.distance_bin_counts.get(name, 0)
            for name in directions:
                record["dir_{}".format(name)] = row.direction_counts.get(name, 0)
            writer.writerow(record)


def write_aggregate_csv(path: Path, rows: Sequence[Dict], key_fields: Sequence[str]) -> None:
    """写出 sequence/day 等聚合层级 CSV。"""

    class_keys = sorted({k for row in rows for k in row["selected_class_counts"].keys()}, key=natural_sort_key)
    fieldnames = list(key_fields) + [
        "clips", "label_frames", "image_frames", "aligned_frames",
        "frames_with_selected", "frames_with_any_known",
        "objects_total", "objects_known", "objects_unknown", "objects_selected",
        "invalid_box_size", "avg_selected_per_label_frame", "avg_selected_per_nonempty_frame",
    ]
    fieldnames += ["class_{}".format(x) for x in class_keys]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {name: row.get(name, "") for name in fieldnames}
            record["avg_selected_per_label_frame"] = "{:.6f}".format(row["avg_selected_per_label_frame"])
            record["avg_selected_per_nonempty_frame"] = "{:.6f}".format(row["avg_selected_per_nonempty_frame"])
            for name in class_keys:
                record["class_{}".format(name)] = row["selected_class_counts"].get(name, 0)
            writer.writerow(record)


def total_summary(rows: Sequence[ClipStats], args: argparse.Namespace) -> Dict:
    """生成 summary.json 的主体统计。"""

    selected_counts = Counter()
    raw_counts = Counter()
    mapped_counts = Counter()
    distance_counts = Counter()
    direction_counts = Counter()
    per_frame_counts: List[int] = []
    totals = Counter()
    for row in rows:
        selected_counts.update(row.selected_class_counts)
        raw_counts.update(row.raw_class_counts)
        mapped_counts.update(row.mapped_class_counts)
        distance_counts.update(row.distance_bin_counts)
        direction_counts.update(row.direction_counts)
        per_frame_counts.extend(row.selected_objects_per_frame)
        for name in (
            "label_frames", "image_frames", "aligned_frames",
            "frames_with_selected", "frames_with_any_known",
            "objects_total", "objects_known", "objects_unknown",
            "objects_selected", "invalid_box_size",
        ):
            totals[name] += int(getattr(row, name))

    sorted_counts = sorted(per_frame_counts)
    def percentile(p: float) -> float:
        if not sorted_counts:
            return 0.0
        idx = min(len(sorted_counts) - 1, max(0, int(round((len(sorted_counts) - 1) * p))))
        return float(sorted_counts[idx])

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(args.data_root),
        "datasets": args.datasets,
        "class_space": args.class_space,
        "selected_classes": args.classes or "all",
        "distance_bins": args.distance_bins,
        "num_clips": len(rows),
        "num_sequences": len({(x.dataset, x.sequence) for x in rows}),
        "totals": {k: int(v) for k, v in sorted(totals.items(), key=lambda x: natural_sort_key(x[0]))},
        "raw_class_counts": counter_to_plain(raw_counts),
        "mapped_class_counts": counter_to_plain(mapped_counts),
        "selected_class_counts": counter_to_plain(selected_counts),
        "distance_bin_counts": counter_to_plain(distance_counts),
        "direction_counts": counter_to_plain(direction_counts),
        "selected_objects_per_frame": {
            "mean": float(sum(per_frame_counts) / len(per_frame_counts)) if per_frame_counts else 0.0,
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": max(per_frame_counts) if per_frame_counts else 0,
        },
    }


def save_plot_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[int], xlabel: str, ylabel: str) -> None:
    """保存简单柱状图。"""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_width = max(8.0, min(18.0, 0.6 * max(1, len(labels))))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_plot_hist(path: Path, title: str, values: Sequence[int], xlabel: str, ylabel: str) -> None:
    """保存目标数直方图。"""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if values:
        max_value = max(values)
        bins = range(0, max_value + 2)
        ax.hist(values, bins=bins, color="#59a14f", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_dataset_class_plot(path: Path, dataset_rows: Sequence[Dict]) -> None:
    """保存 day/dataset 维度的类别堆叠图。"""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_names = sorted({k for row in dataset_rows for k in row["selected_class_counts"].keys()}, key=natural_sort_key)
    labels = [row["dataset"] for row in dataset_rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * max(1, len(labels))), 5.0))
    bottoms = [0] * len(labels)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6"]
    for idx, class_name in enumerate(class_names):
        values = [row["selected_class_counts"].get(class_name, 0) for row in dataset_rows]
        ax.bar(labels, values, bottom=bottoms, label=class_name, color=colors[idx % len(colors)])
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_title("selected class distribution by dataset")
    ax.set_xlabel("dataset")
    ax.set_ylabel("objects")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_plots(output_dir: Path, rows: Sequence[ClipStats], dataset_rows: Sequence[Dict], summary: Dict) -> None:
    """按可视化需要写出几张 png 图；没有 matplotlib 时跳过。"""

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        selected_counts = Counter(summary["selected_class_counts"])
        save_plot_bar(
            plot_dir / "class_counts.png",
            "selected class counts",
            list(selected_counts.keys()),
            list(selected_counts.values()),
            "class",
            "objects",
        )
        distance_counts = Counter(summary["distance_bin_counts"])
        save_plot_bar(
            plot_dir / "distance_bins.png",
            "selected object distance bins",
            list(distance_counts.keys()),
            list(distance_counts.values()),
            "distance",
            "objects",
        )
        direction_counts = Counter(summary["direction_counts"])
        save_plot_bar(
            plot_dir / "direction_counts.png",
            "selected object directions",
            list(direction_counts.keys()),
            list(direction_counts.values()),
            "direction",
            "objects",
        )
        per_frame = []
        for row in rows:
            per_frame.extend(row.selected_objects_per_frame)
        save_plot_hist(
            plot_dir / "objects_per_frame_hist.png",
            "selected objects per label frame",
            per_frame,
            "objects per frame",
            "frames",
        )
        if dataset_rows:
            save_dataset_class_plot(plot_dir / "dataset_class_counts.png", dataset_rows)

        top_sequences = merge_stats(rows, ["dataset", "sequence"])
        top_sequences = sorted(top_sequences, key=lambda x: x["objects_selected"], reverse=True)[:30]
        labels = ["{}/{}".format(x["dataset"], x["sequence"]) for x in top_sequences]
        values = [int(x["objects_selected"]) for x in top_sequences]
        save_plot_bar(
            plot_dir / "top_sequences_by_objects.png",
            "top sequences by selected objects",
            labels,
            values,
            "sequence",
            "objects",
        )
    except ImportError as exc:
        print("[warn] 当前环境缺少 matplotlib，已跳过画图：{}".format(exc), file=sys.stderr)


def iter_refs_with_progress(refs: Sequence[ClipRef], disable_progress: bool) -> Iterable[ClipRef]:
    """遍历 clip 时显示进度条；没有 tqdm 时回退到普通日志。

    tqdm 默认会显示已用时间和预计剩余时间，这里显式指定 bar_format，让长时间
    扫描 10 万级帧数据时能稳定看到 processed/total、elapsed、remaining 和速度。
    """

    if not disable_progress and tqdm is not None:
        progress = tqdm(
            refs,
            total=len(refs),
            desc="[scan]",
            unit="clip",
            dynamic_ncols=True,
            mininterval=1.0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        for ref in progress:
            progress.set_postfix_str("{}/{}/{}".format(ref.dataset, ref.sequence, ref.clip), refresh=False)
            yield ref
        return

    if not disable_progress and tqdm is None:
        print("[warn] 当前环境没有 tqdm，scan 阶段使用普通日志显示进度", file=sys.stderr)
    for index, ref in enumerate(refs, 1):
        if index == 1 or index % 20 == 0 or index == len(refs):
            print("[scan] {}/{} {} / {} / {}".format(index, len(refs), ref.dataset, ref.sequence, ref.clip))
        yield ref


def parse_distance_bins(values: Sequence[float]) -> List[float]:
    """清洗距离分段，要求递增且非负。"""

    bins = sorted({float(x) for x in values})
    if not bins:
        return []
    if bins[0] < 0:
        raise SystemExit("--distance-bins 不能包含负数")
    if len(bins) == 1:
        raise SystemExit("--distance-bins 至少需要两个边界，或完全不传")
    return bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/N7_704_256"), help="N7 数据根目录，可以是 704x256 缓存或 1600x900 原图目录。")
    parser.add_argument("--datasets", "--days", nargs="+", default=None, help="需要分析的数据集/日期目录；不传则扫描 data-root 下所有一级目录。")
    parser.add_argument("--sequences", nargs="+", default=None, help="只分析指定 sequence，默认不限制。")
    parser.add_argument("--clips", nargs="+", default=None, help="只分析指定 clip，默认不限制。")
    parser.add_argument("--classes", nargs="+", default=None, help="重点统计的类别；不传表示统计全部已知类别。类别空间由 --class-space 控制。")
    parser.add_argument("--class-space", choices=["mapped", "raw"], default="mapped", help="--classes 使用原始标签类别还是映射后的训练类别。")
    parser.add_argument("--distance-bins", nargs="+", type=float, default=DEFAULT_DISTANCE_BINS, help="距离分段边界，单位米，最后一段自动表示 >= 最后边界。")
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/n7_dataset_analysis"), help="分析结果输出目录。")
    parser.add_argument("--name", default=None, help="输出子目录名称；不传则自动用时间戳。")
    parser.add_argument("--no-plots", action="store_true", help="只输出 csv/json，不生成 png 图。")
    parser.add_argument("--no-progress", action="store_true", help="关闭 scan 阶段 tqdm 进度条。")
    parser.add_argument("--max-examples", type=int, default=5, help="每个 clip 最多记录多少个异常示例。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.classes = normalize_name_list(args.classes)
    args.datasets = discover_datasets(args.data_root, args.datasets)
    args.distance_bins = parse_distance_bins(args.distance_bins)

    output_name = args.name or "n7_distribution_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = args.output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_refs: List[ClipRef] = []
    for dataset in args.datasets:
        refs = discover_refs(args.data_root, dataset, args.sequences, args.clips)
        print("[discover] {} clips under dataset {}".format(len(refs), dataset))
        all_refs.extend(refs)
    if not all_refs:
        raise SystemExit("没有发现可分析的 3D_OD/lidar 标签目录")

    rows: List[ClipStats] = []
    for ref in iter_refs_with_progress(all_refs, args.no_progress):
        rows.append(scan_clip(ref, args))

    sequence_rows = merge_stats(rows, ["dataset", "sequence"])
    dataset_rows = merge_stats(rows, ["dataset"])
    summary = total_summary(rows, args)

    write_clip_csv(output_dir / "clip_stats.csv", rows)
    write_aggregate_csv(output_dir / "sequence_stats.csv", sequence_rows, ["dataset", "sequence"])
    write_aggregate_csv(output_dir / "dataset_stats.csv", dataset_rows, ["dataset"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.no_plots:
        write_plots(output_dir, rows, dataset_rows, summary)

    print("[done] clips={} sequences={} label_frames={} selected_objects={}".format(
        summary["num_clips"],
        summary["num_sequences"],
        summary["totals"].get("label_frames", 0),
        summary["totals"].get("objects_selected", 0),
    ))
    print("[out] {}".format(output_dir))


if __name__ == "__main__":
    main()
