#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据 N7 数据集分布分析结果生成 train/val/test manifest。

这个脚本主要消费 ``tools/data_converter/n7/analyze_n7_dataset_distribution.py`` 已经输出的
``clip_stats.csv``。``sequence_stats.csv`` 仍然建议保留给人工审阅，但不是
运行本脚本的必须输入。

它不会重新扫描原始标签和图片，因此适合在分析完成后反复调整划分比例、随机种子
和划分粒度。默认以 sequence 为最小切分单元，避免同一个连续片段同时出现在
train/val/test 中造成时序泄漏。

输出的 manifest 文件和 ``tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py`` 约定一致：
每一行都是 ``dataset/sequence/clip``，文件名为：

    <dataset>_train_clips.txt
    <dataset>_val_clips.txt
    <dataset>_test_clips.txt

常用示例：

    # 基于一次分析结果，按 sequence 做 8:1:1 划分，输出到 work_dirs 下。
    python tools/data_converter/n7/split_n7_dataset_from_analysis.py \
        --analysis-dir work_dirs/n7_dataset_analysis_vehicle/n7_distribution_20260625_101500 \
        --output-dir work_dirs/n7_dataset_splits/n7_vehicle_split_seed2026 \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
        --seed 2026

    # 直接把 manifest 写到数据根目录的 manifests 下，converter 会自动读取。
    python tools/data_converter/n7/split_n7_dataset_from_analysis.py \
        --analysis-dir work_dirs/n7_dataset_analysis_vehicle/n7_distribution_20260625_101500 \
        --manifest-dir data/N7_704_256/manifests \
        --seed 2026

    # 只做 train/val，不保留 test。
    python tools/data_converter/n7/split_n7_dataset_from_analysis.py \
        --analysis-dir work_dirs/n7_dataset_analysis_vehicle/n7_distribution_20260625_101500 \
        --train-ratio 0.9 --val-ratio 0.1 --test-ratio 0.0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ClipRow:
    """clip_stats.csv 中的一行。"""

    dataset: str
    sequence: str
    clip: str
    stats: Dict[str, float]
    class_counts: Counter = field(default_factory=Counter)


@dataclass
class SplitUnit:
    """用于划分的最小单元，默认是 sequence，也可选择 clip。"""

    dataset: str
    unit_id: str
    clips: List[ClipRow]
    stats: Dict[str, float]
    class_counts: Counter = field(default_factory=Counter)
    split: str = "train"

    @property
    def manifest_lines(self) -> List[str]:
        """转换为 converter manifest 需要的 dataset/sequence/clip 行。"""

        return ["{}/{}/{}".format(c.dataset, c.sequence, c.clip) for c in self.clips]


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """保留 epilog 示例换行，同时展示默认值。"""


def natural_sort_key(value):
    """自然排序 key，避免 clip_10 排在 clip_2 前面。

    N7 的 sequence/clip 名称经常以数字结尾，普通字符串排序会得到
    1, 10, 2 的顺序。manifest 后续用于推理和可视化时需要稳定的时间/数字顺序，
    因此这里把字符串中的连续数字按整数比较。
    """

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


def to_float(value, default=0.0) -> float:
    """CSV 字段转 float，空值按 default 处理。"""

    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def normalize_name_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    """清洗命令行名称列表，保持顺序去重。"""

    if not values:
        return None
    names = [str(x).strip() for x in values if str(x).strip()]
    return list(dict.fromkeys(names)) or None


def read_clip_stats(path: Path, datasets: Optional[Sequence[str]]) -> List[ClipRow]:
    """读取 clip_stats.csv。"""

    if not path.exists():
        raise SystemExit("找不到 clip_stats.csv：{}".format(path))
    dataset_filter = set(normalize_name_list(datasets) or [])
    rows: List[ClipRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for item in reader:
            dataset = item.get("dataset", "").strip()
            sequence = item.get("sequence", "").strip()
            clip = item.get("clip", "").strip()
            if not dataset or not sequence or not clip:
                continue
            if dataset_filter and dataset not in dataset_filter:
                continue
            class_counts = Counter()
            stats = {}
            for key, value in item.items():
                if key.startswith("class_"):
                    class_counts[key[len("class_"):]] = int(to_float(value))
                elif key not in {"dataset", "sequence", "clip", "examples"}:
                    stats[key] = to_float(value)
            rows.append(ClipRow(dataset=dataset, sequence=sequence, clip=clip, stats=stats, class_counts=class_counts))
    if not rows:
        raise SystemExit("clip_stats.csv 中没有可划分的 clip，请检查 --datasets 或分析结果")
    return rows


def merge_unit_stats(clips: Sequence[ClipRow]) -> Tuple[Dict[str, float], Counter]:
    """把若干 clip 聚合成一个划分单元的统计。"""

    stats = defaultdict(float)
    class_counts = Counter()
    for clip in clips:
        for key, value in clip.stats.items():
            stats[key] += float(value)
        class_counts.update(clip.class_counts)
    stats["clips"] = float(len(clips))
    return dict(stats), class_counts


def build_units(clip_rows: Sequence[ClipRow], unit: str) -> List[SplitUnit]:
    """按 sequence 或 clip 构建划分单元。"""

    grouped: Dict[Tuple[str, str], List[ClipRow]] = defaultdict(list)
    if unit == "sequence":
        for row in clip_rows:
            grouped[(row.dataset, row.sequence)].append(row)
    elif unit == "clip":
        for row in clip_rows:
            grouped[(row.dataset, "{}/{}".format(row.sequence, row.clip))].append(row)
    else:
        raise ValueError("unsupported split unit {}".format(unit))

    units: List[SplitUnit] = []
    for (dataset, unit_id), clips in sorted(grouped.items(), key=lambda x: natural_sort_key("/".join(x[0]))):
        clips = sorted(clips, key=lambda x: (natural_sort_key(x.sequence), natural_sort_key(x.clip)))
        stats, class_counts = merge_unit_stats(clips)
        units.append(SplitUnit(dataset=dataset, unit_id=unit_id, clips=clips, stats=stats, class_counts=class_counts))
    return units


def split_target_count(num_units: int, ratio: float, requested_min: int) -> int:
    """根据比例和最小单元数确定 val/test 目标单元数量。"""

    if ratio <= 0 or num_units <= 1:
        return 0
    count = int(round(num_units * ratio))
    if requested_min > 0:
        count = max(requested_min, count)
    return min(count, max(num_units - 1, 0))


def class_error(selected: Counter, target: Counter, class_names: Sequence[str]) -> float:
    """衡量当前选择的类别分布和目标类别分布的差异。"""

    if not class_names:
        return 0.0
    err = 0.0
    for name in class_names:
        denom = max(float(target.get(name, 0)), 1.0)
        err += abs(float(selected.get(name, 0)) - float(target.get(name, 0))) / denom
    return err / float(len(class_names))


def greedy_pick_units(
    candidates: List[SplitUnit],
    target_weight: float,
    target_count: int,
    balance_key: str,
    class_names: Sequence[str],
    target_class_counts: Counter,
) -> List[SplitUnit]:
    """从候选单元中贪心挑选一组接近目标规模和类别分布的单元。"""

    if target_count <= 0 or not candidates:
        return []

    selected: List[SplitUnit] = []
    selected_weight = 0.0
    selected_classes = Counter()
    remaining = list(candidates)

    while remaining and len(selected) < target_count:
        best_idx = 0
        best_score = None
        for idx, unit in enumerate(remaining):
            weight = float(unit.stats.get(balance_key, 0.0))
            next_weight = selected_weight + weight
            next_classes = selected_classes + unit.class_counts
            weight_score = abs(next_weight - target_weight) / max(target_weight, 1.0)
            cls_score = class_error(next_classes, target_class_counts, class_names)
            # 规模优先，类别分布作为次级约束。这样不会为了类别均衡选出过大/过小的 val/test。
            score = weight_score + 0.25 * cls_score
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        unit = remaining.pop(best_idx)
        selected.append(unit)
        selected_weight += float(unit.stats.get(balance_key, 0.0))
        selected_classes.update(unit.class_counts)

    return selected


def assign_one_dataset(units: List[SplitUnit], args: argparse.Namespace) -> None:
    """对单个 dataset/day 内部做 train/val/test 划分，不跨 day 混合。"""

    if not units:
        return
    # Python 内置 hash 会受进程随机种子影响，不能用于可复现划分。
    dataset_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(units[0].dataset))
    rng = random.Random(args.seed + dataset_seed)
    shuffled = list(units)
    rng.shuffle(shuffled)

    total_weight = sum(float(x.stats.get(args.balance_key, 0.0)) for x in shuffled)
    total_classes = Counter()
    for unit in shuffled:
        total_classes.update(unit.class_counts)
    class_names = sorted((k for k, v in total_classes.items() if v > 0), key=natural_sort_key)

    val_count = split_target_count(len(shuffled), args.val_ratio, args.min_val_units)
    test_count = split_target_count(len(shuffled) - val_count, args.test_ratio, args.min_test_units)
    if val_count + test_count >= len(shuffled):
        # 无论如何至少保留一个训练单元，避免小数据集被 val/test 吃完。
        overflow = val_count + test_count - (len(shuffled) - 1)
        reduce_test = min(test_count, overflow)
        test_count -= reduce_test
        overflow -= reduce_test
        if overflow > 0:
            val_count = max(0, val_count - overflow)

    test_target_weight = total_weight * args.test_ratio
    test_target_classes = Counter({k: total_classes[k] * args.test_ratio for k in class_names})
    test_units = greedy_pick_units(
        shuffled, test_target_weight, test_count, args.balance_key, class_names, test_target_classes)
    test_set = set(id(x) for x in test_units)
    remaining = [x for x in shuffled if id(x) not in test_set]

    val_target_weight = total_weight * args.val_ratio
    val_target_classes = Counter({k: total_classes[k] * args.val_ratio for k in class_names})
    val_units = greedy_pick_units(
        remaining, val_target_weight, val_count, args.balance_key, class_names, val_target_classes)
    val_set = set(id(x) for x in val_units)

    for unit in shuffled:
        if id(unit) in test_set:
            unit.split = "test"
        elif id(unit) in val_set:
            unit.split = "val"
        else:
            unit.split = "train"


def assign_splits(units: List[SplitUnit], args: argparse.Namespace) -> None:
    """按 dataset/day 分组划分。"""

    by_dataset: Dict[str, List[SplitUnit]] = defaultdict(list)
    for unit in units:
        by_dataset[unit.dataset].append(unit)
    for dataset in sorted(by_dataset, key=natural_sort_key):
        assign_one_dataset(by_dataset[dataset], args)


def aggregate(units: Iterable[SplitUnit], keys: Sequence[str]) -> List[Dict]:
    """聚合 split 统计，用于输出 csv/json。"""

    grouped: Dict[Tuple[str, ...], Dict] = {}
    for unit in units:
        value_map = {
            "dataset": unit.dataset,
            "split": unit.split,
        }
        key = tuple(value_map[k] for k in keys)
        if key not in grouped:
            grouped[key] = {k: v for k, v in zip(keys, key)}
            grouped[key].update(dict(units=0, clips=0, class_counts=Counter()))
        item = grouped[key]
        item["units"] += 1
        item["clips"] += len(unit.clips)
        item["class_counts"].update(unit.class_counts)
        for name, value in unit.stats.items():
            item[name] = item.get(name, 0.0) + float(value)
    return sorted(grouped.values(), key=lambda x: tuple(natural_sort_key(x.get(k, "")) for k in keys))


def write_manifest_files(units: Sequence[SplitUnit], manifest_dir: Path) -> Dict[str, Dict[str, int]]:
    """写 converter 可直接读取的 per-dataset manifest。"""

    manifest_dir.mkdir(parents=True, exist_ok=True)
    by_dataset_split: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for unit in units:
        by_dataset_split[(unit.dataset, unit.split)].extend(unit.manifest_lines)

    summary: Dict[str, Dict[str, int]] = defaultdict(dict)
    datasets = sorted({unit.dataset for unit in units}, key=natural_sort_key)
    for dataset in datasets:
        for split in ("train", "val", "test"):
            lines = sorted(by_dataset_split.get((dataset, split), []), key=natural_sort_key)
            path = manifest_dir / "{}_{}_clips.txt".format(dataset, split)
            with path.open("w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            summary[dataset][split] = len(lines)
    return summary


def write_units_csv(path: Path, units: Sequence[SplitUnit]) -> None:
    """写每个 sequence/clip 单元的划分结果。"""

    class_names = sorted({name for unit in units for name in unit.class_counts.keys()}, key=natural_sort_key)
    stat_names = sorted({name for unit in units for name in unit.stats.keys()}, key=natural_sort_key)
    fieldnames = ["dataset", "unit_id", "split", "num_clips"] + stat_names + ["class_{}".format(x) for x in class_names]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for unit in sorted(units, key=lambda x: (natural_sort_key(x.dataset), x.split, natural_sort_key(x.unit_id))):
            row = dict(dataset=unit.dataset, unit_id=unit.unit_id, split=unit.split, num_clips=len(unit.clips))
            for name in stat_names:
                row[name] = unit.stats.get(name, 0.0)
            for name in class_names:
                row["class_{}".format(name)] = unit.class_counts.get(name, 0)
            writer.writerow(row)


def write_aggregate_csv(path: Path, rows: Sequence[Dict]) -> None:
    """写 dataset/split 聚合统计。"""

    class_names = sorted({name for row in rows for name in row["class_counts"].keys()}, key=natural_sort_key)
    stat_names = sorted({k for row in rows for k in row.keys() if k not in {"dataset", "split", "class_counts"}}, key=natural_sort_key)
    fieldnames = ["dataset", "split"] + stat_names + ["class_{}".format(x) for x in class_names]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {name: row.get(name, "") for name in fieldnames}
            for name in class_names:
                record["class_{}".format(name)] = row["class_counts"].get(name, 0)
            writer.writerow(record)


def write_summary_json(path: Path, args: argparse.Namespace, units: Sequence[SplitUnit], manifest_counts: Dict[str, Dict[str, int]]) -> None:
    """写机器可读划分摘要。"""

    aggregate_rows = aggregate(units, ["dataset", "split"])
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_dir": str(args.analysis_dir),
        "unit": args.unit,
        "seed": args.seed,
        "ratios": dict(train=args.train_ratio, val=args.val_ratio, test=args.test_ratio),
        "balance_key": args.balance_key,
        "manifest_counts": manifest_counts,
        "dataset_split_stats": [],
    }
    for row in aggregate_rows:
        item = {k: v for k, v in row.items() if k != "class_counts"}
        item["class_counts"] = dict(sorted(row["class_counts"].items(), key=lambda x: natural_sort_key(x[0])))
        summary["dataset_split_stats"].append(item)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=RawDefaultsHelpFormatter)
    parser.add_argument("--analysis-dir", type=Path, required=True, help="tools/data_converter/n7/analyze_n7_dataset_distribution.py 输出目录，必须包含 clip_stats.csv。")
    parser.add_argument("--output-dir", type=Path, default=None, help="划分结果输出目录；不传则写到 analysis-dir/splits/split_<time>。")
    parser.add_argument("--manifest-dir", type=Path, default=None, help="manifest 输出目录；建议设为 data/N7_704_256/manifests，让 converter 自动读取。")
    parser.add_argument("--datasets", nargs="+", default=None, help="只划分指定 dataset/day，默认使用 clip_stats.csv 中全部 dataset。")
    parser.add_argument("--unit", choices=["sequence", "clip"], default="sequence", help="划分最小单元；正式训练建议 sequence。")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例。")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例。")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例。")
    parser.add_argument("--seed", type=int, default=2026, help="稳定随机种子。")
    parser.add_argument("--balance-key", choices=["label_frames", "objects_selected", "frames_with_selected"], default="label_frames", help="用于控制 val/test 规模接近目标比例的主统计量。")
    parser.add_argument("--min-val-units", type=int, default=1, help="每个 dataset 中 val 至少包含多少个单元；小数据集会自动退让。")
    parser.add_argument("--min-test-units", type=int, default=1, help="每个 dataset 中 test 至少包含多少个单元；小数据集会自动退让。")
    return parser.parse_args()


def validate_ratios(args: argparse.Namespace) -> None:
    """检查划分比例。"""

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(x < 0 for x in ratios):
        raise SystemExit("train/val/test ratio 不能为负数")
    total = sum(ratios)
    if total <= 0:
        raise SystemExit("train/val/test ratio 之和必须大于 0")
    args.train_ratio /= total
    args.val_ratio /= total
    args.test_ratio /= total


def main() -> None:
    args = parse_args()
    validate_ratios(args)
    output_dir = args.output_dir or (args.analysis_dir / "splits" / "split_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.manifest_dir or (output_dir / "manifests")

    clip_rows = read_clip_stats(args.analysis_dir / "clip_stats.csv", args.datasets)
    units = build_units(clip_rows, args.unit)
    assign_splits(units, args)

    manifest_counts = write_manifest_files(units, manifest_dir)
    write_units_csv(output_dir / "split_units.csv", units)
    aggregate_rows = aggregate(units, ["dataset", "split"])
    write_aggregate_csv(output_dir / "split_dataset_stats.csv", aggregate_rows)
    write_summary_json(output_dir / "split_summary.json", args, units, manifest_counts)

    print("[done] unit={} units={} clips={}".format(args.unit, len(units), len(clip_rows)))
    for dataset in sorted(manifest_counts, key=natural_sort_key):
        print("[manifest] {} train={} val={} test={}".format(
            dataset,
            manifest_counts[dataset].get("train", 0),
            manifest_counts[dataset].get("val", 0),
            manifest_counts[dataset].get("test", 0)))
    print("[out] {}".format(output_dir))
    print("[manifest-dir] {}".format(manifest_dir))


if __name__ == "__main__":
    main()
