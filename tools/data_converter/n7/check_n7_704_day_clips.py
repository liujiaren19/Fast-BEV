#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 N7 缓存/训练数据根目录中某一天的 sequence / clip 完整性。

职责边界：
    这个脚本只读本地数据目录，不访问 ozone，也不会修改任何文件。它只检查
    文件和帧级完整性：label 帧、image 帧、相机数量和图片尺寸；不读取 3D_OD
    内容做类别/box 分布统计。类别映射后分布、range 过滤后有效 box 数等训练
    相关统计，应由 tools/data_converter/n7/n7_raw_3dod_to_fastbev_pkl.py 或专门分析脚本输出。

目录约定：
    <root>/<day>/<sequence>/parsed_data/<clip>/frames/<lidar_ts>/images/...
    <root>/<day>/<sequence>/output/<clip>/3D_OD/lidar/*.json

效率说明：
    N7 每帧 images 目录通常是 ``images/<cam_id>/<image_timestamp>.jpg``。
    本脚本优先按这类结构做一层目录扫描，避免对每帧执行递归 rglob。只有遇到
    非标准扁平结构时才退化为有限扫描。

常用示例：
    # 快速检查 20251017，默认每个 clip 抽样少量帧检查 704x256 尺寸。
    python tools/data_converter/n7/check_n7_704_day_clips.py --root data/N7_704_256 --day 20251017

    # rclone/resize 还在运行时，只看问题 clip，且不读图片尺寸，速度最快。
    python tools/data_converter/n7/check_n7_704_day_clips.py --day 20251017 --only-problems --check-size-mode none

    # 只检查指定 sequence。
    python tools/data_converter/n7/check_n7_704_day_clips.py --day 20251017 --sequences 20251017_143548

    # 最终严格检查：每张图片都检查尺寸，且发现问题返回非 0 退出码。
    python tools/data_converter/n7/check_n7_704_day_clips.py --day 20251017 --check-size-mode full --fail-on-problems
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def natural_sort_key(value):
    """自然排序 key，避免 clip_10 排在 clip_2 前面。"""

    key = []
    for part in re.split(r"(\d+)", str(value)):
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


@dataclass
class ClipReport:
    """一个 clip 的检查结果。"""

    sequence: str
    clip: str
    status: str
    label_frames: int
    image_frames: int
    aligned_frames: int
    missing_image_frames: int
    extra_image_frames: int
    bad_camera_frames: int
    bad_size_images: int
    sampled_size_images: int
    examples: str


@dataclass
class FrameImageSummary:
    """单帧 images 目录的轻量检查结果。"""

    timestamp: str
    cameras: set
    image_files_for_size: list


def is_image_name(name):
    """判断文件名是否是常见图片格式。"""

    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def list_dir_names(path):
    """用 os.scandir 快速列出一级子目录名称。"""

    try:
        with os.scandir(path) as entries:
            return sorted((entry.name for entry in entries if entry.is_dir()), key=natural_sort_key)
    except FileNotFoundError:
        return []


def list_sequence_names(day_dir):
    """列出 day 目录下的 sequence 名称。"""

    if not day_dir.exists():
        raise SystemExit("day 目录不存在：{}".format(day_dir))
    return list_dir_names(day_dir)


def normalize_sequence_name(sequence_arg, day):
    """把用户输入的 sequence 参数统一成 sequence 名称。"""

    text = sequence_arg.strip("/")
    parts = text.split("/")
    if len(parts) == 1:
        return parts[0]
    if len(parts) >= 2 and parts[0] == day:
        return parts[1]
    raise ValueError("sequence 必须是 sequence 名或 {}/sequence 格式：{}".format(day, sequence_arg))


def select_sequence_names(day_dir, day, requested_sequences):
    """根据命令行参数决定需要检查哪些 sequence。"""

    if not requested_sequences:
        return list_sequence_names(day_dir)

    selected = []
    missing = []
    for sequence_arg in requested_sequences:
        sequence_name = normalize_sequence_name(sequence_arg, day)
        if (day_dir / sequence_name).is_dir():
            selected.append(sequence_name)
        else:
            missing.append(sequence_name)

    selected = list(dict.fromkeys(selected))
    if missing:
        print("[warn] sequence 目录不存在，已跳过：{}".format(missing), file=sys.stderr)
    if not selected:
        raise SystemExit("没有可检查的 sequence，请确认 --day 和 --sequences 是否正确")
    return selected


def list_clip_names(sequence_dir):
    """同时从 parsed_data 和 output 下收集 clip 名称。"""

    names = set()
    parsed_root = sequence_dir / "parsed_data"
    output_root = sequence_dir / "output"
    names.update(list_dir_names(parsed_root))
    names.update(list_dir_names(output_root))
    return sorted(names, key=natural_sort_key)


def collect_label_timestamps(sequence_dir, clip):
    """收集某个 clip 的 3D_OD 标签时间戳集合。"""

    label_dir = sequence_dir / "output" / clip / "3D_OD" / "lidar"
    if not label_dir.exists():
        return set()
    timestamps = set()
    try:
        with os.scandir(label_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".json"):
                    timestamps.add(os.path.splitext(entry.name)[0])
    except FileNotFoundError:
        return set()
    return timestamps


def should_check_size(frame_index, args):
    """判断当前帧是否需要读取图片头检查尺寸。"""

    if args.check_size_mode == "none":
        return False
    if args.check_size_mode == "full":
        return True
    return frame_index < args.size_sample_frames


def first_image_in_camera_dir(cam_dir):
    """在 images/<cam_id> 目录中找一张图片，避免递归扫描。"""

    best_name = None
    try:
        with os.scandir(cam_dir) as entries:
            for entry in entries:
                if not entry.is_file() or not is_image_name(entry.name):
                    continue
                if best_name is None or entry.name < best_name:
                    best_name = entry.name
    except FileNotFoundError:
        return None
    if best_name is None:
        return None
    return Path(cam_dir) / best_name


def list_images_in_camera_dir(cam_dir):
    """列出单个相机目录中的图片；full 尺寸检查才会调用。"""

    files = []
    try:
        with os.scandir(cam_dir) as entries:
            for entry in entries:
                if entry.is_file() and is_image_name(entry.name):
                    files.append(Path(cam_dir) / entry.name)
    except FileNotFoundError:
        return []
    return sorted(files, key=lambda p: natural_sort_key(p.name))


def collect_flat_images(images_dir):
    """兼容 images 目录直接放图片的历史结构。"""

    files = []
    try:
        with os.scandir(images_dir) as entries:
            for entry in entries:
                if entry.is_file() and is_image_name(entry.name):
                    files.append(Path(images_dir) / entry.name)
    except FileNotFoundError:
        return []
    return sorted(files, key=lambda p: natural_sort_key(p.name))


def summarize_frame_images(frame_dir, frame_index, args):
    """快速汇总一帧图片：相机集合，以及需要检查尺寸的图片列表。"""

    images_dir = frame_dir / "images"
    cameras = set()
    image_files_for_size = []
    if not images_dir.exists():
        return FrameImageSummary(frame_dir.name, cameras, image_files_for_size)

    check_size = should_check_size(frame_index, args)
    saw_camera_dir = False
    try:
        with os.scandir(images_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    saw_camera_dir = True
                    if check_size and args.check_size_mode == "full":
                        cam_files = list_images_in_camera_dir(entry.path)
                        if cam_files:
                            cameras.add(entry.name)
                            image_files_for_size.extend(cam_files)
                    else:
                        first_image = first_image_in_camera_dir(entry.path)
                        if first_image is not None:
                            cameras.add(entry.name)
                            if check_size:
                                image_files_for_size.append(first_image)
                elif entry.is_file() and is_image_name(entry.name):
                    cameras.add(os.path.splitext(entry.name)[0])
                    if check_size:
                        image_files_for_size.append(Path(entry.path))
    except FileNotFoundError:
        return FrameImageSummary(frame_dir.name, cameras, image_files_for_size)

    # 如果 images 下没有相机子目录，但上面也没扫到图片，再做一次扁平兜底。
    if not saw_camera_dir and not cameras:
        flat_images = collect_flat_images(images_dir)
        cameras = {p.stem for p in flat_images}
        if check_size:
            image_files_for_size = flat_images

    return FrameImageSummary(frame_dir.name, cameras, image_files_for_size)


def check_image_size(image_path, expected_size):
    """读取一张图片的尺寸，返回是否符合期望。"""

    if Image is None:
        raise RuntimeError("当前环境没有 Pillow，不能检查图片尺寸")
    with Image.open(image_path) as img:
        return img.size == expected_size, img.size


def collect_frame_dirs(frames_root):
    """快速列出 frames/<timestamp> 目录。"""

    try:
        with os.scandir(frames_root) as entries:
            return sorted((Path(entry.path) for entry in entries if entry.is_dir()), key=lambda p: natural_sort_key(p.name))
    except FileNotFoundError:
        return []


def collect_image_frames(sequence_dir, clip, args):
    """收集图片帧信息，并按配置检查相机数量和图片尺寸。"""

    frames_root = sequence_dir / "parsed_data" / clip / "frames"
    frame_dirs = collect_frame_dirs(frames_root)
    if not frame_dirs:
        return set(), [], [], 0, 0, 0

    frame_timestamps = set()
    bad_camera_examples = []
    bad_size_examples = []
    bad_camera_frames = 0
    bad_size_images = 0
    sampled_size_images = 0
    expected_size = (args.expected_width, args.expected_height)

    for frame_index, frame_dir in enumerate(frame_dirs):
        summary = summarize_frame_images(frame_dir, frame_index, args)
        frame_timestamps.add(summary.timestamp)

        if args.expected_cameras > 0 and len(summary.cameras) != args.expected_cameras:
            bad_camera_frames += 1
            if len(bad_camera_examples) < args.max_examples:
                bad_camera_examples.append(
                    "{}:camera_count={}:{}".format(summary.timestamp, len(summary.cameras), sorted(summary.cameras, key=natural_sort_key))
                )

        for image_path in summary.image_files_for_size:
            sampled_size_images += 1
            try:
                ok, size = check_image_size(image_path, expected_size)
            except Exception as exc:
                ok, size = False, "read_error:{}".format(exc)
            if not ok:
                bad_size_images += 1
                if len(bad_size_examples) < args.max_examples:
                    bad_size_examples.append("{}:{}".format(image_path, size))

    return frame_timestamps, bad_camera_examples, bad_size_examples, sampled_size_images, bad_camera_frames, bad_size_images


def make_status(label_ts, image_ts, bad_camera_examples, bad_size_examples):
    """根据标签、图片和检查结果生成 clip 状态。"""

    if label_ts and not image_ts:
        return "label_only"
    if image_ts and not label_ts:
        return "image_only"
    if not label_ts and not image_ts:
        return "empty"
    if label_ts == image_ts and not bad_camera_examples and not bad_size_examples:
        return "complete"
    return "partial"


def scan_clip(sequence_dir, clip, args):
    """检查一个 clip，并返回结构化报告。"""

    label_ts = collect_label_timestamps(sequence_dir, clip)
    (
        image_ts, bad_camera_examples, bad_size_examples, sampled_size_images,
        bad_camera_frames, bad_size_images,
    ) = collect_image_frames(sequence_dir, clip, args)
    aligned_ts = label_ts & image_ts
    missing_image = sorted(label_ts - image_ts, key=natural_sort_key)
    extra_image = sorted(image_ts - label_ts, key=natural_sort_key)
    status = make_status(label_ts, image_ts, bad_camera_examples, bad_size_examples)

    examples = []
    if missing_image:
        examples.append("missing_image_frames={}".format(missing_image[:args.max_examples]))
    if extra_image:
        examples.append("extra_image_frames={}".format(extra_image[:args.max_examples]))
    if bad_camera_examples:
        examples.append("bad_camera={}".format(bad_camera_examples[:args.max_examples]))
    if bad_size_examples:
        examples.append("bad_size={}".format(bad_size_examples[:args.max_examples]))

    return ClipReport(
        sequence=sequence_dir.name,
        clip=clip,
        status=status,
        label_frames=len(label_ts),
        image_frames=len(image_ts),
        aligned_frames=len(aligned_ts),
        missing_image_frames=len(missing_image),
        extra_image_frames=len(extra_image),
        bad_camera_frames=bad_camera_frames,
        bad_size_images=bad_size_images,
        sampled_size_images=sampled_size_images,
        examples="; ".join(examples),
    )


def write_csv(path, reports):
    """把检查结果写成 CSV，方便后续筛选。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sequence", "clip", "status", "label_frames", "image_frames", "aligned_frames",
        "missing_image_frames", "extra_image_frames", "bad_camera_frames",
        "bad_size_images", "sampled_size_images", "examples",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow({field: getattr(report, field) for field in fields})
    print("[csv] {}".format(path))


def ratio_text(num, den):
    """把比例格式化成百分比字符串。"""

    if den <= 0:
        return "0.00%"
    return "{:.2f}%".format(float(num) * 100.0 / float(den))


def problem_score(report):
    """问题排序分数，越大越需要优先查看。"""

    status_weight = {"partial": 4, "label_only": 3, "image_only": 2, "empty": 1, "complete": 0}
    return (
        status_weight.get(report.status, 0),
        report.missing_image_frames + report.extra_image_frames,
        report.bad_camera_frames,
        report.bad_size_images,
        report.label_frames + report.image_frames,
    )


def print_report(day, reports, args):
    """打印适合大数据集阅读的完整性报告。

    默认报告模式只打印汇总和问题 clip，避免一天几百个 complete clip 把终端刷屏。
    如果需要逐 clip 全量明细，可以使用 ``--report-mode all``；CSV 输出始终包含
    全量结果，适合后续筛选。
    """

    reports = sorted(reports, key=lambda r: (natural_sort_key(r.sequence), natural_sort_key(r.clip)))
    status_counter = Counter(r.status for r in reports)
    sequence_counter = defaultdict(Counter)
    sequence_totals = defaultdict(lambda: Counter())
    totals = {
        "label_frames": sum(r.label_frames for r in reports),
        "image_frames": sum(r.image_frames for r in reports),
        "aligned_frames": sum(r.aligned_frames for r in reports),
        "missing_image_frames": sum(r.missing_image_frames for r in reports),
        "extra_image_frames": sum(r.extra_image_frames for r in reports),
        "bad_camera_frames": sum(r.bad_camera_frames for r in reports),
        "bad_size_images": sum(r.bad_size_images for r in reports),
        "sampled_size_images": sum(r.sampled_size_images for r in reports),
    }
    for report in reports:
        sequence_counter[report.sequence][report.status] += 1
        sequence_totals[report.sequence]["label_frames"] += report.label_frames
        sequence_totals[report.sequence]["image_frames"] += report.image_frames
        sequence_totals[report.sequence]["aligned_frames"] += report.aligned_frames
        sequence_totals[report.sequence]["missing_image_frames"] += report.missing_image_frames
        sequence_totals[report.sequence]["extra_image_frames"] += report.extra_image_frames
        sequence_totals[report.sequence]["bad_camera_frames"] += report.bad_camera_frames
        sequence_totals[report.sequence]["bad_size_images"] += report.bad_size_images

    complete_clips = status_counter.get("complete", 0)
    problem_reports = [r for r in reports if r.status != "complete"]
    print("[summary] day={} sequences={} clips={} complete={} problems={} complete_rate={} status={}".format(
        day, len(sequence_counter), len(reports), complete_clips, len(problem_reports),
        ratio_text(complete_clips, len(reports)), dict(status_counter)))
    print("[summary-frames] label={} image={} aligned={} align_rate={} missing_image={} extra_image={} bad_camera_frames={} bad_size_images={} sampled_size_images={}".format(
        totals["label_frames"], totals["image_frames"], totals["aligned_frames"],
        ratio_text(totals["aligned_frames"], totals["label_frames"]),
        totals["missing_image_frames"], totals["extra_image_frames"],
        totals["bad_camera_frames"], totals["bad_size_images"], totals["sampled_size_images"]))

    if problem_reports:
        limit = max(0, int(args.problem_limit))
        shown = sorted(problem_reports, key=problem_score, reverse=True)[:limit]
        print("[problem-top] showing={}/{} by status/missing/extra/bad_camera/bad_size".format(len(shown), len(problem_reports)))
        for report in shown:
            print(
                "  [{status}] {sequence}/{clip} labels={label_frames} image_frames={image_frames} aligned={aligned_frames} "
                "missing={missing_image_frames} extra={extra_image_frames} bad_cam={bad_camera_frames} "
                "bad_size={bad_size_images} {examples}".format(**report.__dict__)
            )

    report_mode = args.report_mode
    if args.only_problems:
        report_mode = "problems"
    if report_mode == "summary":
        return

    for sequence in sorted(sequence_counter, key=natural_sort_key):
        seq_total = sequence_totals[sequence]
        seq_clips = sum(sequence_counter[sequence].values())
        print("\n[sequence] {} clips={} complete_rate={} status={} label={} image={} aligned={} align_rate={} missing={} extra={} bad_cam={} bad_size={}".format(
            sequence, seq_clips, ratio_text(sequence_counter[sequence].get("complete", 0), seq_clips),
            dict(sequence_counter[sequence]), seq_total["label_frames"], seq_total["image_frames"],
            seq_total["aligned_frames"], ratio_text(seq_total["aligned_frames"], seq_total["label_frames"]),
            seq_total["missing_image_frames"], seq_total["extra_image_frames"],
            seq_total["bad_camera_frames"], seq_total["bad_size_images"]))
        for report in [r for r in reports if r.sequence == sequence]:
            if report_mode == "problems" and report.status == "complete":
                continue
            print(
                "  [{status}] {clip} labels={label_frames} image_frames={image_frames} aligned={aligned_frames} "
                "missing={missing_image_frames} extra={extra_image_frames} "
                "bad_cam={bad_camera_frames} bad_size={bad_size_images} sampled_size={sampled_size_images} {examples}".format(
                    **report.__dict__
                )
            )

def iter_clip_tasks(tasks, args):
    """遍历 clip 任务，默认用 tqdm 显示进度、已用时间和预计剩余时间。"""

    if not args.no_progress and tqdm is not None:
        progress = tqdm(
            tasks,
            total=len(tasks),
            desc="[check]",
            unit="clip",
            dynamic_ncols=True,
            mininterval=1.0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        for sequence_dir, clip in progress:
            progress.set_postfix_str("{}/{}".format(sequence_dir.name, clip), refresh=False)
            yield sequence_dir, clip
        return

    if not args.no_progress and tqdm is None:
        print("[warn] 当前环境没有 tqdm，使用普通日志显示进度", file=sys.stderr)
    for index, task in enumerate(tasks, 1):
        if index == 1 or index % 20 == 0 or index == len(tasks):
            sequence_dir, clip = task
            print("[check] {}/{} {}/{}".format(index, len(tasks), sequence_dir.name, clip))
        yield task


def build_clip_tasks(day_dir, sequence_names):
    """构建需要检查的 clip 任务列表。"""

    tasks = []
    reports = []
    for sequence_name in sequence_names:
        sequence_dir = day_dir / sequence_name
        clip_names = list_clip_names(sequence_dir)
        if not clip_names:
            reports.append(ClipReport(sequence_name, "<no_clip>", "empty", 0, 0, 0, 0, 0, 0, 0, 0, "no parsed_data/output clips"))
            continue
        for clip in clip_names:
            tasks.append((sequence_dir, clip))
    return tasks, reports


def parse_args():
    parser = argparse.ArgumentParser(description="检查 N7 缓存/训练数据根目录某一天内每个 sequence / clip 的完整性。")
    parser.add_argument("--root", default="data/N7_704_256", help="N7 缓存或训练数据根目录，例如 data/N7_704_256 或 data/N7_raw_tmp")
    parser.add_argument("--day", default="20251017", help="需要检查的日期目录")
    parser.add_argument("--sequences", nargs="*", default=None, help="只检查指定 sequence；可写 sequence 名或 day/sequence")
    parser.add_argument("--expected-cameras", type=int, default=9, help="每帧期望图片相机数量；设为 0 表示不检查")
    parser.add_argument("--expected-width", type=int, default=704, help="期望图片宽度")
    parser.add_argument("--expected-height", type=int, default=256, help="期望图片高度")
    parser.add_argument("--check-size-mode", choices=["none", "sample", "full"], default="sample", help="图片尺寸检查方式")
    parser.add_argument("--size-sample-frames", type=int, default=3, help="sample 模式下每个 clip 抽查前多少帧")
    parser.add_argument("--max-examples", type=int, default=5, help="每类问题最多打印多少个示例")
    parser.add_argument("--only-problems", action="store_true", help="只打印非 complete 的 clip；等价于 --report-mode problems")
    parser.add_argument("--report-mode", choices=["summary", "problems", "all"], default="problems", help="终端报告详细程度；CSV 始终写全量结果")
    parser.add_argument("--problem-limit", type=int, default=20, help="summary 中最多展示多少个优先问题 clip")
    parser.add_argument("--csv-out", default=None, help="可选：把完整结果写入 CSV")
    parser.add_argument("--fail-on-problems", action="store_true", help="存在非 complete clip 时返回非 0")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条")
    return parser.parse_args()


def main():
    args = parse_args()
    day_dir = Path(args.root) / args.day
    print("[scan] {}".format(day_dir))
    print("[config] expected_cameras={} expected_size={}x{} check_size_mode={}".format(
        args.expected_cameras, args.expected_width, args.expected_height, args.check_size_mode))

    sequence_names = select_sequence_names(day_dir, args.day, args.sequences)
    print("[sequence-filter] {}".format(sequence_names if args.sequences else "all"))

    tasks, reports = build_clip_tasks(day_dir, sequence_names)
    for sequence_dir, clip in iter_clip_tasks(tasks, args):
        reports.append(scan_clip(sequence_dir, clip, args))

    print_report(args.day, reports, args)
    if args.csv_out:
        write_csv(args.csv_out, reports)

    bad = [r for r in reports if r.status != "complete"]
    if args.fail_on_problems and bad:
        print("[fail] non_complete_clips={}".format(len(bad)), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
