#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把本地 N7 raw tmp 原图批量 resize 到指定缓存尺寸，并做任务级校验。

职责边界：
    这个脚本只负责本地离线缓存 resize，例如 1600x900 -> 704x256，或
    2560x1440 -> 1600x900 / 704x256。脚本内的 --verify-mode 是 resize
    任务级校验，用于确认本次扫描到的 raw 图片是否都处理成功，以及是否可以
    安全删除 raw；它不替代 tools/data_converter/n7/check_n7_704_day_clips.py 的数据集级完整性检查。

何时需要运行：
    1. ozone 源图是 1600x900：本步骤可选。若直接训练 1600x900，可跳过；
       若希望减少训练 IO/CPU、并保持板端强制 resize 行为一致，建议缓存到 704x256。
    2. ozone 源图是 2560x1440：本步骤必须运行，目标尺寸可选 1600x900 或 704x256。

目录约定：
    输入原图：data/N7_raw_tmp/<day>/<sequence>/parsed_data/<clip>/frames/.../images/...
    输出图片：<output-root>/<day>/<sequence>/parsed_data/<clip>/frames/.../images/...
    标签目录：<output-root>/<day>/<sequence>/output/<clip>/3D_OD/lidar/*.json

效率说明：
    N7 标准图片结构是 ``parsed_data/<clip>/frames/<lidar_ts>/images/<cam_id>/<img>.jpg``。
    旧版在 sequence 级别使用 ``rglob('*')`` 全量递归，目录很多时扫描阶段会很慢。
    这里优先按 clip/frame/images/cam 四层结构扫描，只在遇到非标准目录时才退回
    os.walk 兜底。

示例：
    # 缓存到 Fast-BEV/板端一致的 704x256。
    python tools/data_converter/n7/resize_n7_raw_tmp.py \
      --days 20251017 20251030 20251031 20251203 \
      --width 704 --height 256 \
      --workers 96 \
      --resize-chunk-size 64 \
      --verify-mode fast

    # 如果源是 2560x1440，也可以先缓存到 1600x900。
    python tools/data_converter/n7/resize_n7_raw_tmp.py \
      --days 20251017 \
      --source-image-size 1440 2560 \
      --width 1600 --height 900 \
      --output-root data/N7_1600_900

    # 确认输出没有问题后，校验通过即删除对应 raw tmp。
    python tools/data_converter/n7/resize_n7_raw_tmp.py \
      --days 20251017 \
      --verify-mode full \
      --delete-raw-after-verify
"""

import argparse
import concurrent.futures as futures
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LABEL_EXTS = (".json",)


@dataclass
class WorkUnit:
    """一个本地 resize 单元。"""

    name: str
    mode: str
    image_tmp: Path
    image_out: Path
    label_out: Path
    delete_tmp: Path


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


def is_image_name(name):
    """判断文件名是否是图片。"""

    return os.path.splitext(str(name))[1].lower() in IMAGE_EXTS


def is_frame_image_path(path):
    """判断路径是否位于 frames/<lidar_ts>/images 目录下。"""

    normalized = str(path).replace("\\", "/")
    parts = normalized.split("/")
    return is_image_name(normalized) and "frames" in parts and "images" in parts


def scandir_dirs(path):
    """快速列出一级子目录 Path。"""

    try:
        with os.scandir(path) as entries:
            return sorted((Path(entry.path) for entry in entries if entry.is_dir()), key=lambda p: p.name)
    except FileNotFoundError:
        return []


def scandir_image_files(path):
    """快速列出一级目录中的图片文件。"""

    files = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and is_image_name(entry.name):
                    files.append(Path(entry.path))
    except FileNotFoundError:
        return []
    return sorted(files)


def iter_progress(items, desc, unit, no_progress):
    """根据环境选择 tqdm 或普通迭代。"""

    if not no_progress and tqdm is not None:
        return tqdm(
            items,
            total=len(items) if hasattr(items, "__len__") else None,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            mininterval=1.0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
    return items


def discover_clip_roots(src_root):
    """发现标准 N7 clip 根目录。"""

    src_root = Path(src_root)
    if (src_root / "frames").is_dir():
        return [src_root]
    clip_roots = []
    for clip_dir in scandir_dirs(src_root):
        if (clip_dir / "frames").is_dir():
            clip_roots.append(clip_dir)
    return clip_roots


def collect_images_in_frame(frame_dir):
    """收集单帧 images 目录下的图片。"""

    images_dir = Path(frame_dir) / "images"
    if not images_dir.exists():
        return []

    files = []
    saw_camera_dir = False
    try:
        with os.scandir(images_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    saw_camera_dir = True
                    files.extend(scandir_image_files(entry.path))
                elif entry.is_file() and is_image_name(entry.name):
                    files.append(Path(entry.path))
    except FileNotFoundError:
        return []

    # 标准结构中不会走到这里；这里只是兼容 images 下还有一层未知目录的历史数据。
    if not saw_camera_dir and not files:
        for root, _, filenames in os.walk(images_dir):
            for filename in filenames:
                if is_image_name(filename):
                    files.append(Path(root) / filename)
    return sorted(files)


def collect_images_in_clip(clip_root):
    """按 frames/<timestamp>/images 结构收集一个 clip 的图片。"""

    frames_root = Path(clip_root) / "frames"
    files = []
    for frame_dir in scandir_dirs(frames_root):
        files.extend(collect_images_in_frame(frame_dir))
    return files


def collect_images_fallback(root):
    """非标准目录兜底扫描。"""

    files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if is_frame_image_path(path):
                files.append(path)
    return sorted(files)


def collect_image_files(root, no_progress=False, log=True):
    """结构化收集某个根目录下所有 frames/*/images 图片。"""

    root = Path(root)
    start = time.time()
    if not root.exists():
        if log:
            print("[scan-images] root missing: {}".format(root))
        return []

    clip_roots = discover_clip_roots(root)
    files = []
    if clip_roots:
        iterable = iter_progress(clip_roots, "[scan-images]", "clip", no_progress)
        for clip_root in iterable:
            if tqdm is not None and hasattr(iterable, "set_postfix_str"):
                iterable.set_postfix_str(clip_root.name, refresh=False)
            files.extend(collect_images_in_clip(clip_root))
    else:
        if log:
            print("[scan-images] 非标准目录，使用 os.walk 兜底：{}".format(root))
        files = collect_images_fallback(root)

    files = sorted(files)
    if log:
        print("[scan-images-done] root={} images={} elapsed={}".format(
            root, len(files), format_seconds(time.time() - start)))
    return files


def collect_local_images(root):
    """收集某个本地根目录下所有 frames/*/images 图片的相对路径集合。"""

    root = Path(root)
    if not root.exists():
        return set()
    return {p.relative_to(root).as_posix() for p in collect_image_files(root, no_progress=True, log=False)}


def count_local_labels(root):
    """统计标签 json 数量。"""

    root = Path(root)
    if not root.exists():
        return 0
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in LABEL_EXTS:
                count += 1
    return count


def pil_bilinear_resample():
    """兼容不同 Pillow 版本的 BILINEAR 常量。"""

    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def image_save_format(dst):
    """根据输出后缀决定保存格式。"""

    suffix = Path(dst).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    if suffix == ".bmp":
        return "BMP"
    return "JPEG"


def output_path_for_source(src, src_root, dst_root):
    """根据临时原图路径计算最终输出路径。"""

    src = Path(src)
    rel = src.relative_to(src_root)
    return Path(dst_root) / rel


def resize_one_image(src, src_root, dst_root, size, quality, overwrite, backend, jpeg_draft):
    """把一张本地临时原图 resize 到最终输出目录。"""

    src = Path(src)
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst = output_path_for_source(src, src_root, dst_root)
    if dst.exists() and not overwrite:
        return "skip"

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.stem + ".tmp" + dst.suffix)
    save_format = image_save_format(dst)
    if backend == "cv2":
        if cv2 is None:
            raise RuntimeError("当前环境没有 cv2，不能使用 --resize-backend cv2")
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("OpenCV 读取图片失败：{}".format(src))
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        params = []
        if save_format == "JPEG":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        if not cv2.imwrite(str(tmp), resized, params):
            raise RuntimeError("OpenCV 保存图片失败：{}".format(tmp))
    else:
        with Image.open(src) as img:
            if jpeg_draft and save_format == "JPEG":
                img.draft("RGB", size)
            img = img.convert("RGB")
            img = img.resize(size, pil_bilinear_resample())
            if save_format == "JPEG":
                img.save(tmp, format=save_format, quality=quality, optimize=False)
            else:
                img.save(tmp, format=save_format)
    os.replace(tmp, dst)
    return "ok"


def iter_chunks(items, chunk_size):
    """把图片列表切成批，减少多进程任务提交次数。"""

    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def merge_resize_counters(total, part):
    """合并 worker 返回的 resize 统计。"""

    for key in ("ok", "skip", "fail", "processed"):
        total[key] = total.get(key, 0) + int(part.get(key, 0))
    if part.get("errors"):
        total.setdefault("errors", [])
        total["errors"].extend(part["errors"][:max(0, 10 - len(total["errors"]))])


def resize_image_chunk(paths, src_root, dst_root, size, quality, overwrite, backend, jpeg_draft):
    """一个 worker 一次处理多张图。"""

    counters = {"ok": 0, "skip": 0, "fail": 0, "processed": 0, "errors": []}
    for path in paths:
        counters["processed"] += 1
        try:
            result = resize_one_image(path, src_root, dst_root, size, quality, overwrite, backend, jpeg_draft)
            counters[result] = counters.get(result, 0) + 1
        except Exception as exc:
            counters["fail"] += 1
            if len(counters["errors"]) < 5:
                counters["errors"].append("{}: {}".format(path, exc))
    return counters


def print_resize_progress(done, total, start_time, counters):
    """普通日志模式下打印 resize 实时进度。"""

    elapsed = max(time.time() - start_time, 1e-6)
    speed = done / elapsed
    remain = max(total - done, 0)
    eta = "unknown" if speed <= 0 else format_seconds(remain / speed)
    percent = 100.0 if total <= 0 else min(100.0, done * 100.0 / total)
    print(
        "[resize-progress] {}/{} ({:.1f}%) ok={} skip={} fail={} speed={:.1f} img/s eta={}".format(
            done, total, percent, counters.get("ok", 0), counters.get("skip", 0),
            counters.get("fail", 0), speed, eta
        ),
        flush=True,
    )


def filter_existing_outputs(image_files, src_root, dst_root, overwrite):
    """根据输出文件是否已存在过滤 todo 图片。"""

    if overwrite:
        return list(image_files), 0
    dst_root = Path(dst_root)
    if not dst_root.exists():
        return list(image_files), 0
    todo = []
    skip = 0
    for src in image_files:
        if output_path_for_source(src, src_root, dst_root).exists():
            skip += 1
        else:
            todo.append(src)
    return todo, skip


def resize_images(
    src_root, dst_root, size, quality, workers, overwrite, backend, jpeg_draft,
    chunk_size, progress_interval, progress_seconds, no_progress,
):
    """多进程 resize 一个目录下所有图片，并返回统计信息。"""

    src_root = Path(src_root)
    dst_root = Path(dst_root)
    image_files = collect_image_files(src_root, no_progress=no_progress, log=True)
    todo_files, skip_existing = filter_existing_outputs(image_files, src_root, dst_root, overwrite)
    counters = {
        "ok": 0,
        "skip": skip_existing,
        "fail": 0,
        "processed": 0,
        "total": len(image_files),
        "submitted": len(todo_files),
        "errors": [],
    }
    if not image_files:
        return counters

    print("[resize] total={} todo={} skip_existing={} backend={} chunk_size={} : {} -> {}".format(
        len(image_files), len(todo_files), skip_existing, backend, chunk_size, src_root, dst_root))
    if not todo_files:
        return counters

    total_todo = len(todo_files)
    chunk_list = list(iter_chunks(todo_files, chunk_size))
    progress_interval = max(0, int(progress_interval))
    progress_seconds = max(0.0, float(progress_seconds))
    progress_start = time.time()
    last_report_time = progress_start
    next_report = progress_interval if progress_interval > 0 else total_todo

    pbar = None
    if not no_progress and tqdm is not None:
        pbar = tqdm(
            total=total_todo,
            desc="[resize]",
            unit="img",
            dynamic_ncols=True,
            mininterval=1.0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    def report_manual_if_needed(done):
        """没有 tqdm 时按数量或时间阈值输出进度。"""

        nonlocal next_report, last_report_time
        if pbar is not None:
            return
        should_report = done >= total_todo
        if progress_interval > 0 and done >= next_report:
            should_report = True
        if progress_seconds > 0 and time.time() - last_report_time >= progress_seconds:
            should_report = True
        if not should_report:
            return
        print_resize_progress(done, total_todo, progress_start, counters)
        last_report_time = time.time()
        while progress_interval > 0 and next_report <= done:
            next_report += progress_interval

    done = 0
    try:
        if workers <= 1:
            for chunk in chunk_list:
                part = resize_image_chunk(chunk, src_root, dst_root, size, quality, overwrite, backend, jpeg_draft)
                merge_resize_counters(counters, part)
                done += int(part.get("processed", len(chunk)))
                if pbar is not None:
                    pbar.update(int(part.get("processed", len(chunk))))
                    pbar.set_postfix(ok=counters.get("ok", 0), skip=counters.get("skip", 0), fail=counters.get("fail", 0))
                report_manual_if_needed(done)
        else:
            with futures.ProcessPoolExecutor(max_workers=workers) as executor:
                future_list = [
                    executor.submit(resize_image_chunk, chunk, src_root, dst_root, size, quality, overwrite, backend, jpeg_draft)
                    for chunk in chunk_list
                ]
                for future in futures.as_completed(future_list):
                    part = future.result()
                    merge_resize_counters(counters, part)
                    processed = int(part.get("processed", 0))
                    done += processed
                    if pbar is not None:
                        pbar.update(processed)
                        pbar.set_postfix(ok=counters.get("ok", 0), skip=counters.get("skip", 0), fail=counters.get("fail", 0))
                    report_manual_if_needed(done)
    finally:
        if pbar is not None:
            pbar.close()

    for message in counters.get("errors", [])[:10]:
        print("[resize-fail] {}".format(message), file=sys.stderr)
    return counters


def ensure_under(path, root):
    """删除临时目录前做路径保护。"""

    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise RuntimeError("拒绝删除非 raw tmp 目录：{} 不在 {} 下".format(path, root))


def delete_tmp_dir(path, raw_tmp_root, dry_run=False):
    """删除一个已校验通过的临时原图目录。"""

    path = Path(path)
    if not path.exists():
        print("[clean] 临时目录不存在，跳过：{}".format(path))
        return
    ensure_under(path, raw_tmp_root)
    if dry_run:
        print("[dry-run] 删除临时目录：{}".format(path))
        return
    print("[clean] 删除临时目录：{}".format(path))
    shutil.rmtree(path)
    if path.exists():
        raise RuntimeError("删除后目录仍存在：{}".format(path))


def parse_clip_scope(clip_scope):
    """解析 day/sequence/parsed_data/clip 格式。"""

    parts = clip_scope.strip("/").split("/")
    if len(parts) < 4 or parts[2] != "parsed_data":
        raise ValueError("clip 必须是 day/sequence/parsed_data/clip 格式：{}".format(clip_scope))
    return parts[0], parts[1], parts[3]


def make_sequence_unit(args, sequence_scope):
    """构造 sequence resize 单元。"""

    sequence_scope = sequence_scope.strip("/")
    return WorkUnit(
        name=sequence_scope,
        mode="sequence",
        image_tmp=Path(args.raw_tmp_root) / sequence_scope / "parsed_data",
        image_out=Path(args.output_root) / sequence_scope / "parsed_data",
        label_out=Path(args.output_root) / sequence_scope / "output",
        delete_tmp=Path(args.raw_tmp_root) / sequence_scope,
    )


def make_clip_unit(args, clip_scope):
    """构造 clip resize 单元。"""

    clip_scope = clip_scope.strip("/")
    day, sequence, clip = parse_clip_scope(clip_scope)
    label_scope = "/".join([day, sequence, "output", clip, "3D_OD", "lidar"])
    return WorkUnit(
        name=clip_scope,
        mode="clip",
        image_tmp=Path(args.raw_tmp_root) / clip_scope,
        image_out=Path(args.output_root) / clip_scope,
        label_out=Path(args.output_root) / label_scope,
        delete_tmp=Path(args.raw_tmp_root) / clip_scope,
    )


def list_local_sequences(raw_tmp_root, day):
    """列出本地某一天下已经下载的 sequence。"""

    day_dir = Path(raw_tmp_root) / day
    if not day_dir.exists():
        print("[list] {}: local day dir missing: {}".format(day, day_dir))
        return []
    seq_names = [p.name for p in scandir_dirs(day_dir)]
    print("[list] {}: {} local sequences".format(day, len(seq_names)))
    return sorted(seq_names)


def expand_work_units(args):
    """根据 days / sequences / clips 展开本地 resize 单元。"""

    units = []
    if args.clips:
        units.extend(make_clip_unit(args, clip) for clip in args.clips)
    if args.sequences:
        units.extend(make_sequence_unit(args, sequence) for sequence in args.sequences)
    if args.days:
        for day in args.days:
            day = day.strip("/")
            for seq_name in list_local_sequences(args.raw_tmp_root, day):
                units.append(make_sequence_unit(args, "/".join([day, seq_name])))
    return units


def verify_resize(unit, args, resize_counters):
    """校验 resize 结果，返回是否通过。"""

    if resize_counters.get("fail", 0) > 0:
        print("[verify-fail] resize 有失败，保留 raw：{}".format(unit.delete_tmp))
        return False
    total = int(resize_counters.get("total", 0))
    finished = int(resize_counters.get("ok", 0)) + int(resize_counters.get("skip", 0))
    if total == 0 and not args.allow_empty_images:
        print("[verify-fail] raw 图片数为 0，保留 raw：{}".format(unit.delete_tmp))
        return False
    if finished < total:
        print("[verify-fail] 完成数不足 finished={} total={}，保留 raw".format(finished, total))
        return False

    label_count = count_local_labels(unit.label_out)
    if not args.images_only and label_count == 0:
        print("[verify-warn] 未看到标签 json：{}".format(unit.label_out))
        if args.require_labels:
            return False

    if args.verify_mode == "fast":
        print("[verify] mode=fast total={} finished={} label_json={}".format(total, finished, label_count))
        return True
    if args.verify_mode == "none":
        print("[verify] mode=none，仅检查 resize 失败数和完成数")
        return True

    raw_rels = collect_local_images(unit.image_tmp)
    out_rels = collect_local_images(unit.image_out)
    missing = raw_rels - out_rels
    print("[verify] mode=full raw_images={} resized_images={} missing={} label_json={}".format(
        len(raw_rels), len(out_rels), len(missing), label_count))
    if missing:
        print("[verify-fail] 示例缺失：{}".format(sorted(missing)[:5]))
        return False
    return True


def process_unit(unit, args):
    """处理一个本地 resize 单元。"""

    start = time.time()
    print("\n[unit] {} ({})".format(unit.name, unit.mode))
    step_start = time.time()
    counters = resize_images(
        unit.image_tmp,
        unit.image_out,
        (args.width, args.height),
        args.quality,
        args.workers,
        args.overwrite,
        args.resize_backend,
        args.jpeg_draft,
        args.resize_chunk_size,
        args.progress_interval,
        args.progress_seconds,
        args.no_progress,
    )
    elapsed = time.time() - step_start
    total = int(counters.get("total", 0))
    throughput = 0.0 if total <= 0 else total / max(elapsed, 1e-6)
    print("[resize-done] {} elapsed={} throughput={:.1f} img/s".format(
        counters, format_seconds(elapsed), throughput))

    step_start = time.time()
    ok = verify_resize(unit, args, counters)
    print("[verify-done] ok={} elapsed={}".format(ok, format_seconds(time.time() - step_start)))
    if ok and args.delete_raw_after_verify:
        delete_tmp_dir(unit.delete_tmp, args.raw_tmp_root, dry_run=args.dry_run)
    print("[unit-done] {} elapsed={}".format(unit.name, format_seconds(time.time() - start)))
    return ok


def parse_args():
    parser = argparse.ArgumentParser(description="把本地 N7 raw tmp 原图批量 resize 到指定缓存尺寸，并做任务级校验。")
    parser.add_argument("--raw-tmp-root", default="data/N7_raw_tmp", help="远端原图临时根目录")
    parser.add_argument("--output-root", default="data/N7_704_256", help="缓存图片和标签根目录")
    parser.add_argument("--days", nargs="*", default=None, help="按本地 day 处理，例如 20251017")
    parser.add_argument("--sequences", nargs="*", default=None, help="按 sequence 处理，例如 20251031/20251031_164821")
    parser.add_argument("--clips", nargs="*", default=None, help="按 clip 处理，格式 day/sequence/parsed_data/clip")
    parser.add_argument("--width", type=int, default=704, help="输出图像宽度")
    parser.add_argument("--source-image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None, help="raw_tmp 源图尺寸，仅用于日志说明，例如 900 1600 或 1440 2560")
    parser.add_argument("--height", type=int, default=256, help="输出图像高度")
    parser.add_argument("--quality", type=int, default=90, help="JPEG 输出质量")
    parser.add_argument("--workers", type=int, default=max(1, min((os.cpu_count() or 8), 96)), help="本地 resize 进程数")
    parser.add_argument("--resize-chunk-size", type=int, default=64, help="每个 resize 任务一次处理多少张图")
    parser.add_argument("--progress-interval", type=int, default=2000, help="无 tqdm 时每处理多少张待 resize 图片打印一次进度；设为 0 表示只按时间打印")
    parser.add_argument("--progress-seconds", type=float, default=10.0, help="无 tqdm 时距离上次进度日志超过多少秒也打印一次；设为 0 表示关闭时间触发")
    parser.add_argument("--resize-backend", default="pillow", choices=["pillow", "cv2"], help="resize 后端；pillow 更接近训练 pipeline")
    parser.add_argument("--jpeg-draft", action="store_true", help="Pillow JPEG 粗解码加速，像素会与严格 resize 略有差异")
    parser.add_argument("--verify-mode", default="full", choices=["full", "fast", "none"], help="校验方式")
    parser.add_argument("--require-labels", action="store_true", help="没有标签 json 时视为校验失败")
    parser.add_argument("--images-only", action="store_true", help="只检查图片，不要求标签")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有缓存输出图片")
    parser.add_argument("--allow-empty-images", action="store_true", help="允许 raw 图片数为 0")
    parser.add_argument("--delete-raw-after-verify", action="store_true", help="校验通过后删除对应 raw tmp；默认不删除")
    parser.add_argument("--dry-run", action="store_true", help="只打印删除操作，不实际删除；resize 仍会执行")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条，改用普通日志")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.days and not args.sequences and not args.clips:
        raise SystemExit("必须指定 --days、--sequences 或 --clips 至少一种")
    units = expand_work_units(args)
    print("[plan] {} work units".format(len(units)))
    print("[root] raw_tmp={} output={}".format(args.raw_tmp_root, args.output_root))
    print("[image-size] source_hxw={} target_hxw=({}, {})".format(args.source_image_size, args.height, args.width))
    print("[resize] workers={} chunk_size={} progress_interval={} progress_seconds={} backend={} verify_mode={} delete_raw_after_verify={}".format(
        args.workers, args.resize_chunk_size, args.progress_interval, args.progress_seconds,
        args.resize_backend, args.verify_mode, args.delete_raw_after_verify))
    failed = 0
    for unit in units:
        if not process_unit(unit, args):
            failed += 1
    print("[done] resize finished, failed_units={}".format(failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
