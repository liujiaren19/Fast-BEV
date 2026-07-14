#!/usr/bin/env python3
"""校验 N7 converter 生成的 Fast-BEV pkl 数据门禁和几何字段。

该脚本用于 converter 后、训练/可视化前的自动化检查，职责和
``visualize_n7_fastbev_pkl.py`` 分开：

1. 全量检查 train/val/test token/clip 零重叠、camera/尺寸/K/畸变、GT 字段、
   类别与 box 分布，以及单帧/时序 pose 和历史 offset；
2. 检查 pkl 中 ``sensor2lidar_rotation/translation`` 和 ``cam_intrinsic`` 能否
   按 ``CustomMultiViewDataset`` 的读取逻辑组成正确的 ``lidar2img``。
3. 检查 converter 写入的 ``intrinsic_width/height`` 与 ``image_width/height``
   字段是否会被 dataset/pipeline 以一致的方式传递。
4. 可选检查 ``prev/next`` 相邻帧在 pose 存在时是否能补偿到 key frame lidar。

示例：
  # 单帧 S0 正式门禁：同时传 train/val/test，报告写入 data_gate.json/.md。
  python tools/data_converter/n7/validate_n7_fastbev_pkl.py \
--gt-pkl /path/train.pkl /path/val.pkl /path/test.pkl \
--output-dir work_dirs/validate_n7_704 \
--expected-camera-ids cam0 --data-mode single-frame \
--expected-image-size 256 704 --expected-intrinsic-size 900 1600 \
--require-splits train val test --strict-data \
--geometry-check-count 200 --strict-geometry

  # 同时检查 prev/next 时序补偿几何，并打印少量相邻帧相对位姿。
  python tools/data_converter/n7/validate_n7_fastbev_pkl.py \
--gt-pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl \
--output-dir work_dirs/validate_n7_704_temporal \
--expected-camera-ids cam0 --data-mode temporal --strict-data \
--check-temporal-geometry --check-poses --pose-check-count 20 --strict-geometry
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from tools.data_converter.n7.fastbev_pkl_data_gate import (
        audit_fastbev_splits,
        write_data_gate_report,
    )
    from tools.data_converter.n7.fastbev_geometry import (
        check_geometry_consistency,
        check_temporal_poses,
        write_geometry_report,
    )
except ModuleNotFoundError:
    # 支持从仓库根目录执行，也支持直接在 tools/data_converter/n7 附近调试脚本。
    from fastbev_pkl_data_gate import (
        audit_fastbev_splits,
        write_data_gate_report,
    )
    from fastbev_geometry import (
        check_geometry_consistency,
        check_temporal_poses,
        write_geometry_report,
    )


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RawDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """保留 docstring 示例换行，同时继续展示 argparse 默认值。"""


def load_fastbev_pkl(path: Path) -> Tuple[List[Dict], Dict]:
    """读取 converter 新格式 pkl，也兼容旧版只保存 info 列表的 pkl。"""
    with path.open('rb') as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        infos = payload.get('infos', [])
        metadata = payload.get('metadata', {})
    elif isinstance(payload, list):
        infos = payload
        metadata = {}
    else:
        raise TypeError(f'Unsupported pkl payload type: {type(payload)}')

    if not isinstance(infos, list):
        raise TypeError(f'pkl infos should be a list, got {type(infos)}')
    return infos, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='校验 N7 Fast-BEV pkl 几何字段和时序字段。',
        formatter_class=RawDefaultsHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--gt-pkl', '--pkl', dest='gt_pkl', nargs='+', required=True,
                        help='converter 输出的 GT/info pkl；可同时传 train/val/test，--pkl 是兼容别名')
    parser.add_argument('--output-dir', required=True,
                        help='data_gate.json/.md 和 geometry_check.json 输出目录')
    parser.add_argument('--camera-ids', nargs='+', default=None,
                        help='指定要检查的相机 id；默认使用六目检查顺序并追加 pkl 中其他相机')
    parser.add_argument('--expected-camera-ids', nargs='+', default=None,
                        help='数据门禁要求每个 info 严格匹配的相机顺序；mono-front 传 cam0')
    parser.add_argument('--data-mode', choices=['auto', 'single-frame', 'temporal'], default='auto',
                        help='pose/history 门禁模式；S0 显式传 single-frame，B0 传 temporal')
    parser.add_argument('--expected-image-size', nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'),
                        default=None, help='要求 data_path 图片字段尺寸，例如 256 704')
    parser.add_argument('--expected-intrinsic-size', nargs=2, type=int,
                        metavar=('HEIGHT', 'WIDTH'), default=None,
                        help='要求 K/distortion 对应原生尺寸，例如 900 1600')
    parser.add_argument('--visibility-camera-id', default=None,
                        help='统计当前 pinhole 口径几何可见 GT 的相机；mono-front 默认 cam0')
    parser.add_argument('--data-root', default=None,
                        help='配合 --check-image-files 解析 pkl 相对 data_path')
    parser.add_argument('--check-image-files', action='store_true',
                        help='全量检查图片存在、可读且 header 尺寸与 pkl 一致；大数据会增加 IO')
    parser.add_argument('--require-complete-history', action='store_true',
                        help='temporal 模式把 clip 起始处历史不足也视为失败；默认只警告并统计 fallback 比例')
    parser.add_argument('--require-splits', nargs='+', default=None,
                        choices=['train', 'val', 'test'],
                        help='要求输入中必须包含的 split；正式门禁建议传 train val test')
    parser.add_argument('--max-data-examples', type=int, default=50,
                        help='JSON/Markdown 最多保存多少条 failure/warning 示例')
    parser.add_argument('--strict-data', action='store_true',
                        help='data gate 为 FAIL 时非零退出；未开启时仍生成完整报告')
    parser.add_argument('--skip-data-gate', action='store_true',
                        help='只运行旧几何检查；一般不建议在训练前使用')
    parser.add_argument('--geometry-check-count', type=int, default=200,
                        help='最多检查多少个 key frame info；<=0 表示不检查')
    parser.add_argument('--geometry-tol', type=float, default=1e-3,
                        help='几何一致性检查允许的最大绝对误差')
    parser.add_argument('--check-temporal-geometry', action='store_true',
                        help='继续检查 prev/next 相邻帧补偿到 key frame lidar 后的几何')
    parser.add_argument('--check-poses', action='store_true',
                        help='打印少量 key frame 和历史帧的相对位姿，便于人工确认时序字段')
    parser.add_argument('--pose-check-count', type=int, default=10,
                        help='启用 --check-poses 时最多打印多少组 key/prev 位姿')
    parser.add_argument('--strict-geometry', action='store_true',
                        help='几何检查失败时直接退出，适合自动化校验')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    pkl_paths = [Path(value) for value in args.gt_pkl]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = []
    for pkl_path in pkl_paths:
        infos, metadata = load_fastbev_pkl(pkl_path)
        payloads.append((pkl_path, infos, metadata))
        logger.info('Loaded %d infos from %s', len(infos), pkl_path)
        logger.info('metadata.coordinate=%s', metadata.get('coordinate', 'unknown'))
        logger.info('metadata.data_root=%s', metadata.get('data_root', 'unknown'))

    data_gate_failed = False
    if not args.skip_data_gate:
        expected_camera_ids = args.expected_camera_ids
        if expected_camera_ids is None and args.camera_ids is not None:
            expected_camera_ids = args.camera_ids
        data_gate = audit_fastbev_splits(
            payloads,
            expected_camera_ids=expected_camera_ids,
            mode=args.data_mode,
            expected_image_size=(
                tuple(args.expected_image_size)
                if args.expected_image_size is not None else None),
            expected_intrinsic_size=(
                tuple(args.expected_intrinsic_size)
                if args.expected_intrinsic_size is not None else None),
            visibility_camera_id=args.visibility_camera_id,
            check_image_files=args.check_image_files,
            data_root=Path(args.data_root) if args.data_root else None,
            require_complete_history=args.require_complete_history,
            required_splits=args.require_splits,
            max_examples=max(args.max_data_examples, 1))
        json_path, markdown_path = write_data_gate_report(output_dir, data_gate)
        logger.info(
            'Data gate status=%s failures=%d warnings=%d JSON=%s Markdown=%s',
            data_gate['status'], data_gate['failed_checks'],
            data_gate['warning_checks'], json_path, markdown_path)
        data_gate_failed = data_gate['status'] != 'PASS'

    if args.check_poses:
        for pkl_path, infos, _ in payloads:
            logger.info('Pose samples for %s', pkl_path)
            check_temporal_poses(infos, args.pose_check_count)

    if args.geometry_check_count <= 0:
        logger.warning('Skip geometry check because --geometry-check-count <= 0')
    else:
        multiple_inputs = len(payloads) > 1
        for pkl_path, infos, _ in payloads:
            report = check_geometry_consistency(
                infos=infos,
                camera_ids=args.camera_ids,
                max_infos=args.geometry_check_count,
                tolerance=args.geometry_tol,
                include_temporal=args.check_temporal_geometry,
                strict=args.strict_geometry,
            )
            geometry_output_dir = output_dir / pkl_path.stem if multiple_inputs else output_dir
            report_path = write_geometry_report(geometry_output_dir, report)
            logger.info('Geometry report saved to %s', report_path)

    if args.strict_data and data_gate_failed:
        raise AssertionError('N7 pkl data gate failed; see {}/data_gate.md'.format(output_dir))


if __name__ == '__main__':
    main()
