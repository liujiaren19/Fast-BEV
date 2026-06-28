#!/usr/bin/env python3
"""校验 N7 converter 生成的 Fast-BEV pkl 几何字段。

该脚本用于 converter 后、训练/可视化前的自动化检查，职责和
``visualize_n7_fastbev_pkl.py`` 分开：

1. 检查 pkl 中 ``sensor2lidar_rotation/translation`` 和 ``cam_intrinsic`` 能否
   按 ``CustomMultiViewDataset`` 的读取逻辑组成正确的 ``lidar2img``。
2. 检查 converter 写入的 ``intrinsic_width/height`` 与 ``image_width/height``
   字段是否会被 dataset/pipeline 以一致的方式传递。
3. 可选检查 ``prev/next`` 相邻帧在 pose 存在时是否能补偿到 key frame lidar。

示例：
  # 常规检查：默认只检查 key frame 几何，速度快，适合每次生成 pkl 后执行。
  python tools/data_converter/n7/validate_n7_fastbev_pkl.py \
--gt-pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl \
--output-dir work_dirs/validate_n7_704 \
--geometry-check-count 200 --strict-geometry

  # 同时检查 prev/next 时序补偿几何，并打印少量相邻帧相对位姿。
  python tools/data_converter/n7/validate_n7_fastbev_pkl.py \
--gt-pkl data/N7_704_256/pkl/custom_fastbev_20251031_infos_train_20260624.pkl \
--output-dir work_dirs/validate_n7_704_temporal \
--check-temporal-geometry --check-poses --pose-check-count 20 --strict-geometry
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from tools.data_converter.n7.fastbev_geometry import (
        check_geometry_consistency,
        check_temporal_poses,
        write_geometry_report,
    )
except ModuleNotFoundError:
    # 支持从仓库根目录执行，也支持直接在 tools/data_converter/n7 附近调试脚本。
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
    parser.add_argument('--gt-pkl', '--pkl', dest='gt_pkl', required=True,
                        help='converter 输出的 GT/info pkl 路径；--pkl 是兼容旧命令的别名')
    parser.add_argument('--output-dir', required=True, help='geometry_check.json 输出目录')
    parser.add_argument('--camera-ids', nargs='+', default=None,
                        help='指定要检查的相机 id；默认使用六目检查顺序并追加 pkl 中其他相机')
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

    pkl_path = Path(args.gt_pkl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infos, metadata = load_fastbev_pkl(pkl_path)
    logger.info('Loaded %d infos from %s', len(infos), pkl_path)
    logger.info('metadata.coordinate=%s', metadata.get('coordinate', 'unknown'))
    logger.info('metadata.data_root=%s', metadata.get('data_root', 'unknown'))

    if args.check_poses:
        check_temporal_poses(infos, args.pose_check_count)

    if args.geometry_check_count <= 0:
        logger.warning('Skip geometry check because --geometry-check-count <= 0')
        return

    report = check_geometry_consistency(
        infos=infos,
        camera_ids=args.camera_ids,
        max_infos=args.geometry_check_count,
        tolerance=args.geometry_tol,
        include_temporal=args.check_temporal_geometry,
        strict=args.strict_geometry,
    )
    report_path = write_geometry_report(output_dir, report)
    logger.info('Geometry report saved to %s', report_path)


if __name__ == '__main__':
    main()
