#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N7 图像增强目标统计工具的合成回归。"""

import argparse
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / 'tools' / 'data_converter' / 'n7'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    'analyze_n7_image_aug_targets_for_test',
    SCRIPT_DIR / 'analyze_n7_image_aug_targets.py',
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
analyze = MODULE.analyze
distance_bin_name = MODULE.distance_bin_name
validate_args = MODULE.validate_args
write_outputs = MODULE.write_outputs


class N7ImageAugTargetStatsTest(unittest.TestCase):
    def test_distance_bins_cover_below_and_above_range(self):
        bins = [0, 20, 40, 60, 80]
        self.assertEqual(distance_bin_name(-1.0, bins), '<0m')
        self.assertEqual(distance_bin_name(20.0, bins), '20-40m')
        self.assertEqual(distance_bin_name(80.0, bins), '>=80m')

    def test_synthetic_near_far_and_edge_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pkl_path = root / 'infos.pkl'
            output_dir = root / 'output'
            cam_to_lidar = np.array([
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ], dtype=np.float32)
            camera = dict(
                data_path='unused.jpg',
                cam_intrinsic=np.array([
                    [1000.0, 0.0, 800.0],
                    [0.0, 900.0, 450.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float32),
                sensor2lidar_rotation=cam_to_lidar,
                sensor2lidar_translation=np.array(
                    [0.0, 0.0, 1.8], dtype=np.float32),
                distortion=np.array(
                    [-0.05, 0.01, 0.0, 0.0, 0.0], dtype=np.float32),
                intrinsic_width=1600,
                intrinsic_height=900,
                image_width=704,
                image_height=256,
            )
            infos = [dict(
                token='synthetic',
                timestamp=1,
                cams={'cam0': camera},
                gt_boxes=np.array([
                    [10.0, 0.0, 1.0, 4.0, 2.0, 1.6, 0.0],
                    [50.0, 8.0, 1.0, 4.5, 2.2, 2.0, 0.2],
                ], dtype=np.float32),
                gt_names=np.array(['car', 'truck'], dtype=object),
            )]
            with pkl_path.open('wb') as stream:
                pickle.dump({'infos': infos, 'metadata': {}}, stream)

            args = argparse.Namespace(
                pkl=str(pkl_path),
                output_dir=str(output_dir),
                camera_id='cam0',
                classes=['car', 'truck'],
                roi=[0, -35, -5, 80, 35, 3],
                distance_bins=[0, 20, 40, 60, 80],
                edge_thresholds=[0, 4, 8, 16, 32],
                samples_per_edge=17,
                min_depth=0.1,
                start_index=0,
                stride=1,
                max_frames=-1,
            )
            validate_args(args)
            records, summary = analyze(args)
            write_outputs(records, summary, output_dir)

            self.assertEqual(len(records), 2)
            self.assertEqual(summary['counters']['used_targets'], 2)
            self.assertEqual(
                summary['groups']['class=car|distance=0-20m']['count'], 1)
            self.assertEqual(
                summary['groups']['class=truck|distance=40-60m']['count'], 1)
            near = next(row for row in records if row['class_name'] == 'car')
            far = next(row for row in records if row['class_name'] == 'truck')
            self.assertGreater(
                near['bbox_width_px'], far['bbox_width_px'])
            self.assertGreater(
                near['bbox_height_px'], far['bbox_height_px'])
            self.assertTrue((output_dir / 'target_records.csv').is_file())
            self.assertTrue((output_dir / 'summary.md').is_file())
            persisted = json.loads(
                (output_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(
                persisted['counters']['projected_targets'], 2)


if __name__ == '__main__':
    unittest.main()
