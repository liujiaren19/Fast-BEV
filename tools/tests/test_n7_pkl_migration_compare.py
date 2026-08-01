#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N7 1600x900 PKL 迁移严格对比工具回归。"""

import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = (
    ROOT / 'tools' / 'data_converter' / 'n7' /
    'compare_n7_pkl_migration.py')
SPEC = importlib.util.spec_from_file_location('compare_n7_pkl_migration', TOOL_PATH)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def camera(path, height, width):
    return dict(
        data_path=path,
        image_height=height,
        image_width=width,
        intrinsic_height=900,
        intrinsic_width=1600,
        cam_intrinsic=np.asarray([
            [790.0, 0.0, 800.0],
            [0.0, 790.0, 450.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32),
        distortion=np.asarray([-0.3, 0.1, 0, 0, -0.02], dtype=np.float32),
        sensor2lidar_rotation=np.eye(3, dtype=np.float32),
        sensor2lidar_translation=np.asarray([0.0, 0.0, 1.8], dtype=np.float32))


def info(split, native):
    return dict(
        token='{}_token'.format(split),
        dataset='20251017', sequence='seq', clip='clip_1',
        timestamp=123456,
        gt_boxes=np.asarray([[10, 0, -1, 4, 2, 1.5, 0]], dtype=np.float32),
        gt_names=np.asarray(['car']),
        cams=dict(cam0=camera(
            '20251017/seq/parsed_data/clip_1/frames/123/images/cam0/123.jpg',
            900 if native else 256,
            1600 if native else 704)))


def write_pkl(path, split, native):
    payload = dict(
        infos=[info(split, native)],
        metadata=dict(set=split, gt_box_origin='center', manifest_sources=[]))
    with path.open('wb') as stream:
        pickle.dump(payload, stream)


class N7PklMigrationCompareTest(unittest.TestCase):
    def test_allowed_path_and_size_changes_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / 'old.pkl'
            new_path = root / 'new.pkl'
            write_pkl(old_path, 'train', False)
            write_pkl(new_path, 'train', True)
            report = TOOL.compare_split(
                'train', old_path, new_path, (900, 1600), 20)
            self.assertEqual(report['status'], 'PASS')
            self.assertTrue(report['token_set_equal'])
            self.assertEqual(report['changed_critical_fields'], {})

    def test_gt_change_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / 'old.pkl'
            new_path = root / 'new.pkl'
            write_pkl(old_path, 'val', False)
            write_pkl(new_path, 'val', True)
            payload_infos, metadata = TOOL.load_payload(new_path)
            payload_infos[0]['gt_boxes'][0, 0] += 0.01
            with new_path.open('wb') as stream:
                pickle.dump(dict(infos=payload_infos, metadata=metadata), stream)
            report = TOOL.compare_split(
                'val', old_path, new_path, (900, 1600), 20)
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(report['changed_critical_fields']['gt_boxes'], 1)


if __name__ == '__main__':
    unittest.main()
