#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N7 mAVE、速度分桶和 eval GT 汇总回归。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.n7_eval_velocity import summarize_velocity_errors

try:
    from mmdet3d.datasets.custom_multiview_dataset import CustomMultiViewDataset
except ImportError:
    CustomMultiViewDataset = None


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / 'tools' / 'eval_epoch_checkpoints.py'
SPEC = importlib.util.spec_from_file_location(
    'eval_epoch_checkpoints_velocity_test', TOOL_PATH)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def box(x, vx=None, vy=None):
    values = [x, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    if vx is not None and vy is not None:
        values.extend([vx, vy])
    return np.asarray(values, dtype=np.float32)


def synthetic_record():
    metrics = {
        'eval/metric_schema_version': 3,
        'mAP': 0.3,
        'mAP/center_dist': 0.3,
        'mAP/bev_iou@0.5': 0.31,
        'mATE@2m': 0.9,
        'mAOE_deg@2m': 5.0,
        'mASE@2m': 0.2,
        'mAVE@2m': 1.5,
        'mAVE': 1.5,
        'eval/gt_num': 30,
        'eval/det_num': 40,
    }
    for class_name, gt_num, det_num, ave in (
            ('car', 20, 25, 1.0), ('truck', 10, 15, 2.0)):
        metrics.update({
            '{}/gt_num'.format(class_name): gt_num,
            '{}/det_num'.format(class_name): det_num,
            '{}/AVE@2m'.format(class_name): ave,
            '{}/velocity_TP_num@2m'.format(class_name): 8,
            '{}/velocity_l2_p50@2m'.format(class_name): ave - 0.2,
            '{}/velocity_l2_p90@2m'.format(class_name): ave + 0.5,
            '{}/vx_abs_error_mean@2m'.format(class_name): 0.7,
            '{}/vx_abs_error_p50@2m'.format(class_name): 0.6,
            '{}/vx_abs_error_p90@2m'.format(class_name): 1.2,
            '{}/vy_abs_error_mean@2m'.format(class_name): 0.4,
            '{}/vy_abs_error_p50@2m'.format(class_name): 0.3,
            '{}/vy_abs_error_p90@2m'.format(class_name): 0.8,
            '{}/vx_error_mean@2m'.format(class_name): 0.1,
            '{}/vy_error_mean@2m'.format(class_name): -0.2,
        })
        prefix = '{}/range_0_20m'.format(class_name)
        metrics.update({
            '{}/gt_num'.format(prefix): gt_num,
            '{}/tp_num'.format(prefix): 8,
            '{}/recall@2m'.format(prefix): 8.0 / gt_num,
            '{}/velocity_tp_num'.format(prefix): 8,
            '{}/velocity_l2_mean'.format(prefix): ave,
            '{}/velocity_l2_p50'.format(prefix): ave - 0.2,
            '{}/velocity_l2_p90'.format(prefix): ave + 0.5,
            '{}/vx_abs_mean'.format(prefix): 0.7,
            '{}/vx_abs_p50'.format(prefix): 0.6,
            '{}/vx_abs_p90'.format(prefix): 1.2,
            '{}/vy_abs_mean'.format(prefix): 0.4,
            '{}/vy_abs_p50'.format(prefix): 0.3,
            '{}/vy_abs_p90'.format(prefix): 0.8,
            '{}/vx_mean'.format(prefix): 0.1,
            '{}/vy_mean'.format(prefix): -0.2,
        })
    return {
        'epoch': 1,
        'status': 'ok',
        'checkpoint': '/tmp/epoch_1.pth',
        'result_pkl': '/tmp/epoch_1_val_results.pkl',
        'metrics': metrics,
        'score_distribution': {'classes': {}},
    }


class N7EvalVelocityTest(unittest.TestCase):
    def test_velocity_l2_components_ranges_and_invalid_boxes(self):
        matches = [
            dict(
                pred_box=box(10.0, 2.0, 4.0),
                gt_box=box(10.0, 1.0, 2.0)),
            dict(
                pred_box=box(30.0, -2.0, 1.0),
                gt_box=box(30.0, -1.0, 1.0)),
            # 旧 7 维预测框没有速度，不能按零误差计入。
            dict(pred_box=box(10.0), gt_box=box(10.0, 0.0, 0.0)),
        ]
        stats = summarize_velocity_errors(
            matches, ((0, 20), (20, 40), (40, 60)))
        overall = stats['overall']
        self.assertEqual(overall['velocity_tp_num'], 2)
        self.assertAlmostEqual(
            overall['velocity_l2_mean'], (np.sqrt(5.0) + 1.0) / 2.0,
            places=6)
        self.assertAlmostEqual(overall['vx_abs_mean'], 1.0, places=6)
        self.assertAlmostEqual(overall['vx_mean'], 0.0, places=6)
        self.assertAlmostEqual(overall['vy_abs_mean'], 1.0, places=6)
        self.assertAlmostEqual(overall['vy_mean'], 1.0, places=6)
        self.assertEqual(stats['ranges']['0_20m']['velocity_tp_num'], 1)
        self.assertEqual(stats['ranges']['20_40m']['velocity_tp_num'], 1)
        self.assertEqual(stats['ranges']['40_60m']['velocity_tp_num'], 0)
        self.assertIsNone(stats['ranges']['40_60m']['velocity_l2_mean'])

    def test_summary_contains_mave_gt_and_velocity_details(self):
        record = synthetic_record()
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            TOOL.write_json(result_dir / 'epoch_1_metrics.json', record)
            TOOL.write_summaries(result_dir, 'mAP')

            markdown = (result_dir / 'eval_summary.md').read_text(
                encoding='utf-8')
            self.assertIn(
                '- eval_gt: `30` (car=20, truck=10); shared by all '
                'evaluated epochs', markdown)
            self.assertIn('mAVE@2m | eval_gt | eval_det', markdown)
            self.assertIn('## TP Velocity Detail By Class', markdown)
            self.assertIn(
                '## TP Velocity Absolute Error By GT X Range', markdown)
            self.assertIn(
                '## TP Velocity Signed Bias By GT X Range', markdown)

            summary = json.loads(
                (result_dir / 'eval_summary.json').read_text(
                    encoding='utf-8'))
            self.assertEqual(summary['metric_schema_version'], 3)
            self.assertEqual(summary['rows'][0]['eval_gt'], 30)
            self.assertEqual(summary['rows'][0]['mAVE@2m'], 1.5)
            self.assertEqual(
                summary['velocity_class_rows'][0]['velocity_tp'], 8)
            self.assertTrue(
                (result_dir / 'eval_velocity_summary.csv').is_file())

    def test_eval_gt_falls_back_to_class_counts_for_old_record(self):
        record = synthetic_record()
        record['metrics'].pop('eval/gt_num')
        row = TOOL.flatten_record(record, best_epoch=1)
        self.assertEqual(row['eval_gt'], 30)

    def test_cfg_options_require_key_value_form(self):
        with self.assertRaisesRegex(
                ValueError, 'Expected key=value option'):
            TOOL.parse_key_value_options(['data.test.samples_per_g'])

    @unittest.skipIf(
        CustomMultiViewDataset is None,
        '当前本地环境没有 legacy mmcv/mmdet3d 依赖')
    def test_legacy_dataset_parses_gt_velocity_and_reuses_tp_matches(self):
        dataset = CustomMultiViewDataset.__new__(CustomMultiViewDataset)
        dataset.CLASSES = ('car', 'truck')
        dataset.eval_range = None
        dataset.eval_score_thr = 0.0
        dataset.eval_max_dets_per_sample = None
        dataset.filter_gt_visible_camera = None
        info = {
            'gt_boxes': np.asarray([
                [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            ], dtype=np.float32),
            'gt_names': np.asarray(['car']),
            'gt_velocity': np.asarray([[1.0, 2.0]], dtype=np.float32),
        }
        gt_boxes, gt_labels = dataset._parse_gt_info(info)
        self.assertEqual(gt_boxes.shape, (1, 9))
        np.testing.assert_array_equal(gt_boxes[0, 7:9], [1.0, 2.0])
        np.testing.assert_array_equal(gt_labels, [0])

        pred_boxes, pred_scores, pred_labels = dataset._parse_det_result({
            'boxes_3d': np.asarray([
                [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 2.0, 4.0],
            ], dtype=np.float32),
            'scores_3d': np.asarray([0.9], dtype=np.float32),
            'labels_3d': np.asarray([0], dtype=np.int64),
            'box_origin': 'center',
        })
        self.assertEqual(pred_boxes.shape, (1, 9))
        np.testing.assert_array_equal(pred_boxes[0, 7:9], [2.0, 4.0])
        np.testing.assert_allclose(pred_scores, [0.9], rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(pred_labels, [0])

        stats = dataset._summarize_tp_errors([
            dict(pred_box=pred_boxes[0], gt_box=gt_boxes[0]),
        ])
        self.assertEqual(stats['velocity_tp_num'], 1)
        self.assertAlmostEqual(stats['ave'], np.sqrt(5.0), places=6)


if __name__ == '__main__':
    unittest.main()
