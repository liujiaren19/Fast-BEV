#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""旧版板端结果固化后的 standalone golden 回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

import tools.run_mono_front_board_inference as standalone
from tools.mono_front_board_assets import validate_semantic_output_names
from tools.mono_front_board_core import _semantic_output_map
from tools.tests.mono_front_golden import array_fingerprint, load_golden


def one_anchor_head(num_classes: int) -> dict:
    return {
        "num_classes": num_classes,
        "box_code_size": 7,
        "num_dir_bins": 2,
        "class_names": [f"class_{index}" for index in range(num_classes)],
        "score_thr": 0.5,
        "nms_pre": 100,
        "max_num": 10,
        "nms_type_list": ["rotate"] * num_classes,
        "nms_thr_list": [0.2] * num_classes,
        "nms_radius_thr_list": [1.0] * num_classes,
        "nms_rescale_factor": [1.0] * num_classes,
        "circle_nms_post_max_size": 83,
        "dir_offset": 0.0,
        "dir_limit_offset": 0.0,
    }


class BoardGoldenTest(unittest.TestCase):

    def test_unknown_3d_output_names_fail_closed(self):
        outputs = [
            {"name": "output_0", "shape": [1, 1, 1, 1]},
            {"name": "output_1", "shape": [1, 7, 1, 1]},
            {"name": "output_2", "shape": [1, 2, 1, 1]},
        ]
        with self.assertRaisesRegex(ValueError, "禁止按 output index"):
            _semantic_output_map(outputs)
        with self.assertRaisesRegex(RuntimeError, "拒绝使用可能由位置回退"):
            validate_semantic_output_names({
                "head_cls": "output_0",
                "head_bbox": "output_1",
                "head_dir": "output_2",
            })

    def test_postprocess_matches_frozen_legacy_matrix(self):
        fixture = load_golden()
        anchors = np.asarray(
            [[0, 0, 0, 2, 4, 2, 0]], dtype=np.float32)
        for case in fixture["postprocess_cases"]:
            with self.subTest(case=case["id"]):
                num_classes = int(case["num_classes"])
                logits = {
                    "head_cls": np.asarray(
                        case["cls_values"], dtype=np.float32).reshape(
                            1, num_classes, 1, 1),
                    "head_bbox": np.zeros((1, 7, 1, 1), dtype=np.float32),
                    "head_dir": np.asarray(
                        [[[[1.0]], [[0.0]]]], dtype=np.float32),
                }
                decoded, trace = standalone.postprocess(
                    logits, anchors, one_anchor_head(num_classes))
                for name, expected in case["expected"].items():
                    self.assertEqual(
                        array_fingerprint(decoded[name]), expected, name)
                self.assertEqual(
                    {key: trace[key] for key in case["trace"]},
                    case["trace"])


if __name__ == "__main__":
    unittest.main()
