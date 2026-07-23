#!/usr/bin/env python3
"""单目前视生产可视化的最小 box yaw 坐标系回归测试。"""

import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


visualizer = load_module(
    "n7_visualizer_geometry_test",
    "tools/data_converter/n7/visualize_n7_fastbev_pkl.py")
standalone = load_module(
    "mono_front_standalone_geometry_test",
    "tools/run_mono_front_board_inference.py")


class MonoFrontVisualizationGeometryTest(unittest.TestCase):

    CENTER = np.asarray([20.0, 0.0, 0.0], dtype=np.float32)
    DIMS = np.asarray([4.0, 2.0, 2.0], dtype=np.float32)
    BEV_RANGE = (0.0, -35.0, 80.0, 35.0)
    BEV_SIZE = 700

    @classmethod
    def box(cls, yaw: float) -> np.ndarray:
        return np.asarray([[*cls.CENTER, *cls.DIMS, yaw]], dtype=np.float32)

    def test_cardinal_yaws_follow_x_forward_y_left_contract(self):
        center_px = visualizer.bev_to_pixel(
            self.CENTER[None, :2], self.BEV_RANGE, self.BEV_SIZE)[0]
        cases = (
            (0.0, "up"),
            (math.pi / 2.0, "left"),
            (-math.pi / 2.0, "right"),
            (6.216, "up_right"),
        )
        for yaw, expected_direction in cases:
            with self.subTest(yaw=yaw):
                box = self.box(yaw)
                corners = visualizer.corners_from_boxes(box)[0]
                np.testing.assert_allclose(
                    standalone.box_corners_3d(box)[0], corners, atol=1e-6)
                front_center = corners[[0, 1], :2].mean(axis=0, keepdims=True)
                forward = (
                    front_center[0] - self.CENTER[:2]) / (self.DIMS[0] * 0.5)
                np.testing.assert_allclose(
                    forward, [math.cos(yaw), math.sin(yaw)], atol=1e-6)
                front_px = visualizer.bev_to_pixel(
                    front_center, self.BEV_RANGE, self.BEV_SIZE)[0]
                delta = front_px - center_px
                if expected_direction == "up":
                    self.assertAlmostEqual(float(delta[0]), 0.0, places=4)
                    self.assertLess(float(delta[1]), 0.0)
                elif expected_direction == "left":
                    self.assertLess(float(delta[0]), 0.0)
                    self.assertAlmostEqual(float(delta[1]), 0.0, places=4)
                elif expected_direction == "right":
                    self.assertGreater(float(delta[0]), 0.0)
                    self.assertAlmostEqual(float(delta[1]), 0.0, places=4)
                else:
                    self.assertGreater(float(delta[0]), 0.0)
                    self.assertLess(float(delta[1]), 0.0)

    def test_wrapped_yaw_has_equivalent_corners_and_pixels(self):
        wrapped = visualizer.corners_from_boxes(self.box(6.216))[0]
        normalized = visualizer.corners_from_boxes(
            self.box(6.216 - 2.0 * math.pi))[0]
        rounded = visualizer.corners_from_boxes(self.box(-0.0672))[0]
        np.testing.assert_allclose(wrapped, normalized, atol=1e-6)
        np.testing.assert_allclose(wrapped, rounded, atol=5e-5)

        wrapped_px = visualizer.bev_to_pixel(
            wrapped[:4, :2], self.BEV_RANGE, self.BEV_SIZE)
        normalized_px = visualizer.bev_to_pixel(
            normalized[:4, :2], self.BEV_RANGE, self.BEV_SIZE)
        np.testing.assert_allclose(wrapped_px, normalized_px, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
