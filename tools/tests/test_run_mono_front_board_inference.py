#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立单文件板端参考入口的回归测试。"""

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import tools.run_mono_front_board_inference as standalone


PREPROCESS = {
    "input_size_hw": [256, 704],
    "resize_backend": "pipeline_pil_bicubic",
    "to_rgb": True,
    "mean": [123.675, 116.28, 103.53],
    "std": [58.395, 57.12, 57.375],
}
GEOMETRY = {
    "feature_shape_nchw": [1, 64, 64, 176],
    "n_voxels": [160, 140, 4],
    "bev_shape_nchw": [1, 256, 160, 140],
    "point_cloud_range": [0.0, -35.0, -5.0, 80.0, 35.0, 3.0],
}


def head_contract(num_classes=1):
    return {
        "class_names": ["car"] if num_classes == 1 else ["car", "truck"],
        "num_classes": num_classes,
        "box_code_size": 7,
        "num_dir_bins": 2,
        "score_thr": 0.05,
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


class StandaloneBoardInferenceTest(unittest.TestCase):

    class _FakeSession:

        def __init__(self, providers):
            self._providers = providers

        def get_providers(self):
            return self._providers

    def test_script_has_no_project_imports(self):
        """正式单文件不能从 tools/mmdet3d 等项目模块导入实现。"""
        source_path = Path(standalone.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        project_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(("tools", "mmdet3d", "mmdet")):
                    project_imports.append(module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("tools", "mmdet3d", "mmdet")):
                        project_imports.append(alias.name)
        self.assertEqual(project_imports, [])

    def test_preprocess_shape_and_float32(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            image = np.zeros((256, 704, 3), dtype=np.uint8)
            image[:, :, 0] = 10
            image[:, :, 1] = 20
            image[:, :, 2] = 30
            Image.fromarray(image[:, :, ::-1]).save(image_path)
            tensor, loaded_bgr, image_size = standalone.preprocess_image(
                image_path, PREPROCESS)
            self.assertEqual(tensor.shape, (1, 3, 256, 704))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertEqual(image_size, [704, 256])
            self.assertEqual(loaded_bgr[0, 0].tolist(), [10, 20, 30])
            normalized = loaded_bgr.astype(np.float32)
            import cv2
            cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB, normalized)
            cv2.subtract(
                normalized,
                np.float64(np.asarray(
                    PREPROCESS["mean"], dtype=np.float32).reshape(1, -1)),
                normalized)
            cv2.multiply(
                normalized,
                1.0 / np.float64(np.asarray(
                    PREPROCESS["std"], dtype=np.float32).reshape(1, -1)),
                normalized)
            expected = normalized.transpose(2, 0, 1)[None]
            self.assertTrue(np.array_equal(tensor, expected))

    def test_extract_timestamp_from_production_image_name(self):
        image_path = Path("1709058510936_1709058510936_1.jpg")
        self.assertEqual(
            standalone.extract_image_timestamp(image_path),
            "1709058510936")
        with self.assertRaisesRegex(ValueError, "无法提取时间戳"):
            standalone.extract_image_timestamp(Path("frame_1.jpg"))

    def test_lut_zc_channel_order(self):
        feature = np.zeros((1, 64, 64, 176), dtype=np.float32)
        feature[0, 0, 0, 0] = 1.0
        feature[0, 1, 0, 0] = 10.0
        feature[0, 0, 0, 1] = 2.0
        feature[0, 1, 0, 1] = 20.0
        bev = standalone.apply_lut(
            feature,
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            GEOMETRY,
        )
        self.assertEqual(bev.shape, (1, 256, 160, 140))
        self.assertEqual(float(bev[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(bev[0, 1, 0, 0]), 10.0)
        self.assertEqual(float(bev[0, 64, 0, 0]), 2.0)
        self.assertEqual(float(bev[0, 65, 0, 0]), 20.0)

    def test_nchw_nhwc_is_converted_exactly_once(self):
        nchw = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
        nhwc = nchw.transpose(0, 2, 3, 1)
        self.assertTrue(np.array_equal(
            standalone.feature_to_nchw_once(nchw, "nchw"), nchw))
        self.assertTrue(np.array_equal(
            standalone.feature_to_nchw_once(nhwc, "nhwc"), nchw))

    def test_named_3d_outputs_ignore_graph_order(self):
        class Node:
            def __init__(self, name):
                self.name = name

        class Session:
            def get_inputs(self):
                return [Node("bev")]

            def run(self, requested, feeds):
                del feeds
                values = {
                    "cls_score": np.ones((1, 1, 1, 1), dtype=np.float32),
                    "bbox_pred": np.ones((1, 7, 1, 1), dtype=np.float32),
                    "dir_cls_preds": np.ones((1, 2, 1, 1), dtype=np.float32),
                }
                return [values[name] for name in requested]

        outputs = {
            "head_cls": {"name": "cls_score", "shape": [1, 1, 1, 1], "dtype": "float32"},
            "head_bbox": {"name": "bbox_pred", "shape": [1, 7, 1, 1], "dtype": "float32"},
            "head_dir": {"name": "dir_cls_preds", "shape": [1, 2, 1, 1], "dtype": "float32"},
        }
        result = standalone.run_3d_onnx(
            Session(), np.zeros((1, 1, 1, 1), dtype=np.float32),
            {name: item["name"] for name, item in outputs.items()},
            {"input": {
                "name": "bev", "shape": [1, 1, 1, 1],
                "dtype": "float32",
            }, "outputs": outputs})
        self.assertEqual(list(result), ["head_cls", "head_bbox", "head_dir"])

    def test_postprocess_covers_empty_single_and_multiclass_heads(self):
        anchors = np.asarray([[0, 0, 0, 2, 4, 2, 0]], dtype=np.float32)
        empty, empty_trace = standalone.postprocess({
            "head_cls": np.asarray([[[[-10.0]]]], dtype=np.float32),
            "head_bbox": np.zeros((1, 7, 1, 1), dtype=np.float32),
            "head_dir": np.asarray([[[[1.0]], [[0.0]]]], dtype=np.float32),
        }, anchors, head_contract())
        self.assertEqual(empty["boxes"].shape, (0, 7))
        self.assertEqual(empty_trace["sigmoid_count"], 1)

        decoded, trace = standalone.postprocess({
            "head_cls": np.asarray([[[[10.0]]]], dtype=np.float32),
            "head_bbox": np.zeros((1, 7, 1, 1), dtype=np.float32),
            "head_dir": np.asarray([[[[1.0]], [[0.0]]]], dtype=np.float32),
        }, anchors, head_contract())
        self.assertEqual(trace["num_classes"], 1)
        self.assertEqual(decoded["class_names"], ["car"])
        self.assertEqual(decoded["labels"].tolist(), [0])
        self.assertEqual(decoded["box_origin"], "center")
        self.assertAlmostEqual(float(decoded["boxes"][0, 2]), 1.0, places=6)

        multi_head = head_contract(2)
        multi, _ = standalone.postprocess({
            "head_cls": np.asarray(
                [[[[-10.0]], [[10.0]]]], dtype=np.float32),
            "head_bbox": np.zeros((1, 7, 1, 1), dtype=np.float32),
            "head_dir": np.asarray([[[[1.0]], [[0.0]]]], dtype=np.float32),
        }, anchors, multi_head)
        self.assertEqual(multi["labels"].tolist(), [1])

    def test_rotate_and_circle_nms_cover_multiple_boxes(self):
        anchors = np.asarray([
            [0, 0, 0, 2, 4, 2, 0],
            [0, 0, 0, 2, 4, 2, 0],
        ], dtype=np.float32)
        logits = {
            "head_cls": np.asarray([[[[5.0]], [[4.0]]]], dtype=np.float32),
            "head_bbox": np.zeros((1, 14, 1, 1), dtype=np.float32),
            "head_dir": np.asarray(
                [[[[1.0]], [[0.0]], [[1.0]], [[0.0]]]], dtype=np.float32),
        }
        for nms_type in ("rotate", "circle"):
            head = head_contract()
            head["nms_type_list"] = [nms_type]
            head["nms_radius_thr_list"] = [1.0]
            decoded, _ = standalone.postprocess(logits, anchors, head)
            self.assertEqual(len(decoded["boxes"]), 1, nms_type)

        separated = anchors.copy()
        separated[1, 0] = 20.0
        decoded, _ = standalone.postprocess(
            logits, separated, head_contract())
        self.assertEqual(len(decoded["boxes"]), 2)

    def test_topk_tie_boundary_is_traced(self):
        anchors = np.asarray([
            [0, 0, 0, 2, 4, 2, 0],
            [10, 0, 0, 2, 4, 2, 0],
            [20, 0, 0, 2, 4, 2, 0],
        ], dtype=np.float32)
        logits = {
            "head_cls": np.ones((1, 3, 1, 1), dtype=np.float32),
            "head_bbox": np.zeros((1, 21, 1, 1), dtype=np.float32),
            "head_dir": np.tile(
                np.asarray([[[[1.0]], [[0.0]]]], dtype=np.float32),
                (1, 3, 1, 1)),
        }
        head = head_contract()
        head["nms_pre"] = 2
        _, trace = standalone.postprocess(logits, anchors, head)
        self.assertTrue(trace["topk_boundary_tie"]["present"])

    def test_circle_nms_keeps_board_default_limit(self):
        centers = np.stack((
            np.arange(100, dtype=np.float32) * 10.0,
            np.zeros(100, dtype=np.float32),
        ), axis=1)
        scores = np.linspace(1.0, 0.1, 100, dtype=np.float32)
        selected = standalone.circle_nms_cpu(
            centers, scores, threshold=1.0)
        self.assertEqual(len(selected), 83)

    def test_business_output_is_sorted_and_limited(self):
        decoded = {
            "boxes": np.zeros((3, 7), dtype=np.float32),
            "scores": np.asarray([0.8, 0.9, 0.7], dtype=np.float32),
            "labels": np.asarray([0, 0, 1], dtype=np.int64),
            "source_anchor_indices": np.asarray([5, 4, 3], dtype=np.int64),
            "box_origin": "center",
            "class_names": ["car", "truck"],
        }
        selected, rows = standalone.select_business_output(
            decoded, ["car"], 0.2, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_anchor_index"], 4)
        self.assertEqual(selected["source_anchor_indices"].tolist(), [4])

    def test_business_car_filter_runs_after_full_multiclass_nms(self):
        anchors = np.asarray([[0, 0, 0, 2, 4, 2, 0]], dtype=np.float32)
        decoded, trace = standalone.postprocess({
            "head_cls": np.asarray(
                [[[[10.0]], [[9.0]]]], dtype=np.float32),
            "head_bbox": np.zeros((1, 7, 1, 1), dtype=np.float32),
            "head_dir": np.asarray(
                [[[[1.0]], [[0.0]]]], dtype=np.float32),
        }, anchors, head_contract(2))
        self.assertEqual(decoded["labels"].tolist(), [0, 1])
        self.assertEqual(
            [item["selected_after_nms"] for item in trace["per_class"]],
            [1, 1])
        selected, rows = standalone.select_business_output(
            decoded, ["car"], 0.2, 100)
        self.assertEqual(selected["labels"].tolist(), [0])
        self.assertEqual([row["class_name"] for row in rows], ["car"])

    def test_business_frame_record_has_required_fields(self):
        row = {
            "label": 0,
            "class_name": "car",
            "score": 0.7546,
            "box": [
                1.23456, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3,
            ],
        }
        record = standalone.business_frame_record(
            "1709058510936", [row])
        self.assertEqual(record["timestamp"], "1709058510936")
        self.assertNotIn("image", record)
        self.assertNotIn("image_name", record)
        self.assertEqual(record["box_origin"], "center")
        self.assertEqual(record["count"], 1)
        self.assertEqual(record["targets"][0]["label"], "car")
        self.assertEqual(len(record["targets"][0]["bbox"]), 9)
        self.assertEqual(record["targets"][0]["bbox"][0], 1.235)
        self.assertEqual(record["targets"][0]["score"], 0.755)

    def test_json_lines_writes_one_frame_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "business_predictions.jsonl"
            standalone.write_json_lines(output_path, [
                {"timestamp": "1", "targets": []},
                {"timestamp": "2", "targets": []},
            ])
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["timestamp"] for row in rows], ["1", "2"])

    def test_front_visualization_draws_timestamp(self):
        image = np.zeros((256, 704, 3), dtype=np.uint8)
        decoded = {
            "boxes": np.zeros((0, 9), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "class_names": ["car", "truck"],
        }
        calibration = {
            "intrinsic_width": 704,
            "intrinsic_height": 256,
            "cam_intrinsic": [
                [500.0, 0.0, 352.0],
                [0.0, 500.0, 128.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion": [],
        }
        rendered = standalone.draw_camera_boxes(
            image, decoded, calibration, "1709058510936")
        self.assertTrue(np.any(rendered != 0))

    def test_bev_coordinate_legend_matches_vehicle_axes(self):
        decoded = {
            "boxes": np.zeros((0, 9), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        }
        bev = standalone.draw_bev(
            decoded, 360, GEOMETRY["point_cloud_range"])
        # 360 尺寸下图例原点约为 (342,342)：+X 向上为红色，+Y 向左为绿色。
        self.assertGreater(int(bev[322, 342, 2]), 200)
        self.assertGreater(int(bev[342, 322, 1]), 200)
        self.assertFalse(np.any(bev[:, -1, 2] > 100))

    def test_explicit_cuda_must_not_silently_fall_back_to_cpu(self):
        cuda = self._FakeSession([
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ])
        cpu = self._FakeSession(["CPUExecutionProvider"])
        self.assertEqual(
            standalone.validate_session_providers(cuda, cuda, "cuda"),
            "CUDAExecutionProvider")
        with self.assertRaisesRegex(RuntimeError, "创建失败并回退"):
            standalone.validate_session_providers(cpu, cpu, "cuda")

    def test_partial_lut_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            lut_dir = Path(tmp)
            (lut_dir / "LUT").mkdir()
            np.asarray([0], dtype=np.int32).tofile(
                lut_dir / "LUT" / "gather_new_0.bin")
            with self.assertRaisesRegex(FileNotFoundError, "固定 LUT 不完整"):
                standalone.load_lut(lut_dir, {"geometry": GEOMETRY})

    def test_missing_manifest_cli_exits_nonzero_with_init_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights"
            lut_dir = root / "vehicle"
            weights.mkdir()
            lut_dir.mkdir()
            image = root / "1700000000000_frame.png"
            Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(image)
            completed = subprocess.run([
                sys.executable, str(Path(standalone.__file__)),
                "--images", str(image),
                "--weights", str(weights),
                "--lut-dir", str(lut_dir),
                "--output-dir", str(root / "out"),
            ], text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(standalone.ASSET_INIT_HINT, completed.stderr)

    def test_complete_lut_reads_only_declared_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            lut_dir = Path(tmp)
            (lut_dir / "LUT").mkdir()
            np.asarray([0], dtype=np.int32).tofile(
                lut_dir / "LUT" / "gather_new_0.bin")
            np.asarray([0], dtype=np.int32).tofile(
                lut_dir / "LUT" / "scatter_nd_new_0.bin")
            np.asarray([1], dtype=np.int32).tofile(
                lut_dir / "LUT" / "featurePointLength.bin")
            (lut_dir / "LUT_arr").mkdir()
            np.save(lut_dir / "LUT_arr" / "gather_0.npy", np.asarray([0]))
            np.save(lut_dir / "LUT_arr" / "scatter_nd_0.npy", np.asarray([0]))
            paths = {
                "LUT/gather_new_0.bin": lut_dir / "LUT" / "gather_new_0.bin",
                "LUT/scatter_nd_new_0.bin": lut_dir / "LUT" / "scatter_nd_new_0.bin",
                "LUT/featurePointLength.bin": lut_dir / "LUT" / "featurePointLength.bin",
                "LUT_arr/gather_0.npy": lut_dir / "LUT_arr" / "gather_0.npy",
                "LUT_arr/scatter_nd_0.npy": lut_dir / "LUT_arr" / "scatter_nd_0.npy",
            }
            metadata = {
                "n_images": 1,
                "n_times": 1,
                "n_voxels": [160, 140, 4],
                "feature_shape": [1, 64, 64, 176],
                "source_image_size": [704, 256],
                "model_asset_profile": "test",
                "model_asset_contract_sha256": "a" * 64,
                "geometry_contract_hash": "a" * 64,
                "lut_file_sha256": {
                    name: standalone.file_sha256(path)
                    for name, path in paths.items()
                },
            }
            (lut_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            gather, scatter, loaded = standalone.load_lut(lut_dir, {
                "geometry": GEOMETRY,
                "asset_profile": "test",
                "asset_contract_sha256": "a" * 64,
                "geometry_contract_hash": "a" * 64,
            })
            self.assertEqual(gather.tolist(), [0])
            self.assertEqual(scatter.tolist(), [0])
            self.assertEqual(loaded["source_image_size"], [704, 256])
            with self.assertRaisesRegex(RuntimeError, "geometry_contract_hash"):
                standalone.load_lut(lut_dir, {
                    "geometry": GEOMETRY,
                    "asset_profile": "test",
                    "asset_contract_sha256": "a" * 64,
                    "geometry_contract_hash": "b" * 64,
                })


if __name__ == "__main__":
    unittest.main()
