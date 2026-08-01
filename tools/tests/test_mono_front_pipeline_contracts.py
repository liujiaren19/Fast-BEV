#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 manifest、INT8 I/O 和 analyzer tensor dump 回归测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from tools.mono_front_board_core import (
    _build_onnx_backend_contract,
    _onnx_static_io,
    build_board_model_assets,
)
from tools.mono_front_board_assets import validate_model_assets
from tools.tests.mono_front_golden import array_fingerprint, load_golden
import tools.run_mono_front_board_inference as standalone
import tools.infer_mono_front_image as production
import tools.analyze_mono_front_pipeline as analyzer


REPO_ROOT = Path(__file__).resolve().parents[2]


def save_model(path: Path, graph) -> None:
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    # 内网 legacy 环境仍使用 ONNX Runtime 1.10；测试模型只依赖 opset 13，
    # 不需要 IR v10。固定为 IR v8 可同时兼容旧 checker/ORT 与当前环境。
    model.ir_version = min(int(onnx.IR_VERSION), 8)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_qdq_float_model(path: Path) -> None:
    scale = numpy_helper.from_array(np.asarray(0.25, dtype=np.float32), "scale")
    zero = numpy_helper.from_array(np.asarray(0, dtype=np.int8), "zero")
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["input", "scale", "zero"], ["raw"]),
            helper.make_node("DequantizeLinear", ["raw", "scale", "zero"], ["output"]),
        ],
        "qdq-float",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1, 1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 1])],
        [scale, zero],
    )
    save_model(path, graph)


def make_raw_int8_2d(path: Path) -> None:
    scale = numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "scale")
    zero = numpy_helper.from_array(np.asarray(0, dtype=np.int8), "zero")
    graph = helper.make_graph(
        [
            helper.make_node("DequantizeLinear", ["image", "scale", "zero"], ["float"]),
            helper.make_node("QuantizeLinear", ["float", "scale", "zero"], ["features"]),
        ],
        "raw-int8-2d",
        [helper.make_tensor_value_info("image", TensorProto.INT8, [1, 3, 1, 1])],
        [helper.make_tensor_value_info("features", TensorProto.INT8, [1, 3, 1, 1])],
        [scale, zero],
    )
    save_model(path, graph)


def make_per_channel_int8_2d(path: Path) -> None:
    scale = numpy_helper.from_array(
        np.asarray([0.1, 0.2, 0.5], dtype=np.float32), "scale")
    zero = numpy_helper.from_array(
        np.asarray([0, 1, -2], dtype=np.int8), "zero")
    graph = helper.make_graph(
        [
            helper.make_node(
                "DequantizeLinear", ["image", "scale", "zero"], ["float"],
                axis=1),
            helper.make_node(
                "QuantizeLinear", ["float", "scale", "zero"], ["features"],
                axis=1),
        ],
        "per-channel-int8-2d",
        [helper.make_tensor_value_info(
            "image", TensorProto.INT8, [1, 3, 1, 1])],
        [helper.make_tensor_value_info(
            "features", TensorProto.INT8, [1, 3, 1, 1])],
        [scale, zero],
    )
    save_model(path, graph)


def make_raw_int8_3d(path: Path) -> None:
    scale = numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "scale")
    zero = numpy_helper.from_array(np.asarray(0, dtype=np.int8), "zero")
    constants = {
        "cls_float": np.asarray([[[[1.0]]]], dtype=np.float32),
        "bbox_float": np.zeros((1, 7, 1, 1), dtype=np.float32),
        "dir_float": np.asarray([[[[1.0]], [[0.0]]]], dtype=np.float32),
    }
    initializers = [scale, zero]
    nodes = [helper.make_node(
        "DequantizeLinear", ["bev", "scale", "zero"], ["bev_float"])]
    for name, value in constants.items():
        initializers.append(numpy_helper.from_array(value, name))
    for source, output in (
        ("cls_float", "cls_score"),
        ("bbox_float", "bbox_pred"),
        ("dir_float", "dir_cls_preds"),
    ):
        nodes.append(helper.make_node(
            "QuantizeLinear", [source, "scale", "zero"], [output]))
    graph = helper.make_graph(
        nodes,
        "raw-int8-3d",
        [helper.make_tensor_value_info("bev", TensorProto.INT8, [1, 3, 1, 1])],
        [
            helper.make_tensor_value_info("cls_score", TensorProto.INT8, [1, 1, 1, 1]),
            helper.make_tensor_value_info("bbox_pred", TensorProto.INT8, [1, 7, 1, 1]),
            helper.make_tensor_value_info("dir_cls_preds", TensorProto.INT8, [1, 2, 1, 1]),
        ],
        initializers,
    )
    save_model(path, graph)


def make_float_2d(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["image"], ["features"])],
        "float-2d",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 1, 1])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3, 1, 1])],
    )
    save_model(path, graph)


def make_float_3d(path: Path) -> None:
    values = {
        "cls_score": np.ones((1, 1, 1, 1), dtype=np.float32),
        "bbox_pred": np.zeros((1, 7, 1, 1), dtype=np.float32),
        "dir_cls_preds": np.asarray([[[[1.0]], [[0.0]]]], dtype=np.float32),
    }
    nodes = []
    initializers = []
    for name, value in values.items():
        source = name + "_constant"
        initializers.append(numpy_helper.from_array(value, source))
        nodes.append(helper.make_node("Identity", [source], [name]))
    graph = helper.make_graph(
        nodes,
        "float-3d",
        [helper.make_tensor_value_info("bev", TensorProto.FLOAT, [1, 3, 1, 1])],
        [
            helper.make_tensor_value_info("cls_score", TensorProto.FLOAT, [1, 1, 1, 1]),
            helper.make_tensor_value_info("bbox_pred", TensorProto.FLOAT, [1, 7, 1, 1]),
            helper.make_tensor_value_info("dir_cls_preds", TensorProto.FLOAT, [1, 2, 1, 1]),
        ],
        initializers,
    )
    save_model(path, graph)


def make_sigmoid_3d(path: Path) -> None:
    values = {
        "cls_constant": np.ones((1, 1, 1, 1), dtype=np.float32),
        "bbox_pred": np.zeros((1, 7, 1, 1), dtype=np.float32),
        "dir_cls_preds": np.asarray([[[[1.0]], [[0.0]]]], dtype=np.float32),
    }
    initializers = [
        numpy_helper.from_array(value, name)
        for name, value in values.items()
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Sigmoid", ["cls_constant"], ["cls_score"]),
            helper.make_node("Identity", ["bbox_pred"], ["bbox_output"]),
            helper.make_node("Identity", ["dir_cls_preds"], ["dir_output"]),
        ],
        "sigmoid-3d",
        [helper.make_tensor_value_info(
            "bev", TensorProto.FLOAT, [1, 3, 1, 1])],
        [
            helper.make_tensor_value_info(
                "cls_score", TensorProto.FLOAT, [1, 1, 1, 1]),
            helper.make_tensor_value_info(
                "bbox_output", TensorProto.FLOAT, [1, 7, 1, 1]),
            helper.make_tensor_value_info(
                "dir_output", TensorProto.FLOAT, [1, 2, 1, 1]),
        ],
        initializers,
    )
    save_model(path, graph)


def tensor_contract(item, layout):
    return {**item, "layout": layout}


class MonoFrontPipelineContractTest(unittest.TestCase):

    def test_three_entrypoints_expose_exact_asset_init_hint(self):
        hint = standalone.ASSET_INIT_HINT
        for module in (production, standalone, analyzer):
            with self.subTest(module=module.__name__):
                self.assertIn(hint, module.__doc__)
                completed = subprocess.run(
                    [sys.executable, str(Path(module.__file__)), "--help"],
                    cwd=REPO_ROOT, text=True, capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(hint, completed.stdout)

    def test_analyzer_classifies_first_divergence_boundaries(self):
        cases = (
            ("feature_2d_raw.npy", "quantization error"),
            ("feature_2d_dequant.npy", "ONNX graph/runtime error"),
            ("projection_gather.npy", "LUT geometry error"),
            ("center_boxes.npy", "postprocess numeric difference"),
        )
        for filename, category in cases:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                actual = root / "actual"
                reference = root / "reference"
                actual.mkdir()
                reference.mkdir()
                np.save(actual / filename, np.asarray([1], dtype=np.float32))
                np.save(reference / filename, np.asarray([0], dtype=np.float32))
                report = analyzer.compare_stage_dirs(
                    actual, reference, atol=0.0, rtol=0.0)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["classification"], category)

    def test_analyzer_business_comparison_keeps_discrete_fields_exact(self):
        reference = {
            "box_origin": "center",
            "count": 1,
            "source_anchor_indices": [17],
            "predictions": [{
                "label": 0,
                "class_name": "car",
                "score": 0.9,
                "box": [1.0, 2.0, 3.0],
                "source_anchor_index": 17,
                "box_origin": "center",
            }],
        }
        actual = json.loads(json.dumps(reference))
        actual["predictions"][0]["box"][0] += 5e-7
        within = analyzer.compare_business_output_values(
            actual, reference, atol=1e-6, rtol=0.0)
        self.assertEqual(within["status"], "PASS")
        self.assertFalse(within["exact"])
        self.assertTrue(within["discrete_exact"])
        self.assertTrue(within["numeric_within_tolerance"])

        exact = analyzer.compare_business_output_values(
            actual, reference, atol=0.0, rtol=0.0)
        self.assertEqual(exact["status"], "FAIL")
        self.assertTrue(exact["discrete_exact"])
        self.assertFalse(exact["numeric_within_tolerance"])

        actual["source_anchor_indices"] = [18]
        discrete = analyzer.compare_business_output_values(
            actual, reference, atol=1e-6, rtol=0.0)
        self.assertEqual(discrete["status"], "FAIL")
        self.assertFalse(discrete["discrete_exact"])

    def test_analyzer_reads_legacy_golden_tensor_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "input.npy", np.zeros(
                (1, 1, 3, 1, 1), dtype=np.float32))
            np.savez(
                root / "head_logits.npz",
                head_cls_0=np.ones((1, 1, 1, 1), dtype=np.float32),
                head_bbox_0=np.zeros((1, 7, 1, 1), dtype=np.float32),
                head_dir_0=np.zeros((1, 2, 1, 1), dtype=np.float32),
            )
            stages = analyzer.load_stage_inputs(root)
            self.assertEqual(stages["input_tensor"].shape, (1, 3, 1, 1))
            self.assertEqual(
                set(stages) & {"head_cls", "head_bbox", "head_dir"},
                {"head_cls", "head_bbox", "head_dir"})

    def test_analyzer_attaches_canonical_source_anchor_indices(self):
        decoded = {
            "boxes": np.zeros((2, 7), dtype=np.float32),
            "scores": np.asarray([0.9, 0.8], dtype=np.float32),
            "labels": np.asarray([0, 1], dtype=np.int64),
            "class_names": ["car", "truck"],
            "box_origin": "center",
        }
        trace = {"selected": [
            {"output_index": 0, "source_anchor_index": 17},
            {"output_index": 1, "source_anchor_index": 3},
        ]}
        actual = analyzer.attach_canonical_source_anchor_indices(
            decoded, trace)
        self.assertEqual(
            actual["source_anchor_indices"].tolist(), [17, 3])
        selected, rows = standalone.select_business_output(
            actual, ["all"], score_threshold=0.0, max_predictions=0)
        self.assertEqual(
            selected["source_anchor_indices"].tolist(), [17, 3])
        self.assertEqual(
            [row["source_anchor_index"] for row in rows], [17, 3])
        with self.assertRaisesRegex(RuntimeError, "输出顺序不完整"):
            analyzer.attach_canonical_source_anchor_indices(
                decoded, {"selected": list(reversed(trace["selected"]))})

    def test_production_s0_never_auto_repeats_current_image(self):
        image = Path("frame.jpg")
        args = SimpleNamespace(temporal_images=None, temporal_policy="error")
        self.assertEqual(production.make_temporal_paths(image, 1, args), [image])
        with self.assertRaisesRegex(ValueError, "禁止自动复制当前图"):
            production.make_temporal_paths(image, 4, args)

    def test_production_vehicle_output_name_preserves_unicode_safely(self):
        self.assertEqual(
            production.safe_output_name("9852_HK_海豹06"),
            "9852_HK_海豹06")
        self.assertEqual(
            production.safe_output_name("../9852 HK/海豹06"),
            "9852_HK_海豹06")
        self.assertEqual(production.safe_output_name(".."), "vehicle")

    def test_qdq_float_io_is_not_manually_quantized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qdq.onnx"
            make_qdq_float_model(path)
            contract = _onnx_static_io(path)
            self.assertEqual(contract["graph_format"], "qdq")
            self.assertEqual(contract["inputs"][0]["dtype"], "float32")
            self.assertIsNone(contract["inputs"][0]["quantization"])
            value = np.asarray([[[[0.3]]]], dtype=np.float32)
            encoded = standalone.quantize_external(
                value, tensor_contract(contract["inputs"][0], "nchw"))
            self.assertTrue(np.array_equal(encoded, value))

    def test_builder_rejects_cls_output_with_upstream_sigmoid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_2d = root / "2d.onnx"
            model_3d = root / "3d.onnx"
            make_float_2d(model_2d)
            make_sigmoid_3d(model_3d)
            expected = {
                "2d": {
                    "input_shape": [1, 3, 1, 1],
                    "output_shape": [1, 3, 1, 1],
                    "output_layout": "nchw",
                },
                "3d": {
                    "input_shape": [1, 3, 1, 1],
                    "semantic_outputs": {
                        "head_cls": {"shape": [1, 1, 1, 1]},
                        "head_bbox": {"shape": [1, 7, 1, 1]},
                        "head_dir": {"shape": [1, 2, 1, 1]},
                    },
                },
            }
            with self.assertRaisesRegex(ValueError, "已经包含 Sigmoid"):
                _build_onnx_backend_contract(
                    model_2d, model_3d, "nchw", expected, "onnx-fp")

    def test_raw_int8_contract_quantizes_and_dequantizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_2d.onnx"
            model_3d = Path(tmp) / "raw_3d.onnx"
            make_raw_int8_2d(path)
            make_raw_int8_3d(model_3d)
            contract = _onnx_static_io(path)
            input_spec = tensor_contract(contract["inputs"][0], "nchw")
            output_spec = tensor_contract(contract["outputs"][0], "nchw")
            self.assertEqual(input_spec["quantization"]["mode"], "per-tensor")
            self.assertEqual(input_spec["quantization"]["round_policy"],
                             "round-to-nearest-ties-to-even")
            value = np.asarray([[[[-20.0]], [[0.26]], [[20.0]]]], dtype=np.float32)
            raw = standalone.quantize_external(value, input_spec)
            self.assertEqual(raw.dtype, np.int8)
            self.assertEqual(raw.reshape(-1).tolist(), [-128, 3, 127])
            dequant = standalone.dequantize_external(raw, output_spec)
            self.assertTrue(np.allclose(
                dequant.reshape(-1), [-12.8, 0.3, 12.7], atol=1e-6))
            backend_contract = _build_onnx_backend_contract(
                path, model_3d, "nchw", {
                    "2d": {
                        "input_shape": [1, 3, 1, 1],
                        "output_shape": [1, 3, 1, 1],
                        "output_layout": "nchw",
                    },
                    "3d": {
                        "input_shape": [1, 3, 1, 1],
                        "semantic_outputs": {
                            "head_cls": {"shape": [1, 1, 1, 1]},
                            "head_bbox": {"shape": [1, 7, 1, 1]},
                            "head_dir": {"shape": [1, 2, 1, 1]},
                        },
                    },
                }, "onnx-int8")
            self.assertEqual(
                backend_contract["feature_to_bev_bridge"],
                "raw-quantized-direct")

    def test_per_channel_raw_int8_contract_uses_recorded_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_channel.onnx"
            make_per_channel_int8_2d(path)
            io = _onnx_static_io(path)
            spec = tensor_contract(io["inputs"][0], "nchw")
            self.assertEqual(spec["quantization"]["mode"], "per-channel")
            self.assertEqual(spec["quantization"]["channel_axis"], 1)
            value = np.asarray(
                [[[[0.2]], [[0.4]], [[1.0]]]], dtype=np.float32)
            raw = standalone.quantize_external(value, spec)
            self.assertEqual(raw.reshape(-1).tolist(), [2, 3, 0])
            dequant = standalone.dequantize_external(raw, spec)
            self.assertTrue(np.allclose(dequant, value, atol=1e-6))

    def test_builder_writes_single_v2_manifest_with_fp_and_int8_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights"
            assets = root / "assets"
            weights.mkdir()
            model_2d = weights / "export_2d_model.onnx"
            model_3d = weights / "export_3d_model.onnx"
            make_float_2d(model_2d)
            make_float_3d(model_3d)
            config = root / "config.py"
            config.write_text(
                """
class_names = ['car']
model = dict(
    n_images=1,
    n_voxels=[[1, 1, 1]],
    voxel_size=[[1.0, 1.0, 1.0]],
    use_distortion=False,
    bbox_head=dict(
        num_classes=1,
        dir_offset=0.0,
        dir_limit_offset=0.0,
        loss_cls=dict(use_sigmoid=True),
        bbox_coder=dict(code_size=7),
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0.0, -1.0, 0.0, 1.0, 1.0, 0.0]],
            sizes=[[2.0, 4.0, 2.0]],
            scales=[1],
            rotations=[0.0],
            custom_values=[],
            reshape_out=True,
            size_per_range=True,
            align_corner=False,
        ),
    ),
    test_cfg=dict(
        score_thr=0.05,
        nms_pre=10,
        max_num=10,
        nms_type_list=['rotate'],
        nms_thr_list=[0.2],
        nms_radius_thr_list=[1.0],
        nms_rescale_factor=[1.0],
    ),
)
data = dict(test=dict(
    n_images=1,
    n_times=1,
    sequential=False,
    camera_types=['cam0'],
    pipeline=[
        dict(type='RandomAugImageMultiViewImage', force_resize=True,
             data_config=dict(test_input_size=[1, 1])),
        dict(type='NormalizeMultiviewImage', mean=[0, 0, 0],
             std=[1, 1, 1], to_rgb=True),
        dict(type='KittiSetOrigin',
             point_cloud_range=[0.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
    ],
))
""",
                encoding="utf-8")
            export_metadata = {
                "config": str(config),
                "config_sha256": standalone.file_sha256(config),
                "checkpoint": "/not-required-for-asset-build/epoch.pth",
                "checkpoint_sha256": "c" * 64,
                "exports": {
                    "2d": {"raw": str(model_2d), "output_layout": "nchw"},
                    "3d": {"raw": str(model_3d)},
                },
            }
            (weights / "export_metadata.json").write_text(
                json.dumps(export_metadata), encoding="utf-8")
            result = build_board_model_assets(
                weights, config, assets, "test", force=True)
            spec = result["spec"]
            self.assertEqual(spec["schema_version"], 2)
            self.assertTrue(spec["onnx_contracts"]["onnx-fp"]["available"])
            self.assertEqual(
                spec["onnx_contracts"]["onnx-fp"]["2d"]["input"]["dtype"],
                "float32")
            self.assertFalse(spec["onnx_contracts"]["onnx-int8"]["available"])
            self.assertEqual(
                spec["onnx_contracts"]["onnx-int8"]["statement"],
                "未做真实 INT8 模型数值验证")
            validated = validate_model_assets(weights, assets, "test")
            self.assertEqual(validated["geometry_contract_hash"],
                             spec["asset_contract_sha256"])
            image_path = root / "1700000000000_frame.png"
            Image.fromarray(
                np.zeros((1, 1, 3), dtype=np.uint8)).save(image_path)
            partial_lut = assets / "vehicle"
            (partial_lut / "LUT").mkdir(parents=True)
            np.asarray([0], dtype=np.int32).tofile(
                partial_lut / "LUT" / "gather_new_0.bin")
            runtime_command = [
                sys.executable, str(Path(standalone.__file__)),
                "--images", str(image_path),
                "--weights", str(weights),
                "--lut-dir", str(partial_lut),
                "--output-dir", str(root / "runtime_output"),
                "--provider", "cpu",
            ]
            completed = subprocess.run(
                runtime_command, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("固定 LUT 不完整", completed.stderr)
            self.assertIn(standalone.ASSET_INIT_HINT, completed.stderr)

            changed_config = root / "changed_config.py"
            changed_config.write_text(
                config.read_text(encoding="utf-8") + "\n# contract drift\n",
                encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "--config.*board_model_spec SHA256"
            ):
                build_board_model_assets(
                    weights, changed_config, assets, "test", force=False)

            spec_path = weights / "board_model_spec.json"
            original_spec_text = spec_path.read_text(encoding="utf-8")
            mismatched = json.loads(original_spec_text)
            mismatched["geometry_contract_hash"] = "e" * 64
            spec_path.write_text(json.dumps(mismatched), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "geometry_contract_hash"):
                validate_model_assets(weights, assets, "test")
            completed = subprocess.run(
                runtime_command, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("geometry_contract_hash", completed.stderr)
            spec_path.write_text(original_spec_text, encoding="utf-8")

            mismatched = json.loads(original_spec_text)
            mismatched["head"]["score_thr"] = 0.25
            spec_path.write_text(json.dumps(mismatched), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "model_contract_sha256"):
                validate_model_assets(weights, assets, "test")
            completed = subprocess.run(
                runtime_command, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("model_contract_sha256", completed.stderr)

            mismatched = json.loads(original_spec_text)
            mismatched["geometry"]["origin"][0] += 1.0
            mismatched["model_contract_sha256"] = standalone.json_sha256({
                "config_sha256": mismatched["config_sha256"],
                "checkpoint_sha256": mismatched["checkpoint_sha256"],
                "preprocess": mismatched["preprocess"],
                "geometry": mismatched["geometry"],
                "head": mismatched["head"],
            })
            spec_path.write_text(json.dumps(mismatched), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "geometry_contract_hash"):
                validate_model_assets(weights, assets, "test")
            spec_path.write_text(original_spec_text, encoding="utf-8")

            with model_2d.open("ab") as stream:
                stream.write(b"hash-mismatch")
            with self.assertRaisesRegex(RuntimeError, "ONNX SHA256 不匹配"):
                validate_model_assets(weights, assets, "test")

    def test_full_fp_chain_matches_frozen_legacy_golden(self):
        """standalone FP 全链逐段匹配删除前冻结的旧版结果。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_2d = root / "export_2d_model.onnx"
            model_3d = root / "export_3d_model.onnx"
            make_float_2d(model_2d)
            make_float_3d(model_3d)
            golden = load_golden()["fp_chain"]
            inputs = golden["input"]
            image_path = root / "1700000000000_frame.png"
            Image.fromarray(
                np.asarray([[inputs["image_rgb"]]], dtype=np.uint8)).save(
                    image_path)
            input_tensor, _raw, _size = standalone.preprocess_image(
                image_path, inputs["preprocess"])
            session_2d = standalone.ort.InferenceSession(
                str(model_2d), providers=["CPUExecutionProvider"])
            session_3d = standalone.ort.InferenceSession(
                str(model_3d), providers=["CPUExecutionProvider"])
            io_2d = _onnx_static_io(model_2d)
            contract_2d = {
                "input": tensor_contract(io_2d["inputs"][0], "nchw"),
                "output": tensor_contract(io_2d["outputs"][0], "nchw"),
            }
            feature = standalone.run_2d_onnx(
                session_2d, input_tensor, contract_2d)
            bev = standalone.apply_lut(
                feature,
                np.asarray(inputs["gather"], dtype=np.int64),
                np.asarray(inputs["scatter"], dtype=np.int64),
                inputs["geometry"])
            io_3d = _onnx_static_io(model_3d)
            by_name = {item["name"]: item for item in io_3d["outputs"]}
            outputs = {
                "head_cls": tensor_contract(by_name["cls_score"], "nchw"),
                "head_bbox": tensor_contract(by_name["bbox_pred"], "nchw"),
                "head_dir": tensor_contract(
                    by_name["dir_cls_preds"], "nchw"),
            }
            logits = standalone.run_3d_onnx(
                session_3d, bev,
                {name: item["name"] for name, item in outputs.items()},
                {
                    "input": tensor_contract(io_3d["inputs"][0], "nchw-zc"),
                    "outputs": outputs,
                })
            decoded, trace = standalone.postprocess(
                logits, np.asarray(inputs["anchors"], dtype=np.float32),
                inputs["head"])
            actual = {
                "input_tensor": input_tensor,
                "feature_2d": feature,
                "bev_input": bev,
                **logits,
                **{
                    f"decoded_{name}": decoded[name]
                    for name in (
                        "boxes", "scores", "labels",
                        "source_anchor_indices")
                },
            }
            for name, expected in golden["expected"].items():
                self.assertEqual(
                    array_fingerprint(actual[name]), expected, name)
            self.assertEqual(
                {key: trace[key] for key in golden["trace"]},
                golden["trace"])

    def test_raw_and_simplified_fp_contracts_keep_same_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_2d = root / "raw_2d.onnx"
            simplified_2d = root / "simplified_2d.onnx"
            raw_3d = root / "raw_3d.onnx"
            simplified_3d = root / "simplified_3d.onnx"
            make_float_2d(raw_2d)
            make_float_2d(simplified_2d)
            make_float_3d(raw_3d)
            make_float_3d(simplified_3d)
            input_tensor = np.asarray(
                [[[[1.0]], [[2.0]], [[3.0]]]], dtype=np.float32)
            bev = input_tensor.copy()
            pairs = ((raw_2d, simplified_2d, input_tensor),
                     (raw_3d, simplified_3d, bev))
            for raw_path, simplified_path, value in pairs:
                raw_session = standalone.ort.InferenceSession(
                    str(raw_path), providers=["CPUExecutionProvider"])
                simplified_session = standalone.ort.InferenceSession(
                    str(simplified_path), providers=["CPUExecutionProvider"])
                raw_outputs = raw_session.run(
                    None, {raw_session.get_inputs()[0].name: value})
                simplified_outputs = simplified_session.run(
                    None, {simplified_session.get_inputs()[0].name: value})
                self.assertEqual(len(raw_outputs), len(simplified_outputs))
                for raw_value, simplified_value in zip(
                    raw_outputs, simplified_outputs
                ):
                    self.assertTrue(np.array_equal(
                        raw_value, simplified_value))

    def test_analyzer_saves_raw_and_dequant_int8_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights"
            asset_root = root / "lut"
            vehicle = asset_root / "vehicle"
            common = asset_root / "common" / "test"
            output = root / "analysis"
            weights.mkdir()
            (vehicle / "LUT").mkdir(parents=True)
            (vehicle / "LUT_arr").mkdir()
            common.mkdir(parents=True)
            model_2d = weights / "int8_2d.onnx"
            model_3d = weights / "int8_3d.onnx"
            make_raw_int8_2d(model_2d)
            make_raw_int8_3d(model_3d)
            io_2d = _onnx_static_io(model_2d)
            io_3d = _onnx_static_io(model_3d)
            semantic = {
                "head_cls": io_3d["outputs"][0],
                "head_bbox": io_3d["outputs"][1],
                "head_dir": io_3d["outputs"][2],
            }
            int8_contract = {
                "backend": "onnx-int8",
                "available": True,
                "numerical_validation": "not_performed",
                "external_io_mode": "raw-int8",
                "manual_external_quantization": True,
                "feature_to_bev_bridge": "raw-quantized-direct",
                "models": {
                    "2d": {"file": str(model_2d), "sha256": standalone.file_sha256(model_2d)},
                    "3d": {"file": str(model_3d), "sha256": standalone.file_sha256(model_3d)},
                },
                "2d": {
                    "input": tensor_contract(io_2d["inputs"][0], "nchw"),
                    "output": tensor_contract(io_2d["outputs"][0], "nchw"),
                },
                "3d": {
                    "input": tensor_contract(io_3d["inputs"][0], "nchw-zc"),
                    "outputs": {
                        name: tensor_contract(item, "nchw")
                        for name, item in semantic.items()
                    },
                    "semantic_output_names": {
                        name: item["name"] for name, item in semantic.items()
                    },
                },
            }
            anchors = np.asarray([[0, 0, 0, 2, 4, 2, 0]], dtype=np.float32)
            anchors_path = common / "anchors.npy"
            points_path = common / "points.npy"
            np.save(anchors_path, anchors)
            np.save(points_path, np.zeros((3, 1, 1, 1), dtype=np.float32))
            gather_path = vehicle / "LUT" / "gather_new_0.bin"
            scatter_path = vehicle / "LUT" / "scatter_nd_new_0.bin"
            length_path = vehicle / "LUT" / "featurePointLength.bin"
            np.asarray([0], dtype=np.int32).tofile(gather_path)
            np.asarray([0], dtype=np.int32).tofile(scatter_path)
            np.asarray([1], dtype=np.int32).tofile(length_path)
            gather_arr = vehicle / "LUT_arr" / "gather_0.npy"
            scatter_arr = vehicle / "LUT_arr" / "scatter_nd_0.npy"
            np.save(gather_arr, np.asarray([0], dtype=np.int64))
            np.save(scatter_arr, np.asarray([0], dtype=np.int64))
            lut_files = {
                "LUT/gather_new_0.bin": gather_path,
                "LUT/scatter_nd_new_0.bin": scatter_path,
                "LUT/featurePointLength.bin": length_path,
                "LUT_arr/gather_0.npy": gather_arr,
                "LUT_arr/scatter_nd_0.npy": scatter_arr,
            }
            geometry = {
                "n_images": 1,
                "n_times": 1,
                "camera_types": ["cam0"],
                "feature_shape_nchw": [1, 3, 1, 1],
                "n_voxels": [1, 1, 1],
                "voxel_size": [1.0, 1.0, 1.0],
                "origin": [0.5, 0.0, 0.0],
                "bev_shape_nchw": [1, 3, 1, 1],
                "stride": 1,
                "point_cloud_range": [0, -1, -1, 1, 1, 1],
                "channel_layout": "ZC",
                "use_distortion": False,
                "data_config": {},
            }
            head = {
                "class_names": ["car"],
                "num_classes": 1,
                "feature_map_hw": [1, 1],
                "num_anchors_per_location": 1,
                "anchor_generator": {
                    "ranges": [[0, 0, 0, 1, 1, 1]],
                    "sizes": [[2, 4, 2]],
                    "rotations": [0],
                    "reshape_out": True,
                },
                "box_code_size": 7,
                "num_dir_bins": 2,
                "output_shapes": {
                    "head_cls": [1, 1, 1, 1],
                    "head_bbox": [1, 7, 1, 1],
                    "head_dir": [1, 2, 1, 1],
                },
                "score_thr": 0.05,
                "nms_pre": 10,
                "max_num": 10,
                "nms_type_list": ["rotate"],
                "nms_thr_list": [0.2],
                "nms_radius_thr_list": [1.0],
                "nms_rescale_factor": [1.0],
                "circle_nms_post_max_size": 83,
                "dir_offset": 0.0,
                "dir_limit_offset": 0.0,
                "sigmoid_count": 1,
                "public_box_origin": "center",
            }
            preprocess = {
                "input_size_hw": [1, 1],
                "resize_backend": "pipeline_pil_bicubic",
                "to_rgb": True,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            }
            asset_contract = {
                "geometry": {
                    key: geometry[key]
                    for key in (
                        "n_images", "n_times", "camera_types", "n_voxels",
                        "voxel_size", "origin", "feature_shape_nchw",
                        "bev_shape_nchw", "stride", "channel_layout",
                        "use_distortion", "data_config")
                },
                "head": {
                    key: head[key]
                    for key in (
                        "feature_map_hw", "num_anchors_per_location",
                        "box_code_size", "anchor_generator")
                },
            }
            geometry_hash = standalone.json_sha256(asset_contract)
            metadata = {
                "n_images": 1,
                "n_times": 1,
                "n_voxels": [1, 1, 1],
                "feature_shape": [1, 3, 1, 1],
                "source_image_size": [1, 1],
                "model_asset_profile": "test",
                "model_asset_contract_sha256": geometry_hash,
                "geometry_contract_hash": geometry_hash,
                "lut_file_sha256": {
                    name: standalone.file_sha256(path)
                    for name, path in lut_files.items()
                },
            }
            (vehicle / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            spec = {
                "schema_version": 2,
                "asset_profile": "test",
                "asset_contract_sha256": geometry_hash,
                "geometry_contract_hash": geometry_hash,
                "config_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "model_contract_sha256": standalone.json_sha256({
                    "config_sha256": "b" * 64,
                    "checkpoint_sha256": "c" * 64,
                    "preprocess": preprocess,
                    "geometry": geometry,
                    "head": head,
                }),
                "preprocess": preprocess,
                "geometry": geometry,
                "head": head,
                "assets": {
                    "anchors_relative": "common/test/anchors.npy",
                    "anchors_shape": [1, 7],
                    "anchors_sha256": standalone.file_sha256(anchors_path),
                    "points_relative": "common/test/points.npy",
                    "points_shape": [3, 1, 1, 1],
                    "points_sha256": standalone.file_sha256(points_path),
                },
                "onnx_contracts": {
                    "onnx-fp": {"available": False},
                    "onnx-int8": int8_contract,
                },
            }
            (weights / "board_model_spec.json").write_text(
                json.dumps(spec), encoding="utf-8")
            image_path = root / "frame.png"
            Image.fromarray(np.asarray([[[10, 20, 30]]], dtype=np.uint8)).save(image_path)
            command = [
                sys.executable,
                str(REPO_ROOT / "tools" / "analyze_mono_front_pipeline.py"),
                "--backend", "onnx-int8",
                "--geometry", "fixed",
                "--preprocess", "board",
                "--postprocess", "board",
                "--weights", str(weights),
                "--lut-dir", str(vehicle),
                "--image", str(image_path),
                "--output-dir", str(output),
            ]
            completed = subprocess.run(
                command, cwd=REPO_ROOT, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            raw = np.load(output / "tensors" / "head_cls_raw.npy")
            dequant = np.load(output / "tensors" / "head_cls_dequant.npy")
            self.assertEqual(raw.dtype, np.int8)
            self.assertEqual(dequant.dtype, np.float32)
            report = json.loads(
                (output / "analysis_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["postprocess"]["sigmoid_count"], 1)
            self.assertEqual(
                report["int8_validation_statement"], "未做真实 INT8 模型数值验证")


if __name__ == "__main__":
    unittest.main()
