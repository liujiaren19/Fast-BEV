#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEOM1-A 原生 1600x900 等比缩放/中心裁剪回归。"""

import ast
import argparse
import copy
import hashlib
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from test_force_resize_image_augmentation import (  # noqa: E402
    RandomAugImageMultiViewImage,
    _camera_aug,
    _project,
    find_pipeline_step,
    load_py_config,
    test_image_aug_params as _test_image_aug_params,
)


DIST_CONFIG = (
    ROOT / 'configs' / 'fastbev' / 'custom' /
    'custom_fastbev_mono_front_single_frame_r18_n7_1600x900_scale_crop_dist_train.py')
CONFIG = DIST_CONFIG
LEGACY_NESTED_CONFIG = DIST_CONFIG.with_name(
    'custom_fastbev_mono_front_single_frame_r18_n7_1600x900_scale_crop.py')
BASELINE_DIST_CONFIG = CONFIG.with_name(
    'custom_fastbev_mono_front_single_frame_r18_dist_train.py')

# 扁平化前的 6 层继承链已在修改前规范化冻结。只哈希实际参与运行的
# model/data/pipeline/optimizer/runtime 字段，排除旧链遗留但未被 data 使用的
# 704x256 custom_dataset_common 等孤儿 helper。
NESTED_REFERENCE_EFFECTIVE_SHA256 = (
    'dcfd9882e1e8d2f75348d8d693d0bc411adbb7930e47be306da18b8c67d961d4')
EFFECTIVE_CONFIG_KEYS = (
    'model', 'point_cloud_range', 'class_names', 'dataset_type', 'data_root',
    'input_modality', 'img_norm_cfg', 'data_config', 'file_client_args',
    'train_pipeline', 'test_pipeline', 'data', 'optimizer', 'optimizer_config',
    'lr_config', 'total_epochs', 'checkpoint_config', 'log_config', 'evaluation',
    'dist_params', 'find_unused_parameters', 'log_level', 'load_from',
    'resume_from', 'workflow', 'fp16', 'experiment_id', 'work_dir')


def load_scale_crop_tool():
    path = (
        ROOT / 'tools' / 'data_converter' / 'n7' /
        'analyze_n7_scale_crop_geometry.py')
    spec = importlib.util.spec_from_file_location('analyze_n7_scale_crop_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCALE_CROP_TOOL = load_scale_crop_tool()


def data_config():
    return dict(
        input_size=(256, 704),
        resize=(0.0, 0.0),
        crop=(0.0, 0.0),
        rot=(0.0, 0.0),
        flip=False,
        test_input_size=(256, 704),
        test_resize=0.0,
        test_rotate=0.0,
        test_flip=False,
        pad=(0, 0, 0, 0),
        pad_divisor=32,
        pad_color=(0, 0, 0))


def native_image():
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    image[440:461, 790:811] = np.asarray([255, 32, 16], dtype=np.uint8)
    return image


def native_results():
    camera = _camera_aug()
    camera['image_width'] = 1600
    camera['image_height'] = 900
    camera['tran'] = np.zeros(3, dtype=np.float32)
    return dict(
        img=[native_image()],
        view_layout=dict(n_images=1, n_times=1, sequential=False),
        lidar2img=dict(
            lidar2img_aug=[camera],
            extrinsic=[np.eye(4, dtype=np.float32)]))


def rng_state_equal(left, right):
    return (
        left[0] == right[0] and
        np.array_equal(left[1], right[1]) and
        left[2:] == right[2:]
    )


def effective_config_sha256(config):
    effective = {key: config[key] for key in EFFECTIVE_CONFIG_KEYS}
    payload = json.dumps(
        effective,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


class N7ScaleCropGeometryTest(unittest.TestCase):
    def test_formal_config_is_standalone_and_matches_nested_reference(self):
        source = DIST_CONFIG.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(DIST_CONFIG))
        assigned_names = {
            target.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        self.assertNotIn('_base_', assigned_names)
        self.assertFalse(
            LEGACY_NESTED_CONFIG.exists(),
            '旧 GEOM1-A 嵌套 base 不应继续作为候选配置')

        config = load_py_config(DIST_CONFIG)
        # 正式训练允许把 work_dir 从 experiment 根目录固定到唯一 run 目录；
        # 对比扁平化前的有效训练契约时先归一化该非模型字段。
        reference_config = copy.deepcopy(config)
        reference_config['work_dir'] = (
            'work_dirs/n7_mono_1600_900_scale_crop/EXP-MONO-GEOM1-A')
        self.assertEqual(
            effective_config_sha256(reference_config),
            NESTED_REFERENCE_EFFECTIVE_SHA256)
        self.assertNotIn('custom_dataset_common', config)
        self.assertEqual(config['dataset_common']['data_root'], './data/N7_1600_900/')

    def test_front_wide_intrinsic_is_declared_in_1600x900_coordinates(self):
        info_json = json.loads((
            ROOT / 'data' / 'info_json' /
            '2025_04_18_2k_byd_info.json').read_text(encoding='utf-8'))
        front_wide = next(
            sensor for sensor in info_json['sensors']
            if sensor.get('name') == 'front_wide')
        self.assertEqual((front_wide['height'], front_wide['width']), (900, 1600))
        intrinsic = np.asarray(front_wide['intrinsic']['K'], dtype=np.float64)
        self.assertLess(abs(intrinsic[0, 2] - 800.0), 1.0)
        self.assertLess(abs(intrinsic[1, 2] - 450.0), 5.0)

    def test_exact_resize_crop_and_post_affine(self):
        transform = RandomAugImageMultiViewImage(
            data_config=data_config(),
            force_resize=False,
            enable_random_aug=False,
            n_images=1)
        params = transform.sample_augmentation(
            900, 1600, is_train=True, randomize=False)
        resize, resize_dims, crop, flip, rotate, _ = params
        self.assertEqual(resize, 0.44)
        self.assertEqual(resize_dims, (704, 396))
        self.assertEqual(crop, (0, 70, 704, 326))
        self.assertFalse(flip)
        self.assertEqual(rotate, 0.0)

        output = transform(native_results())
        self.assertEqual(output['img'][0].shape, (256, 704, 3))
        camera = output['lidar2img']['lidar2img_aug'][0]
        np.testing.assert_array_equal(
            camera['post_rot'],
            np.diag([np.float32(0.44), np.float32(0.44), 1.0]))
        np.testing.assert_array_equal(
            camera['post_tran'], np.asarray([0.0, -70.0, 0.0]))
        # 原图中心标记应落到 crop 后的网络输入中心附近。
        patch = output['img'][0][124:133, 348:357]
        self.assertGreater(int(patch[..., 0].max()), 200)

    def test_lidar2img_marker_projection_is_synchronized(self):
        transform = RandomAugImageMultiViewImage(
            data_config=data_config(),
            force_resize=False,
            enable_random_aug=False,
            n_images=1)
        output = transform(native_results())
        point = np.asarray([[0.0, 0.0, 10.0]], dtype=np.float32)
        uv = _project(output['lidar2img']['extrinsic'][0], point)[0]
        np.testing.assert_allclose(uv, [352.0, 128.0], atol=1e-4, rtol=0)

        raw_uv = np.asarray([800.0, 450.0], dtype=np.float32)
        expected = raw_uv * np.float32(0.44) + np.asarray([0.0, -70.0])
        np.testing.assert_allclose(uv, expected, atol=1e-4, rtol=0)

    def test_train_test_determinism_and_numpy_rng(self):
        outputs = []
        states = []
        for is_train in (True, False):
            transform = RandomAugImageMultiViewImage(
                data_config=data_config(),
                is_train=is_train,
                force_resize=False,
                enable_random_aug=False,
                n_images=1)
            np.random.seed(20260728)
            before = copy.deepcopy(np.random.get_state())
            output = transform(native_results())
            after = copy.deepcopy(np.random.get_state())
            self.assertTrue(rng_state_equal(before, after))
            outputs.append(output)
            states.append(after)
        np.testing.assert_array_equal(outputs[0]['img'][0], outputs[1]['img'][0])
        np.testing.assert_array_equal(
            outputs[0]['lidar2img']['extrinsic'][0],
            outputs[1]['lidar2img']['extrinsic'][0])
        self.assertTrue(rng_state_equal(states[0], states[1]))

    def test_lut_geometry_matches_real_test_transform(self):
        config = load_py_config(CONFIG)
        step = find_pipeline_step(
            config['data']['test']['pipeline'],
            'RandomAugImageMultiViewImage')
        camera = _camera_aug()
        camera['image_width'] = 1600
        camera['image_height'] = 900
        expected_rot, expected_tran, expected_shape = _test_image_aug_params(
            camera, step['data_config'], force_resize=step['force_resize'])

        transform = RandomAugImageMultiViewImage(
            data_config=step['data_config'],
            is_train=False,
            force_resize=step['force_resize'],
            enable_random_aug=step['enable_random_aug'],
            n_images=step['n_images'])
        output = transform(native_results())
        actual_camera = output['lidar2img']['lidar2img_aug'][0]
        np.testing.assert_array_equal(actual_camera['post_rot'], expected_rot)
        np.testing.assert_array_equal(actual_camera['post_tran'], expected_tran)
        self.assertEqual(output['img_shape'][0], expected_shape)

    def test_resolved_config_contract(self):
        config = load_py_config(DIST_CONFIG)
        baseline = load_py_config(BASELINE_DIST_CONFIG)
        self.assertEqual(config['experiment_id'], 'EXP-MONO-GEOM1-A')
        self.assertEqual(
            config['work_dir'],
            'work_dirs/n7_mono_1600_900_scale_crop/EXP-MONO-GEOM1-A/'
            '20251017_20251030_20251031_20251203_gpu4_batch64_work_8_260729')
        self.assertEqual(config['model']['n_images'], 1)
        self.assertEqual(config['data_config']['input_size'], (256, 704))
        self.assertEqual(config['data_config']['test_input_size'], (256, 704))
        self.assertEqual(config['optimizer']['lr'], 0.0001)
        self.assertEqual(config['total_epochs'], 15)
        self.assertEqual(config['seed'], 0)
        self.assertIn('cascade_mask_rcnn_r18', config['load_from'])
        # GEOM1-A 只改变图片/PKL 几何与独立实验目录；模型、ROI、anchor、
        # optimizer、schedule、初始化和 BEV 增强保持 S0 原样。
        for key in (
                'model', 'point_cloud_range', 'class_names', 'optimizer',
                'optimizer_config', 'lr_config', 'total_epochs', 'fp16',
                'load_from', 'workflow'):
            self.assertEqual(config[key], baseline[key], key)
        for step_type in ('RandomFlip3D', 'GlobalRotScaleTrans'):
            self.assertEqual(
                find_pipeline_step(config['data']['train']['pipeline'], step_type),
                find_pipeline_step(baseline['data']['train']['pipeline'], step_type))
        for split in ('train', 'val', 'test'):
            dataset = config['data'][split]
            self.assertIn('data/N7_1600_900', dataset['data_root'])
            self.assertIn('data/N7_1600_900', dataset['ann_file'])
            self.assertFalse(dataset['sequential'])
            self.assertEqual(dataset['n_times'], 1)
            step = find_pipeline_step(
                dataset['pipeline'], 'RandomAugImageMultiViewImage')
            self.assertFalse(step['force_resize'])
            self.assertFalse(step['enable_random_aug'])
            self.assertEqual(step['n_images'], 1)
        self.assertEqual(
            config['data']['test']['ann_file'],
            config['data']['val']['ann_file'])

    def test_strategy_contract_has_geom1_a_center_crop(self):
        contracts = SCALE_CROP_TOOL.strategy_contracts(
            (900, 1600), (256, 704), [70])
        geom = contracts[1]
        self.assertEqual(geom['resize_dims'], [704, 396])
        self.assertEqual(geom['crop'], [0, 70, 704, 326])
        np.testing.assert_array_equal(
            geom['post_rot'], [[0.44, 0.0], [0.0, 0.44]])
        np.testing.assert_array_equal(geom['post_tran'], [0.0, -70.0])

    def test_batched_edge_projection_matches_single_box_metrics(self):
        camera = dict(
            cam_intrinsic=np.asarray([
                [1000.0, 0.0, 800.0],
                [0.0, 900.0, 450.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),
            distortion=np.zeros(5, dtype=np.float32),
            sensor2lidar_rotation=np.eye(3, dtype=np.float32),
            sensor2lidar_translation=np.zeros(3, dtype=np.float32),
            intrinsic_width=1600,
            intrinsic_height=900,
            image_width=1600,
            image_height=900,
        )
        box = np.asarray(
            [0.0, 0.0, 10.0, 4.0, 2.0, 2.0, 0.25],
            dtype=np.float32)
        contract = SCALE_CROP_TOOL.strategy_contracts(
            (900, 1600), (256, 704), [70])[1]
        expected = SCALE_CROP_TOOL.projection_record(
            box, camera, contract, (256, 704), 17, 0.1)
        edge_uv, edge_depth = SCALE_CROP_TOOL.project_box_edges_distorted(
            box.reshape(1, -1), camera, 17)
        actual = SCALE_CROP_TOOL.projection_record_from_edges(
            edge_uv[0], edge_depth[0], contract, (256, 704), 0.1)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=5, msg=key)

    def test_full_visibility_uses_all_frames_while_pixels_are_sampled(self):
        camera = dict(
            cam_intrinsic=np.asarray([
                [1000.0, 0.0, 800.0],
                [0.0, 900.0, 450.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),
            distortion=np.zeros(5, dtype=np.float32),
            sensor2lidar_rotation=np.eye(3, dtype=np.float32),
            sensor2lidar_translation=np.zeros(3, dtype=np.float32),
            intrinsic_width=1600,
            intrinsic_height=900,
            image_width=1600,
            image_height=900,
            data_path='unused.jpg',
        )
        infos = [
            dict(
                token='token-{}'.format(index),
                gt_boxes=np.asarray(
                    [[0.0, 0.0, 10.0, 4.0, 2.0, 2.0, 0.25]],
                    dtype=np.float32),
                gt_names=np.asarray(['car']),
                cams={'cam0': camera})
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            pkl_path = Path(temp_dir) / 'infos_train.pkl'
            with pkl_path.open('wb') as stream:
                pickle.dump({'infos': infos, 'metadata': {'set': 'train'}}, stream)
            args = argparse.Namespace(
                pkl=[pkl_path], output_dir=Path(temp_dir), camera_id='cam0',
                classes=['car'], roi=[-10, -10, -10, 10, 10, 20],
                source_image_size=[900, 1600], target_image_size=[256, 704],
                crop_top_offsets=[70], distance_bins=[-10, 0, 20],
                yaw_bins=[-180, 0, 180], samples_per_edge=17,
                min_depth=0.1, start_index=0, stride=1, max_frames=-1,
                metric_stride=2, selection_count=2,
                max_eval_only_ratio=0.0, strict_visibility=True)
            records, summary, _ = SCALE_CROP_TOOL.analyze(args)

        self.assertEqual(summary['visibility_frames'], 3)
        self.assertEqual(summary['metric_frames'], 2)
        self.assertEqual(summary['population_records'], 6)
        self.assertEqual(summary['metric_records'], 4)
        self.assertEqual(len(records), 4)
        self.assertEqual(summary['status'], 'PASS')
        for row in summary['visibility']:
            self.assertEqual(row['eval_keep'], 3)
            self.assertEqual(row['train_keep'], 3)
        center = summary['groups']['strategy=scale_crop_top_70']
        self.assertEqual(center['count'], 3)
        self.assertEqual(center['sample_count'], 2)


if __name__ == '__main__':
    unittest.main()
