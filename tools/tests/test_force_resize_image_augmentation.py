#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N7 force_resize 基础几何与随机图像增强回归测试。"""

import ast
import copy
import hashlib
import json
import math
import os
import random as prandom
import string
import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    ROOT / 'tools' / 'tests' / 'fixtures' /
    'force_resize_legacy_v1.json')
N7_TOOL_DIR = ROOT / 'tools' / 'data_converter' / 'n7'
if str(N7_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(N7_TOOL_DIR))

from build_fastbev_lut import (  # noqa: E402
    find_pipeline_step,
    load_py_config,
    test_image_aug_params,
)


class _Registry:
    def register_module(self, *args, **kwargs):
        if args and isinstance(args[0], type):
            return args[0]

        def decorate(obj):
            return obj

        return decorate


def _load_transform_class():
    """只加载目标类，避免本地缺少 legacy mmcv/mmdet。"""
    source_path = (
        ROOT / 'mmdet3d' / 'datasets' / 'pipelines' / 'transforms_3d.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and
        node.name == 'RandomAugImageMultiViewImage')
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(
        PIPELINES=_Registry(),
        np=np,
        torch=torch,
        Image=Image,
        prandom=prandom,
        string=string,
        os=os,
        cv2=cv2,
        draw_lidar_bbox3d_on_img=lambda *args, **kwargs: None,
    )
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['RandomAugImageMultiViewImage']


RandomAugImageMultiViewImage = _load_transform_class()


class _Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
        return results


def _load_multi_view_pipeline_class():
    """只加载 MultiViewPipeline，避免本地缺少 legacy mmdet。"""
    source_path = (
        ROOT / 'mmdet3d' / 'datasets' / 'pipelines' / 'multi_view.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and
        node.name == 'MultiViewPipeline')
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(
        PIPELINES=_Registry(),
        Compose=_Compose,
        np=np,
    )
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['MultiViewPipeline']


MultiViewPipeline = _load_multi_view_pipeline_class()


def _data_config():
    return dict(
        input_size=(256, 704),
        resize=(-0.06, 0.11),
        crop=(-0.05, 0.05),
        rot=(-5.4, 5.4),
        flip=True,
        test_input_size=(256, 704),
        test_resize=0.0,
        test_rotate=0.0,
        test_flip=False,
        pad=(0, 0, 0, 0),
        pad_divisor=32,
        pad_color=(0, 0, 0),
    )


def _camera_aug(camera_index=0):
    intrinsic = np.array([
        [1000.0, 0.0, 800.0],
        [0.0, 900.0, 450.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return dict(
        intrin=intrinsic,
        rot=np.eye(3, dtype=np.float32),
        tran=np.array([0.1 * camera_index, 0.0, 0.0], dtype=np.float32),
        post_rot=np.eye(3, dtype=np.float32),
        post_tran=np.zeros(3, dtype=np.float32),
        intrinsic_width=1600,
        intrinsic_height=900,
        image_width=704,
        image_height=256,
    )


def _image(camera_index=0):
    yy, xx = np.indices((256, 704))
    image = np.stack([
        (xx + camera_index * 13) % 256,
        (yy * 3 + camera_index * 17) % 256,
        (xx + yy * 2 + camera_index * 19) % 256,
    ], axis=-1)
    return image.astype(np.uint8)


def _results(n_images, n_times, image_factory=_image):
    images = []
    cameras = []
    for _time_index in range(n_times):
        for camera_index in range(n_images):
            images.append(image_factory(camera_index))
            cameras.append(_camera_aug(camera_index))
    return dict(
        img=images,
        lidar2img=dict(
            lidar2img_aug=cameras,
            extrinsic=[np.eye(4, dtype=np.float32) for _ in images],
        ),
    )


def _normalized_tensor(image):
    image = image.astype(np.float32)
    image = image[..., ::-1]
    mean = np.asarray(
        [123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(
        [58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
    return np.ascontiguousarray(
        ((image - mean) / std).transpose(2, 0, 1))


def _project(projection, points):
    """按固定 float32 运算顺序投影，避免 NumPy/BLAS 末位差异。"""
    points = np.asarray(points, dtype=np.float32)
    homogeneous = np.concatenate([
        points,
        np.ones((points.shape[0], 1), dtype=np.float32),
    ], axis=1)
    projection = np.asarray(projection[:3, :4], dtype=np.float32)
    projected = np.empty((points.shape[0], 3), dtype=np.float32)
    # 小矩阵的 ``@`` 可能随 NumPy/BLAS 版本选择不同的累加顺序，数值只差
    # 约 1e-4 px，却会让原始字节 SHA 不同。测试辅助计算固定为逐项累加，
    # 主实现产出的 lidar2img 仍由 fixture 做逐字节严格校验。
    for point_index in range(points.shape[0]):
        for row_index in range(3):
            value = np.float32(0.0)
            for column_index in range(4):
                value = np.float32(
                    value + np.float32(
                        homogeneous[point_index, column_index] *
                        projection[row_index, column_index]))
            projected[point_index, row_index] = value

    normalized = np.empty((points.shape[0], 2), dtype=np.float32)
    for point_index in range(points.shape[0]):
        normalized[point_index, 0] = np.float32(
            projected[point_index, 0] / projected[point_index, 2])
        normalized[point_index, 1] = np.float32(
            projected[point_index, 1] / projected[point_index, 2])
    return normalized


def _array_fingerprint(value):
    array = np.ascontiguousarray(value)
    return dict(
        shape=list(array.shape),
        dtype=str(array.dtype),
        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
    )


def _output_snapshot(output):
    images = np.stack([
        np.asarray(image) for image in output['img']
    ])
    camera_aug = output['lidar2img']['lidar2img_aug']
    post_rot = np.stack([
        np.asarray(item['post_rot']) for item in camera_aug
    ])
    post_tran = np.stack([
        np.asarray(item['post_tran']) for item in camera_aug
    ])
    lidar2img = np.stack([
        np.asarray(item) for item in output['lidar2img']['extrinsic']
    ])
    normalized = np.stack([
        _normalized_tensor(image) for image in output['img']
    ])
    points = np.array([
        [2.0, -0.5, 8.0],
        [5.0, 1.0, 20.0],
        [12.0, -2.0, 35.0],
    ], dtype=np.float32)
    projected = np.stack([
        _project(projection, points)
        for projection in output['lidar2img']['extrinsic']
    ])

    rng_state = np.random.get_state()
    next_random = np.random.random(8)
    np.random.set_state(rng_state)
    return dict(
        images=_array_fingerprint(images),
        post_rot=_array_fingerprint(post_rot),
        post_tran=_array_fingerprint(post_tran),
        lidar2img=_array_fingerprint(lidar2img),
        normalized_tensor=_array_fingerprint(normalized),
        projected_points=_array_fingerprint(projected),
        img_shape=_array_fingerprint(
            np.asarray(output['img_shape'], dtype=np.int64)),
        numpy_rng_state=dict(
            algorithm=rng_state[0],
            keys=_array_fingerprint(rng_state[1]),
            position=int(rng_state[2]),
            has_gauss=int(rng_state[3]),
            cached_gaussian=_array_fingerprint(
                np.asarray(rng_state[4], dtype=np.float64)),
            next_random=_array_fingerprint(next_random),
        ),
    )


def _load_legacy_fixture():
    with FIXTURE_PATH.open('r', encoding='utf-8') as stream:
        fixture = json.load(stream)
    if fixture.get('schema_version') != 1:
        raise RuntimeError(
            '不支持的 force_resize fixture schema: {}'.format(
                fixture.get('schema_version')))
    return fixture


def _expected_post(base_rot, resize, crop, flip, rotate):
    post_rot = np.asarray(base_rot, dtype=np.float32).copy()
    post_tran = np.zeros(2, dtype=np.float32)
    post_rot *= np.float32(resize)
    post_tran -= np.asarray(crop[:2], dtype=np.float32)
    if flip:
        matrix = np.asarray([[-1, 0], [0, 1]], dtype=np.float32)
        bias = np.asarray([crop[2] - crop[0], 0], dtype=np.float32)
        post_rot = matrix @ post_rot
        post_tran = matrix @ post_tran + bias
    theta = float(rotate) / 180.0 * math.pi
    matrix = np.asarray([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)],
    ], dtype=np.float32)
    center = np.asarray([
        crop[2] - crop[0],
        crop[3] - crop[1],
    ], dtype=np.float32) / 2.0
    bias = matrix @ (-center) + center
    post_rot = matrix @ post_rot
    post_tran = matrix @ post_tran + bias
    result_rot = np.eye(3, dtype=np.float64)
    result_tran = np.zeros(3, dtype=np.float64)
    result_rot[:2, :2] = post_rot
    result_tran[:2] = post_tran
    return result_rot, result_tran


class ForceResizeImageAugmentationTest(unittest.TestCase):
    def test_default_switch_preserves_legacy_contract(self):
        force_resize = RandomAugImageMultiViewImage(
            data_config=_data_config(),
            force_resize=True,
        )
        ordinary = RandomAugImageMultiViewImage(
            data_config=_data_config(),
            force_resize=False,
        )
        self.assertFalse(force_resize.enable_random_aug)
        self.assertTrue(ordinary.enable_random_aug)

    def test_frozen_legacy_scenarios_are_bit_exact(self):
        fixture = _load_legacy_fixture()
        self.assertEqual(
            fixture['source']['commit'],
            'c1aea0d7c6beef930737bb3502d8c3aea7696d81')
        self.assertEqual(
            fixture['source']['file_sha256'],
            '27c6da0c6a9b4662d28dcd8265066292919b0ec6a93702ef2fbf9dbaea157c52')
        # 评审文本将全组合误记为 7 组；这里冻结 3 种布局的
        # train/test 加普通路径 train/test，共 8 组完整超集。
        self.assertEqual(len(fixture['cases']), 8)
        for case in fixture['cases']:
            with self.subTest(case=case['id']):
                transform = RandomAugImageMultiViewImage(
                    data_config=_data_config(),
                    is_train=case['is_train'],
                    force_resize=case['force_resize'],
                    enable_random_aug=case['enable_random_aug'],
                    n_images=case['configured_n_images'],
                )
                np.random.seed(case['seed'])
                output = transform(_results(
                    n_images=case['n_images'],
                    n_times=case['n_times']))
                actual = _output_snapshot(output)
                self.assertEqual(set(actual), set(case['expected']))
                for field_name, expected_value in case['expected'].items():
                    with self.subTest(
                            case=case['id'], field=field_name):
                        self.assertEqual(
                            actual[field_name], expected_value)

    def test_eval_is_deterministic_and_matches_lut_builder(self):
        inputs = _results(n_images=6, n_times=4)
        transform = RandomAugImageMultiViewImage(
            data_config=_data_config(),
            is_train=False,
            force_resize=True,
            enable_random_aug=True,
            n_images=6,
        )
        np.random.seed(1)
        first = transform(copy.deepcopy(inputs))
        np.random.seed(999)
        second = transform(copy.deepcopy(inputs))
        for view_index in range(24):
            np.testing.assert_array_equal(
                first['img'][view_index], second['img'][view_index])
            np.testing.assert_array_equal(
                first['lidar2img']['extrinsic'][view_index],
                second['lidar2img']['extrinsic'][view_index])
            cam_aug = first['lidar2img']['lidar2img_aug'][view_index]
            lut_rot, lut_tran, lut_shape = test_image_aug_params(
                cam_aug, _data_config(), force_resize=True)
            np.testing.assert_array_equal(cam_aug['post_rot'], lut_rot)
            np.testing.assert_array_equal(cam_aug['post_tran'], lut_tran)
            self.assertEqual(first['img_shape'][view_index], lut_shape)

    def test_fixed_seed_and_actual_view_organizations(self):
        for n_images, n_times in ((1, 1), (1, 4), (6, 4)):
            with self.subTest(n_images=n_images, n_times=n_times):
                transform = RandomAugImageMultiViewImage(
                    data_config=_data_config(),
                    is_train=True,
                    force_resize=True,
                    enable_random_aug=True,
                    n_images=n_images,
                )
                inputs = _results(n_images=n_images, n_times=n_times)
                np.random.seed(314159)
                first = transform(copy.deepcopy(inputs))
                np.random.seed(314159)
                second = transform(copy.deepcopy(inputs))
                for view_index in range(n_images * n_times):
                    np.testing.assert_array_equal(
                        first['img'][view_index], second['img'][view_index])
                    np.testing.assert_array_equal(
                        first['lidar2img']['extrinsic'][view_index],
                        second['lidar2img']['extrinsic'][view_index])

                for camera_index in range(n_images):
                    reference = first['lidar2img']['lidar2img_aug'][
                        camera_index]
                    for time_index in range(1, n_times):
                        current = first['lidar2img']['lidar2img_aug'][
                            time_index * n_images + camera_index]
                        np.testing.assert_array_equal(
                            current['post_rot'], reference['post_rot'])
                        np.testing.assert_array_equal(
                            current['post_tran'], reference['post_tran'])
                        np.testing.assert_array_equal(
                            first['img'][time_index * n_images + camera_index],
                            first['img'][camera_index])

                if n_images > 1:
                    transforms = {
                        (
                            tuple(item['post_rot'].reshape(-1)),
                            tuple(item['post_tran'].reshape(-1)),
                        )
                        for item in first['lidar2img']['lidar2img_aug'][
                            :n_images]
                    }
                    self.assertGreater(len(transforms), 1)

    def test_multi_view_pipeline_records_actual_layout(self):
        def load_image(results):
            results['img'] = np.zeros((4, 8, 3), dtype=np.uint8)
            return results

        sequential = MultiViewPipeline(
            transforms=[load_image],
            n_images=1,
            n_times=4,
            sequential=True)
        sequential_results = dict(
            img_prefix=[None] * 4,
            img_info=[dict(filename=str(index)) for index in range(4)],
            lidar2img=dict(
                extrinsic=[np.eye(4, dtype=np.float32) for _ in range(4)]))
        sequential_output = sequential(sequential_results)
        self.assertEqual(
            sequential_output['view_layout'],
            dict(n_images=1, n_times=4, sequential=True))

        single_time = MultiViewPipeline(
            transforms=[load_image],
            n_images=1,
            n_times=4,
            sequential=False)
        single_time_results = dict(
            img_prefix=[None],
            img_info=[dict(filename='0')],
            lidar2img=dict(extrinsic=[np.eye(4, dtype=np.float32)]))
        single_time_output = single_time(single_time_results)
        self.assertEqual(
            single_time_output['view_layout'],
            dict(n_images=1, n_times=1, sequential=False))

    def test_view_layout_is_validated_on_every_branch(self):
        conflict = RandomAugImageMultiViewImage(
            data_config=_data_config(),
            is_train=False,
            force_resize=False,
            enable_random_aug=False,
            n_images=2)
        conflict_inputs = _results(n_images=1, n_times=4)
        conflict_inputs['view_layout'] = dict(
            n_images=1, n_times=4, sequential=True)
        with self.assertRaisesRegex(ValueError, 'conflicts with view_layout'):
            conflict(conflict_inputs)

        count_mismatch = RandomAugImageMultiViewImage(
            data_config=_data_config(),
            is_train=False,
            force_resize=False,
            enable_random_aug=False)
        mismatch_inputs = _results(n_images=1, n_times=1)
        mismatch_inputs['view_layout'] = dict(
            n_images=1, n_times=4, sequential=True)
        with self.assertRaisesRegex(ValueError, 'does not match view_layout'):
            count_mismatch(mismatch_inputs)

        inferred = RandomAugImageMultiViewImage(
            data_config=_data_config(),
            is_train=True,
            force_resize=True,
            enable_random_aug=True)
        inferred_inputs = _results(n_images=1, n_times=4)
        inferred_inputs['view_layout'] = dict(
            n_images=1, n_times=4, sequential=True)
        np.random.seed(20260728)
        inferred_output = inferred(inferred_inputs)
        reference = inferred_output['lidar2img']['lidar2img_aug'][0]
        for item in inferred_output['lidar2img']['lidar2img_aug'][1:]:
            np.testing.assert_array_equal(item['post_rot'], reference['post_rot'])
            np.testing.assert_array_equal(
                item['post_tran'], reference['post_tran'])

    def test_training_size_contract_is_independent_from_randomization(self):
        config = _data_config()
        config['test_input_size'] = (128, 352)
        transform = RandomAugImageMultiViewImage(
            data_config=config,
            is_train=True,
            force_resize=False,
            enable_random_aug=False)

        np.random.seed(20260728)
        expected_next = np.random.random(8)
        np.random.seed(20260728)
        params = transform.sample_augmentation(
            H=256, W=704, is_train=True, randomize=False)
        actual_next = np.random.random(8)
        np.testing.assert_array_equal(actual_next, expected_next)
        resize, resize_dims, crop, flip, rotate, _pad = params
        self.assertEqual(resize, 1.0)
        self.assertEqual(resize_dims, (704, 256))
        self.assertEqual(crop, (0, 0, 704, 256))
        self.assertFalse(flip)
        self.assertEqual(rotate, 0.0)

        output = transform(_results(n_images=1, n_times=1))
        self.assertEqual(output['img_shape'], [(256, 704, 3)])

        np.random.seed(31415)
        expected_eval_next = np.random.random(8)
        np.random.seed(31415)
        eval_params = transform.sample_augmentation(
            H=256, W=704, is_train=False, randomize=True)
        actual_eval_next = np.random.random(8)
        np.testing.assert_array_equal(actual_eval_next, expected_eval_next)
        self.assertEqual(eval_params[1], (352, 128))

    def test_random_affine_image_post_and_projection_stay_synchronized(self):
        marker_xy = np.array([264.0, 96.0], dtype=np.float32)
        native_xy = np.array([
            marker_xy[0] * 1600.0 / 704.0,
            marker_xy[1] * 900.0 / 256.0,
        ], dtype=np.float32)

        def marker_image(_camera_index):
            image = np.zeros((256, 704, 3), dtype=np.uint8)
            x, y = marker_xy.astype(int)
            image[y - 4:y + 5, x - 4:x + 5] = 255
            return image

        cases = [
            ('crop_min', 1.08, (0, -13, 704, 243), False, 0.0),
            ('crop_max', 1.08, (56, 20, 760, 276), False, 0.0),
            ('horizontal_flip', 1.0, (0, 0, 704, 256), True, 0.0),
            ('rotate_positive', 1.0, (0, 0, 704, 256), False, 5.4),
            ('rotate_negative', 1.0, (0, 0, 704, 256), False, -5.4),
        ]
        for name, resize, crop, flip, rotate in cases:
            with self.subTest(case=name):
                resize_dims = (
                    int(704 * resize),
                    int(256 * resize),
                )
                params = (
                    resize,
                    resize_dims,
                    crop,
                    flip,
                    rotate,
                    ((0, 0, 0, 0), (0, 0, 0)),
                )
                transform = RandomAugImageMultiViewImage(
                    data_config=_data_config(),
                    is_train=True,
                    force_resize=True,
                    enable_random_aug=True,
                    n_images=1,
                )

                def fixed_sample(
                        _self, H, W, is_train=None, randomize=None):
                    return params

                transform.sample_augmentation = types.MethodType(
                    fixed_sample, transform)
                output = transform(_results(
                    n_images=1,
                    n_times=1,
                    image_factory=marker_image))
                cam_aug = output['lidar2img']['lidar2img_aug'][0]
                expected_rot, expected_tran = _expected_post(
                    np.diag([
                        704.0 / 1600.0,
                        256.0 / 900.0,
                    ]),
                    resize,
                    crop,
                    flip,
                    rotate,
                )
                np.testing.assert_allclose(
                    cam_aug['post_rot'], expected_rot, rtol=0.0, atol=1e-6)
                np.testing.assert_allclose(
                    cam_aug['post_tran'], expected_tran, rtol=0.0, atol=1e-5)

                source_h = np.array(
                    [native_xy[0], native_xy[1], 1.0], dtype=np.float64)
                expected_xy = (
                    expected_rot @ source_h + expected_tran)[:2]
                depth = 10.0
                intrinsic = cam_aug['intrin']
                point = np.array([[
                    (native_xy[0] - intrinsic[0, 2]) / intrinsic[0, 0] * depth,
                    (native_xy[1] - intrinsic[1, 2]) / intrinsic[1, 1] * depth,
                    depth,
                ]], dtype=np.float32)
                np.testing.assert_allclose(
                    _project(output['lidar2img']['extrinsic'][0], point)[0],
                    expected_xy,
                    rtol=0.0,
                    atol=2e-4,
                )

                image = output['img'][0].astype(np.float32).mean(axis=2)
                ys, xs = np.nonzero(image > 32.0)
                self.assertGreater(xs.size, 0)
                weights = image[ys, xs]
                observed_xy = np.array([
                    np.average(xs, weights=weights),
                    np.average(ys, weights=weights),
                ])
                np.testing.assert_allclose(
                    observed_xy, expected_xy, rtol=0.0, atol=2.0)

    def test_random_aug_requires_valid_dynamic_camera_count(self):
        missing = RandomAugImageMultiViewImage(
            data_config=_data_config(),
            is_train=True,
            force_resize=True,
            enable_random_aug=True,
        )
        with self.assertRaisesRegex(ValueError, 'n_images is required'):
            missing(_results(n_images=1, n_times=1))

        invalid = RandomAugImageMultiViewImage(
            data_config=_data_config(),
            is_train=True,
            force_resize=True,
            enable_random_aug=True,
            n_images=6,
        )
        with self.assertRaisesRegex(ValueError, 'not divisible'):
            invalid(_results(n_images=1, n_times=4))

    def test_product_configs_keep_random_image_aug_disabled(self):
        cases = [
            (
                'configs/fastbev/custom/'
                'custom_fastbev_mono_front_single_frame_r18.py',
                1,
                1,
                False,
            ),
            (
                'configs/fastbev/custom/custom_fastbev_mono_front_'
                'single_frame_r18_dist_train.py',
                1,
                1,
                False,
            ),
            (
                'configs/fastbev/custom/custom_fastbev_mono_front_r18.py',
                1,
                4,
                True,
            ),
            (
                'configs/fastbev/custom/'
                'custom_fastbev_mono_front_r18_dist_train.py',
                1,
                4,
                True,
            ),
            (
                'configs/fastbev/custom/'
                'custom_fastbev_6v_r18_n7_704x256.py',
                6,
                4,
                True,
            ),
            (
                'configs/fastbev/custom/'
                'custom_fastbev_6v_r18_n7_704x256_dist_train.py',
                6,
                4,
                True,
            ),
        ]
        for relative_path, n_images, n_times, sequential in cases:
            with self.subTest(config=relative_path):
                config = load_py_config(ROOT / relative_path)
                train = config['data']['train']
                test = config['data']['test']
                self.assertEqual(train['n_times'], n_times)
                self.assertEqual(train['sequential'], sequential)
                self.assertEqual(config['model']['n_images'], n_images)
                for dataset in (train, test):
                    step = find_pipeline_step(
                        dataset['pipeline'],
                        'RandomAugImageMultiViewImage',
                    )
                    self.assertTrue(step['force_resize'])
                    self.assertFalse(step['enable_random_aug'])
                    self.assertEqual(step['n_images'], n_images)


if __name__ == '__main__':
    unittest.main()
