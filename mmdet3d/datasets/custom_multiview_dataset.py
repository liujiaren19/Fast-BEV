import os
import os.path as osp
import random
from collections import defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from mmdet.datasets import DATASETS

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.evaluation import indoor_eval
from tools.n7_box_origin import boxes_to_numpy as n7_boxes_to_numpy
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class CustomMultiViewDataset(Custom3DDataset):
    """Multi-view camera dataset for converted in-house Fast-BEV infos."""

    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
    N7_X_RANGE_BINS = ((0, 20), (20, 40), (40, 60), (60, 80))
    N7_TP_MATCH_DISTANCE = 2.0
    N7_EVAL_METRIC_SCHEMA_VERSION = 2

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 camera_types=None,
                 sequential=False,
                 n_times=1,
                 prev_only=True,
                 train_adj_ids=None,
                 test_adj_ids=None,
                 temporal_compensate=True,
                 max_interval=3,
                 min_interval=0,
                 shuffle=False,
                 seed=0,
                 eval_iou_thr=(0.25, 0.5),
                 eval_distance_thr=(0.5, 1.0, 2.0, 4.0),
                 eval_range=None,
                 eval_score_thr=0.0,
                 eval_max_dets_per_sample=None,
                 filter_gt_visible_camera=None,
                 filter_gt_visible_min_depth=0.1):
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.camera_types = camera_types
        self.sequential = sequential
        self.n_times = n_times
        self.prev_only = prev_only
        self.train_adj_ids = train_adj_ids
        self.test_adj_ids = test_adj_ids
        self.temporal_compensate = temporal_compensate
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.shuffle = shuffle
        self.seed = seed
        self.eval_iou_thr = eval_iou_thr
        self.eval_distance_thr = eval_distance_thr
        self.eval_range = eval_range
        self.eval_score_thr = float(eval_score_thr)
        self.eval_max_dets_per_sample = eval_max_dets_per_sample
        self.filter_gt_visible_camera = self._normalize_visible_camera_filter(
            filter_gt_visible_camera)
        self.filter_gt_visible_min_depth = float(filter_gt_visible_min_depth)

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        self.metadata = data.get('metadata', {})
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(data_infos)
        return data_infos

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        cat_ids = []
        for name in set(info.get('gt_names', [])):
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def _resolve_path(self, path):
        if osp.isabs(path):
            return path
        return osp.join(self.data_root, path)

    @staticmethod
    def _quat_wxyz_to_matrix(quat):
        """Convert nuScenes-style ``[w, x, y, z]`` quaternion to a rotation matrix."""
        quat = np.asarray(quat, dtype=np.float32).reshape(4)
        norm = np.linalg.norm(quat)
        if norm <= 0:
            raise ValueError('zero-norm quaternion')
        w, x, y, z = quat / norm
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)

    @classmethod
    def _info_lidar2global(cls, info):
        """Return ``global_from_lidar`` pose from one converter info.

        The converter writes clip-local poses as ``lidar2global_*`` in the same
        Fast-BEV lidar axes as the boxes and camera extrinsics.  The coordinate
        named "global" can be a clip reference frame; temporal compensation only
        needs relative transforms, so absolute world meaning is not required.
        """
        rot = info.get('lidar2global_rotation')
        tran = info.get('lidar2global_translation')
        if rot is None or tran is None:
            return None
        return cls._quat_wxyz_to_matrix(rot), np.asarray(tran, dtype=np.float32).reshape(3)

    def _sensor2reference_lidar(self, cam_info, frame_info, ref_info):
        """Express a camera-to-lidar transform in the key-frame lidar system.

        For the key frame itself, the stored ``sensor2lidar`` transform is
        already correct.  For an adjacent frame, Fast-BEV temporal fusion expects
        that adjacent camera rays are projected into the key-frame BEV volume.
        With ``global_from_lidar`` poses, the required rigid transform is:

            key_from_adj = inverse(global_from_key) @ global_from_adj
            key_lidar_from_adj_cam = key_from_adj @ adj_lidar_from_adj_cam

        新版 N7 pkl 必须写入相邻帧位姿；缺失时直接报错，避免时序融合静默退化。
        """
        sensor2lidar_r = np.asarray(
            cam_info['sensor2lidar_rotation'], dtype=np.float32)
        sensor2lidar_t = np.asarray(
            cam_info['sensor2lidar_translation'], dtype=np.float32).reshape(3)

        if (not self.temporal_compensate or ref_info is None or
                frame_info.get('token') == ref_info.get('token')):
            return sensor2lidar_r, sensor2lidar_t, False

        ref_pose = self._info_lidar2global(ref_info)
        frame_pose = self._info_lidar2global(frame_info)
        if ref_pose is None or frame_pose is None:
            raise KeyError(
                'CustomMultiViewDataset requires lidar2global_rotation and '
                'lidar2global_translation for temporal compensation in N7 pkl')

        ref_to_global_r, ref_to_global_t = ref_pose
        frame_to_global_r, frame_to_global_t = frame_pose
        key_from_adj_r = ref_to_global_r.T @ frame_to_global_r
        key_from_adj_t = ref_to_global_r.T @ (frame_to_global_t - ref_to_global_t)

        compensated_r = key_from_adj_r @ sensor2lidar_r
        compensated_t = key_from_adj_r @ sensor2lidar_t + key_from_adj_t
        return compensated_r.astype(np.float32), compensated_t.astype(np.float32), True

    @staticmethod
    def _lidar2img_from_cam_info(cam_info, sensor2lidar_r=None,
                                 sensor2lidar_t=None,
                                 temporal_compensated=False,
                                 intrinsic_override=None):
        intrinsic = np.asarray(
            intrinsic_override if intrinsic_override is not None else cam_info['cam_intrinsic'],
            dtype=np.float32)
        if sensor2lidar_r is None:
            sensor2lidar_r = np.asarray(
                cam_info['sensor2lidar_rotation'], dtype=np.float32)
        if sensor2lidar_t is None:
            sensor2lidar_t = np.asarray(
                cam_info['sensor2lidar_translation'], dtype=np.float32)
        sensor2lidar_t = np.asarray(sensor2lidar_t, dtype=np.float32).reshape(3)

        lidar2cam_r = np.linalg.inv(sensor2lidar_r)
        lidar2cam_t = sensor2lidar_t @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4, dtype=np.float32)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img = viewpad @ lidar2cam_rt.T

        distortion = np.asarray(
            cam_info.get('distortion', []), dtype=np.float32).reshape(-1)
        # intrinsic_* 表示 cam_intrinsic 对应的图像坐标系；image_* 表示当前
        # data_path 指向图片的实际尺寸。新版 N7 pkl 必须显式提供这四个字段。
        required_size_keys = (
            'intrinsic_width', 'intrinsic_height', 'image_width', 'image_height')
        missing_size_keys = [key for key in required_size_keys if key not in cam_info]
        if missing_size_keys:
            raise KeyError(
                'N7 camera info missing required size fields: {}'.format(
                    ', '.join(missing_size_keys)))
        intrinsic_width = int(cam_info['intrinsic_width'])
        intrinsic_height = int(cam_info['intrinsic_height'])
        image_width = int(cam_info['image_width'])
        image_height = int(cam_info['image_height'])
        if min(intrinsic_width, intrinsic_height, image_width, image_height) <= 0:
            raise ValueError(
                'N7 camera size fields must be positive: '
                'intrinsic={}x{}, image={}x{}'.format(
                    intrinsic_width, intrinsic_height, image_width, image_height))
        lidar2img_aug = dict(
            intrin=intrinsic,
            rot=sensor2lidar_r,
            tran=sensor2lidar_t,
            post_rot=np.eye(3, dtype=np.float32),
            post_tran=np.zeros(3, dtype=np.float32),
            temporal_compensated=temporal_compensated,
            distortion=distortion,
            intrinsic_width=intrinsic_width,
            intrinsic_height=intrinsic_height,
            image_width=image_width,
            image_height=image_height,
        )
        lidar2img_extra = dict(
            distortion=distortion,
            temporal_compensated=temporal_compensated,
            intrinsic_width=intrinsic_width,
            intrinsic_height=intrinsic_height,
            image_width=image_width,
            image_height=image_height,
        )
        return lidar2img.astype(np.float32), lidar2img_aug, lidar2img_extra

    @staticmethod
    def _normalize_visible_camera_filter(camera_ids):
        """归一化 GT 可见性过滤相机配置。"""
        if camera_ids is None:
            return []
        if isinstance(camera_ids, str):
            return [camera_ids]
        return [str(x) for x in camera_ids]

    @staticmethod
    def _box_corners_from_boxes(boxes):
        """按 LiDARInstance3DBoxes.corners 约定生成 box 角点。"""
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            return np.zeros((0, 8, 3), dtype=np.float32)
        boxes = boxes.reshape(-1, boxes.shape[-1])[:, :7]
        corners = np.zeros((boxes.shape[0], 8, 3), dtype=np.float32)
        for idx, box in enumerate(boxes):
            x, y, z, length, width, height, yaw = [float(v) for v in box[:7]]
            if length <= 0 or width <= 0 or height <= 0:
                continue
            local = np.array([
                [ length / 2,  width / 2, -height / 2],
                [ length / 2, -width / 2, -height / 2],
                [-length / 2, -width / 2, -height / 2],
                [-length / 2,  width / 2, -height / 2],
                [ length / 2,  width / 2,  height / 2],
                [ length / 2, -width / 2,  height / 2],
                [-length / 2, -width / 2,  height / 2],
                [-length / 2,  width / 2,  height / 2],
            ], dtype=np.float32)
            c, s = np.cos(yaw), np.sin(yaw)
            local_xy = local[:, :2].copy()
            local[:, 0] = local_xy[:, 0] * c + local_xy[:, 1] * s
            local[:, 1] = -local_xy[:, 0] * s + local_xy[:, 1] * c
            corners[idx] = local + np.array([x, y, z], dtype=np.float32)
        return corners

    @staticmethod
    def _image_space_intrinsic(cam_info):
        """把 pkl 标定内参缩放到当前 data_path 图片坐标系。"""
        intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32).copy()
        intrinsic_width = int(cam_info['intrinsic_width'])
        intrinsic_height = int(cam_info['intrinsic_height'])
        image_width = int(cam_info['image_width'])
        image_height = int(cam_info['image_height'])
        if min(intrinsic_width, intrinsic_height, image_width, image_height) <= 0:
            raise ValueError(
                'N7 camera size fields must be positive: intrinsic={}x{}, image={}x{}'.format(
                    intrinsic_width, intrinsic_height, image_width, image_height))
        intrinsic[0, :] *= image_width / float(intrinsic_width)
        intrinsic[1, :] *= image_height / float(intrinsic_height)
        return intrinsic, (image_height, image_width)

    def _project_points_visible(self, points, projection, image_shape):
        """判断 lidar 点是否投影到指定相机图像内。"""
        if points.size == 0:
            return np.zeros((0,), dtype=np.bool_)
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        points_2d_3 = points_h @ projection[:3, :4].T
        depth = points_2d_3[:, 2]
        eps = np.finfo(np.float32).eps
        xs = points_2d_3[:, 0] / np.maximum(depth, eps)
        ys = points_2d_3[:, 1] / np.maximum(depth, eps)
        height, width = image_shape[:2]
        return ((depth > self.filter_gt_visible_min_depth) &
                (xs >= 0) & (xs < width) &
                (ys >= 0) & (ys < height))

    def _gt_visible_camera_mask(self, info, boxes):
        """按配置相机过滤不可见 GT，供 train ann 和 eval GT 共用。"""
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.bool_)
        boxes = boxes.reshape(-1, boxes.shape[-1])[:, :7]
        if not self.filter_gt_visible_camera:
            return np.ones((boxes.shape[0],), dtype=np.bool_)

        cams = info.get('cams', {}) or {}
        missing = [cam for cam in self.filter_gt_visible_camera if cam not in cams]
        if missing:
            raise KeyError(
                'filter_gt_visible_camera requested cameras missing in token={}: {}'.format(
                    info.get('token'), missing))

        centers = boxes[:, :3]
        corners = self._box_corners_from_boxes(boxes)
        keep = np.zeros((boxes.shape[0],), dtype=np.bool_)
        for cam_id in self.filter_gt_visible_camera:
            cam_info = cams[cam_id]
            intrinsic, image_shape = self._image_space_intrinsic(cam_info)
            projection, _, _ = self._lidar2img_from_cam_info(
                cam_info, intrinsic_override=intrinsic)
            center_visible = self._project_points_visible(
                centers, projection, image_shape)
            corner_visible = self._project_points_visible(
                corners.reshape(-1, 3), projection, image_shape)
            corner_visible = corner_visible.reshape(boxes.shape[0], -1).any(axis=1)
            keep |= center_visible | corner_visible
        return keep

    def _select_adjacent(self, info, time_id):
        adj_ids = self.test_adj_ids if self.test_mode else self.train_adj_ids
        if adj_ids is not None:
            # 显式 adj_ids 路径下 max_interval 不生效，min_interval 也不会参与选择。
            # 这些参数保留是为了贴近 nuScenes 数据集参数格式并兼容现有配置。
            select_id = min(abs(adj_ids[time_id - 1]),
                            len(info.get('prev', [])) - 1)
        else:
            select_id = min(self.min_interval + time_id - 1,
                            len(info.get('prev', [])) - 1)
        return info['prev'][max(select_id, 0)]

    def _collect_one_frame(self, info, ref_info=None):
        image_paths, lidar2img_rts, lidar2img_augs, lidar2img_extras = [], [], [], []
        cam_items = info['cams'].items()
        if self.camera_types is not None:
            cam_items = [(cam, info['cams'][cam]) for cam in self.camera_types]

        for _, cam_info in cam_items:
            image_paths.append(self._resolve_path(cam_info['data_path']))
            sensor2lidar_r, sensor2lidar_t, compensated = self._sensor2reference_lidar(
                cam_info, frame_info=info, ref_info=ref_info)
            lidar2img_rt, lidar2img_aug, lidar2img_extra = self._lidar2img_from_cam_info(
                cam_info,
                sensor2lidar_r=sensor2lidar_r,
                sensor2lidar_t=sensor2lidar_t,
                temporal_compensated=compensated)
            lidar2img_rts.append(lidar2img_rt)
            lidar2img_augs.append(lidar2img_aug)
            lidar2img_extras.append(lidar2img_extra)
        return image_paths, lidar2img_rts, lidar2img_augs, lidar2img_extras

    def get_data_info(self, index):
        info = self.data_infos[index]
        image_paths, lidar2img_rts, lidar2img_augs, lidar2img_extras = self._collect_one_frame(info, ref_info=info)

        if self.sequential:
            for time_id in range(1, self.n_times):
                if not info.get('prev'):
                    adj_info = info
                else:
                    adj_info = self._select_adjacent(info, time_id)
                adj_paths, adj_rts, adj_augs, adj_extras = self._collect_one_frame(adj_info, ref_info=info)
                image_paths.extend(adj_paths)
                lidar2img_rts.extend(adj_rts)
                lidar2img_augs.extend(adj_augs)
                lidar2img_extras.extend(adj_extras)

        n_cameras = len(image_paths)
        input_dict = dict(
            sample_idx=info['token'],
            timestamp=info['timestamp'] / 1e6,
            img_prefix=[None] * n_cameras,
            img_info=[dict(filename=x) for x in image_paths],
            lidar2img=dict(
                extrinsic=[x.astype(np.float32) for x in lidar2img_rts],
                intrinsic=np.eye(4, dtype=np.float32),
                lidar2img_aug=lidar2img_augs,
                lidar2img_extra=lidar2img_extras,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and not (annos['gt_labels_3d'] >= 0).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_bboxes_3d = np.asarray(
            info.get('gt_boxes', np.zeros((0, 7))), dtype=np.float32)
        if gt_bboxes_3d.size == 0:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
        if gt_bboxes_3d.ndim == 1:
            gt_bboxes_3d = gt_bboxes_3d.reshape(1, -1)
        gt_names_3d = np.asarray(info.get('gt_names', []))

        gt_labels_3d = []
        for cat in gt_names_3d:
            gt_labels_3d.append(self.CLASSES.index(cat)
                                if cat in self.CLASSES else -1)
        gt_labels_3d = np.asarray(gt_labels_3d, dtype=np.int64)
        valid_len = min(gt_bboxes_3d.shape[0], gt_labels_3d.shape[0])
        gt_bboxes_3d = gt_bboxes_3d[:valid_len]
        gt_names_3d = gt_names_3d[:valid_len]
        gt_labels_3d = gt_labels_3d[:valid_len]
        gt_velocity = np.asarray(
            info.get('gt_velocity', np.zeros((valid_len, 2))),
            dtype=np.float32)
        if gt_velocity.size == 0:
            gt_velocity = np.zeros((valid_len, 2), dtype=np.float32)
        elif gt_velocity.ndim == 1:
            gt_velocity = gt_velocity.reshape(-1, 2)
        if gt_velocity.shape[0] < valid_len:
            padded_velocity = np.zeros((valid_len, 2), dtype=np.float32)
            padded_velocity[:gt_velocity.shape[0]] = gt_velocity
            gt_velocity = padded_velocity
        else:
            gt_velocity = gt_velocity[:valid_len]
        # 训练侧防御性过滤未知类别，和 eval 侧 _parse_gt_info 的 labels >= 0 保持一致。
        valid_label = gt_labels_3d >= 0
        gt_bboxes_3d = gt_bboxes_3d[valid_label]
        gt_names_3d = gt_names_3d[valid_label]
        gt_labels_3d = gt_labels_3d[valid_label]
        gt_velocity = gt_velocity[valid_label]
        visible_mask = self._gt_visible_camera_mask(info, gt_bboxes_3d[:, :7])
        gt_bboxes_3d = gt_bboxes_3d[visible_mask]
        gt_names_3d = gt_names_3d[visible_mask]
        gt_labels_3d = gt_labels_3d[visible_mask]
        gt_velocity = gt_velocity[visible_mask]

        if self.with_velocity:
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)

    @staticmethod
    def _to_numpy(value):
        """把 torch.Tensor / numpy / list 统一转为 numpy，便于 eval 纯 CPU 计算。"""
        if value is None:
            return np.zeros((0,), dtype=np.float32)
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        elif hasattr(value, 'cpu') and hasattr(value.cpu(), 'numpy'):
            value = value.cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _boxes_to_numpy(cls, boxes, target_origin='tensor', source_origin=None):
        """通过 N7 公共 origin 契约取出前 7 维 box。

        ``LiDARInstance3DBoxes.tensor[:, 2]`` 使用底中心 z，而 N7 converter
        保存的 GT ``location.z`` 使用 3D 框重心。评估 xyz 误差时需要把预测
        统一到重心；BEV IoU 和 xy center AP 不受该转换影响。

        Args:
            boxes: mmdet3d box 对象或 numpy/tensor-like 数组。
            target_origin: ``tensor`` 保留输入 z；``center`` / ``bottom``
                分别转换成重心或底中心 z。
            source_origin: 数组输入的显式语义，可为 ``center`` 或 ``bottom``。
                mmdet3d box 对象会优先读取 ``gravity_center``，无需指定。
        """
        public_target = 'native' if target_origin == 'tensor' else target_origin
        return n7_boxes_to_numpy(
            boxes,
            target_origin=public_target,
            source_origin=source_origin,
            box_dim=7,
            dtype=np.float32)

    @staticmethod
    def _signed_polygon_area(poly):
        """计算多边形有向面积；输入为 Nx2 点。"""
        if len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @classmethod
    def _polygon_area(cls, poly):
        """计算多边形面积。"""
        return abs(cls._signed_polygon_area(poly))

    @staticmethod
    def _cross_2d(a, b):
        """二维叉积。"""
        return float(a[0] * b[1] - a[1] * b[0])

    @classmethod
    def _box_bev_corners(cls, box):
        """把 [x, y, z, l, w, h, yaw] 转成 BEV 四角点。"""
        x, y, _, length, width, _, yaw = [float(v) for v in box[:7]]
        half_l = max(length, 0.0) * 0.5
        half_w = max(width, 0.0) * 0.5
        corners = np.asarray([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ], dtype=np.float32)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot = np.asarray([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]], dtype=np.float32)
        return corners @ rot.T + np.asarray([x, y], dtype=np.float32)

    @classmethod
    def _convex_intersection(cls, subject, clipper):
        """Sutherland-Hodgman 凸多边形裁剪，用于无额外依赖计算旋转框 BEV IoU。"""
        if len(subject) == 0 or len(clipper) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        output = [np.asarray(p, dtype=np.float32) for p in subject]
        orientation = 1.0 if cls._signed_polygon_area(clipper) >= 0 else -1.0

        def inside(point, edge_start, edge_end):
            return orientation * cls._cross_2d(edge_end - edge_start, point - edge_start) >= -1e-6

        def intersection(line_start, line_end, edge_start, edge_end):
            line_vec = line_end - line_start
            edge_vec = edge_end - edge_start
            denom = cls._cross_2d(line_vec, edge_vec)
            if abs(denom) < 1e-8:
                return line_end
            t = cls._cross_2d(edge_start - line_start, edge_vec) / denom
            return line_start + t * line_vec

        for idx in range(len(clipper)):
            edge_start = np.asarray(clipper[idx], dtype=np.float32)
            edge_end = np.asarray(clipper[(idx + 1) % len(clipper)], dtype=np.float32)
            input_points = output
            output = []
            if not input_points:
                break
            prev = input_points[-1]
            for cur in input_points:
                cur_inside = inside(cur, edge_start, edge_end)
                prev_inside = inside(prev, edge_start, edge_end)
                if cur_inside:
                    if not prev_inside:
                        output.append(intersection(prev, cur, edge_start, edge_end))
                    output.append(cur)
                elif prev_inside:
                    output.append(intersection(prev, cur, edge_start, edge_end))
                prev = cur
        if not output:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack(output, axis=0).astype(np.float32)

    @classmethod
    def _bev_iou(cls, box_a, box_b):
        """计算两个 3D box 在 BEV 平面的旋转 IoU。"""
        poly_a = cls._box_bev_corners(box_a)
        poly_b = cls._box_bev_corners(box_b)
        area_a = cls._polygon_area(poly_a)
        area_b = cls._polygon_area(poly_b)
        if area_a <= 0 or area_b <= 0:
            return 0.0
        inter_poly = cls._convex_intersection(poly_a, poly_b)
        inter_area = cls._polygon_area(inter_poly)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    @classmethod
    def _prepare_bev_eval_cache(cls, boxes):
        """预计算 GT BEV 几何，避免评估时为每个预测框重复构造。"""
        if len(boxes) == 0:
            return dict(
                boxes=boxes,
                corners=[],
                areas=np.zeros((0,), dtype=np.float32),
                mins=np.zeros((0, 2), dtype=np.float32),
                maxs=np.zeros((0, 2), dtype=np.float32))

        corners = [cls._box_bev_corners(box) for box in boxes]
        areas = np.asarray(
            [cls._polygon_area(poly) for poly in corners], dtype=np.float32)
        mins = np.asarray([poly.min(axis=0) for poly in corners], dtype=np.float32)
        maxs = np.asarray([poly.max(axis=0) for poly in corners], dtype=np.float32)
        return dict(
            boxes=boxes,
            corners=corners,
            areas=areas,
            mins=mins,
            maxs=maxs)

    @classmethod
    def _bev_iou_values_from_cache(cls, pred_box, gt_cache):
        """计算一个预测框与同帧所有 GT 的 BEV IoU。

        先用 axis-aligned BEV bbox 做精确预筛；AABB 不相交时旋转框 IoU
        必然为 0，因此可以跳过较慢的多边形裁剪。
        """
        gt_num = len(gt_cache['corners'])
        values = np.zeros((gt_num,), dtype=np.float32)
        if gt_num == 0:
            return values

        pred_poly = cls._box_bev_corners(pred_box)
        pred_area = cls._polygon_area(pred_poly)
        if pred_area <= 0:
            return values

        pred_min = pred_poly.min(axis=0)
        pred_max = pred_poly.max(axis=0)
        candidates = np.where(
            (gt_cache['areas'] > 0) &
            (gt_cache['maxs'][:, 0] >= pred_min[0]) &
            (gt_cache['mins'][:, 0] <= pred_max[0]) &
            (gt_cache['maxs'][:, 1] >= pred_min[1]) &
            (gt_cache['mins'][:, 1] <= pred_max[1]))[0]

        for gt_idx in candidates:
            inter_poly = cls._convex_intersection(
                pred_poly, gt_cache['corners'][gt_idx])
            inter_area = cls._polygon_area(inter_poly)
            union = pred_area + float(gt_cache['areas'][gt_idx]) - inter_area
            if union > 0:
                values[gt_idx] = float(inter_area / union)
        return values

    @staticmethod
    def _average_precision(recalls, precisions):
        """按 continuous AP 计算 PR 曲线面积。"""
        if recalls.size == 0:
            return 0.0
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        change = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))

    def _normalize_thresholds(self, values):
        """把阈值配置统一为 float list。"""
        if values is None:
            return []
        if isinstance(values, (int, float)):
            return [float(values)]
        return [float(x) for x in values]

    def _eval_range_mask(self, boxes):
        """根据 eval_range 过滤评估范围外的 GT/预测框。"""
        if boxes.shape[0] == 0 or self.eval_range is None:
            return np.ones((boxes.shape[0],), dtype=bool)
        eval_range = np.asarray(self.eval_range, dtype=np.float32).reshape(-1)
        if eval_range.size == 6:
            x_min, y_min, z_min, x_max, y_max, z_max = eval_range.tolist()
            return ((boxes[:, 0] >= x_min) & (boxes[:, 0] <= x_max) &
                    (boxes[:, 1] >= y_min) & (boxes[:, 1] <= y_max) &
                    (boxes[:, 2] >= z_min) & (boxes[:, 2] <= z_max))
        if eval_range.size == 4:
            x_min, y_min, x_max, y_max = eval_range.tolist()
            return ((boxes[:, 0] >= x_min) & (boxes[:, 0] <= x_max) &
                    (boxes[:, 1] >= y_min) & (boxes[:, 1] <= y_max))
        raise ValueError('eval_range must be [x_min,y_min,x_max,y_max] or point_cloud_range')

    def _parse_det_result(self, result):
        """解析 Fast-BEV/bbox3d2result 输出。"""
        if 'pts_bbox' in result:
            result = result['pts_bbox']
        # N7 pkl GT 使用重心 z；mmdet3d 预测对象使用底中心 z。这里先统一成
        # 重心再做 eval_range 和 xyz 误差统计。portable numpy 结果应显式写
        # ``box_origin``，旧数组未声明时保持原值以兼容历史文件。
        boxes = self._boxes_to_numpy(
            result.get('boxes_3d'),
            target_origin='center',
            source_origin=result.get('box_origin'))
        scores = self._to_numpy(result.get('scores_3d')).astype(np.float32).reshape(-1)
        labels = self._to_numpy(result.get('labels_3d')).astype(np.int64).reshape(-1)
        valid_len = min(boxes.shape[0], scores.shape[0], labels.shape[0])
        boxes = boxes[:valid_len]
        scores = scores[:valid_len]
        labels = labels[:valid_len]
        valid = scores >= self.eval_score_thr
        valid &= self._eval_range_mask(boxes)
        boxes = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]

        max_dets = self.eval_max_dets_per_sample
        if max_dets is not None:
            max_dets = int(max_dets)
            if max_dets > 0 and scores.shape[0] > max_dets:
                topk = np.argpartition(-scores, max_dets - 1)[:max_dets]
                topk = topk[np.argsort(-scores[topk])]
                boxes = boxes[topk]
                scores = scores[topk]
                labels = labels[topk]
        return boxes, scores, labels

    def _parse_gt_info(self, info):
        """解析 pkl info 中的 GT，并过滤未知类别和评估范围外目标。"""
        boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))), dtype=np.float32)
        if boxes.size == 0:
            boxes = np.zeros((0, 7), dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        boxes = boxes[:, :7]
        labels = []
        for cat in info.get('gt_names', []):
            labels.append(self.CLASSES.index(cat) if cat in self.CLASSES else -1)
        labels = np.asarray(labels, dtype=np.int64)
        valid_len = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:valid_len]
        labels = labels[:valid_len]
        valid = labels >= 0
        valid &= self._eval_range_mask(boxes)
        valid &= self._gt_visible_camera_mask(info, boxes)
        return boxes[valid], labels[valid]

    def _collect_eval_items(self, results):
        """把 results 和 data_infos 整理成按类别检索的 GT/预测集合。"""
        assert isinstance(results, list), 'results must be a list, got {}'.format(type(results))
        assert len(results) <= len(self.data_infos), (
            'results length {} exceeds data_infos length {}'.format(len(results), len(self.data_infos)))

        class_ids = list(range(len(self.CLASSES)))
        gt_by_class = {class_id: defaultdict(list) for class_id in class_ids}
        pred_by_class = {class_id: [] for class_id in class_ids}

        for sample_id, (result, info) in enumerate(zip(results, self.data_infos)):
            gt_boxes, gt_labels = self._parse_gt_info(info)
            for class_id in class_ids:
                gt_by_class[class_id][sample_id] = gt_boxes[gt_labels == class_id]

            pred_boxes, pred_scores, pred_labels = self._parse_det_result(result)
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                label = int(label)
                if label not in pred_by_class:
                    continue
                pred_by_class[label].append((float(score), sample_id, box.astype(np.float32)))
        return gt_by_class, pred_by_class

    @staticmethod
    def _yaw_error(pred_yaw, gt_yaw):
        """计算 yaw 最小夹角误差，单位 rad。"""
        diff = float(pred_yaw) - float(gt_yaw)
        return abs((diff + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _scale_error(pred_box, gt_box):
        """nuScenes-like ASE：对齐中心和方向后，用尺寸 IoU 表示尺度误差。"""
        pred_size = np.maximum(np.asarray(pred_box[3:6], dtype=np.float32), 0.0)
        gt_size = np.maximum(np.asarray(gt_box[3:6], dtype=np.float32), 0.0)
        pred_vol = float(np.prod(pred_size))
        gt_vol = float(np.prod(gt_size))
        if pred_vol <= 0 or gt_vol <= 0:
            return 1.0
        inter = float(np.prod(np.minimum(pred_size, gt_size)))
        union = pred_vol + gt_vol - inter
        if union <= 0:
            return 1.0
        return float(1.0 - inter / union)

    @staticmethod
    def _percentile_or_zero(values, percentile):
        if len(values) == 0:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))

    @staticmethod
    def _range_key(start, end):
        return '{}_{}m'.format(int(start), int(end))

    @staticmethod
    def _range_label(start, end):
        return '{}-{}m'.format(int(start), int(end))

    def _count_gt_by_x_range(self, gt_for_class, range_bins=None):
        """统计经过当前 eval 过滤后的各距离段 GT 数量。"""
        range_bins = range_bins or self.N7_X_RANGE_BINS
        counts = {
            self._range_key(start, end): 0
            for start, end in range_bins
        }
        for boxes in gt_for_class.values():
            boxes = np.asarray(boxes)
            if boxes.size == 0:
                continue
            gt_x = boxes[:, 0]
            for start, end in range_bins:
                key = self._range_key(start, end)
                counts[key] += int(np.count_nonzero(
                    (gt_x >= float(start)) & (gt_x < float(end))))
        return counts

    def _summarize_tp_errors(self, matches, range_bins=None):
        """基于 center distance TP 匹配计算 nuScenes-like 和业务分桶误差。

        *_abs_mean 是逐目标先取绝对值再求均值，即各轴 MAE，表示下游
        更关心的实际偏移量；*_mean 保留有符号均值，只用于诊断系统偏置。
        """
        range_bins = range_bins or self.N7_X_RANGE_BINS
        stats = {
            'tp_num': len(matches),
            'ate': 0.0,
            'aoe': 0.0,
            'aoe_deg': 0.0,
            'ase': 0.0,
            'x_abs_mean': 0.0,
            'x_abs_p50': 0.0,
            'x_abs_p90': 0.0,
            'y_abs_mean': 0.0,
            'y_abs_p50': 0.0,
            'y_abs_p90': 0.0,
            'z_abs_mean': 0.0,
            'z_abs_p50': 0.0,
            'z_abs_p90': 0.0,
            'x_mean': 0.0,
            'y_mean': 0.0,
            'z_mean': 0.0,
            'ranges': {},
        }
        for start, end in range_bins:
            key = self._range_key(start, end)
            stats['ranges'][key] = dict(
                label=self._range_label(start, end),
                tp_num=0,
                x_abs_mean=0.0,
                x_abs_p50=0.0,
                x_abs_p90=0.0,
                y_abs_mean=0.0,
                y_abs_p50=0.0,
                y_abs_p90=0.0,
                z_abs_mean=0.0,
                z_abs_p50=0.0,
                z_abs_p90=0.0,
                x_mean=0.0,
                y_mean=0.0,
                z_mean=0.0)
        if not matches:
            return stats

        center_errors = []
        yaw_errors = []
        scale_errors = []
        xyz_errors = []
        per_range_errors = {key: [] for key in stats['ranges']}
        for match in matches:
            pred_box = match['pred_box']
            gt_box = match['gt_box']
            xyz_error = np.asarray(pred_box[:3] - gt_box[:3], dtype=np.float32)
            xyz_errors.append(xyz_error)
            center_errors.append(float(np.linalg.norm(xyz_error[:2])))
            yaw_errors.append(self._yaw_error(pred_box[6], gt_box[6]))
            scale_errors.append(self._scale_error(pred_box, gt_box))

            gt_x = float(gt_box[0])
            for start, end in range_bins:
                if start <= gt_x < end:
                    per_range_errors[self._range_key(start, end)].append(
                        xyz_error)
                    break

        xyz_errors = np.stack(xyz_errors, axis=0)
        abs_xyz_errors = np.abs(xyz_errors)
        stats.update(
            ate=float(np.mean(center_errors)),
            aoe=float(np.mean(yaw_errors)),
            aoe_deg=float(np.degrees(np.mean(yaw_errors))),
            ase=float(np.mean(scale_errors)),
            x_abs_mean=float(np.mean(abs_xyz_errors[:, 0])),
            x_abs_p50=self._percentile_or_zero(abs_xyz_errors[:, 0], 50),
            x_abs_p90=self._percentile_or_zero(abs_xyz_errors[:, 0], 90),
            y_abs_mean=float(np.mean(abs_xyz_errors[:, 1])),
            y_abs_p50=self._percentile_or_zero(abs_xyz_errors[:, 1], 50),
            y_abs_p90=self._percentile_or_zero(abs_xyz_errors[:, 1], 90),
            z_abs_mean=float(np.mean(abs_xyz_errors[:, 2])),
            z_abs_p50=self._percentile_or_zero(abs_xyz_errors[:, 2], 50),
            z_abs_p90=self._percentile_or_zero(abs_xyz_errors[:, 2], 90),
            x_mean=float(np.mean(xyz_errors[:, 0])),
            y_mean=float(np.mean(xyz_errors[:, 1])),
            z_mean=float(np.mean(xyz_errors[:, 2])))

        for key, values in per_range_errors.items():
            if not values:
                continue
            errors = np.stack(values, axis=0)
            abs_errors = np.abs(errors)
            stats['ranges'][key].update(
                tp_num=int(errors.shape[0]),
                x_abs_mean=float(np.mean(abs_errors[:, 0])),
                x_abs_p50=self._percentile_or_zero(abs_errors[:, 0], 50),
                x_abs_p90=self._percentile_or_zero(abs_errors[:, 0], 90),
                y_abs_mean=float(np.mean(abs_errors[:, 1])),
                y_abs_p50=self._percentile_or_zero(abs_errors[:, 1], 50),
                y_abs_p90=self._percentile_or_zero(abs_errors[:, 1], 90),
                z_abs_mean=float(np.mean(abs_errors[:, 2])),
                z_abs_p50=self._percentile_or_zero(abs_errors[:, 2], 50),
                z_abs_p90=self._percentile_or_zero(abs_errors[:, 2], 90),
                x_mean=float(np.mean(errors[:, 0])),
                y_mean=float(np.mean(errors[:, 1])),
                z_mean=float(np.mean(errors[:, 2])))
        return stats

    def _eval_single_class(self, gt_for_class, pred_for_class, thresholds, mode, collect_matches=False):
        """对单个类别计算多个阈值下的 AP/recall。

        大规模 N7 full-val 会产生百万级预测框。这里保持原有贪心匹配
        口径，但把多个阈值合并成一次按 score 的扫描，并复用同一个预测框
        与同帧 GT 的几何计算，避免每个阈值重复完整遍历。
        """
        thresholds = [float(thr) for thr in thresholds]
        pred_for_class = sorted(pred_for_class, key=lambda x: -x[0])
        gt_for_class = {
            sample_id: boxes
            for sample_id, boxes in gt_for_class.items()
            if len(boxes) > 0
        }
        npos = int(sum(len(v) for v in gt_for_class.values()))
        results = {}
        if npos == 0:
            for thr in thresholds:
                results[thr] = dict(ap=0.0, recall=0.0, precision=0.0, gt_num=0, det_num=len(pred_for_class))
            return results

        if not thresholds:
            return results

        if mode not in ('bev_iou', 'center_distance'):
            raise ValueError('Unsupported eval mode {}'.format(mode))

        thr_num = len(thresholds)
        det_num = len(pred_for_class)
        matched_by_thr = [
            {sample_id: np.zeros((len(boxes),), dtype=bool)
             for sample_id, boxes in gt_for_class.items()}
            for _ in thresholds
        ]
        tp_by_thr = np.zeros((thr_num, det_num), dtype=bool)
        matches_by_thr = [[] for _ in thresholds] if collect_matches else None
        bev_cache_by_sample = {}
        if mode == 'bev_iou':
            bev_cache_by_sample = {
                sample_id: self._prepare_bev_eval_cache(boxes)
                for sample_id, boxes in gt_for_class.items()
            }

        for det_idx, (score, sample_id, pred_box) in enumerate(pred_for_class):
            gt_boxes = gt_for_class.get(sample_id)
            if gt_boxes is None:
                continue

            if mode == 'bev_iou':
                match_values = self._bev_iou_values_from_cache(
                    pred_box, bev_cache_by_sample[sample_id])
            else:
                match_values = np.linalg.norm(
                    gt_boxes[:, :2] - pred_box[:2], axis=1)

            for thr_idx, thr in enumerate(thresholds):
                matched = matched_by_thr[thr_idx][sample_id]
                valid = ~matched
                if not np.any(valid):
                    continue

                valid_indices = np.flatnonzero(valid)
                if mode == 'bev_iou':
                    valid_values = match_values[valid_indices]
                    best_idx = int(valid_indices[np.argmax(valid_values)])
                    is_match = match_values[best_idx] >= thr
                else:
                    valid_values = match_values[valid_indices]
                    best_idx = int(valid_indices[np.argmin(valid_values)])
                    is_match = match_values[best_idx] <= thr

                if is_match:
                    tp_by_thr[thr_idx, det_idx] = True
                    matched[best_idx] = True
                    if collect_matches:
                        matches_by_thr[thr_idx].append(dict(
                            score=float(score),
                            sample_id=sample_id,
                            pred_box=pred_box.astype(np.float32),
                            gt_box=gt_boxes[best_idx].astype(np.float32),
                            match_value=float(match_values[best_idx])))

        for thr_idx, thr in enumerate(thresholds):
            tp = tp_by_thr[thr_idx].astype(np.float32)
            fp = (~tp_by_thr[thr_idx]).astype(np.float32)
            fp_cum = np.cumsum(fp)
            tp_cum = np.cumsum(tp)
            recalls = tp_cum / max(float(npos), 1.0)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float32).eps)
            ap = self._average_precision(recalls, precisions)
            results[thr] = dict(
                ap=ap,
                recall=float(recalls[-1]) if recalls.size else 0.0,
                precision=float(precisions[-1]) if precisions.size else 0.0,
                gt_num=npos,
                det_num=len(pred_for_class))
            if collect_matches:
                results[thr]['matches'] = matches_by_thr[thr_idx]
        return results

    @staticmethod
    def _format_thr(thr):
        """把阈值转成适合日志 key 的短字符串。"""
        return ('{:.2f}'.format(float(thr))).rstrip('0').rstrip('.')

    def _log_custom_eval(self, class_rows, range_rows, overall_metrics=None,
                         logger=None):
        """打印 nuScenes-like 摘要和业务分桶误差，便于训练日志查看。"""
        if not class_rows:
            return
        header = [
            'class', 'GT', 'det', 'center_AP', 'Recall@2m',
            'ATE@2m', 'AOE@2m(deg)', 'ASE@2m', 'BEV_AP@0.5'
        ]
        table_rows = [header]
        for row in class_rows:
            metrics = row['metrics']
            table_rows.append([
                row['class_name'],
                row['gt_num'],
                row['det_num'],
                '{:.4f}'.format(metrics.get('center_AP', 0.0)),
                '{:.4f}'.format(metrics.get('dist_recall@2m', 0.0)),
                '{:.3f}'.format(metrics.get('ATE@2m', 0.0)),
                '{:.2f}'.format(metrics.get('AOE_deg@2m', 0.0)),
                '{:.3f}'.format(metrics.get('ASE@2m', 0.0)),
                '{:.4f}'.format(metrics.get('bev_AP@0.5', 0.0)),
            ])
        print_log('\nN7 nuScenes-like eval summary\n' + AsciiTable(table_rows).table, logger=logger)

        if overall_metrics is not None:
            overall_rows = [[
                'mAP(center)', 'BEV_mAP@0.5', 'mATE@2m',
                'mAOE@2m(deg)', 'mASE@2m', 'GT', 'det'
            ], [
                '{:.4f}'.format(overall_metrics.get('mAP', 0.0)),
                '{:.4f}'.format(
                    overall_metrics.get('mAP/bev_iou@0.5', 0.0)),
                '{:.3f}'.format(overall_metrics.get('mATE@2m', 0.0)),
                '{:.2f}'.format(
                    overall_metrics.get('mAOE_deg@2m', 0.0)),
                '{:.3f}'.format(overall_metrics.get('mASE@2m', 0.0)),
                overall_metrics.get('eval/gt_num', 0),
                overall_metrics.get('eval/det_num', 0),
            ]]
            print_log(
                '\nN7 eval overall\n' + AsciiTable(overall_rows).table,
                logger=logger)

        if range_rows:
            detail_header = [
                'class', 'range', 'GT', 'TP@2m', 'Recall@2m',
                'x_MAE', 'y_MAE', 'z_MAE'
            ]
            detail_rows = [detail_header]
            for row in range_rows:
                has_tp = row['tp_num'] > 0
                detail_rows.append([
                    row['class_name'],
                    row['range_label'],
                    row['gt_num'],
                    row['tp_num'],
                    '{:.4f}'.format(row['recall']),
                    '{:.3f}'.format(row['x_abs_mean']) if has_tp else '-',
                    '{:.3f}'.format(row['y_abs_mean']) if has_tp else '-',
                    '{:.3f}'.format(row['z_abs_mean']) if has_tp else '-',
                ])
            print_log(
                '\nN7 recall and TP xyz MAE by gt x range; TP@2m uses '
                'unique center-distance matches <= 2m\n' +
                AsciiTable(detail_rows).table,
                logger=logger)

    def _evaluate_n7_custom(self, results, metric=None, logger=None):
        """N7 自采数据轻量评估：BEV IoU AP + 中心距离 AP。"""
        iou_thrs = self._normalize_thresholds(self.eval_iou_thr if metric is None else metric)
        distance_thrs = self._normalize_thresholds(self.eval_distance_thr)
        gt_by_class, pred_by_class = self._collect_eval_items(results)

        ret_dict = {
            'eval/metric_schema_version': self.N7_EVAL_METRIC_SCHEMA_VERSION,
            'eval/tp_match_distance_m': self.N7_TP_MATCH_DISTANCE,
        }
        class_rows = []
        bev_ap_by_thr = {thr: [] for thr in iou_thrs}
        dist_ap_by_thr = {thr: [] for thr in distance_thrs}
        center_map_by_class = []
        ate_by_class = []
        aoe_by_class = []
        ase_by_class = []
        total_gt = 0
        total_det = 0
        range_rows = []

        for class_id, class_name in enumerate(self.CLASSES):
            gt_for_class = gt_by_class[class_id]
            pred_for_class = pred_by_class[class_id]
            gt_num = int(sum(len(v) for v in gt_for_class.values()))
            det_num = len(pred_for_class)
            if gt_num == 0 and det_num == 0:
                continue

            bev_results = self._eval_single_class(gt_for_class, pred_for_class, iou_thrs, 'bev_iou')
            dist_eval_thrs = list(distance_thrs)
            if self.N7_TP_MATCH_DISTANCE not in dist_eval_thrs:
                dist_eval_thrs.append(self.N7_TP_MATCH_DISTANCE)
            dist_results = self._eval_single_class(
                gt_for_class, pred_for_class, dist_eval_thrs, 'center_distance',
                collect_matches=True)
            total_gt += gt_num
            total_det += det_num

            row_metrics = {}
            for thr in iou_thrs:
                thr_key = self._format_thr(thr)
                ap = bev_results[thr]['ap']
                recall = bev_results[thr]['recall']
                row_metrics['bev_AP@{}'.format(thr_key)] = ap
                ret_dict['{}/bev_AP@{}'.format(class_name, thr_key)] = ap
                ret_dict['{}/bev_recall@{}'.format(class_name, thr_key)] = recall
                if gt_num > 0:
                    bev_ap_by_thr[thr].append(ap)
            for thr in distance_thrs:
                thr_key = self._format_thr(thr)
                ap = dist_results[thr]['ap']
                recall = dist_results[thr]['recall']
                row_metrics['dist_AP@{}m'.format(thr_key)] = ap
                row_metrics['dist_recall@{}m'.format(thr_key)] = recall
                ret_dict['{}/dist_AP@{}m'.format(class_name, thr_key)] = ap
                ret_dict['{}/dist_recall@{}m'.format(class_name, thr_key)] = recall
                if gt_num > 0:
                    dist_ap_by_thr[thr].append(ap)
            class_center_map_values = [
                dist_results[thr]['ap'] for thr in distance_thrs
            ]
            center_dist_map = float(np.mean(class_center_map_values)) if class_center_map_values else 0.0
            row_metrics['center_AP'] = center_dist_map
            row_metrics['center_dist_mAP'] = center_dist_map
            ret_dict['{}/center_AP'.format(class_name)] = center_dist_map
            # 兼容历史结果；新报表和终端统一使用 center_AP。
            ret_dict['{}/center_dist_mAP'.format(class_name)] = center_dist_map
            if gt_num > 0:
                center_map_by_class.append(center_dist_map)

            tp_metric_thr = self.N7_TP_MATCH_DISTANCE
            tp_stats = self._summarize_tp_errors(
                dist_results[tp_metric_thr].get('matches', [])
                if tp_metric_thr is not None else [])
            tp_recall = dist_results[tp_metric_thr]['recall']
            row_metrics['dist_recall@2m'] = tp_recall
            row_metrics['ATE@2m'] = tp_stats['ate']
            row_metrics['AOE@2m'] = tp_stats['aoe']
            row_metrics['AOE_deg@2m'] = tp_stats['aoe_deg']
            row_metrics['ASE@2m'] = tp_stats['ase']
            ret_dict['{}/TP_num@2m'.format(class_name)] = int(tp_stats['tp_num'])
            ret_dict['{}/dist_recall@2m'.format(class_name)] = tp_recall
            ret_dict['{}/ATE@2m'.format(class_name)] = tp_stats['ate']
            ret_dict['{}/AOE@2m'.format(class_name)] = tp_stats['aoe']
            ret_dict['{}/AOE_deg@2m'.format(class_name)] = tp_stats['aoe_deg']
            ret_dict['{}/ASE@2m'.format(class_name)] = tp_stats['ase']
            for axis in ['x', 'y', 'z']:
                ret_dict['{}/{}_abs_error_mean@2m'.format(class_name, axis)] = tp_stats['{}_abs_mean'.format(axis)]
                ret_dict['{}/{}_abs_error_p50@2m'.format(class_name, axis)] = tp_stats['{}_abs_p50'.format(axis)]
                ret_dict['{}/{}_abs_error_p90@2m'.format(class_name, axis)] = tp_stats['{}_abs_p90'.format(axis)]
                ret_dict['{}/{}_error_mean@2m'.format(class_name, axis)] = tp_stats['{}_mean'.format(axis)]
            if gt_num > 0:
                ate_by_class.append(tp_stats['ate'])
                aoe_by_class.append(tp_stats['aoe'])
                ase_by_class.append(tp_stats['ase'])
            range_gt_counts = self._count_gt_by_x_range(gt_for_class)
            for range_key, range_stats in tp_stats['ranges'].items():
                range_gt_num = int(range_gt_counts.get(range_key, 0))
                range_recall = (
                    float(range_stats['tp_num']) / float(range_gt_num)
                    if range_gt_num > 0 else 0.0)
                for axis in ['x', 'y', 'z']:
                    ret_dict['{}/range_{}/{}_abs_mean'.format(class_name, range_key, axis)] = range_stats['{}_abs_mean'.format(axis)]
                    ret_dict['{}/range_{}/{}_abs_p50'.format(class_name, range_key, axis)] = range_stats['{}_abs_p50'.format(axis)]
                    ret_dict['{}/range_{}/{}_abs_p90'.format(class_name, range_key, axis)] = range_stats['{}_abs_p90'.format(axis)]
                    ret_dict['{}/range_{}/{}_mean'.format(class_name, range_key, axis)] = range_stats['{}_mean'.format(axis)]
                ret_dict['{}/range_{}/tp_num'.format(class_name, range_key)] = int(range_stats['tp_num'])
                ret_dict['{}/range_{}/gt_num'.format(
                    class_name, range_key)] = range_gt_num
                ret_dict['{}/range_{}/recall@2m'.format(
                    class_name, range_key)] = range_recall
                range_rows.append(dict(
                    class_name=class_name,
                    range_label=range_stats['label'],
                    gt_num=range_gt_num,
                    tp_num=int(range_stats['tp_num']),
                    recall=range_recall,
                    x_abs_mean=range_stats['x_abs_mean'],
                    x_abs_p50=range_stats['x_abs_p50'],
                    x_abs_p90=range_stats['x_abs_p90'],
                    y_abs_mean=range_stats['y_abs_mean'],
                    y_abs_p50=range_stats['y_abs_p50'],
                    y_abs_p90=range_stats['y_abs_p90'],
                    z_abs_mean=range_stats['z_abs_mean'],
                    z_abs_p50=range_stats['z_abs_p50'],
                    z_abs_p90=range_stats['z_abs_p90'],
                    x_mean=range_stats['x_mean'],
                    y_mean=range_stats['y_mean'],
                    z_mean=range_stats['z_mean']))
            ret_dict['{}/gt_num'.format(class_name)] = gt_num
            ret_dict['{}/det_num'.format(class_name)] = det_num

            class_rows.append(dict(
                class_name=class_name,
                gt_num=gt_num,
                det_num=det_num,
                metrics=row_metrics))

        for thr in iou_thrs:
            key = 'mAP/bev_iou@{}'.format(self._format_thr(thr))
            values = bev_ap_by_thr[thr]
            ret_dict[key] = float(np.mean(values)) if values else 0.0
        for thr in distance_thrs:
            key = 'mAP/center_dist@{}m'.format(self._format_thr(thr))
            values = dist_ap_by_thr[thr]
            ret_dict[key] = float(np.mean(values)) if values else 0.0
        ret_dict['mAP'] = float(np.mean(center_map_by_class)) if center_map_by_class else 0.0
        # 兼容旧脚本/历史 JSON；canonical mAP 从本版本起定义为 center mAP。
        ret_dict['mAP/center_dist'] = ret_dict['mAP']
        ret_dict['mATE@2m'] = float(np.mean(ate_by_class)) if ate_by_class else 0.0
        ret_dict['mAOE@2m'] = float(np.mean(aoe_by_class)) if aoe_by_class else 0.0
        ret_dict['mAOE_deg@2m'] = float(np.degrees(ret_dict['mAOE@2m']))
        ret_dict['mASE@2m'] = float(np.mean(ase_by_class)) if ase_by_class else 0.0
        ret_dict['eval/gt_num'] = int(total_gt)
        ret_dict['eval/det_num'] = int(total_det)
        self._log_custom_eval(
            class_rows, range_rows, overall_metrics=ret_dict, logger=logger)
        return ret_dict

    def _evaluate_indoor_legacy(self, results, metric=None, logger=None):
        """保留旧 indoor_eval 入口，便于排查或兼容历史实验。"""
        if metric is None:
            metric = self.eval_iou_thr
        if isinstance(metric, float):
            metric = [metric]

        dt_annos = []
        for result in results:
            if 'pts_bbox' in result:
                result = result['pts_bbox']
            dt_annos.append(result)

        gt_annos = []
        for info in self.data_infos[:len(dt_annos)]:
            gt_boxes = np.asarray(
                info.get('gt_boxes', np.zeros((0, 7))), dtype=np.float32)
            labels = []
            for cat in info.get('gt_names', []):
                labels.append(self.CLASSES.index(cat)
                              if cat in self.CLASSES else -1)
            labels = np.asarray(labels, dtype=np.int64)
            valid = labels >= 0
            gt_annos.append({
                'gt_num': int(valid.sum()),
                'gt_boxes_upright_depth': gt_boxes[valid],
                'class': labels[valid],
            })

        label2cat = {i: name for i, name in enumerate(self.CLASSES)}
        return indoor_eval(
            gt_annos,
            dt_annos,
            metric,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 **kwargs):
        """评估 CustomMultiViewDataset 预测结果。

        默认使用不依赖 nuScenes devkit 的 N7 自定义轻量评估：
        - BEV 旋转框 IoU AP，用于观察框尺寸/朝向/位置综合质量；
        - 中心点距离 AP，用于观察目标中心定位能力，接近 nuScenes 的距离匹配思想。

        如果确实需要旧逻辑，可以在 config 的 evaluation 中传 ``metric='indoor'``。
        """
        if isinstance(metric, str) and metric == 'indoor':
            ret_dict = self._evaluate_indoor_legacy(results, metric=self.eval_iou_thr, logger=logger)
        else:
            ret_dict = self._evaluate_n7_custom(results, metric=metric, logger=logger)

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return ret_dict
