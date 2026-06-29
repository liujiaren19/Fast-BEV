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
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class CustomMultiViewDataset(Custom3DDataset):
    """Multi-view camera dataset for converted in-house Fast-BEV infos."""

    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

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
                 eval_score_thr=0.0):
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
                                 temporal_compensated=False):
        intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
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

    def _select_adjacent(self, info, time_id):
        adj_ids = self.test_adj_ids if self.test_mode else self.train_adj_ids
        if adj_ids is not None:
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
        gt_names_3d = np.asarray(info.get('gt_names', []))

        gt_labels_3d = []
        for cat in gt_names_3d:
            gt_labels_3d.append(self.CLASSES.index(cat)
                                if cat in self.CLASSES else -1)
        gt_labels_3d = np.asarray(gt_labels_3d, dtype=np.int64)

        if self.with_velocity:
            gt_velocity = np.asarray(
                info.get('gt_velocity',
                         np.zeros((gt_bboxes_3d.shape[0], 2))),
                dtype=np.float32)
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
    def _boxes_to_numpy(cls, boxes):
        """从 mmdet3d 预测框对象或 ndarray 中取出前 7 维 box。"""
        if boxes is None:
            return np.zeros((0, 7), dtype=np.float32)
        if hasattr(boxes, 'tensor'):
            boxes = boxes.tensor
        boxes = cls._to_numpy(boxes).astype(np.float32)
        if boxes.size == 0:
            return np.zeros((0, 7), dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if boxes.shape[1] < 7:
            raise ValueError('3D boxes must have at least 7 dims, got {}'.format(boxes.shape))
        return boxes[:, :7]

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
        rot = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
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
        boxes = self._boxes_to_numpy(result.get('boxes_3d'))
        scores = self._to_numpy(result.get('scores_3d')).astype(np.float32).reshape(-1)
        labels = self._to_numpy(result.get('labels_3d')).astype(np.int64).reshape(-1)
        valid_len = min(boxes.shape[0], scores.shape[0], labels.shape[0])
        boxes = boxes[:valid_len]
        scores = scores[:valid_len]
        labels = labels[:valid_len]
        valid = scores >= self.eval_score_thr
        valid &= self._eval_range_mask(boxes)
        return boxes[valid], scores[valid], labels[valid]

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

    def _eval_single_class(self, gt_for_class, pred_for_class, thresholds, mode):
        """对单个类别计算多个阈值下的 AP/recall。"""
        pred_for_class = sorted(pred_for_class, key=lambda x: -x[0])
        npos = int(sum(len(v) for v in gt_for_class.values()))
        results = {}
        if npos == 0:
            for thr in thresholds:
                results[thr] = dict(ap=0.0, recall=0.0, precision=0.0, gt_num=0, det_num=len(pred_for_class))
            return results

        for thr in thresholds:
            matched = {sample_id: np.zeros((len(boxes),), dtype=bool)
                       for sample_id, boxes in gt_for_class.items()}
            tp = np.zeros((len(pred_for_class),), dtype=np.float32)
            fp = np.zeros((len(pred_for_class),), dtype=np.float32)

            for det_idx, (_, sample_id, pred_box) in enumerate(pred_for_class):
                gt_boxes = gt_for_class.get(sample_id, np.zeros((0, 7), dtype=np.float32))
                if len(gt_boxes) == 0:
                    fp[det_idx] = 1.0
                    continue

                best_idx = -1
                if mode == 'bev_iou':
                    best_value = -1.0
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if matched[sample_id][gt_idx]:
                            continue
                        value = self._bev_iou(pred_box, gt_box)
                        if value > best_value:
                            best_value = value
                            best_idx = gt_idx
                    is_match = best_idx >= 0 and best_value >= thr
                elif mode == 'center_distance':
                    best_value = np.inf
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if matched[sample_id][gt_idx]:
                            continue
                        value = float(np.linalg.norm(pred_box[:2] - gt_box[:2]))
                        if value < best_value:
                            best_value = value
                            best_idx = gt_idx
                    is_match = best_idx >= 0 and best_value <= thr
                else:
                    raise ValueError('Unsupported eval mode {}'.format(mode))

                if is_match:
                    tp[det_idx] = 1.0
                    matched[sample_id][best_idx] = True
                else:
                    fp[det_idx] = 1.0

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
        return results

    @staticmethod
    def _format_thr(thr):
        """把阈值转成适合日志 key 的短字符串。"""
        return ('{:.2f}'.format(float(thr))).rstrip('0').rstrip('.')

    def _log_custom_eval(self, class_rows, logger=None):
        """打印自定义评估表，便于直接从训练日志看主要结果。"""
        if not class_rows:
            return
        header = ['class', 'gt', 'det'] + class_rows[0]['metric_names']
        table_rows = [header]
        for row in class_rows:
            table_rows.append([
                row['class_name'], row['gt_num'], row['det_num'],
                *['{:.4f}'.format(row['metrics'][name]) for name in row['metric_names']]
            ])
        print_log('\n' + AsciiTable(table_rows).table, logger=logger)

    def _evaluate_n7_custom(self, results, metric=None, logger=None):
        """N7 自采数据轻量评估：BEV IoU AP + 中心距离 AP。"""
        iou_thrs = self._normalize_thresholds(self.eval_iou_thr if metric is None else metric)
        distance_thrs = self._normalize_thresholds(self.eval_distance_thr)
        gt_by_class, pred_by_class = self._collect_eval_items(results)

        ret_dict = {}
        class_rows = []
        bev_metric_names = ['bev_AP@{}'.format(self._format_thr(x)) for x in iou_thrs]
        dist_metric_names = ['dist_AP@{}m'.format(self._format_thr(x)) for x in distance_thrs]
        metric_names = bev_metric_names + dist_metric_names

        bev_ap_by_thr = {thr: [] for thr in iou_thrs}
        dist_ap_by_thr = {thr: [] for thr in distance_thrs}
        total_gt = 0
        total_det = 0

        for class_id, class_name in enumerate(self.CLASSES):
            gt_for_class = gt_by_class[class_id]
            pred_for_class = pred_by_class[class_id]
            gt_num = int(sum(len(v) for v in gt_for_class.values()))
            det_num = len(pred_for_class)
            if gt_num == 0 and det_num == 0:
                continue

            bev_results = self._eval_single_class(gt_for_class, pred_for_class, iou_thrs, 'bev_iou')
            dist_results = self._eval_single_class(gt_for_class, pred_for_class, distance_thrs, 'center_distance')
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
                ret_dict['{}/dist_AP@{}m'.format(class_name, thr_key)] = ap
                ret_dict['{}/dist_recall@{}m'.format(class_name, thr_key)] = recall
                if gt_num > 0:
                    dist_ap_by_thr[thr].append(ap)
            ret_dict['{}/gt_num'.format(class_name)] = gt_num
            ret_dict['{}/det_num'.format(class_name)] = det_num

            class_rows.append(dict(
                class_name=class_name,
                gt_num=gt_num,
                det_num=det_num,
                metric_names=metric_names,
                metrics={name: row_metrics.get(name, 0.0) for name in metric_names}))

        for thr in iou_thrs:
            key = 'mAP/bev_iou@{}'.format(self._format_thr(thr))
            values = bev_ap_by_thr[thr]
            ret_dict[key] = float(np.mean(values)) if values else 0.0
        for thr in distance_thrs:
            key = 'mAP/center_dist@{}m'.format(self._format_thr(thr))
            values = dist_ap_by_thr[thr]
            ret_dict[key] = float(np.mean(values)) if values else 0.0
        ret_dict['eval/gt_num'] = int(total_gt)
        ret_dict['eval/det_num'] = int(total_det)
        if iou_thrs:
            ret_dict['mAP'] = ret_dict['mAP/bev_iou@{}'.format(self._format_thr(iou_thrs[-1]))]

        self._log_custom_eval(class_rows, logger=logger)
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
