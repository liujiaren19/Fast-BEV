import os
import os.path as osp
import random

import mmcv
import numpy as np
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
                 max_interval=3,
                 min_interval=0,
                 shuffle=False,
                 seed=0,
                 eval_iou_thr=(0.25, 0.5)):
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.camera_types = camera_types
        self.sequential = sequential
        self.n_times = n_times
        self.prev_only = prev_only
        self.train_adj_ids = train_adj_ids
        self.test_adj_ids = test_adj_ids
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.shuffle = shuffle
        self.seed = seed
        self.eval_iou_thr = eval_iou_thr

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
    def _lidar2img_from_cam_info(cam_info):
        intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
        sensor2lidar_r = np.asarray(
            cam_info['sensor2lidar_rotation'], dtype=np.float32)
        sensor2lidar_t = np.asarray(
            cam_info['sensor2lidar_translation'], dtype=np.float32)

        lidar2cam_r = np.linalg.inv(sensor2lidar_r)
        lidar2cam_t = sensor2lidar_t @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4, dtype=np.float32)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img = viewpad @ lidar2cam_rt.T

        lidar2img_aug = dict(
            intrin=intrinsic,
            rot=sensor2lidar_r,
            tran=sensor2lidar_t,
            post_rot=np.eye(3, dtype=np.float32),
            post_tran=np.zeros(3, dtype=np.float32),
        )
        return lidar2img.astype(np.float32), lidar2img_aug

    def _select_adjacent(self, info, time_id):
        adj_ids = self.test_adj_ids if self.test_mode else self.train_adj_ids
        if adj_ids is not None:
            select_id = min(abs(adj_ids[time_id - 1]),
                            len(info.get('prev', [])) - 1)
        else:
            select_id = min(self.min_interval + time_id - 1,
                            len(info.get('prev', [])) - 1)
        return info['prev'][max(select_id, 0)]

    def _collect_one_frame(self, info):
        image_paths, lidar2img_rts, lidar2img_augs = [], [], []
        cam_items = info['cams'].items()
        if self.camera_types is not None:
            cam_items = [(cam, info['cams'][cam]) for cam in self.camera_types]

        for _, cam_info in cam_items:
            image_paths.append(self._resolve_path(cam_info['data_path']))
            lidar2img_rt, lidar2img_aug = self._lidar2img_from_cam_info(cam_info)
            lidar2img_rts.append(lidar2img_rt)
            lidar2img_augs.append(lidar2img_aug)
        return image_paths, lidar2img_rts, lidar2img_augs

    def get_data_info(self, index):
        info = self.data_infos[index]
        image_paths, lidar2img_rts, lidar2img_augs = self._collect_one_frame(info)

        if self.sequential:
            for time_id in range(1, self.n_times):
                if not info.get('prev'):
                    adj_info = info
                else:
                    adj_info = self._select_adjacent(info, time_id)
                adj_paths, adj_rts, adj_augs = self._collect_one_frame(adj_info)
                image_paths.extend(adj_paths)
                lidar2img_rts.extend(adj_rts)
                lidar2img_augs.extend(adj_augs)

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
                lidar2img_extra=[],
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

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 **kwargs):
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
        ret_dict = indoor_eval(
            gt_annos,
            dt_annos,
            metric,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return ret_dict
