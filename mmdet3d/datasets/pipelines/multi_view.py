import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose, RandomFlip, LoadImageFromFile
import ipdb


@PIPELINES.register_module()
class MultiViewPipeline:
    def __init__(self, transforms, n_images, n_times=2, sequential=False):
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.n_times = n_times
        self.sequential = sequential

    def __sort_list(self, old_list, order):
        new_list = []
        for i in order:
            new_list.append(old_list[i])
        return new_list

    def __call__(self, results):
        imgs = []
        extrinsics = []
        if not self.sequential:
            assert len(results['img_info']) >= self.n_images
            ids = np.arange(len(results['img_info']))
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
            ids_list = sorted(ids)  # sort & tolist
        else:
            expected = self.n_images * self.n_times
            assert len(results['img_info']) == expected, \
                f'img info: {len(results["img_info"])}, expected: {expected}, n_times: {self.n_times}, n_images: {self.n_images}'
            ids_list = np.arange(len(results['img_info'])).tolist()
        for i in ids_list:
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs
        # 记录实际输出的视图组织，供后续几何增强校验相机/时序契约。
        # 非时序模式只输出一个时刻，即使构造参数保留了历史默认 n_times。
        results['view_layout'] = dict(
            n_images=int(self.n_images),
            n_times=int(self.n_times if self.sequential else 1),
            sequential=bool(self.sequential))
        # resort 2d box by random ids
        if 'gt_bboxes' in results.keys():
            gt_bboxes = self.__sort_list(results['gt_bboxes'], ids_list)
            gt_labels = self.__sort_list(results['gt_labels'], ids_list)
            gt_bboxes_ignore = self.__sort_list(results['gt_bboxes_ignore'], ids_list)
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels
            results['gt_bboxes_ignore'] = gt_bboxes_ignore

        results['lidar2img']['extrinsic'] = extrinsics
        return results


@PIPELINES.register_module()
class RandomShiftOrigin:
    def __init__(self, std):
        self.std = std

    def __call__(self, results):
        shift = np.random.normal(.0, self.std, 3)
        results['lidar2img']['origin'] += shift
        return results


@PIPELINES.register_module()
class KittiSetOrigin:
    def __init__(self, point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

    def __call__(self, results):
        results['lidar2img']['origin'] = self.origin.copy()
        return results


@PIPELINES.register_module()
class FrontCameraVisibleObjectFilter:
    """按相机可见性过滤 3D GT。"""

    def __init__(self, n_images=1, min_depth=0.1, keep_if_no_boxes=True):
        self.n_images = n_images
        self.min_depth = min_depth
        self.keep_if_no_boxes = keep_if_no_boxes

    def _project_visible(self, points, projection, image_shape):
        pts = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        pts_2d_3 = pts @ projection.T
        depth = pts_2d_3[:, 2]
        eps = np.finfo(np.float32).eps
        xs = pts_2d_3[:, 0] / np.maximum(depth, eps)
        ys = pts_2d_3[:, 1] / np.maximum(depth, eps)
        height, width = image_shape[:2]
        return ((depth > self.min_depth) & (xs >= 0) & (xs < width) &
                (ys >= 0) & (ys < height))

    def __call__(self, results):
        if 'gt_bboxes_3d' not in results or 'gt_labels_3d' not in results:
            return results

        gt_bboxes_3d = results['gt_bboxes_3d']
        num_boxes = len(gt_bboxes_3d)
        if num_boxes == 0:
            return results

        projections = results['lidar2img']['extrinsic'][:self.n_images]
        image_shapes = results.get('img_shape', [])[:self.n_images]
        if len(projections) == 0 or len(image_shapes) == 0:
            if self.keep_if_no_boxes:
                return results
            keep = np.zeros(num_boxes, dtype=np.bool_)
        else:
            centers = gt_bboxes_3d.tensor[:, :3].detach().cpu().numpy()
            corners = gt_bboxes_3d.corners.detach().cpu().numpy()
            keep = np.zeros(num_boxes, dtype=np.bool_)
            for projection, image_shape in zip(projections, image_shapes):
                projection = np.asarray(projection[:3, :4], dtype=np.float32)
                center_visible = self._project_visible(
                    centers, projection, image_shape)
                corner_points = corners.reshape(-1, 3)
                corner_visible = self._project_visible(
                    corner_points, projection, image_shape).reshape(
                        num_boxes, -1).any(axis=1)
                keep |= center_visible | corner_visible

        mask = gt_bboxes_3d.tensor.new_tensor(
            keep, dtype=gt_bboxes_3d.tensor.dtype).bool()
        results['gt_bboxes_3d'] = gt_bboxes_3d[mask]
        results['gt_labels_3d'] = results['gt_labels_3d'][keep]
        if 'gt_bboxes' in results:
            results['gt_bboxes'] = [
                b for b, k in zip(results['gt_bboxes'], keep) if k]
        if 'gt_labels' in results:
            results['gt_labels'] = [
                l for l, k in zip(results['gt_labels'], keep) if k]
        if 'gt_bboxes_ignore' in results:
            results['gt_bboxes_ignore'] = [
                b for b, k in zip(results['gt_bboxes_ignore'], keep) if k]
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(n_images={self.n_images}, '
                f'min_depth={self.min_depth})')


@PIPELINES.register_module()
class KittiRandomFlip:
    def __call__(self, results):
        if results['flip']:
            results['lidar2img']['intrinsic'][0, 2] = -results['lidar2img']['intrinsic'][0, 2] + \
                                                      results['ori_shape'][1]
            flip_matrix_0 = np.eye(4, dtype=np.float32)
            flip_matrix_0[0, 0] *= -1
            flip_matrix_1 = np.eye(4, dtype=np.float32)
            flip_matrix_1[1, 1] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0]
            extrinsic = flip_matrix_0 @ extrinsic @ flip_matrix_1.T
            results['lidar2img']['extrinsic'][0] = extrinsic
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 0], -center[:, 1]) - alpha
            center_flip = center
            center_flip[:, 1] *= -1
            alpha_flip = np.arctan2(center_flip[:, 0], -center_flip[:, 1]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results


@PIPELINES.register_module()
class SunRgbdSetOrigin:
    def __call__(self, results):
        intrinsic = results['lidar2img']['intrinsic'][:3, :3]
        extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
        projection = intrinsic @ extrinsic
        h, w, _ = results['ori_shape']
        center_2d_3 = np.array([w / 2, h / 2, 1], dtype=np.float32)
        center_2d_3 *= 3
        origin = np.linalg.inv(projection) @ center_2d_3
        results['lidar2img']['origin'] = origin
        return results


@PIPELINES.register_module()
class SunRgbdTotalLoadImageFromFile(LoadImageFromFile):
    def __call__(self, results):
        file_name = results['img_info']['filename']
        flip = file_name.endswith('_flip.jpg')
        if flip:
            results['img_info']['filename'] = file_name.replace('_flip.jpg', '.jpg')
        results = super().__call__(results)
        if flip:
            results['img'] = results['img'][:, ::-1]
        return results


@PIPELINES.register_module()
class SunRgbdRandomFlip:
    def __call__(self, results):
        if results['flip']:
            flip_matrix = np.eye(3)
            flip_matrix[0, 0] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
            results['lidar2img']['extrinsic'][0][:3, :3] = flip_matrix @ extrinsic @ flip_matrix.T
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 1], center[:, 0]) - alpha
            center_flip = center @ flip_matrix
            alpha_flip = np.arctan2(center_flip[:, 1], center_flip[:, 0]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results
