# -*- coding: utf-8 -*-
import math
import os
import shutil
# FASTBEV_PROFILE 计时依赖 time.perf_counter；正常训练不开启该开关时只保留轻量判断。
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16

import copy
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@DETECTORS.register_module()
class FastBEV(BaseDetector):
    def __init__(
        self,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        bbox_head,
        seg_head,
        n_voxels,
        voxel_size,
        bbox_head_2d=None,
        train_cfg=None,
        test_cfg=None,
        train_cfg_2d=None,
        test_cfg_2d=None,
        pretrained=None,
        init_cfg=None,
        extrinsic_noise=0,
        seq_detach=False,
        multi_scale_id=None,
        multi_scale_3d_scaler=None,
        with_cp=False,
        backproject='inplace',
        style='v4',
        n_images=6,
        feature_resize_mode='bilinear',
        use_distortion=False,
        calibration_save_dir=None,
        max_calibration_samples=500,
        onnx_custom_op_path=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        
        # style
        # v1: fastbev wo/ ms
        # v2: fastbev + img ms
        # v3: fastbev + bev ms
        # v4: fastbev + img/bev ms
        self.style = style
        assert self.style in ['v1', 'v2', 'v3', 'v4'], self.style
        self.multi_scale_id = multi_scale_id
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            self.bbox_head.voxel_size = voxel_size
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        if bbox_head_2d is not None:
            bbox_head_2d.update(train_cfg=train_cfg_2d)
            bbox_head_2d.update(test_cfg=test_cfg_2d)
            self.bbox_head_2d = build_head(bbox_head_2d)
        else:
            self.bbox_head_2d = None

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        # detach adj feature
        self.seq_detach = seq_detach
        self.backproject = backproject
        self.n_images = n_images
        self.feature_resize_mode = feature_resize_mode
        assert self.feature_resize_mode in ['bilinear', 'nearest'], self.feature_resize_mode
        self.use_distortion = use_distortion
        self.calibration_save_dir = calibration_save_dir
        self.max_calibration_samples = max_calibration_samples
        self.onnx_custom_op_path = onnx_custom_op_path
        self._calibration_samples_saved = 0
        # profile 计数器只限制前若干次打印，避免长时间训练持续做 CUDA 同步计时。
        self._profile_extract_iter = 0
        self._profile_train_iter = 0
        # BEV 网格点缓存用于复用相同 n_voxels/voxel_size/origin 下的体素中心点。
        self._points_cache = {}
        # checkpoint
        self.with_cp = with_cp

        self.backbone_session = None
        self.head_session = None
        test_mode = (self.test_cfg or {}).get('test_mode', None)
        if test_mode in ['test_onnx', 'test_custom']:
            self.backbone_session = self._init_onnx_session(
                self.test_cfg.get('backbone_onnx'))
            self.head_session = self._init_onnx_session(
                self.test_cfg.get('head_onnx'))

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def _init_onnx_session(self, onnx_path):
        if ort is None:
            raise ImportError('test_onnx/test_custom 需要安装 onnxruntime')
        if not onnx_path:
            raise ValueError('test_onnx/test_custom 需要在 test_cfg 中配置 backbone_onnx/head_onnx')

        ort.set_default_logger_severity(3)
        try:
            return ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        except Exception:
            custom_op_path = (self.test_cfg or {}).get(
                'onnx_custom_op_path', self.onnx_custom_op_path)
            if not custom_op_path:
                raise
            session_options = ort.SessionOptions()
            session_options.register_custom_ops_library(custom_op_path)
            return ort.InferenceSession(
                onnx_path, session_options, providers=['CPUExecutionProvider'])

    @staticmethod
    def _onnx_infer(session, input_data):
        input_names = [item.name for item in session.get_inputs()]
        input_dict = {name: input_data[i] for i, name in enumerate(input_names)}
        return session.run(None, input_dict)

    @staticmethod
    def _slice_view_meta(img_meta, start, end):
        img_meta = copy.deepcopy(img_meta)
        lidar2img = img_meta.get("lidar2img", {})
        for key in ("extrinsic", "lidar2img_aug", "lidar2img_extra"):
            value = lidar2img.get(key)
            if isinstance(value, list):
                lidar2img[key] = value[start:end]
        for key in ("img_shape", "ori_shape", "pad_shape", "img_info", "filename"):
            value = img_meta.get(key)
            if isinstance(value, list) and len(value) >= end:
                img_meta[key] = value[start:end]
        return img_meta

    def _resize_feature(self, feat, size):
        kwargs = dict(size=size, mode=self.feature_resize_mode)
        if self.feature_resize_mode == 'bilinear':
            kwargs['align_corners'] = False
        return resize(feat, **kwargs)

    @staticmethod
    def _cache_key_values(values, ndigits=6):
        # 浮点配置转成可哈希 key 时做少量四舍五入，避免等价配置因浮点尾差缓存失效。
        return tuple(round(float(v), ndigits) for v in values)

    def _get_cached_points(self, n_voxels, voxel_size, origin, device, dtype):
        # BEV 体素中心网格只由 n_voxels、voxel_size、origin 决定；同一配置下所有 batch/时序帧复用。
        n_voxels_key = tuple(int(v) for v in n_voxels)
        voxel_size_key = self._cache_key_values(voxel_size)
        origin_key = self._cache_key_values(origin)
        key = (n_voxels_key, voxel_size_key, origin_key, str(device), str(dtype))
        cached = self._points_cache.get(key)
        if cached is None:
            # 沿用原始 CPU 构造方式以兼容旧版 torch.arange(tensor) 行为，然后只搬到目标设备一次。
            cached = get_points(
                n_voxels=torch.tensor(n_voxels),
                voxel_size=torch.tensor(voxel_size, dtype=dtype),
                origin=torch.tensor(origin, dtype=dtype),
            ).to(device=device, dtype=dtype)
            self._points_cache[key] = cached
        return cached

    @staticmethod
    def _profile_enabled():
        # 通过环境变量临时打开性能剖析，不改 config，便于在线上/调试脚本快速 A/B。
        return os.getenv('FASTBEV_PROFILE', '0').lower() not in ('0', 'false', 'no', '')

    @staticmethod
    def _profile_limit():
        # 默认只打印前 10 次，防止日志过多，也避免 profiling 的同步开销污染完整训练。
        return int(os.getenv('FASTBEV_PROFILE_ITERS', '10'))

    @staticmethod
    def _profile_now():
        # CUDA kernel 默认异步执行；计时前同步一次，才能看清各阶段真实耗时。
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _extract_onnx_backbone(self, img):
        if self.backbone_session is None:
            raise RuntimeError('ONNX backbone session 尚未初始化')
        outputs = []
        for i in range(img.size(0)):
            input_data = img[i].unsqueeze(0).detach().cpu().float().numpy()
            onnx_output = self._onnx_infer(self.backbone_session, [input_data])
            outputs.append(torch.from_numpy(onnx_output[0]).to(img.device))
        return [torch.cat(outputs, dim=0)]

    @staticmethod
    def _volume_to_deploy_input(volume):
        bs, channels, size_x, size_y, size_z = volume.shape
        return volume.permute(0, 4, 1, 2, 3).reshape(
            bs, size_z * channels, size_x, size_y)

    def _onnx_head_forward(self, deploy_head_inputs, device):
        if self.head_session is None:
            raise RuntimeError('ONNX head session 尚未初始化')
        input_data = [x.detach().cpu().float().numpy() for x in deploy_head_inputs]
        outputs = self._onnx_infer(self.head_session, input_data)
        return tuple([torch.from_numpy(out).to(device)] for out in outputs)

    def _save_backbone_calibration(self, img_metas):
        test_cfg = self.test_cfg or {}
        if not test_cfg.get('save_calibrate_data_flag', False):
            return
        data_path = test_cfg.get('backbone_data_path')
        if not data_path or not img_metas:
            return
        os.makedirs(data_path, exist_ok=True)
        for item in img_metas[0].get('img_info', []):
            filename = item.get('filename')
            if filename and os.path.exists(filename):
                shutil.copy2(filename, data_path)

    def _save_head_calibration(self, deploy_head_inputs):
        test_cfg = self.test_cfg or {}
        save_dir = test_cfg.get('head_data_path', self.calibration_save_dir)
        if not test_cfg.get('save_calibrate_data_flag', False) or not save_dir:
            return
        if self._calibration_samples_saved >= self.max_calibration_samples:
            return
        calibrate_id = test_cfg.get('calibrate_data_id', self._calibration_samples_saved)
        for idx, tensor in enumerate(deploy_head_inputs):
            out_dir = os.path.join(save_dir, str(idx))
            os.makedirs(out_dir, exist_ok=True)
            np.save(
                os.path.join(out_dir, f'{calibrate_id}_{idx}.npy'),
                tensor.detach().cpu().float().numpy())
        self._calibration_samples_saved += 1

    def extract_feat(self, img, img_metas, mode):
        # extract_feat 是 FastBEV 训练最重的路径；这里按阶段统计二维特征、投影、点云网格、回投影和拼接耗时。
        profile = self._profile_enabled() and self._profile_extract_iter < self._profile_limit()
        profile_t0 = self._profile_now() if profile else None
        profile_last = profile_t0
        profile_meta = 0.0
        profile_projection = 0.0
        profile_points = 0.0
        profile_backproject = 0.0
        profile_stack = 0.0
        self._save_backbone_calibration(img_metas)
        batch_size = img.shape[0]
        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [bs, views, 3, h, w] -> [bs*views, 3, h, w]

        if mode in ['test_onnx', 'test_custom']:
            mlvl_feats = self._extract_onnx_backbone(img)
            features_2d = None
        else:
            x = self.backbone(img)

            # use for vovnet
            if isinstance(x, dict):
                tmp = []
                for k in x.keys():
                    tmp.append(x[k])
                x = tmp

            def _inner_forward(x):
                out = self.neck(x)
                return out

            if self.with_cp and x.requires_grad:
                mlvl_feats = cp.checkpoint(_inner_forward, x)
            else:
                mlvl_feats = _inner_forward(x)
            mlvl_feats = list(mlvl_feats)

            features_2d = None
            if self.bbox_head_2d:
                features_2d = mlvl_feats

            if self.multi_scale_id is not None:
                mlvl_feats_ = []
                for msid in self.multi_scale_id:
                    if getattr(self, f'neck_fuse_{msid}', None) is not None:
                        fuse_feats = [mlvl_feats[msid]]
                        for i in range(msid + 1, len(mlvl_feats)):
                            resized_feat = self._resize_feature(
                                mlvl_feats[i], size=mlvl_feats[msid].size()[2:])
                            fuse_feats.append(resized_feat)

                        if len(fuse_feats) > 1:
                            fuse_feats = torch.cat(fuse_feats, dim=1)
                        else:
                            fuse_feats = fuse_feats[0]
                        fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                        mlvl_feats_.append(fuse_feats)
                    else:
                        mlvl_feats_.append(mlvl_feats[msid])
                mlvl_feats = mlvl_feats_

        if profile:
            # 二维 backbone、neck 以及多尺度融合整体计为 backbone_neck_fuse。
            profile_backbone = self._profile_now() - profile_last
            profile_last = self._profile_now()
        else:
            profile_backbone = 0.0

        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        mlvl_volumes = []
        deploy_head_inputs = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])
            total_views = mlvl_feat.shape[0] // batch_size
            assert mlvl_feat.shape[0] == batch_size * total_views, (
                f'feature batch/view mismatch: feat_batch={mlvl_feat.shape[0]}, '
                f'batch_size={batch_size}')
            assert total_views % self.n_images == 0, (
                f'total input views ({total_views}) must be divisible by '
                f'n_images ({self.n_images})')
            if img_metas:
                expected_total_views = len(img_metas[0]["lidar2img"]["extrinsic"])
                assert expected_total_views == total_views, (
                    f'img_meta extrinsics ({expected_total_views}) do not match '
                    f'feature views ({total_views})')

            mlvl_feat = mlvl_feat.reshape(
                [batch_size, total_views] + list(mlvl_feat.shape[1:]))
            mlvl_feat_split = torch.split(mlvl_feat, self.n_images, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]
                    if profile:
                        # meta 计时覆盖 deepcopy 和按当前时序帧切分相机内外参等元信息。
                        profile_stage = self._profile_now()
                    start = seq_id * self.n_images
                    end = (seq_id + 1) * self.n_images
                    img_meta = self._slice_view_meta(seq_img_meta, start, end)
                    assert len(img_meta["lidar2img"]["extrinsic"]) == self.n_images, (
                        f'seq_id={seq_id} expected {self.n_images} extrinsics, '
                        f'got {len(img_meta["lidar2img"]["extrinsic"])}')
                    for key in ('lidar2img_aug', 'lidar2img_extra'):
                        value = img_meta["lidar2img"].get(key)
                        if isinstance(value, list):
                            assert len(value) == self.n_images, (
                                f'seq_id={seq_id} expected {self.n_images} {key} entries, '
                                f'got {len(value)}')
                    if isinstance(img_meta["img_shape"], list):
                        assert len(img_meta["img_shape"]) == self.n_images, (
                            f'seq_id={seq_id} expected {self.n_images} image shapes, '
                            f'got {len(img_meta["img_shape"])}')
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)
                    if profile:
                        profile_meta += self._profile_now() - profile_stage
                        profile_stage = self._profile_now()

                    # projection 由当前时序帧的 lidar2img 外参、图像增强矩阵和特征 stride 共同决定。
                    projection = self._compute_projection(
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    if profile:
                        profile_projection += self._profile_now() - profile_stage
                        profile_stage = self._profile_now()
                    if self.style in ['v1', 'v2']:
                        n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                    else:
                        n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                    # 原实现会在每个 batch/时序/尺度循环中重复生成相同 BEV 网格，训练慢时主要卡在这里。
                    points = self._get_cached_points(
                        n_voxels=n_voxels,
                        voxel_size=voxel_size,
                        origin=img_meta["lidar2img"]["origin"],
                        device=feat_i.device,
                        dtype=projection.dtype,
                    )
                    if profile:
                        profile_points += self._profile_now() - profile_stage
                        profile_stage = self._profile_now()

                    if self.backproject == 'inplace':
                        volume = backproject_inplace(
                            feat_i[:, :, :height, :width], points, projection,
                            img_meta=img_meta, seq_id=seq_id, stride=stride_i,
                            use_distortion=self.use_distortion)
                    else:
                        volume, valid = backproject_vanilla(
                            feat_i[:, :, :height, :width], points, projection)
                        volume = volume.sum(dim=0)
                        valid = valid.sum(dim=0)
                        volume = volume / valid
                        valid = valid > 0
                        volume[:, ~valid[0]] = 0.0

                    if profile:
                        profile_backproject += self._profile_now() - profile_stage
                    volumes.append(volume)
                if profile:
                    # stack/cat 单独统计，便于区分真正的 backproject 计算和张量拼接开销。
                    profile_stage = self._profile_now()
                seq_volume = torch.stack(volumes)
                if lvl == 0:
                    deploy_head_inputs.append(self._volume_to_deploy_input(seq_volume))
                volume_list.append(seq_volume)
                if profile:
                    profile_stack += self._profile_now() - profile_stage

            if profile:
                profile_stage = self._profile_now()
            mlvl_volumes.append(torch.cat(volume_list, dim=1))
            if profile:
                profile_stack += self._profile_now() - profile_stage

        if profile:
            profile_volume_total = self._profile_now() - profile_last
            profile_last = self._profile_now()
        else:
            profile_volume_total = 0.0

        if mode in ['test_onnx', 'test_custom']:
            return self._onnx_head_forward(deploy_head_inputs, img.device), None, None

        self._save_head_calibration(deploy_head_inputs)

        if self.style in ['v1', 'v2']:
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)
        else:
            for i in range(len(mlvl_volumes)):
                mlvl_volume = mlvl_volumes[i]
                bs, c, x, y, z = mlvl_volume.shape
                mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(
                    bs, x, y, z * c).permute(0, 3, 1, 2)

                if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                    mlvl_volume = F.adaptive_avg_pool2d(
                        mlvl_volume, mlvl_volumes[-1].size()[2:4])
                elif self.multi_scale_3d_scaler == 'upsample' and i != 0:
                    mlvl_volume = resize(
                        mlvl_volume,
                        mlvl_volumes[0].size()[2:4],
                        mode='bilinear',
                        align_corners=False)

                mlvl_volume = mlvl_volume.unsqueeze(-1)
                mlvl_volumes[i] = mlvl_volume
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)

        if profile:
            profile_merge = self._profile_now() - profile_last
            profile_last = self._profile_now()
        else:
            profile_merge = 0.0

        x = mlvl_volumes
        def _inner_forward(x):
            out = self.neck_3d(x)
            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        if profile:
            profile_neck3d = self._profile_now() - profile_last
            profile_total = self._profile_now() - profile_t0
            # 输出单行结构化耗时，方便直接从训练日志中定位瓶颈阶段。
            print(
                '[FASTBEV_PROFILE extract_feat #{:03d}] total={:.3f}s backbone_neck_fuse={:.3f}s '
                'volume_total={:.3f}s meta={:.3f}s projection={:.3f}s points={:.3f}s '
                'backproject={:.3f}s stack_cat={:.3f}s merge={:.3f}s neck3d={:.3f}s '
                'batch={} total_views={} n_images={} use_distortion={}'.format(
                    self._profile_extract_iter, profile_total, profile_backbone,
                    profile_volume_total, profile_meta, profile_projection, profile_points,
                    profile_backproject, profile_stack, profile_merge, profile_neck3d,
                    batch_size, total_views if 'total_views' in locals() else 'na',
                    self.n_images, self.use_distortion))
            self._profile_extract_iter += 1

        return x, None, features_2d

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            if kwargs.get("export_2d", False):
                return self.onnx_export_2d(img, img_metas)
            elif kwargs.get("export_3d", False):
                return self.onnx_export_3d(img, img_metas)
            else:
                raise NotImplementedError

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(
        self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs
    ):
        # forward_train 级 profile 用来确认耗时是在特征提取、检测头前向，还是 loss 计算。
        profile = self._profile_enabled() and self._profile_train_iter < self._profile_limit()
        profile_t0 = self._profile_now() if profile else None
        feature_bev, valids, features_2d = self.extract_feat(img, img_metas, "train")
        if profile:
            profile_extract = self._profile_now() - profile_t0
            # 后续 stage 以 extract_feat 结束时间为起点，继续拆分 bbox_head 与 bbox_loss。
            profile_stage = self._profile_now()
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            if profile:
                profile_bbox_head = self._profile_now() - profile_stage
                profile_stage = self._profile_now()
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            if profile:
                profile_bbox_loss = self._profile_now() - profile_stage
                profile_stage = self._profile_now()
            losses.update(loss_det)
        else:
            profile_bbox_head = 0.0
            profile_bbox_loss = 0.0

        if self.seg_head is not None:
            assert len(gt_bev_seg) == 1
            x_bev = self.seg_head(feature_bev)
            gt_bev = gt_bev_seg[0][None, ...].long()
            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        if self.bbox_head_2d is not None:
            gt_bboxes = kwargs["gt_bboxes"][0]
            gt_labels = kwargs["gt_labels"][0]
            assert len(kwargs["gt_bboxes"]) == 1 and len(kwargs["gt_labels"]) == 1
            # hack a img_metas_2d
            img_metas_2d = []
            img_info = img_metas[0]["img_info"]
            for idx, info in enumerate(img_info):
                tmp_dict = dict(
                    filename=info["filename"],
                    ori_filename=info["filename"].split("/")[-1],
                    ori_shape=img_metas[0]["ori_shape"],
                    img_shape=img_metas[0]["img_shape"],
                    pad_shape=img_metas[0]["pad_shape"],
                    scale_factor=img_metas[0]["scale_factor"],
                    flip=False,
                    flip_direction=None,
                )
                img_metas_2d.append(tmp_dict)

            rank, world_size = get_dist_info()
            loss_2d = self.bbox_head_2d.forward_train(
                features_2d, img_metas_2d, gt_bboxes, gt_labels
            )
            losses.update(loss_2d)

        if profile:
            profile_total = self._profile_now() - profile_t0
            # 与 extract_feat 的 profile 行配套输出，便于同一 iteration 对齐分析。
            print(
                '[FASTBEV_PROFILE forward_train #{:03d}] total={:.3f}s extract={:.3f}s '
                'bbox_head={:.3f}s bbox_loss={:.3f}s batch={} img_shape={}'.format(
                    self._profile_train_iter, profile_total, profile_extract,
                    profile_bbox_head, profile_bbox_loss, img.shape[0], tuple(img.shape)))
            self._profile_train_iter += 1

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        test_cfg = self.test_cfg or {}
        if not test_cfg.get('use_tta', False):
            return self.simple_test(
                img, img_metas, mode=test_cfg.get('test_mode', 'test'))
        return self.aug_test(img, img_metas)

    def onnx_export_2d(self, img, img_metas):
        """
        input: 6, 3, 544, 960
        output: 6, 64, 136, 240
        """
        x = self.backbone(img)
        c1, c2, c3, c4 = self.neck(x)
        c2 = self._resize_feature(c2, size=c1.size()[2:])
        c3 = self._resize_feature(c3, size=c1.size()[2:])
        c4 = self._resize_feature(c4, size=c1.size()[2:])
        x = torch.cat([c1, c2, c3, c4], dim=1)
        neck_fuse = getattr(self, 'neck_fuse_0', getattr(self, 'neck_fuse', None))
        if neck_fuse is None:
            raise RuntimeError('onnx_export_2d 需要 neck_fuse 或 neck_fuse_0')
        x = neck_fuse(x)

        if bool(os.getenv("DEPLOY", False)):
            x = x.permute(0, 2, 3, 1)
            return x

        return x

    def onnx_export_3d(self, x, _):
        if isinstance(x, (list, tuple)):
            x = torch.cat(list(x), dim=1)

        if self.style in ["v1", "v3"]:
            x = self.neck_3d(x)
        elif self.style == "v2":
            x = self.neck_3d(x)
            x = [x[0].sum(dim=0, keepdim=True)]
        else:
            raise NotImplementedError

        if self.bbox_head is not None:
            cls_score, bbox_pred, dir_cls_preds = self.bbox_head(x)
            cls_score = [item.sigmoid() for item in cls_score]

        if os.getenv("DEPLOY", False):
            if dir_cls_preds is None:
                x = [cls_score, bbox_pred]
            else:
                x = [cls_score, bbox_pred, dir_cls_preds]
            return x

        return x

    def simple_test(self, img, img_metas, mode='test'):
        bbox_results = []
        feature_bev, _, features_2d = self.extract_feat(img, img_metas, mode)
        if self.bbox_head is not None:
            if mode in ['test_onnx', 'test_custom']:
                x = feature_bev
            else:
                x = self.bbox_head(feature_bev)
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list
            ]

        else:
            bbox_results = [dict()]

        # BEV semantic seg
        if self.seg_head is not None:
            x_bev = self.seg_head(feature_bev)
            bbox_results[0]['bev_seg'] = x_bev

        return bbox_results

    def aug_test(self, imgs, img_metas):
        x_list = []
        img_metas_list = []
        n_tta = 2
        total_views = len(img_metas[0]['lidar2img']['extrinsic'])
        assert total_views % n_tta == 0, 'TTA 输入视角数必须能按增强次数整除'
        n_tta_imgs = total_views // n_tta
        for tta_id in range(n_tta):
            start = n_tta_imgs * tta_id
            end = n_tta_imgs * (tta_id + 1)
            cur_img_metas = [
                self._slice_view_meta(img_meta, start, end)
                for img_meta in img_metas
            ]
            img_metas_list.append(cur_img_metas)

            feature_bev, _, _ = self.extract_feat(
                imgs[:, start:end], cur_img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ],
            indexing='ij'
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


def _flatten_distortion(value, device, dtype):
    if value is None:
        return None
    coeffs = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if coeffs.numel() == 0:
        return None
    return coeffs


def _camera_distortion(img_meta, cam_id, device, dtype):
    lidar2img = img_meta.get('lidar2img', {})
    extra = lidar2img.get('lidar2img_extra', [])
    aug = lidar2img.get('lidar2img_aug', [])
    candidates = []
    if isinstance(extra, list) and cam_id < len(extra):
        candidates.append(extra[cam_id].get('distortion') if isinstance(extra[cam_id], dict) else None)
    if isinstance(aug, list) and cam_id < len(aug):
        candidates.append(aug[cam_id].get('distortion') if isinstance(aug[cam_id], dict) else None)
    for item in candidates:
        coeffs = _flatten_distortion(item, device, dtype)
        if coeffs is not None:
            return coeffs
    return None


def _project_points_with_distortion(points, projection, img_meta, stride):
    """使用相机畸变参数把 lidar 体素点投影到特征图坐标。"""
    if img_meta is None:
        return torch.bmm(projection, points)
    lidar2img = img_meta.get('lidar2img', {})
    aug_infos = lidar2img.get('lidar2img_aug', [])
    if not isinstance(aug_infos, list) or len(aug_infos) < points.shape[0]:
        return torch.bmm(projection, points)

    outputs = []
    device = points.device
    dtype = points.dtype
    eye3 = torch.eye(3, device=device, dtype=dtype)
    zero3 = torch.zeros(3, device=device, dtype=dtype)
    for cam_id in range(points.shape[0]):
        aug = aug_infos[cam_id]
        if not isinstance(aug, dict) or 'rot' not in aug or 'tran' not in aug or 'intrin' not in aug:
            outputs.append(projection[cam_id] @ points[cam_id])
            continue
        distortion = _camera_distortion(img_meta, cam_id, device, dtype)
        if distortion is None:
            outputs.append(projection[cam_id] @ points[cam_id])
            continue

        sensor2lidar_r = torch.as_tensor(aug['rot'], device=device, dtype=dtype).reshape(3, 3)
        sensor2lidar_t = torch.as_tensor(aug['tran'], device=device, dtype=dtype).reshape(3, 1)
        intrinsic = torch.as_tensor(aug['intrin'], device=device, dtype=dtype).reshape(3, 3)
        post_rot = torch.as_tensor(aug.get('post_rot', eye3), device=device, dtype=dtype).reshape(3, 3)
        post_tran = torch.as_tensor(aug.get('post_tran', zero3), device=device, dtype=dtype).reshape(3, 1)

        lidar_points = points[cam_id, :3]
        lidar2cam_r = torch.inverse(sensor2lidar_r)
        cam_points = lidar2cam_r @ (lidar_points - sensor2lidar_t)
        z = cam_points[2].clamp(min=1e-5)
        x_c = cam_points[0] / z
        y_c = cam_points[1] / z
        r2 = x_c * x_c + y_c * y_c

        if distortion.numel() >= 8:
            k1, k2, p1, p2, k3, k4, k5, k6 = distortion[:8]
        else:
            padded = torch.zeros(8, device=device, dtype=dtype)
            padded[:min(distortion.numel(), 5)] = distortion[:min(distortion.numel(), 5)]
            k1, k2, p1, p2, k3, k4, k5, k6 = padded

        radial_num = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        radial_den = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        x_distorted = x_c * radial_num / radial_den + (2 * p1 * x_c * y_c + p2 * (r2 + 2 * x_c ** 2))
        y_distorted = y_c * radial_num / radial_den + (p1 * (r2 + 2 * y_c ** 2) + 2 * p2 * x_c * y_c)

        distorted = torch.stack((x_distorted, y_distorted, torch.ones_like(x_distorted)))
        pixel = intrinsic @ distorted
        pixel = pixel[:3] / pixel[2:3].clamp(min=1e-5)
        pixel = post_rot @ pixel + post_tran
        pixel_xy = pixel[:2] / float(stride)
        outputs.append(torch.stack((pixel_xy[0] * cam_points[2], pixel_xy[1] * cam_points[2], cam_points[2])))
    return torch.stack(outputs)


def backproject_inplace(features, points, projection, img_meta=None, seq_id=0, stride=1, use_distortion=False):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    if use_distortion:
        points_2d_3 = _project_points_with_distortion(points, projection, img_meta, stride)
    else:
        points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume
