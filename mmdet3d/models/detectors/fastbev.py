# -*- coding: utf-8 -*-
import math
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import auto_fp16

import copy
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@DETECTORS.register_module()
class FastBEV(BaseDetector):
    """Fast-BEV detector with N7 6V training/deployment adaptations.

    当前板端只按 ``style='v1'`` 的 R18 6V 配置打通。相比厂家
    ``fastbev_bst.py`` 的临时实现，这里保留论文训练主线的通用结构：
    视角数使用 ``self.n_images``，校准数据由专用工具打开开关，v1/v2 的
    3D head 输入在 detector 内提前折成板端需要的 4D BEV 张量。

    和 ``fastbev_bst.py`` 的主要差异：
    - 不硬编码 6 个相机。所有时序切片都基于 ``self.n_images``，所以 6V
      和未来 1V/其他相机数可以共用同一套逻辑。
    - 不在通用 test API 中按间隔打开校准开关。校准由
      ``tools/generate_calibrate_data.py`` 强制 batch=1 后触发。
    - 不复制 ``extract_feat_custom`` 的整段实现。``test_onnx`` 和
      ``test_custom`` 共享同一条 metadata slicing/backproject 逻辑，只在
      ONNXRuntime session 是否注册 custom op 上有区别。
    - 支持 N7 相机畸变参数、BEV points 缓存、fused final-camera scatter。
    """

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
        # 厂家 demo 在多处写死 6 视角和 seq_id*6；当前主线把每个时序的相机数
        # 提成配置项。6V N7 设置为 6，单目前视时序实验可设置为 1。
        self.n_images = n_images
        # 厂家 demo 的 FPN 多尺度 resize 固定 nearest；当前保留可配置项，默认
        # bilinear 更接近常规特征对齐写法，必要时可在 config 中切回 nearest。
        self.feature_resize_mode = feature_resize_mode
        assert self.feature_resize_mode in ['bilinear', 'nearest'], self.feature_resize_mode
        # N7 pkl 可写入 distortion；开启后 backproject 使用动态畸变投影路径。
        # 这比 pinhole 快速路径更慢，但几何更接近原始相机模型。
        self.use_distortion = use_distortion
        # 校准数据保存相关参数只在 generate_calibrate_data.py 打开。
        # 普通 train/test 不应依赖这些字段，也不应修改 mmdet3d/apis/test.py。
        self.calibration_save_dir = calibration_save_dir
        self.max_calibration_samples = max_calibration_samples
        self.onnx_custom_op_path = onnx_custom_op_path
        self._calibration_samples_saved = 0
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
        """根据当前时序帧的 meta 计算特征图尺度下的 projection。

        ``lidar2img['extrinsic']`` 已经由数据 pipeline 写入图像增强后的
        projection；这里仅把相机内参除以 FPN stride。``fastbev_bst.py``
        也使用同一公式，但它只切 ``extrinsic``，当前主线会同步切
        ``lidar2img_aug/lidar2img_extra/img_shape/img_info`` 等字段。
        """
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
        """初始化 ONNXRuntime session。

        厂家 demo 固定加载当前目录下的 ``bstnnx_cpp2py_export...so``。
        当前主线先尝试普通 CPUExecutionProvider，失败后再读取
        ``test_cfg['onnx_custom_op_path']`` 或构造参数 ``onnx_custom_op_path``。
        这样普通 ONNX 和 custom-op ONNX 可以共用同一份代码。
        """
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
        """切出一个时序片段对应的相机 meta。

        Fast-BEV 输入按 ``[当前时刻所有相机, prev1 所有相机, ...]`` 排列。
        厂家 demo 用 ``seq_id*6:(seq_id+1)*6`` 只处理 6V；这里由调用方传入
        ``start/end``，因此由 ``self.n_images`` 控制。除了 ``extrinsic``，
        N7 动态畸变和 force_resize 还依赖 ``lidar2img_aug``、
        ``lidar2img_extra``、``img_shape`` 等字段，必须一起切片。
        """
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
        """统一 FPN 多尺度特征 resize。

        ``fastbev_bst.py`` 在多尺度融合和导出中直接写 ``nearest``。
        当前主线把 resize mode 做成配置，便于保持 paper 配置或做速度 A/B。
        """
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

    def _extract_onnx_backbone(self, img):
        """用 ONNX backbone 逐图提取 2D 特征。

        ONNXRuntime session 的输入是单张图片，因此这里和厂家 demo 一样逐张
        调 session，再把输出按 batch/view 维拼回 PyTorch tensor。和
        ``fastbev_bst.py`` 不同的是，当前 ``test_onnx`` 和 ``test_custom``
        后续共用 ``extract_feat`` 的同一套 2D->3D 逻辑。
        """
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
        """把 Fast-BEV 5D volume 折成板端 3D head 使用的 4D BEV 输入。

        paper 训练链路原本可以把 ``[bs, c, x, y, z]`` 交给
        ``M2BevNeck`` 再折叠；厂家 ``fastbev_bst.py`` 为了导出/部署把
        折叠提前到了 detector。当前代码沿用这个部署约定，但保留
        ``M2BevNeck`` 的 5D 兼容分支，避免破坏其他 style/config。
        """
        bs, channels, size_x, size_y, size_z = volume.shape
        return volume.permute(0, 4, 1, 2, 3).reshape(
            bs, size_z * channels, size_x, size_y)

    def _onnx_head_forward(self, deploy_head_inputs, device):
        """执行板端 3D head ONNX，并转回 bbox_head.get_bboxes 所需格式。"""
        if self.head_session is None:
            raise RuntimeError('ONNX head session 尚未初始化')
        input_data = [x.detach().cpu().float().numpy() for x in deploy_head_inputs]
        outputs = self._onnx_infer(self.head_session, input_data)
        return tuple([torch.from_numpy(out).to(device)] for out in outputs)

    def _save_backbone_calibration(self, img_metas):
        test_cfg = self.test_cfg or {}
        if not test_cfg.get('save_calibrate_data_flag', False):
            return
        # backbone/head 共用同一个样本计数，避免校准脚本误拷全量数据。
        # ``tools/generate_calibrate_data.py`` 会强制 samples_per_gpu=1；
        # 若这里传入 batch>1，img_metas[0] 只能代表 batch 内第一条样本。
        if self._calibration_samples_saved >= self.max_calibration_samples:
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
        # v1/R18 板端 head ONNX 接收 4 个时序 BEV 输入；每个输入形状为
        # [1, z*c, x, y]。因此校准导出必须保持 batch=1，避免把 batch 维
        # 混进单个 npy，导致量化工具按错误的激活分布统计。
        for idx, tensor in enumerate(deploy_head_inputs):
            out_dir = os.path.join(save_dir, str(idx))
            os.makedirs(out_dir, exist_ok=True)
            np.save(
                os.path.join(out_dir, f'{calibrate_id}_{idx}.npy'),
                tensor.detach().cpu().float().numpy())
        self._calibration_samples_saved += 1

    def extract_feat(self, img, img_metas, mode):
        """提取 BEV 特征或直接返回 ONNX head 输出。

        这是一条统一主线：
        - ``train/test/test_pth`` 使用 PyTorch backbone + PyTorch head；
        - ``test_onnx/test_custom`` 使用 ONNX backbone + ONNX 3D head；
        - 量化校准使用 ``test_pth``，但额外保存 backbone 图片和 3D head
          输入。厂家 demo 为 ``test_custom`` 复制了一份 ``extract_feat_custom``，
          当前主线避免复制，减少 6V hard-code 和 metadata 切片不一致风险。
        """
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

        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            # v3/v4 可能声明多个 BEV scale；当 2D ONNX 只导出一个尺度时，
            # 沿用 paper/demo 的行为，用第一个特征补齐后续尺度。
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        mlvl_volumes = []
        deploy_head_inputs = []
        test_cfg = self.test_cfg or {}
        need_save_head_calibration = (
            test_cfg.get('save_calibrate_data_flag', False)
            and self._calibration_samples_saved < self.max_calibration_samples
        )
        need_deploy_head_inputs = (
            mode in ['test_onnx', 'test_custom']
            or need_save_head_calibration
        )
        # deploy_head_inputs 只服务 ONNX/custom-op 推理和量化校准。普通训练不
        # 额外保留这组中间张量，避免显存和 Python 对象开销。
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
            # split 的宽度由 self.n_images 决定。N7 6V 时每段是 6 张图；
            # 单目前视时序时每段是 1 张图。
            mlvl_feat_split = torch.split(mlvl_feat, self.n_images, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]
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

                    # projection 由当前时序帧的 lidar2img 外参、图像增强矩阵和特征 stride 共同决定。
                    projection = self._compute_projection(
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
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

                    if self.backproject == 'inplace':
                        # 当前主线的 inplace backproject 支持 pinhole 快速路径和
                        # N7 动态畸变路径；厂家 demo 只有 pinhole projection。
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

                    volumes.append(volume)
                seq_volume = torch.stack(volumes)
                if self.style in ['v1', 'v2']:
                    # 当前 N7 板端只实现 v1/R18。这里不再使用
                    # ``fastbev_bst.py`` 中硬编码 6 视角/4 时序的 reshape，而是：
                    # 1. 每个时序 volume 先折成 [bs, z*c, x, y]；
                    # 2. 后续按时序 concat 成 [bs, t*z*c, x, y]；
                    # 3. 训练、3D ONNX 导出和 head 校准看到同一套 4D 布局。
                    seq_volume = self._volume_to_deploy_input(seq_volume)
                    if lvl == 0 and need_deploy_head_inputs:
                        deploy_head_inputs.append(seq_volume)
                elif lvl == 0 and need_deploy_head_inputs:
                    deploy_head_inputs.append(self._volume_to_deploy_input(seq_volume))
                volume_list.append(seq_volume)

            mlvl_volumes.append(torch.cat(volume_list, dim=1))

        if mode in ['test_onnx', 'test_custom']:
            # ONNX/custom-op 推理模式下，extract_feat 的返回值已经是 bbox_head
            # 的三个输出 tuple，不再经过 PyTorch bbox_head。
            return self._onnx_head_forward(deploy_head_inputs, img.device), None, None

        if need_save_head_calibration:
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

        x = mlvl_volumes
        def _inner_forward(x):
            # v1/v2 到这里已经是 4D BEV 特征；M2BevNeck 仍保留 5D fallback，
            # 用于兼容 paper 原始路径或未来非 v1/R18 配置。
            out = self.neck_3d(x)
            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

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
            # 厂家 demo 通过输入 tensor shape 判断导出 2D/3D，这对尺寸变化敏感。
            # 当前主线改为显式 kwargs，导出脚本需传 export_2d/export_3d。
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
        feature_bev, valids, features_2d = self.extract_feat(img, img_metas, "train")
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses.update(loss_det)

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

            loss_2d = self.bbox_head_2d.forward_train(
                features_2d, img_metas_2d, gt_bboxes, gt_labels
            )
            losses.update(loss_2d)

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

        只导出 backbone + FPN + neck_fuse。厂家板端 demo 在 2D ONNX 导出
        中固定使用 nearest resize；这里保持这个导出约定，避免导出的 2D
        backbone 和芯片侧已验证算子/数值路径不一致。训练/普通 pth 推理仍由
        ``feature_resize_mode`` 控制，如需严格做 pth-vs-ONNX 数值对齐，应在
        config 中同步设置 ``model.feature_resize_mode='nearest'``。
        """
        x = self.backbone(img)
        c1, c2, c3, c4 = self.neck(x)
        c2 = resize(c2, size=c1.size()[2:], mode="nearest")
        c3 = resize(c3, size=c1.size()[2:], mode="nearest")
        c4 = resize(c4, size=c1.size()[2:], mode="nearest")
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
        """导出 3D neck + bbox head。

        输入可以是一个原生单帧或多个时序 BEV tensor 的 list/tuple，也可以
        是已经 concat 好的 4D tensor。导出脚本负责按 config 传入正确数量的
        输入，函数内部只按 channel 维拼接；当前产品约束正式支持 n_times=1
        和 n_times=4。

        注意这里返回的是 bbox head 原始输出，不对 ``cls_score`` 做 sigmoid。
        PyTorch 后处理 ``bbox_head.get_bboxes`` 和厂家板端后处理都会自己做
        sigmoid/topk/NMS；如果 ONNX 内提前 sigmoid，test_onnx 再走
        ``get_bboxes`` 会发生二次 sigmoid，板端也会和厂家 demo 输出不一致。
        """
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
            if dir_cls_preds is None:
                return [cls_score, bbox_pred]
            return [cls_score, bbox_pred, dir_cls_preds]

        return x

    def simple_test(self, img, img_metas, mode='test'):
        """单尺度推理入口。

        ``test_onnx`` 和 ``test_custom`` 在当前主线都返回 ONNX head 输出，
        再统一走 ``bbox_head.get_bboxes``。厂家 demo 的 ``test_custom`` 会
        调外部 ``tools.utils.get_bboxes``，这和 PyTorch 后处理可能不完全一致；
        当前训练可用版本优先保持项目内 bbox_head 后处理一致。
        """
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
        """TTA 推理。

        厂家 demo 按 ``24 * tta_id`` 切片，隐含 6V * 4 times。当前主线先从
        meta 中读取总视角数，再按 TTA 次数均分，所以不再绑定 6V。
        """
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
    """把 pkl/meta 中的畸变参数转为当前设备上的 1D tensor。

    ``fastbev_bst.py`` 不读取畸变参数，只使用已经合成好的 pinhole
    projection。N7 当前 pkl 会保留相机 distortion，因此主线需要这组辅助函数
    在 backproject 时按真实相机模型重新投影 BEV 点。
    """
    if value is None:
        return None
    coeffs = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if coeffs.numel() == 0:
        return None
    return coeffs


def _camera_distortion(img_meta, cam_id, device, dtype):
    """从当前相机 meta 中取畸变参数。

    ``lidar2img_extra`` 和 ``lidar2img_aug`` 都可能携带 distortion；优先
    使用 extra，缺失时回退到 aug，保证 converter/dataset/pipeline 的字段
    调整不会让投影静默退回错误参数。
    """
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


def _project_points_with_distortion_loop(points, projection, img_meta, stride):
    """逐相机畸变投影 fallback。

    这是动态畸变路径的保底实现：逐相机取 ``sensor2lidar``、K、post_rot、
    post_tran 和 distortion，再完成 ``lidar -> camera -> distorted pixel``
    的投影。批量实现缺字段时会退回这里，便于定位单个相机 metadata 问题。
    """
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


def _project_points_with_distortion(points, projection, img_meta, stride):
    """使用相机畸变参数把 lidar 体素点投影到特征图坐标。

    这是当前 N7 主线相对 ``fastbev_bst.py`` 的核心几何差异之一。厂家 demo
    只做 ``projection @ [x,y,z,1]``，速度快但忽略 N7 相机畸变；当前实现把
    6 个相机参数 stack 后批量计算，减少 Python loop 开销。若任何相机缺少
    必要字段，则回退到逐相机 loop，避免错误地混用部分参数。
    """
    if img_meta is None:
        return torch.bmm(projection, points)
    lidar2img = img_meta.get('lidar2img', {})
    aug_infos = lidar2img.get('lidar2img_aug', [])
    n_images = points.shape[0]
    if not isinstance(aug_infos, list) or len(aug_infos) < n_images:
        return torch.bmm(projection, points)

    device = points.device
    dtype = points.dtype
    eye3 = torch.eye(3, device=device, dtype=dtype)
    zero3 = torch.zeros(3, device=device, dtype=dtype)
    sensor2lidar_rs = []
    sensor2lidar_ts = []
    intrinsics = []
    post_rots = []
    post_trans = []
    distortions = []
    for cam_id in range(n_images):
        aug = aug_infos[cam_id]
        if not isinstance(aug, dict) or 'rot' not in aug or 'tran' not in aug or 'intrin' not in aug:
            return _project_points_with_distortion_loop(points, projection, img_meta, stride)
        distortion = _camera_distortion(img_meta, cam_id, device, dtype)
        if distortion is None:
            return _project_points_with_distortion_loop(points, projection, img_meta, stride)

        sensor2lidar_rs.append(torch.as_tensor(aug['rot'], device=device, dtype=dtype).reshape(3, 3))
        sensor2lidar_ts.append(torch.as_tensor(aug['tran'], device=device, dtype=dtype).reshape(3, 1))
        intrinsics.append(torch.as_tensor(aug['intrin'], device=device, dtype=dtype).reshape(3, 3))
        post_rots.append(torch.as_tensor(aug.get('post_rot', eye3), device=device, dtype=dtype).reshape(3, 3))
        post_trans.append(torch.as_tensor(aug.get('post_tran', zero3), device=device, dtype=dtype).reshape(3, 1))

        padded = torch.zeros(8, device=device, dtype=dtype)
        if distortion.numel() >= 8:
            padded[:] = distortion[:8]
        else:
            padded[:min(distortion.numel(), 5)] = distortion[:min(distortion.numel(), 5)]
        distortions.append(padded)

    sensor2lidar_r = torch.stack(sensor2lidar_rs)
    sensor2lidar_t = torch.stack(sensor2lidar_ts)
    intrinsic = torch.stack(intrinsics)
    post_rot = torch.stack(post_rots)
    post_tran = torch.stack(post_trans)
    distortion = torch.stack(distortions)

    lidar2cam_r = torch.inverse(sensor2lidar_r)
    cam_points = torch.bmm(lidar2cam_r, points[:, :3] - sensor2lidar_t)
    z_depth = cam_points[:, 2]
    z = z_depth.clamp(min=1e-5)
    x_c = cam_points[:, 0] / z
    y_c = cam_points[:, 1] / z
    r2 = x_c * x_c + y_c * y_c

    k1 = distortion[:, 0:1]
    k2 = distortion[:, 1:2]
    p1 = distortion[:, 2:3]
    p2 = distortion[:, 3:4]
    k3 = distortion[:, 4:5]
    k4 = distortion[:, 5:6]
    k5 = distortion[:, 6:7]
    k6 = distortion[:, 7:8]
    radial_num = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    radial_den = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
    x_distorted = x_c * radial_num / radial_den + (2 * p1 * x_c * y_c + p2 * (r2 + 2 * x_c ** 2))
    y_distorted = y_c * radial_num / radial_den + (p1 * (r2 + 2 * y_c ** 2) + 2 * p2 * x_c * y_c)

    distorted = torch.stack((x_distorted, y_distorted, torch.ones_like(x_distorted)), dim=1)
    pixel = torch.bmm(intrinsic, distorted)
    pixel = pixel / pixel[:, 2:3].clamp(min=1e-5)
    pixel = torch.bmm(post_rot, pixel) + post_tran
    pixel_xy = pixel[:, :2] / float(stride)
    return torch.stack(
        (pixel_xy[:, 0] * z_depth, pixel_xy[:, 1] * z_depth, z_depth),
        dim=1)


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
    n_points = points.shape[-1]
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    if use_distortion:
        # 动态畸变路径需要齐次点，因为内部会重新执行 lidar->camera->pixel
        # 投影并套用 post_rot/post_tran；这对应 N7 真实相机模型。
        points_homo = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
        points_2d_3 = _project_points_with_distortion(points_homo, projection, img_meta, stride)
    else:
        # 无畸变路径避免每次构造齐次坐标，直接执行 R @ xyz + t。
        # 语义等价于厂家 demo 的 projection @ homogeneous points，但少一次
        # 拼接和 4x4 矩阵乘法，适合作为 pinhole 快速路径。
        points_2d_3 = torch.bmm(projection[:, :, :3], points) + projection[:, :, 3:4]
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, n_points), device=features.device
    ).type_as(features)
    # 等价于原先按相机顺序逐个写入、后写覆盖的规则，但每个 BEV 点只写一次。
    # ``fastbev_bst.py`` 逐相机循环 scatter，重复覆盖同一个 BEV 点；当前实现
    # 先求出最终会胜出的相机 index，再一次 gather/scatter。覆盖语义保持
    # “自然相机顺序最后一个有效相机胜出”，但运行开销更低。
    camera_order = torch.arange(
        n_images, device=features.device, dtype=torch.long).view(n_images, 1)
    camera_choice = torch.where(
        valid, camera_order, torch.full_like(camera_order, -1)).max(dim=0).values
    assigned = camera_choice >= 0
    if assigned.any():
        dst_index = assigned.nonzero(as_tuple=False).squeeze(1)
        src_cam = camera_choice[dst_index]
        sampled = features[src_cam, :, y[src_cam, dst_index], x[src_cam, dst_index]]
        volume[:, dst_index] = sampled.transpose(0, 1)

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume
